from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import torch

from mptorch import (
    AccumulateAlgorithm,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
)

__all__ = [
    "binaryK_quantize",
    "superfp_quantize",
    "binaryK_matmul",
    "superfp_matmul",
    "binaryK_matmul_fma",
    "superfp_matmul_fma",
    "binaryK_matmul_mixed",
    "superfp_matmul_mixed",
    "binaryK_matmul_fma_mixed",
    "superfp_matmul_fma_mixed",
]

mantissa_size_mapping: dict[torch.dtype, int] = {
    torch.bfloat16: 7,
    torch.float16: 10,
    torch.float32: 23,
    torch.float64: 52,
}


# --- format defaulting, written once ----------------------------------------
#
# The wrappers below all resolve the same handful of things: the exponent bias
# a binaryK format defaults to, the accumulate format's fallback to the
# multiply format's, and whether a stochastic-rounding request fits inside the
# storage dtype. The palette (mixed) wrappers do the same per entry. What
# follows is that logic in one place, so the eight GEMM wrappers are a
# signature, a docstring and the schema's argument order.


def _binaryK_bias(K: int, P: int, is_signed: bool) -> int:
    """binaryK's default exponent bias: the middle of the exponent range."""
    return 2 ** (K - P - 1) if is_signed else 2 ** (K - P)


def _prng_terms(*terms: tuple[str, int, int]) -> tuple[int, tuple[tuple[str, int, int], ...]]:
    """Fold `(name, prng_bits, man_bits)` constraints into one storage-width bound.

    Stochastic rounding draws its random bits from the storage mantissa *below*
    the target format's, so a `man_bits`-wide format asking for `prng_bits` of
    noise needs a storage dtype carrying at least their sum. The lower bound on
    `prng_bits` is checked here, since it depends on nothing but the format;
    the sum is what the call itself checks against its operand dtype.
    """
    for name, prng_bits, _ in terms:
        assert prng_bits >= 0, f"{name} must be non-negative, got {prng_bits}"
    return max(p + m for _, p, m in terms), terms


def _prng_overflow(terms: tuple[tuple[str, int, int], ...], dtype: torch.dtype) -> str:
    """The message for a `_prng_terms` bound the storage dtype cannot meet.

    Only ever called on the failing branch of an assert, so the formatting is
    free on the path that matters.
    """
    have = mantissa_size_mapping[dtype]
    asked = "; ".join(f"{name}={p} under a {m}-bit mantissa" for name, p, m in terms)
    return (
        f"{dtype} carries {have} mantissa bits, too few for {asked}: a format's "
        "mantissa and its stochastic-rounding bits have to fit in the storage dtype together"
    )


def _assert_prng_fits(dtype: torch.dtype, name: str, prng_bits: int, man_bits: int) -> None:
    """`_prng_terms` and its storage-dtype check together, for a single format."""
    needed, terms = _prng_terms((name, prng_bits, man_bits))
    assert needed <= mantissa_size_mapping[dtype], _prng_overflow(terms, dtype)


def _palette_list(val: int | Sequence[int], n: int, name: str) -> list[int]:
    """Broadcast a scalar to an n-length list, or validate a sequence's length."""
    if isinstance(val, int):
        return [val] * n
    out = list(val)
    if len(out) != n:
        raise ValueError(f"{name} must have length {n} (the palette size), got {len(out)}")
    return out


def _palette_list_or(
    val: int | Sequence[int] | None, default: list[int], n: int, name: str
) -> list[int]:
    """:func:`_palette_list`, but ``None`` falls back to ``default``."""
    return default if val is None else _palette_list(val, n, name)


def _palette_pair(
    first: Sequence[int], second: Sequence[int], second_name: str, fn: str
) -> tuple[list[int], list[int], int]:
    """The two per-entry sequences that define a palette, as lists, with its size.

    Every mixed op takes one such pair -- `(K, P)` or `(man_bits, exp_bits)` --
    whose common length is the palette size every other palette argument is
    broadcast or defaulted against.
    """
    first_l, second_l = list(first), list(second)
    n = len(first_l)
    if n < 1:
        raise ValueError(f"{fn} needs at least one palette format")
    if len(second_l) != n:
        raise ValueError(
            f"{second_name} must have length {n} (the palette size), got {len(second_l)}"
        )
    return first_l, second_l, n


def _binaryK_palette_bias(
    bias: int | Sequence[int] | None, K_l: list[int], P_l: list[int], is_signed: bool, name: str
) -> list[int]:
    """:func:`_binaryK_bias` per palette entry, or the given scalar/sequence broadcast."""
    if bias is None:
        return [_binaryK_bias(k, p, is_signed) for k, p in zip(K_l, P_l, strict=True)]
    return _palette_list(bias, len(K_l), name)


# --- a GEMM call, resolved ---------------------------------------------------


class _GemmSpec(NamedTuple):
    """One GEMM's formats resolved into exactly what the op takes.

    `args` is every schema argument after the operands and the transpose
    flags, in order. `needed` is the narrowest storage mantissa that admits
    the stochastic-rounding request, and `terms` is what that maximum came
    from -- kept unformatted because it is only ever read on failure.

    The split exists so a caller with a fixed format can resolve once and call
    many times: `mptorch.quant.gemm`'s factories build a spec per layer and
    bind it into the layer's math hooks, where the old code re-derived every
    default and re-ran every check on each forward and backward pass.
    """

    op: Callable[..., torch.Tensor]
    args: tuple[Any, ...]
    needed: int
    terms: tuple[tuple[str, int, int], ...]


def _run_gemm(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """Call a resolved GEMM. All that is left per call is the operand dtype."""
    assert spec.needed <= mantissa_size_mapping[a.dtype], _prng_overflow(spec.terms, a.dtype)
    return spec.op(a.contiguous(), b.contiguous(), trans_a, trans_b, *spec.args)


def _run_gemm_mixed(
    spec: _GemmSpec,
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    """:func:`_run_gemm` for the palette ops, whose schema takes `prec_idx` third.

    `prec_idx` is passed through as given: narrowing and packing it is the
    C++ side's job, and it memoizes the bounds check against the tensor it is
    handed (finding G5b), which a `.to()` here would defeat.
    """
    assert spec.needed <= mantissa_size_mapping[a.dtype], _prng_overflow(spec.terms, a.dtype)
    return spec.op(a.contiguous(), b.contiguous(), prec_idx, trans_a, trans_b, *spec.args)


# --- elementwise quantizers --------------------------------------------------


def binaryK_quantize(
    x: torch.Tensor,
    K: int,
    P: int,
    bias: int | None = None,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> torch.Tensor:
    _assert_prng_fits(x.dtype, "prng_bits", prng_bits, P - 1)

    if not bias:
        bias = _binaryK_bias(K, P, is_signed)

    return torch.ops.mptorch.binaryK_quant.default(
        x.contiguous(),
        K,
        P,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
    )


def superfp_quantize(
    x: torch.Tensor,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    _assert_prng_fits(x.dtype, "prng_bits", prng_bits, man_bits)

    return torch.ops.mptorch.superfp_quant.default(
        x.contiguous(),
        man_bits,
        exp_bits,
        normal_binades,
        bias,
        prng_bits,
        is_signed,
        rounding_mode.value,
        saturation_mode.value,
    )


# --- single-format GEMMs -----------------------------------------------------


def _binaryK_spec(
    *,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul`'s formats -- see it for the contract."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if accumulate_quant:
        acc_K = mul_K if acc_K is None else acc_K
        acc_P = mul_P if acc_P is None else acc_P
    else:
        acc_K = acc_K or 0
        acc_P = acc_P or 0

    if not mul_bias:
        mul_bias = _binaryK_bias(mul_K, mul_P, mul_is_signed)
    if accumulate_quant and not acc_bias:
        acc_bias = _binaryK_bias(acc_K, acc_P, acc_is_signed)
    elif acc_bias is None:
        acc_bias = 0

    terms = [("mul_prng_bits", mul_prng_bits, mul_P - 1)]
    if accumulate_quant:
        terms.append(("acc_prng_bits", acc_prng_bits, acc_P - 1))
    needed, terms_t = _prng_terms(*terms)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK.default,
        (
            mul_K,
            mul_P,
            mul_bias,
            mul_is_signed,
            accumulate_quant,
            acc_K,
            acc_P,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        needed,
        terms_t,
    )


def binaryK_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: int,
    mul_P: int,
    mul_bias: int | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> torch.Tensor:
    """
    Quantized GEMM core: computes ``op(a) @ op(b)``, where ``op(x) = x.T``
    if the corresponding ``trans_*`` flag is set, else ``op(x) = x``.

    Unlike quantizing ``a``/``b`` and then calling ``torch.matmul``, this
    quantizes the *arithmetic* of the dot product itself: every partial
    product is cast to the binaryK format given by ``mul_K``/``mul_P``, and
    (when ``accumulate_quant`` is true) the running sum of each dot product
    is additionally cast to the format given by ``acc_K``/``acc_P`` after
    every accumulation step. With ``accumulate_quant=False`` the running sum
    stays in full precision -- only the multiply is quantized.

    ``mul_prng_bits``/``acc_prng_bits`` only matter when ``rounding_mode``
    is ``RoundMode.SR`` (stochastic): they set the number of random
    mantissa bits used by the multiply/accumulate rounding respectively,
    same convention as :func:`binaryK_quantize`'s ``prng_bits``.

    ``a`` and ``b`` must be 2D; batched/rank>2 callers should flatten their
    leading dimensions first (see ``mptorch.quant.gemm`` for an example that
    does this for ``QAffineFormats``). Callers holding one format across many
    calls should use those factories rather than this function: they resolve
    the format once instead of on every call.
    """
    return _run_gemm(
        _binaryK_spec(
            mul_K=mul_K,
            mul_P=mul_P,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_K=acc_K,
            acc_P=acc_P,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
        ),
        a,
        b,
        trans_a,
        trans_b,
    )


def _superfp_spec(
    *,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul`'s formats -- see it for the contract."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if accumulate_quant:
        acc_man_bits = mul_man_bits if acc_man_bits is None else acc_man_bits
        acc_exp_bits = mul_exp_bits if acc_exp_bits is None else acc_exp_bits
        acc_normal_binades = (
            mul_normal_binades if acc_normal_binades is None else acc_normal_binades
        )
        acc_bias = mul_bias if acc_bias is None else acc_bias
    else:
        acc_man_bits = acc_man_bits or 0
        acc_exp_bits = acc_exp_bits or 0
        acc_normal_binades = acc_normal_binades or 0
        acc_bias = acc_bias or 0

    terms = [("mul_prng_bits", mul_prng_bits, mul_man_bits)]
    if accumulate_quant:
        terms.append(("acc_prng_bits", acc_prng_bits, acc_man_bits))
    needed, terms_t = _prng_terms(*terms)

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp.default,
        (
            mul_man_bits,
            mul_exp_bits,
            mul_normal_binades,
            mul_bias,
            mul_is_signed,
            accumulate_quant,
            acc_man_bits,
            acc_exp_bits,
            acc_normal_binades,
            acc_bias,
            acc_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        needed,
        terms_t,
    )


def superfp_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: int,
    mul_exp_bits: int,
    mul_normal_binades: int,
    mul_bias: int,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    """
    superfp analog of :func:`binaryK_matmul` -- see its docstring for the
    general contract (2D-only, ``trans_a``/``trans_b``,
    ``accumulate_quant``, ``mul_prng_bits``/``acc_prng_bits``).
    ``acc_man_bits``/``acc_exp_bits``/``acc_normal_binades``/``acc_bias``
    default to the multiply format's values when omitted and
    ``accumulate_quant`` is true.
    """
    return _run_gemm(
        _superfp_spec(
            mul_man_bits=mul_man_bits,
            mul_exp_bits=mul_exp_bits,
            mul_normal_binades=mul_normal_binades,
            mul_bias=mul_bias,
            mul_is_signed=mul_is_signed,
            mul_prng_bits=mul_prng_bits,
            accumulate_quant=accumulate_quant,
            acc_man_bits=acc_man_bits,
            acc_exp_bits=acc_exp_bits,
            acc_normal_binades=acc_normal_binades,
            acc_bias=acc_bias,
            acc_is_signed=acc_is_signed,
            acc_prng_bits=acc_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
        ),
        a,
        b,
        trans_a,
        trans_b,
    )


def _binaryK_fma_spec(
    *,
    fma_K: int,
    fma_P: int,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma`'s format -- see it for the contract."""
    if not fma_bias:
        fma_bias = _binaryK_bias(fma_K, fma_P, fma_is_signed)
    needed, terms = _prng_terms(("fma_prng_bits", fma_prng_bits, fma_P - 1))

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_binaryK_fma.default,
        (
            fma_quant,
            fma_K,
            fma_P,
            fma_bias,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            subnormals_mode.value,
            fma_prng_bits,
        ),
        needed,
        terms,
    )


def binaryK_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_K: int,
    fma_P: int,
    fma_bias: int | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> torch.Tensor:
    """
    Fused-multiply-add analog of :func:`binaryK_matmul`: every dot-product
    step computes a single hardware-style fused multiply-add (``a*b + acc``,
    one rounding) instead of quantizing the multiply and the accumulate
    separately (two roundings). There is no separate multiply format --  a
    real FMA unit only has one output rounding, so unlike
    :func:`binaryK_matmul` there is a single ``fma_K``/``fma_P`` format,
    not a ``mul_*``/``acc_*`` pair.

    With ``fma_quant=False`` the fused step runs in full fp32 precision (no
    rounding beyond fp32 itself) -- the FMA analog of
    :func:`binaryK_matmul`'s ``accumulate_quant=False``.

    ``fma_prng_bits`` only matters when ``rounding_mode`` is
    ``RoundMode.SR`` (stochastic) -- see :func:`binaryK_matmul`'s
    ``mul_prng_bits``/``acc_prng_bits`` for the convention.

    ``a`` and ``b`` must be 2D; batched/rank>2 callers should flatten their
    leading dimensions first (see ``mptorch.quant.gemm`` for an example that
    does this for ``QAffineFormats``).
    """
    return _run_gemm(
        _binaryK_fma_spec(
            fma_K=fma_K,
            fma_P=fma_P,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
            subnormals_mode=subnormals_mode,
        ),
        a,
        b,
        trans_a,
        trans_b,
    )


def _superfp_fma_spec(
    *,
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma`'s format -- see it for the contract."""
    needed, terms = _prng_terms(("fma_prng_bits", fma_prng_bits, fma_man_bits))

    return _GemmSpec(
        torch.ops.mptorch.custom_matmul_superfp_fma.default,
        (
            fma_quant,
            fma_man_bits,
            fma_exp_bits,
            fma_normal_binades,
            fma_bias,
            fma_is_signed,
            accumulate_algorithm.value,
            rounding_mode.value,
            saturation_mode.value,
            fma_prng_bits,
        ),
        needed,
        terms,
    )


def superfp_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_man_bits: int,
    fma_exp_bits: int,
    fma_normal_binades: int,
    fma_bias: int,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    """superfp analog of :func:`binaryK_matmul_fma` -- see its docstring."""
    return _run_gemm(
        _superfp_fma_spec(
            fma_man_bits=fma_man_bits,
            fma_exp_bits=fma_exp_bits,
            fma_normal_binades=fma_normal_binades,
            fma_bias=fma_bias,
            fma_is_signed=fma_is_signed,
            fma_quant=fma_quant,
            fma_prng_bits=fma_prng_bits,
            accumulate_algorithm=accumulate_algorithm,
            rounding_mode=rounding_mode,
            saturation_mode=saturation_mode,
        ),
        a,
        b,
        trans_a,
        trans_b,
    )


# --- palette (spatially-varying mixed-format) GEMMs --------------------------
#
# These resolve in the wrapper rather than in a separate builder like the four
# above: a `_GemmSpec` is worth splitting out only where something holds one
# across many calls, and no `QAffineFormats` factory takes a palette yet.


def binaryK_matmul_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_K: Sequence[int],
    mul_P: Sequence[int],
    mul_bias: int | Sequence[int] | None = None,
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_K: int | Sequence[int] | None = None,
    acc_P: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> torch.Tensor:
    """
    Spatially-varying (per-output-element) mixed-format analog of
    :func:`binaryK_matmul`. Same ``SplitMac`` arithmetic (quantized multiply
    then quantized accumulate), but the binaryK format each output element's
    dot product runs in is chosen from a *palette* of up to 8 formats by
    ``prec_idx``.

    ``mul_K``/``mul_P`` are per-palette-entry sequences and their common
    length is the palette size ``n`` (1..8). ``mul_bias``/``acc_K``/``acc_P``/
    ``acc_bias`` may each be a scalar (broadcast to every entry), an
    ``n``-length sequence, or ``None`` (same per-entry defaulting as
    :func:`binaryK_matmul`: ``acc_*`` fall back to the ``mul_*`` entry,
    ``*_bias`` to ``2**(K - P - 1)`` / ``2**(K - P)``). ``mul_is_signed``/
    ``acc_is_signed``/``rounding_mode``/``saturation_mode``/
    ``subnormals_mode``/``mul_prng_bits``/``acc_prng_bits`` are shared across
    the whole palette.

    ``prec_idx`` is an integer tensor selecting a palette entry per output
    element. Its shape must be the GEMM's output shape ``[M, N]`` (dense),
    ``[M, 1]`` (per row), or ``[1, N]`` (per column); values must lie in
    ``[0, n)`` (checked host-side). ``a``/``b`` must be 2D, as for
    :func:`binaryK_matmul`.

    Any integer dtype and layout is accepted, but a map that is already
    ``int32``, contiguous and on ``a``'s device is passed through untouched,
    where any other spelling is narrowed and packed on every call. Hold one
    such map and reuse it across calls: the bounds check is memoized against
    the tensor you pass, so a map built fresh each call is re-checked each
    call.
    """
    mul_K_l, mul_P_l, n = _palette_pair(mul_K, mul_P, "mul_P", "binaryK_matmul_mixed")
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed

    mul_bias_l = _binaryK_palette_bias(mul_bias, mul_K_l, mul_P_l, mul_is_signed, "mul_bias")
    if accumulate_quant:
        acc_K_l = _palette_list_or(acc_K, mul_K_l, n, "acc_K")
        acc_P_l = _palette_list_or(acc_P, mul_P_l, n, "acc_P")
        acc_bias_l = _binaryK_palette_bias(acc_bias, acc_K_l, acc_P_l, acc_is_signed, "acc_bias")
    else:
        acc_K_l = [0] * n
        acc_P_l = [0] * n
        acc_bias_l = [0] * n

    prng = [("mul_prng_bits", mul_prng_bits, max(mul_P_l) - 1)]
    if accumulate_quant:
        prng.append(("acc_prng_bits", acc_prng_bits, max(acc_P_l) - 1))
    needed, terms = _prng_terms(*prng)

    return _run_gemm_mixed(
        _GemmSpec(
            torch.ops.mptorch.custom_matmul_binaryK_mixed.default,
            (
                mul_K_l,
                mul_P_l,
                mul_bias_l,
                mul_is_signed,
                accumulate_quant,
                acc_K_l,
                acc_P_l,
                acc_bias_l,
                acc_is_signed,
                accumulate_algorithm.value,
                rounding_mode.value,
                saturation_mode.value,
                subnormals_mode.value,
                mul_prng_bits,
                acc_prng_bits,
            ),
            needed,
            terms,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def superfp_matmul_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_man_bits: Sequence[int],
    mul_exp_bits: Sequence[int],
    mul_normal_binades: int | Sequence[int],
    mul_bias: int | Sequence[int],
    mul_is_signed: bool = True,
    mul_prng_bits: int = 0,
    accumulate_quant: bool = True,
    acc_man_bits: int | Sequence[int] | None = None,
    acc_exp_bits: int | Sequence[int] | None = None,
    acc_normal_binades: int | Sequence[int] | None = None,
    acc_bias: int | Sequence[int] | None = None,
    acc_is_signed: bool | None = None,
    acc_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    """
    superfp analog of :func:`binaryK_matmul_mixed` -- see its docstring for
    the palette / ``prec_idx`` contract. ``mul_man_bits``/``mul_exp_bits``
    are per-palette-entry sequences defining the palette size ``n``;
    ``mul_normal_binades``/``mul_bias`` and every ``acc_*`` may be a scalar,
    an ``n``-length sequence, or (for ``acc_*``) ``None`` to fall back to the
    corresponding ``mul_*`` entry.
    """
    mul_mb_l, mul_eb_l, n = _palette_pair(
        mul_man_bits, mul_exp_bits, "mul_exp_bits", "superfp_matmul_mixed"
    )
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed

    mul_nb_l = _palette_list(mul_normal_binades, n, "mul_normal_binades")
    mul_bias_l = _palette_list(mul_bias, n, "mul_bias")

    if accumulate_quant:
        acc_mb_l = _palette_list_or(acc_man_bits, mul_mb_l, n, "acc_man_bits")
        acc_eb_l = _palette_list_or(acc_exp_bits, mul_eb_l, n, "acc_exp_bits")
        acc_nb_l = _palette_list_or(acc_normal_binades, mul_nb_l, n, "acc_normal_binades")
        acc_bias_l = _palette_list_or(acc_bias, mul_bias_l, n, "acc_bias")
    else:
        acc_mb_l = [0] * n
        acc_eb_l = [0] * n
        acc_nb_l = [0] * n
        acc_bias_l = [0] * n

    prng = [("mul_prng_bits", mul_prng_bits, max(mul_mb_l))]
    if accumulate_quant:
        prng.append(("acc_prng_bits", acc_prng_bits, max(acc_mb_l)))
    needed, terms = _prng_terms(*prng)

    return _run_gemm_mixed(
        _GemmSpec(
            torch.ops.mptorch.custom_matmul_superfp_mixed.default,
            (
                mul_mb_l,
                mul_eb_l,
                mul_nb_l,
                mul_bias_l,
                mul_is_signed,
                accumulate_quant,
                acc_mb_l,
                acc_eb_l,
                acc_nb_l,
                acc_bias_l,
                acc_is_signed,
                accumulate_algorithm.value,
                rounding_mode.value,
                saturation_mode.value,
                mul_prng_bits,
                acc_prng_bits,
            ),
            needed,
            terms,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def binaryK_matmul_fma_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_K: Sequence[int],
    fma_P: Sequence[int],
    fma_bias: int | Sequence[int] | None = None,
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> torch.Tensor:
    """
    Spatially-varying (per-output-element) mixed-format analog of
    :func:`binaryK_matmul_fma`. Each dot-product step is a single
    hardware-style fused multiply-add rounded once, but the binaryK format
    that rounding uses is chosen per output element from a palette of up to
    8 formats by ``prec_idx``.

    ``fma_K``/``fma_P`` are per-palette-entry sequences whose common length
    is the palette size ``n`` (1..8); ``fma_bias`` may be a scalar
    (broadcast), an ``n``-length sequence, or ``None`` (per-entry default
    ``2**(K - P - 1)`` / ``2**(K - P)``). ``fma_is_signed``/
    ``rounding_mode``/``saturation_mode``/``subnormals_mode``/
    ``fma_prng_bits`` are shared across the palette.

    ``prec_idx`` follows the same contract as :func:`binaryK_matmul_mixed`
    (shape ``[M, N]`` / ``[M, 1]`` / ``[1, N]``, values in ``[0, n)``).
    ``fma_quant=False`` is rejected -- ``FusedMac<IdentityAdder>`` carries no
    format, so a palette of it would make ``prec_idx`` a no-op; use
    :func:`binaryK_matmul_fma` for the unquantized fused step.
    """
    fma_K_l, fma_P_l, _ = _palette_pair(fma_K, fma_P, "fma_P", "binaryK_matmul_fma_mixed")
    fma_bias_l = _binaryK_palette_bias(fma_bias, fma_K_l, fma_P_l, fma_is_signed, "fma_bias")
    needed, terms = _prng_terms(("fma_prng_bits", fma_prng_bits, max(fma_P_l) - 1))

    return _run_gemm_mixed(
        _GemmSpec(
            torch.ops.mptorch.custom_matmul_binaryK_fma_mixed.default,
            (
                fma_quant,
                fma_K_l,
                fma_P_l,
                fma_bias_l,
                fma_is_signed,
                accumulate_algorithm.value,
                rounding_mode.value,
                saturation_mode.value,
                subnormals_mode.value,
                fma_prng_bits,
            ),
            needed,
            terms,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def superfp_matmul_fma_mixed(
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_man_bits: Sequence[int],
    fma_exp_bits: Sequence[int],
    fma_normal_binades: int | Sequence[int],
    fma_bias: int | Sequence[int],
    fma_is_signed: bool = True,
    fma_quant: bool = True,
    fma_prng_bits: int = 0,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    """
    superfp analog of :func:`binaryK_matmul_fma_mixed` -- see its docstring
    for the palette / ``prec_idx`` contract and the ``fma_quant=False``
    rejection. ``fma_man_bits``/``fma_exp_bits`` are per-palette-entry
    sequences defining the palette size ``n``; ``fma_normal_binades``/
    ``fma_bias`` may each be a scalar or an ``n``-length sequence.
    """
    fma_mb_l, fma_eb_l, n = _palette_pair(
        fma_man_bits, fma_exp_bits, "fma_exp_bits", "superfp_matmul_fma_mixed"
    )
    fma_nb_l = _palette_list(fma_normal_binades, n, "fma_normal_binades")
    fma_bias_l = _palette_list(fma_bias, n, "fma_bias")
    needed, terms = _prng_terms(("fma_prng_bits", fma_prng_bits, max(fma_mb_l)))

    return _run_gemm_mixed(
        _GemmSpec(
            torch.ops.mptorch.custom_matmul_superfp_fma_mixed.default,
            (
                fma_quant,
                fma_mb_l,
                fma_eb_l,
                fma_nb_l,
                fma_bias_l,
                fma_is_signed,
                accumulate_algorithm.value,
                rounding_mode.value,
                saturation_mode.value,
                fma_prng_bits,
            ),
            needed,
            terms,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )
