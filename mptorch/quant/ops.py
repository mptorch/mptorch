from collections.abc import Callable, Sequence
from functools import partial
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
    """binaryK's default exponent bias, IEEE P3109's: 1.0 encodes to the middle code point."""
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
    # Whether `op` is one of the four palette ops, i.e. takes a `prec_idx`
    # between the operands and the transpose flags. The format vocabulary in
    # `mptorch.quant.mac` reads this to decide whether a call needs a map,
    # rather than re-deriving from the format what the builder already knew.
    mixed: bool = False


def _packed(x: torch.Tensor, trans: bool) -> tuple[torch.Tensor, bool]:
    """An operand the op can read, and the transpose flag to read it with.

    The kernel reads ``op(x)`` through ``x``'s own storage, so a tensor that is
    the transpose of a contiguous one needs no copy -- only the other flag.
    That is exactly ``q @ k.mT``, the shape batched GEMM exists for, which the
    unconditional ``.contiguous()`` this replaces used to materialize in full.
    Values are unchanged either way: the kernel indexes the same elements, and
    RoundMode::SR keys on the output element, not on the operand's layout.

    A contiguous operand takes the first branch, so the common path costs one
    ``is_contiguous()`` where it used to cost a ``.contiguous()`` call.
    """
    if x.is_contiguous():
        return x, trans
    xt = x.transpose(-2, -1)
    if xt.is_contiguous():
        return xt, not trans
    return x.contiguous(), trans


def _run_gemm(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """Call a resolved GEMM on rank-2 or rank-3 operands.

    All that is left per call is the operand dtype and the layout fold. The
    ``torch.matmul`` operand rules -- 1D promotion, leading-dim broadcasting,
    rank > 3 -- are :func:`_matmul_operands`' job, above this.
    """
    assert spec.needed <= mantissa_size_mapping[a.dtype], _prng_overflow(spec.terms, a.dtype)
    a, trans_a = _packed(a, trans_a)
    b, trans_b = _packed(b, trans_b)
    return spec.op(a, b, trans_a, trans_b, *spec.args)


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
    a, trans_a = _packed(a, trans_a)
    b, trans_b = _packed(b, trans_b)
    return spec.op(a, b, prec_idx, trans_a, trans_b, *spec.args)


# --- torch.matmul's operand rules, over a rank-2-or-3 op ----------------------
#
# The op boundary takes rank 2 or rank 3 and one batch dimension whose extent
# each operand either carries or does not (common/gemm_host.h). Everything
# torch.matmul accepts on top of that -- a 1D operand, leading dims of any
# rank, broadcasting between them -- is expressed here, as views wherever
# torch.matmul itself would use one, so the two agree on when a copy happens.


class _MatmulLayout(NamedTuple):
    """What the op is handed, and the shape its result has to come back as.

    `out_shape` already has the promoted dimensions of a 1D operand removed,
    so the caller reshapes the op's [B, M, N] (or [M, N]) result to it and is
    done.
    """

    a: torch.Tensor
    b: torch.Tensor
    trans_a: bool
    trans_b: bool
    out_shape: tuple[int, ...]


def _collapse(x: torch.Tensor) -> torch.Tensor:
    """Fold every leading dim of a rank>2 operand into one, without a copy.

    ``reshape`` alone would copy a transposed view, which is the layout
    :func:`_packed` exists to keep -- so a transposed view is collapsed
    through its own base and handed back transposed.
    """
    # The batch extent is spelled out rather than inferred: `reshape(-1, r, c)`
    # is ambiguous when the tensor is empty, which a zero batch or a zero M is.
    batch = 1
    for d in x.shape[:-2]:
        batch *= d
    if x.is_contiguous() or not x.transpose(-2, -1).is_contiguous():
        return x.reshape(batch, *x.shape[-2:])
    xt = x.transpose(-2, -1)
    return xt.reshape(batch, *xt.shape[-2:]).transpose(-2, -1)


def _operand_3d(x: torch.Tensor, batch: tuple[int, ...]) -> torch.Tensor:
    """One operand as the rank-2 or rank-3 tensor the op takes.

    No leading dims, or all of them 1, means the op reads it at stride 0 --
    shared across the batch, never expanded. Leading dims that are already the
    broadcast batch collapse to one dim as a view. Anything else is a genuine
    partial broadcast (``[4, 1, M, K] @ [1, 3, K, N]``), and expanding it
    copies exactly where ``torch.matmul`` itself copies.
    """
    lead = x.shape[:-2]
    if all(d == 1 for d in lead):
        while x.dim() > 2:  # squeeze(0) is a view whatever the layout
            x = x.squeeze(0)
        return x
    if tuple(lead) != tuple(batch):
        x = x.expand(*batch, *x.shape[-2:])
    return _collapse(x)


def _broadcast_batch(a_lead: Sequence[int], b_lead: Sequence[int]) -> tuple[int, ...]:
    """The leading dims two operands broadcast to.

    ``torch.broadcast_shapes`` spelled out because it costs 8.7 us -- most of
    an entire resolved GEMM call (finding P4) -- for a pair of tuples that are
    usually equal and never long. Same rule, same errors, ~0.3 us.
    """
    if a_lead == b_lead:
        return tuple(a_lead)
    n = max(len(a_lead), len(b_lead))
    out = []
    for i in range(-n, 0):
        x = a_lead[i] if -i <= len(a_lead) else 1
        y = b_lead[i] if -i <= len(b_lead) else 1
        if x != y and x != 1 and y != 1:
            raise ValueError(
                f"matmul operands' leading dimensions do not broadcast: {tuple(a_lead)} "
                f"against {tuple(b_lead)}"
            )
        out.append(x if y == 1 else y)
    return tuple(out)


def _matmul_operands(
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    fold: bool = True,
) -> _MatmulLayout:
    """``torch.matmul``'s operand contract, resolved onto the rank-2/3 op.

    ``trans_a``/``trans_b`` apply to the last two dims, as they would on a
    ``bmm`` of transposed views; they are ignored for a 1D operand, which
    ``torch.matmul`` has no transpose flag for.

    ``fold=False`` keeps the batch a batch. Only a palette op needs that: its
    ``prec_idx`` is indexed by output element, so a map shaped for the
    unfolded ``[M, N]`` output does not describe the folded ``[B*M, N]`` one.
    """
    a_1d, b_1d = a.dim() == 1, b.dim() == 1
    if a_1d:
        a, trans_a = a.unsqueeze(0), False
    if b_1d:
        b, trans_b = b.unsqueeze(-1), False
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("matmul operands must have at least one dimension")

    batch = _broadcast_batch(a.shape[:-2], b.shape[:-2])
    M = a.shape[-1] if trans_a else a.shape[-2]
    N = b.shape[-2] if trans_b else b.shape[-1]
    out_shape = (*batch, *((() if a_1d else (M,)) + (() if b_1d else (N,))))

    a3 = _operand_3d(a, batch)
    b3 = _operand_3d(b, batch)

    # `[..., M, K] @ [K, N]` with a shared, untransposed `a`: folding the batch
    # into M is a view and gives one large GEMM instead of B small ones. It is
    # bit-identical rather than merely equivalent -- an element's SR
    # subsequence is `(b*M + row)*N + col` batched and `row' * N + col` folded,
    # with `row' = b*M + row`, i.e. the same index -- so it needs no gate of
    # its own. This is the QLinear shape, which `gemm.py` folds the same way.
    if fold and a3.dim() == 3 and b3.dim() == 2 and not trans_a and a3.is_contiguous():
        a3 = a3.reshape(-1, a3.shape[-1])

    return _MatmulLayout(a3, b3, trans_a, trans_b, out_shape)


def _gemm_nd(
    spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor, trans_a: bool, trans_b: bool
) -> torch.Tensor:
    """:func:`_run_gemm` under ``torch.matmul``'s operand rules -- what the
    eight flat wrappers call, where the layer hooks in `mptorch.quant.gemm`
    call `_run_gemm` directly on operands they have already flattened."""
    return _matmul_nd(partial(_run_gemm, spec), a, b, trans_a, trans_b)


def _gemm_mixed_nd(
    spec: _GemmSpec,
    a: torch.Tensor,
    b: torch.Tensor,
    prec_idx: torch.Tensor,
    trans_a: bool,
    trans_b: bool,
) -> torch.Tensor:
    """:func:`_gemm_nd` for the palette ops.

    The map's own leading dims are collapsed the same way the operands' are,
    so a per-batch-element map keeps up with a rank>3 call; and the batch is
    never folded into ``M``, because the map is indexed by output element and
    a folded output has a different one.
    """
    if prec_idx.dim() > 3:
        prec_idx = prec_idx.reshape(-1, *prec_idx.shape[-2:])

    def run(a3, b3, ta, tb):
        return _run_gemm_mixed(spec, a3, b3, prec_idx, ta, tb)

    return _matmul_nd(run, a, b, trans_a, trans_b, fold=False)


def _matmul_nd(
    run: Callable[[torch.Tensor, torch.Tensor, bool, bool], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    trans_a: bool = False,
    trans_b: bool = False,
    fold: bool = True,
) -> torch.Tensor:
    """`run` -- a resolved 2D/3D GEMM -- under ``torch.matmul``'s operand rules.

    Two ranks reach the op untouched, and they are the ones every existing
    caller uses: a pair of matrices, and a pair of equal batches. Both are
    already exactly what the op takes, and taking them through the general
    path costs ~12 us of Python (`_matmul_operands`, most of it in the shape
    broadcast) for a result identical to the operands themselves -- against
    the 2.3-6.4 us finding P4 removed from this same layer.
    """
    if a.dim() == 2 and b.dim() == 2:
        return run(a, b, trans_a, trans_b)
    if a.dim() == 3 and b.dim() == 3 and a.shape[0] == b.shape[0]:
        return run(a, b, trans_a, trans_b)
    layout = _matmul_operands(a, b, trans_a, trans_b, fold)
    out = run(layout.a, layout.b, layout.trans_a, layout.trans_b)
    return out if out.shape == layout.out_shape else out.reshape(layout.out_shape)


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
    """
    Round every element of ``x`` to a binaryK floating-point format: the
    parameterized family of IEEE P3109, the draft Standard for Arithmetic
    Formats for Machine Learning (see :class:`mptorch.BinaryK`).

    The format has ``K`` bits in total, ``P`` of them precision (``P - 1``
    stored mantissa bits plus the implicit one), and ``K - P`` exponent bits
    (``K - P + 1`` when ``is_signed`` is false, since there is no sign bit).
    ``bias`` defaults to P3109's, ``2**(K - P - 1)`` signed and ``2**(K - P)``
    unsigned -- one more than IEEE 754 would give the same exponent width.
    Pass it explicitly for a format outside P3109: the OCP 8-bit formats are
    E4M3, ``K=8, P=4, bias=7``, and E5M2, ``K=8, P=3, bias=15``.

    ``x`` may be float32, float64, float16 or bfloat16; the rounding is done
    on the float32 value and the result is returned in ``x``'s dtype, as a new
    tensor. NaN inputs pass through, and so do infinities except under
    ``SaturationMode.SAT_FINITE``, which clamps them. ``prng_bits`` is the number of
    random bits ``RoundMode.SR`` draws below the target mantissa (ignored by
    every other mode); it must fit, together with ``P - 1``, inside the
    storage dtype's mantissa.

    Not differentiable: on a tensor that requires grad under grad mode this
    raises and points at :class:`mptorch.quant.Quantizer`. See
    :class:`mptorch.quant.Quant` for the format-object spelling.
    """
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
    """
    Round every element of ``x`` to a superfp (supernormal) floating-point format.

    The format has ``man_bits`` stored mantissa bits and ``exp_bits`` exponent
    bits, but only the top ``normal_binades`` binades carry the mantissa: the
    remaining binades' encodings become that many further powers of two below
    the normal region, and values below those flush to zero (there are no
    subnormals, hence no ``subnormals_mode``). ``bias`` is required -- the
    format has no default rule for it. See :class:`mptorch.SuperFP`.

    Dtypes, ``prng_bits`` and differentiability are as for
    :func:`binaryK_quantize`.
    """
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
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul`'s formats -- see it for the contract."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_subnormals_mode is None:
        acc_subnormals_mode = subnormals_mode
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
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
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
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
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

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules: 1D operands are
    promoted (and their dimension dropped from the result), leading dimensions
    broadcast against each other, and ``trans_a``/``trans_b`` apply to the last
    two dimensions. A shared operand rides at stride 0 rather than being
    expanded, and ``[..., M, K] @ [K, N]`` folds its batch into ``M`` -- so the
    two shapes a quantized attention block uses cost no copy. See
    :func:`mptorch.quant.qmatmul` for the differentiable entry point.

    Callers holding one format across many calls should use
    ``mptorch.quant.gemm``'s factories rather than this function: they resolve
    the format once instead of on every call.
    """
    return _gemm_nd(
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
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
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
    acc_saturation_mode: SaturationMode | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul`'s formats -- see it for the contract."""
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
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
            acc_saturation_mode.value,
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
    acc_saturation_mode: SaturationMode | None = None,
) -> torch.Tensor:
    """
    superfp analog of :func:`binaryK_matmul` -- see its docstring for the
    general contract (``torch.matmul``'s operand rules, ``trans_a``/
    ``trans_b``, ``accumulate_quant``, ``mul_prng_bits``/``acc_prng_bits``).
    ``acc_man_bits``/``acc_exp_bits``/``acc_normal_binades``/``acc_bias``
    default to the multiply format's values when omitted and
    ``accumulate_quant`` is true.
    """
    return _gemm_nd(
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
            acc_saturation_mode=acc_saturation_mode,
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

    ``a`` and ``b`` follow ``torch.matmul``'s operand rules, as for
    :func:`binaryK_matmul`.
    """
    return _gemm_nd(
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
    return _gemm_nd(
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


def _binaryK_mixed_spec(
    *,
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
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_mixed`'s palette -- see it for the contract."""
    mul_K_l, mul_P_l, n = _palette_pair(mul_K, mul_P, "mul_P", "binaryK_matmul_mixed")
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode
    if acc_subnormals_mode is None:
        acc_subnormals_mode = subnormals_mode

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

    return _GemmSpec(
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
            acc_saturation_mode.value,
            acc_subnormals_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        needed,
        terms,
        mixed=True,
    )


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
    acc_saturation_mode: SaturationMode | None = None,
    acc_subnormals_mode: SubnormalsMode | None = None,
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
    ``[M, 1]`` (per row), or ``[1, N]`` (per column), optionally with a leading
    batch dimension of ``B`` (one map per batch element) or 1; values must lie
    in ``[0, n)`` (checked host-side). ``a``/``b`` follow ``torch.matmul``'s
    operand rules, as for :func:`binaryK_matmul`.

    Any integer dtype and layout is accepted, but a map that is already
    ``int32``, contiguous and on ``a``'s device is passed through untouched,
    where any other spelling is narrowed and packed on every call. Hold one
    such map and reuse it across calls: the bounds check is memoized against
    the tensor you pass, so a map built fresh each call is re-checked each
    call.
    """
    return _gemm_mixed_nd(
        _binaryK_mixed_spec(
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
            acc_saturation_mode=acc_saturation_mode,
            acc_subnormals_mode=acc_subnormals_mode,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _superfp_mixed_spec(
    *,
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
    acc_saturation_mode: SaturationMode | None = None,
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_mixed`'s palette -- see it for the contract."""
    mul_mb_l, mul_eb_l, n = _palette_pair(
        mul_man_bits, mul_exp_bits, "mul_exp_bits", "superfp_matmul_mixed"
    )
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if acc_saturation_mode is None:
        acc_saturation_mode = saturation_mode

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

    return _GemmSpec(
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
            acc_saturation_mode.value,
            mul_prng_bits,
            acc_prng_bits,
        ),
        needed,
        terms,
        mixed=True,
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
    acc_saturation_mode: SaturationMode | None = None,
) -> torch.Tensor:
    """
    superfp analog of :func:`binaryK_matmul_mixed` -- see its docstring for
    the palette / ``prec_idx`` contract. ``mul_man_bits``/``mul_exp_bits``
    are per-palette-entry sequences defining the palette size ``n``;
    ``mul_normal_binades``/``mul_bias`` and every ``acc_*`` may be a scalar,
    an ``n``-length sequence, or (for ``acc_*``) ``None`` to fall back to the
    corresponding ``mul_*`` entry.
    """
    return _gemm_mixed_nd(
        _superfp_mixed_spec(
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
            acc_saturation_mode=acc_saturation_mode,
        ),
        a,
        b,
        prec_idx,
        trans_a,
        trans_b,
    )


def _binaryK_fma_mixed_spec(
    *,
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
) -> _GemmSpec:
    """Resolve :func:`binaryK_matmul_fma_mixed`'s palette -- see it for the contract."""
    fma_K_l, fma_P_l, _ = _palette_pair(fma_K, fma_P, "fma_P", "binaryK_matmul_fma_mixed")
    fma_bias_l = _binaryK_palette_bias(fma_bias, fma_K_l, fma_P_l, fma_is_signed, "fma_bias")
    needed, terms = _prng_terms(("fma_prng_bits", fma_prng_bits, max(fma_P_l) - 1))

    return _GemmSpec(
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
        mixed=True,
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
    return _gemm_mixed_nd(
        _binaryK_fma_mixed_spec(
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
        prec_idx,
        trans_a,
        trans_b,
    )


def _superfp_fma_mixed_spec(
    *,
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
) -> _GemmSpec:
    """Resolve :func:`superfp_matmul_fma_mixed`'s palette -- see it for the contract."""
    fma_mb_l, fma_eb_l, n = _palette_pair(
        fma_man_bits, fma_exp_bits, "fma_exp_bits", "superfp_matmul_fma_mixed"
    )
    fma_nb_l = _palette_list(fma_normal_binades, n, "fma_normal_binades")
    fma_bias_l = _palette_list(fma_bias, n, "fma_bias")
    needed, terms = _prng_terms(("fma_prng_bits", fma_prng_bits, max(fma_mb_l)))

    return _GemmSpec(
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
        mixed=True,
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
    return _gemm_mixed_nd(
        _superfp_fma_mixed_spec(
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
        prec_idx,
        trans_a,
        trans_b,
    )
