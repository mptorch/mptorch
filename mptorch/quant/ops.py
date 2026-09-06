from collections.abc import Sequence

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
    assert 0 <= prng_bits <= mantissa_size_mapping[x.dtype] - (P - 1), (
        "prng_bits should be between 0 and 23 minus the number of mantissa bits (P - 1)"
    )

    if not bias:
        if is_signed:
            bias = 2 ** (K - P - 1)
        else:
            bias = 2 ** (K - P)

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
    assert 0 <= prng_bits <= mantissa_size_mapping[x.dtype] - man_bits, (
        "prng_bits should be between 0 and 23 minus the number of mantissa bits (man_bits)"
    )

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
    does this for ``QAffineFormats``).
    """
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed
    if accumulate_quant:
        acc_K = mul_K if acc_K is None else acc_K
        acc_P = mul_P if acc_P is None else acc_P
    else:
        acc_K = acc_K or 0
        acc_P = acc_P or 0

    if not mul_bias:
        mul_bias = 2 ** (mul_K - mul_P - 1) if mul_is_signed else 2 ** (mul_K - mul_P)
    if accumulate_quant and not acc_bias:
        acc_bias = 2 ** (acc_K - acc_P - 1) if acc_is_signed else 2 ** (acc_K - acc_P)
    elif acc_bias is None:
        acc_bias = 0

    assert 0 <= mul_prng_bits <= mantissa_size_mapping[a.dtype] - (mul_P - 1), (
        "mul_prng_bits should be between 0 and 23 minus the number of mantissa bits (mul_P - 1)"
    )
    if accumulate_quant:
        assert 0 <= acc_prng_bits <= mantissa_size_mapping[a.dtype] - (acc_P - 1), (
            "acc_prng_bits should be between 0 and 23 minus the number of mantissa bits (acc_P - 1)"
        )

    return torch.ops.mptorch.custom_matmul_binaryK.default(
        a.contiguous(),
        b.contiguous(),
        trans_a,
        trans_b,
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

    assert 0 <= mul_prng_bits <= mantissa_size_mapping[a.dtype] - mul_man_bits, (
        "mul_prng_bits should be between 0 and 23 minus the number of mantissa bits (mul_man_bits)"
    )
    if accumulate_quant:
        assert 0 <= acc_prng_bits <= mantissa_size_mapping[a.dtype] - acc_man_bits, (
            "acc_prng_bits should be between 0 and 23 minus the number of "
            "mantissa bits (acc_man_bits)"
        )

    return torch.ops.mptorch.custom_matmul_superfp.default(
        a.contiguous(),
        b.contiguous(),
        trans_a,
        trans_b,
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
    if not fma_bias:
        fma_bias = 2 ** (fma_K - fma_P - 1) if fma_is_signed else 2 ** (fma_K - fma_P)

    assert 0 <= fma_prng_bits <= mantissa_size_mapping[a.dtype] - (fma_P - 1), (
        "fma_prng_bits should be between 0 and 23 minus the number of mantissa bits (fma_P - 1)"
    )

    return torch.ops.mptorch.custom_matmul_binaryK_fma.default(
        a.contiguous(),
        b.contiguous(),
        trans_a,
        trans_b,
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
    assert 0 <= fma_prng_bits <= mantissa_size_mapping[a.dtype] - fma_man_bits, (
        "fma_prng_bits should be between 0 and 23 minus the number of mantissa bits (fma_man_bits)"
    )

    return torch.ops.mptorch.custom_matmul_superfp_fma.default(
        a.contiguous(),
        b.contiguous(),
        trans_a,
        trans_b,
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
    )


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
    mul_K = list(mul_K)
    mul_P = list(mul_P)
    n = len(mul_K)
    if n < 1:
        raise ValueError("binaryK_matmul_mixed needs at least one palette format")
    if len(mul_P) != n:
        raise ValueError(f"mul_P must have length {n} (the palette size), got {len(mul_P)}")
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed

    if mul_bias is None:
        mul_bias_l = [
            2 ** (k - p - 1) if mul_is_signed else 2 ** (k - p)
            for k, p in zip(mul_K, mul_P, strict=True)
        ]
    else:
        mul_bias_l = _palette_list(mul_bias, n, "mul_bias")

    if accumulate_quant:
        acc_K_l = _palette_list_or(acc_K, mul_K, n, "acc_K")
        acc_P_l = _palette_list_or(acc_P, mul_P, n, "acc_P")
        if acc_bias is None:
            acc_bias_l = [
                2 ** (k - p - 1) if acc_is_signed else 2 ** (k - p)
                for k, p in zip(acc_K_l, acc_P_l, strict=True)
            ]
        else:
            acc_bias_l = _palette_list(acc_bias, n, "acc_bias")
    else:
        acc_K_l = [0] * n
        acc_P_l = [0] * n
        acc_bias_l = [0] * n

    _prng_bits_bound = mantissa_size_mapping[a.dtype]
    assert 0 <= mul_prng_bits <= _prng_bits_bound - (max(mul_P) - 1), (
        "mul_prng_bits must be in [0, 23 - (max palette mul_P - 1)]"
    )
    if accumulate_quant:
        assert 0 <= acc_prng_bits <= _prng_bits_bound - (max(acc_P_l) - 1), (
            "acc_prng_bits must be in [0, 23 - (max palette acc_P - 1)]"
        )

    return torch.ops.mptorch.custom_matmul_binaryK_mixed.default(
        a.contiguous(),
        b.contiguous(),
        prec_idx,
        trans_a,
        trans_b,
        mul_K,
        mul_P,
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
    mul_man_bits = list(mul_man_bits)
    mul_exp_bits = list(mul_exp_bits)
    n = len(mul_man_bits)
    if n < 1:
        raise ValueError("superfp_matmul_mixed needs at least one palette format")
    if len(mul_exp_bits) != n:
        raise ValueError(f"mul_exp_bits must have length {n} (the palette size)")
    if acc_is_signed is None:
        acc_is_signed = mul_is_signed

    mul_nb_l = _palette_list(mul_normal_binades, n, "mul_normal_binades")
    mul_bias_l = _palette_list(mul_bias, n, "mul_bias")

    if accumulate_quant:
        acc_mb_l = _palette_list_or(acc_man_bits, mul_man_bits, n, "acc_man_bits")
        acc_eb_l = _palette_list_or(acc_exp_bits, mul_exp_bits, n, "acc_exp_bits")
        acc_nb_l = _palette_list_or(acc_normal_binades, mul_nb_l, n, "acc_normal_binades")
        acc_bias_l = _palette_list_or(acc_bias, mul_bias_l, n, "acc_bias")
    else:
        acc_mb_l = [0] * n
        acc_eb_l = [0] * n
        acc_nb_l = [0] * n
        acc_bias_l = [0] * n

    _prng_bits_bound = mantissa_size_mapping[a.dtype]
    assert 0 <= mul_prng_bits <= _prng_bits_bound - max(mul_man_bits), (
        "mul_prng_bits must be in [0, 23 - max palette mul_man_bits]"
    )
    if accumulate_quant:
        assert 0 <= acc_prng_bits <= _prng_bits_bound - max(acc_mb_l), (
            "acc_prng_bits must be in [0, 23 - max palette acc_man_bits]"
        )

    return torch.ops.mptorch.custom_matmul_superfp_mixed.default(
        a.contiguous(),
        b.contiguous(),
        prec_idx,
        trans_a,
        trans_b,
        mul_man_bits,
        mul_exp_bits,
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
    fma_K = list(fma_K)
    fma_P = list(fma_P)
    n = len(fma_K)
    if n < 1:
        raise ValueError("binaryK_matmul_fma_mixed needs at least one palette format")
    if len(fma_P) != n:
        raise ValueError(f"fma_P must have length {n} (the palette size), got {len(fma_P)}")

    if fma_bias is None:
        fma_bias_l = [
            2 ** (k - p - 1) if fma_is_signed else 2 ** (k - p)
            for k, p in zip(fma_K, fma_P, strict=True)
        ]
    else:
        fma_bias_l = _palette_list(fma_bias, n, "fma_bias")

    assert 0 <= fma_prng_bits <= mantissa_size_mapping[a.dtype] - (max(fma_P) - 1), (
        "fma_prng_bits must be in [0, 23 - (max palette fma_P - 1)]"
    )

    return torch.ops.mptorch.custom_matmul_binaryK_fma_mixed.default(
        a.contiguous(),
        b.contiguous(),
        prec_idx,
        trans_a,
        trans_b,
        fma_quant,
        fma_K,
        fma_P,
        fma_bias_l,
        fma_is_signed,
        accumulate_algorithm.value,
        rounding_mode.value,
        saturation_mode.value,
        subnormals_mode.value,
        fma_prng_bits,
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
    fma_man_bits = list(fma_man_bits)
    fma_exp_bits = list(fma_exp_bits)
    n = len(fma_man_bits)
    if n < 1:
        raise ValueError("superfp_matmul_fma_mixed needs at least one palette format")
    if len(fma_exp_bits) != n:
        raise ValueError(f"fma_exp_bits must have length {n} (the palette size)")

    fma_nb_l = _palette_list(fma_normal_binades, n, "fma_normal_binades")
    fma_bias_l = _palette_list(fma_bias, n, "fma_bias")

    assert 0 <= fma_prng_bits <= mantissa_size_mapping[a.dtype] - max(fma_man_bits), (
        "fma_prng_bits must be in [0, 23 - max palette fma_man_bits]"
    )

    return torch.ops.mptorch.custom_matmul_superfp_fma_mixed.default(
        a.contiguous(),
        b.contiguous(),
        prec_idx,
        trans_a,
        trans_b,
        fma_quant,
        fma_man_bits,
        fma_exp_bits,
        fma_nb_l,
        fma_bias_l,
        fma_is_signed,
        accumulate_algorithm.value,
        rounding_mode.value,
        saturation_mode.value,
        fma_prng_bits,
    )
