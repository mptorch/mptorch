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
    accumulate_quant: bool = True,
    acc_K: int | None = None,
    acc_P: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
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
    accumulate_quant: bool = True,
    acc_man_bits: int | None = None,
    acc_exp_bits: int | None = None,
    acc_normal_binades: int | None = None,
    acc_bias: int | None = None,
    acc_is_signed: bool | None = None,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    """
    superfp analog of :func:`binaryK_matmul` -- see its docstring for the
    general contract (2D-only, ``trans_a``/``trans_b``,
    ``accumulate_quant``). ``acc_man_bits``/``acc_exp_bits``/
    ``acc_normal_binades``/``acc_bias`` default to the multiply format's
    values when omitted and ``accumulate_quant`` is true.
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

    ``a`` and ``b`` must be 2D; batched/rank>2 callers should flatten their
    leading dimensions first (see ``mptorch.quant.gemm`` for an example that
    does this for ``QAffineFormats``).
    """
    if not fma_bias:
        fma_bias = 2 ** (fma_K - fma_P - 1) if fma_is_signed else 2 ** (fma_K - fma_P)

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
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    rounding_mode: RoundMode = RoundMode.RNE,
    saturation_mode: SaturationMode = SaturationMode.OVF_INF,
) -> torch.Tensor:
    """superfp analog of :func:`binaryK_matmul_fma` -- see its docstring."""
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
    )
