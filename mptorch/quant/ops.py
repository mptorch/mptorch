import torch

from mptorch import (
    RoundMode,
    SaturationMode,
    SubnormalsMode,
)

__all__ = [
    "binaryK_quantize",
    "superfp_quantize",
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
