import torch
from mptorch import (
    SubnormalsMode,
    SaturationMode,
    RoundMode,
)
from torch.utils.cpp_extension import load
import os
import platform

__all__ = ["float_quantize_v2", "superfp_quantize_v2", "binaryK_quantize"]


def get_sources(directory):
    sources = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp") or file.endswith(".cu"):
                sources.append(os.path.join(root, file))

    return sources


current_path = os.path.dirname(os.path.realpath(__file__))


def get_extra_cflags():
    match platform.system():
        case "Windows":
            return ["/std:c++20", "/openmp"]
        case "Darwin":
            return ["-std=c++20"]
        case _:
            return ["-std=c++20", "-fopenmp"]


quant_cpu = load(
    name="quant_cpu",
    sources=get_sources(os.path.join(current_path, "quant_cpu")),
    extra_cflags=get_extra_cflags(),
)

if torch.cuda.is_available():
    extra_ldflags = []
    if platform.system() == "Windows":
        extra_ldflags.append("cublas.lib")
    quant_cuda = load(
        name="quant_cuda",
        sources=get_sources(os.path.join(current_path, "quant_cuda")),
        extra_ldflags=extra_ldflags,
        extra_cuda_cflags=["--extended-lambda"],
    )
else:
    quant_cuda = quant_cpu


def assert_wl_fl(wl: int, fl: int, stage: str = ""):
    if wl == -1 and fl != -1:
        raise ValueError("fixed point {} wl {}, fl {}".format(stage, wl, fl))


def get_module(x: torch.Tensor):
    if x.is_cuda:
        quant_module = quant_cuda
    else:
        quant_module = quant_cpu
    return quant_module


def translate_saturation_mode(module, saturation_mode: SaturationMode):
    enum_items = {
        SaturationMode.SAT_FINITE: module.SaturationMode.SAT_FINITE,
        SaturationMode.SAT_PROPAGATE: module.SaturationMode.SAT_PROPAGATE,
        SaturationMode.OVF_INF: module.SaturationMode.OVF_INF,
    }
    assert (
        saturation_mode in enum_items.keys()
    ), f"invalid saturation mode, {saturation_mode}"
    return enum_items[saturation_mode]


def translate_subnormals_mode(module, subnormals_mode: SubnormalsMode):
    enum_items = {
        SubnormalsMode.SUBNORMALS: module.SubnormalsMode.SUBNORMALS,
        SubnormalsMode.EXTENDED_NORMALS: module.SubnormalsMode.EXTENDED_NORMALS,
        SubnormalsMode.NORMALS: module.SubnormalsMode.NORMALS,
    }
    assert (
        subnormals_mode in enum_items.keys()
    ), f"invalid subnormals mode, {subnormals_mode}"
    return enum_items[subnormals_mode]


def translate_rounding_mode(module, rounding_mode: RoundMode):
    enum_items = {
        RoundMode.RNE: module.RoundMode.RNE,
        RoundMode.RNA: module.RoundMode.RNA,
        RoundMode.RU: module.RoundMode.RU,
        RoundMode.RD: module.RoundMode.RD,
        RoundMode.RZ: module.RoundMode.RZ,
        RoundMode.SR: module.RoundMode.SR,
    }
    assert rounding_mode in enum_items.keys(), f"invalid rounding mode, {rounding_mode}"
    return enum_items[rounding_mode]


def normalize_binades(binades: int | tuple[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(binades, int):
        binades_l, binades_h = binades, 0
    elif len(binades) == 1:
        binades_l, binades_h = binades[0], binades[0]
    else:
        binades_l, binades_h = binades[0], binades[1]

    return (binades_l, binades_h)


def float_quantize_v2(
    x: torch.Tensor,
    exp: int,
    man: int,
    bias: int | None = None,
    prng_bits: int = 0,
    saturate: bool = False,
    rounding_mode: RoundMode = RoundMode.RNE,
    subnormals_mode: SubnormalsMode = SubnormalsMode.SUBNORMALS,
) -> torch.Tensor:
    assert (
        0 <= prng_bits <= 23 - man
    ), "prng_bits should be between 0 and 23 minus the number of mantissa bits"

    quant_module = get_module(x)
    rnd_mode = translate_rounding_mode(quant_module, rounding_mode)
    sub_mode = translate_subnormals_mode(quant_module, subnormals_mode)

    if not bias:
        bias = 2 ** (exp - 1) - 1

    return quant_module.fp_quantize(
        x.contiguous(), man, exp, bias, prng_bits, saturate, rnd_mode, sub_mode
    )


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
    assert (
        0 <= prng_bits <= 23 - (P - 1)
    ), "prng_bits should be between 0 and 23 minus the number of mantissa bits (P - 1)"

    quant_module = get_module(x)
    sat_mode = translate_saturation_mode(quant_module, saturation_mode)
    rnd_mode = translate_rounding_mode(quant_module, rounding_mode)
    sub_mode = translate_subnormals_mode(quant_module, subnormals_mode)

    if not bias:
        if is_signed:
            bias = 2 ** (K - P - 1)
        else:
            bias = 2 ** (K - P)

    return quant_module.binaryK_quantize(
        x.contiguous(), K, P, bias, prng_bits, is_signed, rnd_mode, sat_mode, sub_mode
    )


def superfp_quantize_v2(
    x: torch.Tensor,
    exp: int,
    man: int,
    binades: int | tuple[int] | tuple[int, int],
    bias: int | None = None,
    prng_bits: int = 0,
    saturate: bool = False,
    rounding_mode: RoundMode = RoundMode.RNE,
) -> torch.Tensor:
    assert (
        0 <= prng_bits <= 23 - man
    ), "prng_bits should be between 0 and 23 minus the number of mantissa bits"
    quant_module = get_module(x)
    rnd_mode = translate_rounding_mode(quant_module, rounding_mode)
    if not bias:
        bias = 2 ** (exp - 1)

    binades_l, binades_h = normalize_binades(binades)

    return quant_module.superfp_quantize(
        x.contiguous(),
        man,
        exp,
        bias,
        prng_bits,
        binades_l,
        binades_h,
        saturate,
        rnd_mode,
    )
