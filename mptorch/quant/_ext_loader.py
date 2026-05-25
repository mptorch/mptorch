import os
import platform

import torch
from torch.utils.cpp_extension import load

from mptorch import RoundMode, SubnormalsMode


def get_sources(directory):
    sources = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".cpp") or file.endswith(".cu"):
                sources.append(os.path.join(root, file))
    shared_mm = os.path.join(current_path, "mm_common.cpp")
    if os.path.exists(shared_mm):
        sources.append(shared_mm)
    return sources


def get_extra_cflags():
    match platform.system():
        case "Windows":
            return ["/std:c++20", "/openmp"]
        case "Darwin":
            return ["-std=c++20"]
        case _:
            return ["-std=c++20", "-fopenmp"]


current_path = os.path.dirname(os.path.realpath(__file__))


def _load_jit_extensions():
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
    return quant_cpu, quant_cuda


def _load_prebuilt_extensions():
    import mptorch_quant_cpu as quant_cpu

    try:
        import mptorch_quant_cuda as quant_cuda
    except ImportError:
        quant_cuda = quant_cpu
    return quant_cpu, quant_cuda


def load_quant_extensions():
    if os.environ.get("MPTORCH_JIT", "0") == "1":
        return _load_jit_extensions()
    try:
        return _load_prebuilt_extensions()
    except ImportError:
        return _load_jit_extensions()


quant_cpu, quant_cuda = load_quant_extensions()

__all__ = ["quant_cpu", "quant_cuda", "load_quant_extensions"]
