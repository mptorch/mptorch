import os
import platform
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

ROOT = Path(__file__).resolve().parent
QUANT_DIR = ROOT / "mptorch" / "quant"


def collect_sources(subdir: str) -> list[str]:
    sources = []
    for path in (QUANT_DIR / subdir).rglob("*"):
        if path.suffix in {".cpp", ".cu"}:
            sources.append(str(path.relative_to(ROOT)))
    global_ops = QUANT_DIR / "quant_ops.cpp"
    if global_ops.exists():
        sources.append(str(global_ops.relative_to(ROOT)))
    return sources


def get_extra_cflags() -> list[str]:
    if platform.system() == "Windows":
        return ["/std:c++20", "/openmp"]
    if platform.system() == "Darwin":
        return ["-std=c++20"]
    return ["-std=c++20", "-fopenmp"]


ext_modules = [
    CppExtension(
        name="mptorch_quant_cpu",
        sources=collect_sources("quant_cpu"),
        extra_compile_args=get_extra_cflags(),
    )
]

if torch.cuda.is_available():
    ext_modules.append(
        CUDAExtension(
            name="mptorch_quant_cuda",
            sources=collect_sources("quant_cuda"),
            extra_compile_args={"cxx": get_extra_cflags(), "nvcc": ["--extended-lambda"]},
        )
    )

setup(
    name="mptorch",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
