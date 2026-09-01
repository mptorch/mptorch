import os
import subprocess
import tempfile
from pathlib import Path

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

library_name = "mptorch"

if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def compiler_accepts(flag: str) -> bool:
    """True if the C++ compiler that will build the extension accepts `flag`."""
    cxx = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cpp"
        src.write_text("int main() { return 0; }\n")
        try:
            return (
                subprocess.run(
                    [cxx, *flag.split(), "-c", str(src), "-o", str(src.with_suffix(".o"))],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
        except OSError:
            return False


def get_extensions():
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    # -fopenmp is required for at::parallel_for to actually parallelize: with
    # this torch build's AT_PARALLEL_OPENMP backend, ATen/ParallelOpenMP.h's
    # `#pragma omp parallel` is a header template inlined into *our* translation
    # units, so without the flag it compiles to nothing and CPU kernels run
    # single-threaded. torch itself links GNU libgomp, which is the same runtime
    # gcc's -fopenmp uses, so there is no second OpenMP runtime to oversubscribe
    # with. See dev/gemm_perf_audit.md (finding C1).
    extra_link_args = ["-fopenmp"]
    extra_compile_args = {
        "cxx": [
            "-std=c++20",
            "-O3" if not debug_mode else "-O0",
            "-fopenmp",
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "--extended-lambda",
        ],
    }
    # GCC caps how much a translation unit may grow through inlining at 20% of
    # its own size (--param inline-unit-growth). The CPU quantization kernels
    # blow past that: each instantiates a ~150-instruction format cast, and
    # binaryK_kernel.cpp alone has 48 of them (4 dtypes x 6 round modes x
    # signed/unsigned). Past the cap GCC stops inlining the cast into the
    # elementwise loop, which costs a call per element and blocks the constant
    # folding the templated is_signed exists for. Raising the cap is worth
    # 2.8x on binaryK_quantize (14.3 -> 5.3 ns/element, 4M f32, e4m3, RNE,
    # one thread) and is bit-exact. It is a GCC/Clang spelling, hence the
    # probe. See dev/gemm_perf_audit.md (finding C4).
    inline_growth = "--param inline-unit-growth=400"
    if compiler_accepts(inline_growth):
        extra_compile_args["cxx"].extend(inline_growth.split())

    if py_limited_api:
        extra_compile_args["cxx"].append("-DPy_LIMITED_API=0x03090000")
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    root = Path(__file__).resolve().parent
    csrc = root / library_name / "csrc"
    sources = sorted(str(p.relative_to(root)) for p in csrc.glob("*.cpp"))
    sources += sorted(str(p.relative_to(root)) for p in (csrc / "cpu").glob("*.cpp"))
    if use_cuda:
        sources += sorted(str(p.relative_to(root)) for p in (csrc / "cuda").glob("*.cu"))

    include_dirs = [str(csrc)]

    ext_modules = [
        extension(
            f"{library_name}._C",
            sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            py_limited_api=py_limited_api,
        )
    ]

    return ext_modules


setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
