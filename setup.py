import os
import shutil
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


def ensure_ninja_on_path() -> bool:
    """Make torch's ninja probe succeed when ninja is installed but not on PATH.

    `BuildExtension` picks its backend with `is_ninja_available()`, which shells
    out to `ninja --version` -- a plain PATH lookup. The ninja that
    pyproject.toml's `requires` pulls in is a Python package whose executable
    lands in the environment's script directory, so the probe succeeds with the
    venv activated and fails when its interpreter is invoked by path
    (`.venv/bin/pip3 install -e .`, which is how this repo documents building).
    Torch then falls back to distutils, which compiles the nine translation
    units one at a time: 360 s against 103 s on a 16-core machine, with no
    error and one easily-missed warning. See dev/gemm_roadmap.md (finding B0).
    """
    if shutil.which("ninja") is not None:
        return True
    try:
        from ninja import BIN_DIR
    except ImportError:
        return False
    if shutil.which("ninja", path=BIN_DIR) is None:
        return False
    os.environ["PATH"] = os.pathsep.join([BIN_DIR, os.environ.get("PATH", "")])
    return True


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
    else:
        # torch.utils.cpp_extension builds host .cpp files with Python's own
        # CFLAGS (`self.compiler.compiler_so[1:]`), which on a CPython built
        # the usual way carries -g. Nothing here asks for it, and it is
        # expensive: these TUs inline a ~150-instruction format cast into
        # dozens of unrolled loop bodies, and GCC's var-tracking then dominates
        # the compile. Compiled on its own, the CPU GEMM host code (then one
        # cpu/custom_matmul_kernel.cpp, since split into four -- finding B2)
        # was 78 s / 13.7 MB of object with -g and 57 s / 1.0 MB without; the
        # other host TUs are 1.5-1.7x. Debug info was 16.5 MB of the 23.7 MB .so,
        # which drops to 7.3 MB. Bit-exact and codegen-identical: .text and
        # .nv_fatbin come out byte for byte the same (unlike dropping Python's
        # -fno-omit-frame-pointer, which perturbs .text for no measurable
        # gain). Our flags land after Python's on the command line, so the last
        # -g wins. The .cu files never saw -g in the first place: torch's ninja
        # path passes only preprocessor options and our own nvcc flags to
        # nvcc. See dev/gemm_roadmap.md (finding B1).
        extra_compile_args["cxx"].append("-g0")

    root = Path(__file__).resolve().parent
    csrc = root / library_name / "csrc"
    sources = sorted(str(p.relative_to(root)) for p in csrc.glob("*.cpp"))
    sources += sorted(str(p.relative_to(root)) for p in (csrc / "cpu").glob("*.cpp"))
    if use_cuda:
        # Both, and only under use_cuda: since H1 the CUDA GEMM's entry points
        # and its launch-context draw are .cpp files sitting next to the .cu
        # files they drive, because a .cu that never sees at::Tensor compiles
        # its fixed ATen cost in ~3 s instead of ~25 s. A CPU-only build skips
        # this directory whole, exactly as it always did.
        sources += sorted(str(p.relative_to(root)) for p in (csrc / "cuda").glob("*.cu"))
        sources += sorted(str(p.relative_to(root)) for p in (csrc / "cuda").glob("*.cpp"))

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


# Decided here rather than left to BuildExtension's own probe so that the
# serial fallback is a stated outcome instead of a silent one, and so USE_NINJA=0
# can ask for it deliberately (torch offers no env-var opt-out of its own).
use_ninja = os.getenv("USE_NINJA", "1") == "1" and ensure_ninja_on_path()
if not use_ninja:
    print("Building without ninja: translation units will compile one at a time.")

setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=use_ninja)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
