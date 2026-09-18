"""Build script for MPTorch's C++/CUDA extension, ``mptorch._C``.

The extension is one shared object built from every source under
``mptorch/csrc`` (the CUDA directory only when a CUDA toolchain is present).
Most of this file is compiler flags, and several of them are load-bearing for
speed or for numerics rather than cosmetic; each says why where it is added.

Environment switches, all read at build time:

``USE_CUDA=0``
    Build the CPU-only extension even if CUDA is available.
``DEBUG=1``
    Compile with ``-O0 -g`` instead of ``-O3 -g0``.
``USE_NINJA=0``
    Use the serial distutils backend instead of ninja.
``MPTORCH_NO_FP64=1``
    Leave out the float64 GEMM kernels for a faster iteration build.

Always install with ``pip3 install -e . --no-build-isolation``, so that the
extension is compiled against the torch that will import it. An isolated build
resolves its own torch, and an ABI mismatch then shows up as an ``undefined
symbol`` error at import.
"""

import os
import shutil
import subprocess
import sys
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

# From torch 2.6 an extension can target CPython's stable ABI, so one wheel
# (tagged abi3) serves every supported Python instead of one wheel per version.
if torch.__version__ >= "2.6.0":
    py_limited_api = True
else:
    py_limited_api = False


def compiler_accepts(flag: str, source: str = "int main() { return 0; }\n") -> bool:
    """Probe the host C++ compiler for a flag, or for a property of the target.

    Compiles ``source`` with ``flag`` using the compiler the extension build
    will use (``$CXX``, else ``c++``) and reports whether that succeeded. With
    the default source this asks "does the compiler know this flag"; with a
    source holding a ``static_assert`` it asks a question about the target
    instead, which is how the fast-cast probe checks ``FLT_EVAL_METHOD``.

    Args:
        flag: One or more space-separated compiler options to test.
        source: The translation unit to compile with them.

    Returns:
        True if the compile succeeded, False if it failed or no compiler ran.
    """
    cxx = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "probe.cpp"
        src.write_text(source)
        obj = src.with_suffix(".o")
        try:
            return (
                subprocess.run(
                    # -Werror: clang accepts some GCC-only options, such as
                    # --param, with only an "argument unused" warning.
                    [cxx, "-Werror", *flag.split(), "-c", str(src), "-o", str(obj)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            )
        except OSError:
            return False


def ensure_ninja_on_path() -> bool:
    """Put the ``ninja`` package's executable on ``PATH`` if it is not there.

    ``BuildExtension`` chooses its backend by running ``ninja --version``,
    which is a plain ``PATH`` lookup. The ninja that ``pyproject.toml`` requires
    is a Python package whose executable sits in the environment's script
    directory, so the lookup succeeds with the virtualenv activated and fails
    when its interpreter is invoked by path (``.venv/bin/pip3 install -e .``).
    Torch then falls back to distutils without an error, and distutils compiles
    the roughly thirty translation units one at a time: about 360 s instead of
    about 66 s on 16 threads. Prepending the package's ``BIN_DIR`` makes the
    choice independent of how the interpreter was reached.

    Returns:
        True if ``ninja`` is runnable after the call, False if it is not
        installed at all.
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
    """Describe the one extension module, ``mptorch._C``, and its build flags.

    Returns:
        A one-element list holding a ``CUDAExtension`` when a CUDA build is
        possible and wanted, else a ``CppExtension`` over the CPU sources only.
    """
    debug_mode = os.getenv("DEBUG", "0") == "1"
    use_cuda = os.getenv("USE_CUDA", "1") == "1"
    if debug_mode:
        print("Compiling in debug mode")

    # A CUDA build is attempted only if torch sees a device and a toolkit is
    # installed (CUDA_HOME). Missing either, the build degrades to CPU-only
    # rather than failing.
    use_cuda = use_cuda and torch.cuda.is_available() and CUDA_HOME is not None
    extension = CUDAExtension if use_cuda else CppExtension

    # -fopenmp is what makes at::parallel_for parallel. Torch's OpenMP backend
    # implements it as a header template (ATen/ParallelOpenMP.h) containing a
    # `#pragma omp parallel`, and that template is instantiated inside this
    # extension's translation units, not inside libtorch. Without the flag the
    # pragma is ignored and every CPU kernel runs on one thread, silently.
    # Torch links GNU libgomp, the same runtime gcc's -fopenmp uses, so the
    # process still holds one OpenMP runtime and cannot oversubscribe itself.
    #
    # Apple clang has no -fopenmp driver flag. -Xpreprocessor hands it to the
    # front end alone, which honours the pragmas without linking a runtime.
    # torch's macOS wheel ships omp.h (torch/include, already on the include
    # path) and its own libomp.dylib, and nothing is linked here: Python's
    # LDSHARED carries -undefined dynamic_lookup, so the __kmpc_* symbols bind
    # at import to the libomp torch has already loaded, keeping one runtime in
    # the process. Linking a libomp would load a second one (Homebrew's), or
    # fail to load at all (torch's copy has the install name
    # /opt/llvm-openmp/lib/libomp.dylib, which does not exist on disk).
    if sys.platform == "darwin":
        openmp_compile, openmp_link = ["-Xpreprocessor", "-fopenmp"], []
        openmp_probe = f"{' '.join(openmp_compile)} -I{Path(torch.__file__).parent / 'include'}"
        omp_source = "#include <omp.h>\nint main() { return omp_get_max_threads(); }\n"
        if not compiler_accepts(openmp_probe, omp_source):
            print("OpenMP unavailable: CPU kernels will run on one thread.")
            openmp_compile = []
    else:
        openmp_compile, openmp_link = ["-fopenmp"], ["-fopenmp"]
    extra_link_args = list(openmp_link)
    extra_compile_args = {
        "cxx": [
            "-std=c++20",
            "-O3" if not debug_mode else "-O0",
            *openmp_compile,
            "-fdiagnostics-color=always",
        ],
        "nvcc": [
            "-O3" if not debug_mode else "-O0",
            "--extended-lambda",
        ],
    }
    # GCC stops inlining once a translation unit has grown by 20% through
    # inlining (--param inline-unit-growth). The CPU quantization kernels pass
    # that quickly: each instantiates a format cast of about 150 instructions,
    # and binaryK_kernel.cpp alone holds dozens of them (dtypes x round modes x
    # signed/unsigned). Past the cap the cast stays a call per element inside
    # the elementwise loop, which also blocks the constant folding that the
    # `is_signed` template parameter exists for. Raising the cap is worth 2.8x
    # on binaryK_quantize (14.3 to 5.3 ns per element) and changes no result
    # bit. The spelling is GCC's and Clang's, hence the probe.
    inline_growth = "--param inline-unit-growth=400"
    if compiler_accepts(inline_growth):
        extra_compile_args["cxx"].extend(inline_growth.split())

    # The float-arithmetic cast fast paths (MPTORCH_FAST_CAST, documented in
    # common/bit_helper.h) round with a Veltkamp split, `t - (t - x)`, which is
    # exact only if each float operation is evaluated as a float and no two of
    # them are contracted into one. Both are properties of the build, not of
    # the source, so they are requested and checked here, and the paths are
    # admitted only if both hold:
    #
    #   -ffp-contract=off  forbids fusing the split into an FMA, which would
    #                      skip the intermediate rounding the split relies on.
    #                      It costs nothing: no code in csrc wants an implicit
    #                      FMA (gemm_policy.h asks for its FMA explicitly with
    #                      std::fma, which the flag does not affect). On the
    #                      baseline x86-64 target, which has no FMA instruction,
    #                      it is insurance against a CFLAGS or -march that adds
    #                      one; on arm64, which has FMA and where clang contracts
    #                      by default, it is required.
    #   FLT_EVAL_METHOD    must be 0, meaning no x87 excess precision. SSE
    #                      arithmetic is the x86-64 default, so this holds
    #                      there; the static_assert in the probe makes it a
    #                      checked assumption on every other target.
    #
    # The flag and the define go in together or not at all, since either alone
    # is not enough. Neither is passed to nvcc: device code turns the paths on
    # by itself in the header, because CUDA's `_rn` intrinsics carry the
    # no-contraction guarantee in the source.
    fp_contract = "-ffp-contract=off"
    if compiler_accepts(
        fp_contract,
        '#include <cfloat>\nint main() { static_assert(FLT_EVAL_METHOD == 0, ""); return 0; }\n',
    ):
        extra_compile_args["cxx"].extend([fp_contract, "-DMPTORCH_FAST_CAST=1"])
    else:
        print("Fast cast paths off: this compiler cannot promise unfused float32 arithmetic.")

    # -DPy_LIMITED_API is deliberately absent. BuildExtension adds its own
    # -DPy_LIMITED_API=<oldest supported CPython> to every extension built with
    # py_limited_api=True, and adds it after extra_compile_args, so a define
    # here would only produce a "redefined" warning per file and then lose.
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])
    else:
        # torch.utils.cpp_extension compiles host .cpp files with the CFLAGS
        # CPython itself was built with, and those usually include -g. Debug
        # info is expensive here: these files inline a format cast of about 150
        # instructions into dozens of unrolled loop bodies, and GCC's variable
        # tracking then dominates the compile. Measured on the CPU GEMM host
        # code: 78 s and 13.7 MB of object with -g, 57 s and 1.0 MB without;
        # debug info was 16.5 MB of a 23.7 MB shared object. -g0 comes after
        # Python's flags on the command line, so it wins. The generated code is
        # byte for byte the same (.text and .nv_fatbin), unlike dropping
        # Python's -fno-omit-frame-pointer, which perturbs .text for no
        # measurable gain. nvcc never received -g: torch's ninja path hands it
        # only preprocessor options and the nvcc flags listed above.
        extra_compile_args["cxx"].append("-g0")

    # The GEMM kernels that compute in binary64 live in eight translation units
    # of their own, custom_matmul_*_f64.{cu,cpp}, and account for most of what
    # float64 support costs the build. MPTORCH_NO_FP64=1 leaves them out: the
    # sources are skipped by `glob` below, and the define makes the GEMM entry
    # points refuse float64 operands with an error that names this flag
    # (common/dispatch.h) instead of failing at link time. The elementwise
    # quantizers keep their float64 path either way, since it is instantiated
    # in place and costs about a second.
    no_fp64 = os.getenv("MPTORCH_NO_FP64", "0") == "1"
    if no_fp64:
        print("MPTORCH_NO_FP64=1: building without the float64 GEMM kernels.")
        extra_compile_args["cxx"].append("-DMPTORCH_NO_FP64=1")
        extra_compile_args["nvcc"].append("-DMPTORCH_NO_FP64=1")

    root = Path(__file__).resolve().parent
    csrc = root / library_name / "csrc"

    def glob(directory: Path, pattern: str) -> list[str]:
        """List a directory's sources, repo-relative and sorted.

        Sorted so the ninja build file, and with it the incremental rebuild, is
        stable across runs. Relative because setuptools rejects absolute source
        paths. The ``*_f64`` files are dropped under ``MPTORCH_NO_FP64=1``.
        """
        return sorted(
            str(p.relative_to(root))
            for p in directory.glob(pattern)
            if not (no_fp64 and p.stem.endswith("_f64"))
        )

    # Sources are globbed, so a new kernel file only needs to land in the right
    # directory: csrc/ for registrations, csrc/cpu/ for host kernels, csrc/cuda/
    # for anything that needs the CUDA toolkit.
    sources = glob(csrc, "*.cpp")
    sources += glob(csrc / "cpu", "*.cpp")
    if use_cuda:
        # csrc/cuda holds .cpp files as well as .cu files. The CUDA GEMM's entry
        # points and RNG-state draws are host code that handles at::Tensor, and
        # putting ATen's headers through nvcc costs about 25 s per file against
        # about 3 s for a .cu that sees only raw pointers. Keeping the tensors
        # in .cpp files next to the kernels they drive saves that per kernel
        # file. Both patterns stay under use_cuda, so a CPU-only build skips
        # the directory whole.
        sources += glob(csrc / "cuda", "*.cu")
        sources += glob(csrc / "cuda", "*.cpp")

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


# The backend is chosen here rather than left to BuildExtension's own probe for
# two reasons: the serial fallback becomes a printed outcome instead of a silent
# one, and USE_NINJA=0 can ask for it on purpose (torch offers no environment
# switch of its own for that).
use_ninja = os.getenv("USE_NINJA", "1") == "1" and ensure_ninja_on_path()
if not use_ninja:
    print("Building without ninja: translation units will compile one at a time.")

# The wheel is tagged for the stable ABI only when the extension was built
# against it (see py_limited_api above).
setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=use_ninja)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
