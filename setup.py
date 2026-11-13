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
``USE_MPS=0``
    On macOS, build without the Metal kernels of the MPS backend.
``MPTORCH_COMPRESS_FATBIN=0``
    Leave the CUDA device code at nvcc's default compression instead of
    ``--compress-mode=size`` (a larger extension, slightly less build work).

Install with ``uv sync``, which builds MPTorch in ``.venv`` without build
isolation (``pyproject.toml``'s ``no-build-isolation-package``), so that the
extension is compiled against the torch that will import it. An isolated build
resolves its own torch, and an ABI mismatch then shows up as an ``undefined
symbol`` error at import.

The objects are kept, in ``build/temp.<platform>-<key>`` (one directory per
configuration; ``build_config.txt`` inside says which), so a rebuild compiles
only the translation units whose sources, headers or flags changed, whether it
is run by ``uv sync`` or as ``uv run --no-sync python setup.py build_ext
--inplace``.
``rm -rf build`` is how to ask for a clean build. With ``ccache`` (or
``sccache``) on ``PATH``, torch puts it in front of both compilers, and a clean
build then recompiles only what the cache has not seen.
"""

import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
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
    the translation units one at a time: about 360 s instead of about 66 s on
    16 threads when there were thirty of them, and there are eighty now
    (about 200 s with ninja). Prepending the package's ``BIN_DIR`` makes the
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


def tool_version(command: list[str]) -> str:
    """Run a build tool's version command and return what it printed, on one line.

    Args:
        command: The tool and its version option, such as ``["c++", "--version"]``.

    Returns:
        The non-empty lines of its standard output joined by ``"; "``, or
        ``"unavailable"`` if the tool did not run or printed nothing.
    """
    try:
        out = subprocess.run(command, capture_output=True, text=True).stdout
    except OSError:
        return "unavailable"
    return "; ".join(line.strip() for line in out.splitlines() if line.strip()) or "unavailable"


def compiler_cache() -> str | None:
    """Name the compiler cache torch will put in front of the compilers, if any.

    ``torch.utils.cpp_extension`` prepends ``ccache``, else ``sccache``, to the
    host compiler and to nvcc when one is on ``PATH`` (its ``_wrap_compiler``),
    unless ``TORCH_NO_COMPILER_WRAPPER`` is set. Nothing here turns the cache
    on: this only reports what torch is about to do, so that a build without
    one is a printed outcome, like a build without ninja.

    Returns:
        ``"ccache"`` or ``"sccache"``, or None if torch will call the
        compilers directly (no cache installed, the wrapper disabled, or a
        torch from before it had one).
    """
    from torch.utils import cpp_extension

    if not hasattr(cpp_extension, "_wrap_compiler"):
        return None
    if os.environ.get("TORCH_NO_COMPILER_WRAPPER"):
        return None
    for wrapper in ("ccache", "sccache"):
        if shutil.which(wrapper):
            return wrapper
    return None


def build_configuration(extension) -> list[str]:
    """List what a build's objects depend on besides their sources and headers.

    These lines are the identity of a build directory: ``build_directory``
    hashes them into its name, and they are written to ``build_config.txt``
    inside it. They hold two kinds of thing. The tool and library versions are
    what ninja cannot see: it rebuilds an object when its command line or one
    of its recorded headers changes, and a compiler upgrade changes neither
    (the host compiler's depfile leaves system headers out). The flags and
    backends ninja does see, and they are here so that each configuration
    keeps its own objects: ``MPTORCH_NO_FP64=1`` puts a define on every
    command line, and in one shared directory switching it on and back would
    recompile everything twice.

    The source list is deliberately absent, so that adding a kernel file does
    not start a new directory.

    Args:
        extension: The extension ``get_extensions`` built.

    Returns:
        One ``name: value`` line per ingredient.
    """
    cuda = any(source.endswith(".cu") for source in extension.sources)
    mps = any(source.endswith(".mm") for source in extension.sources)
    flags = extension.extra_compile_args
    lines = [
        f"platform: {sysconfig.get_platform()}",
        f"python: {platform.python_version()}",
        f"torch: {torch.__version__}",
        f"backends: {' '.join(['cpu'] + ['cuda'] * cuda + ['mps'] * mps)}",
        f"c++: {tool_version([os.environ.get('CXX', 'c++'), '--version'])}",
        f"c++ flags: {' '.join(flags['cxx'])}",
        f"link flags: {' '.join(extension.extra_link_args)}",
    ]
    if cuda:
        lines.append(f"nvcc: {tool_version([os.path.join(CUDA_HOME, 'bin', 'nvcc'), '--version'])}")
        lines.append(f"nvcc flags: {' '.join(flags['nvcc'])}")
    return lines


def build_directory(configuration: list[str]) -> Path:
    """Name the directory that keeps a configuration's objects between builds.

    Args:
        configuration: ``build_configuration``'s lines.

    Returns:
        ``build/temp.<platform>-<key>`` under the repo root, where the key is
        the first ten hex digits of the configuration's SHA-256.
    """
    key = hashlib.sha256("\n".join(configuration).encode()).hexdigest()[:10]
    root = Path(__file__).resolve().parent
    return root / "build" / f"temp.{sysconfig.get_platform()}-{key}"


class IncrementalBuildExtension(BuildExtension):
    """torch's ``BuildExtension``, compiling into a directory that is kept.

    An editable install (``uv sync``, ``pip3 install -e .``) hands ``build_ext`` a
    ``build_temp`` inside a ``TemporaryDirectory``
    (``setuptools.command.editable_wheel``), so ninja starts from nothing each
    time and every install is a clean build: 340 s for 179 translation units,
    where the same build against the objects of the previous one is 9 s when
    nothing changed (16 s through pip), and one file's compile plus a 2 s link
    when one did.
    ``finalize_options`` therefore points ``build_temp`` at
    ``build_directory``'s, unless ``--build-temp`` names one on the command
    line. ninja's own records in that directory (the command line each object
    was built with, the headers it read) are what make the rebuild sound, and
    its log of how long each object took is what lets ninja 1.12 and later
    start the longest ones first.

    ``force`` is set because distutils' own up-to-date test would get in the
    way: it skips an extension none of whose *sources* is newer than the built
    library, which misses a header edit and a changed flag, and it relinks
    only if an object is newer than the library, which misses a switch to a
    configuration whose objects are older. With ``force`` it always runs the
    compile step, which under ninja is still incremental (torch's ninja path
    does not read the flag), and always links. Under ``USE_NINJA=0`` distutils
    compiles every source regardless, as it did before.
    """

    build_configuration: list[str] = []

    def finalize_options(self) -> None:
        """Choose the kept build directory unless the command line named one."""
        named = self.build_temp is not None
        super().finalize_options()
        if not named:
            self.build_temp = str(build_directory(self.build_configuration))
        self.force = True

    def run(self) -> None:
        """Record the configuration in the build directory, then build."""
        directory = Path(self.build_temp)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "build_config.txt").write_text("\n".join(self.build_configuration) + "\n")
        print(f"Build directory: {directory}")
        super().run()


def embed_metal_sources(mps_dir: Path, header: Path) -> None:
    """Write the MPS backend's Metal sources into a C++ header, as strings.

    The Metal kernels are compiled at run time, by the Metal framework on the
    machine that runs them (there is no offline ``metal`` compiler without
    Xcode's), and that compiler has no include path: a library is one string.
    Each root below (``prelude.metal``, ``gemm.metal``, ``quantize.metal``,
    ``block.metal``)
    is therefore flattened here, its quoted ``#include``\\ s replaced by the
    files they name, recursively and once each, with ``#line`` directives so a
    compile error names the header and line it is in. The shared headers
    under ``csrc/common`` are what the prelude includes, so the kernels are
    compiled from the same casts the CPU and CUDA backends are, and a change
    to one of them reaches the GPU at the next build.

    The header is rewritten only when its content changes, so an unchanged
    source leaves ninja nothing to rebuild.

    Args:
        mps_dir: ``csrc/mps``, which holds the ``.metal`` roots.
        header: The header to write, ``mps_metal_sources.h`` in a directory
            on the include path.
    """
    csrc = mps_dir.parent
    include = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
    delimiter = "mptorch_metal"

    def flatten(path: Path, seen: set[Path], out: list[str]) -> None:
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        name = path.relative_to(csrc.resolve()).as_posix()
        out.append(f'#line 1 "{name}"')
        for number, line in enumerate(path.read_text().splitlines(), 1):
            match = include.match(line)
            if match:
                flatten(path.parent / match.group(1), seen, out)
                out.append(f'#line {number + 1} "{name}"')
            else:
                out.append("" if line.strip() == "#pragma once" else line)

    strings = []
    for root in ("prelude", "gemm", "quantize", "block"):
        lines: list[str] = []
        flatten(mps_dir / f"{root}.metal", set(), lines)
        text = "\n".join(lines) + "\n"
        assert f"){delimiter}" not in text, f"{root}.metal contains the raw-string delimiter"
        strings.append(
            f'inline constexpr const char {root}[] = R"{delimiter}(\n{text}){delimiter}";\n'
        )
    content = (
        "// Generated by setup.py (embed_metal_sources) from mptorch/csrc/mps/*.metal\n"
        "// and the headers they include. Do not edit: edit those and rebuild.\n"
        "#pragma once\n\nnamespace mptorch_mps::metal_sources\n{\n"
        + "\n".join(strings)
        + "} // namespace mptorch_mps::metal_sources\n"
    )
    if not header.exists() or header.read_text() != content:
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(content)


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
            "-std=c++20",
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

    # nvcc stores each .cu file's device code in .nv_fatbin twice: as a native
    # image for the architecture it was built for, and as PTX for any newer
    # one. By default only the PTX is compressed, and the native images are
    # most of the extension: 64.9 MB of an 88.6 MB .nv_fatbin, in a 125 MB
    # shared object (179 translation units, 79 of them .cu). --compress-mode=
    # size compresses both, harder: .nv_fatbin comes to 19.6 MB and the shared
    # object to about 56 MB. The device code is the same (every image extracts
    # to the bytes it had, and the driver decompresses a module when it loads
    # it: the first call into a module measures the same, 5-7 ms), and the
    # price is build work: 86 s of single-threaded packing over the 79 files,
    # a second for a typical one and six for the largest. On 16 threads that
    # should be about 10 s of a clean build's wall time; one clean build each
    # way measured 340 s and 361 s, on a machine that drifts by up to 10%.
    #
    # Only from CUDA 13 on. There every mode needs the same driver (580 or
    # later), so the flag does not narrow where the extension loads; under a
    # CUDA 12 toolkit this mode asks for a newer driver than the default does.
    # MPTORCH_COMPRESS_FATBIN=0 keeps nvcc's default.
    #
    # A flag added to the nvcc list must not contain the substring "arch":
    # torch's _get_cuda_arch_flags takes any such flag for a user-chosen
    # architecture and then adds no -gencode of its own.
    if use_cuda:
        toolkit = re.search(
            r"release (\d+)\.", tool_version([os.path.join(CUDA_HOME, "bin", "nvcc"), "--version"])
        )
        if os.getenv("MPTORCH_COMPRESS_FATBIN", "1") != "1":
            print("MPTORCH_COMPRESS_FATBIN=0: device code left at nvcc's default compression.")
        elif toolkit is not None and int(toolkit.group(1)) >= 13:
            extra_compile_args["nvcc"].append("--compress-mode=size")
        else:
            print(
                "Device code left at nvcc's default compression: "
                "--compress-mode=size is used from CUDA 13 on."
            )

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
    # for anything that needs the CUDA toolkit, csrc/mps/ for anything that
    # needs Metal.
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

    # The MPS backend, csrc/mps/: the ops on Apple GPU tensors, as Metal
    # kernels. Built on macOS whenever this torch was built with MPS, which
    # does not need a GPU at build time; without it the ops have no MPS kernel
    # and an MPS tensor fails at dispatch, as any unregistered device does.
    # Its one Objective-C++ file (metal_runtime.mm) compiles and launches the
    # kernels, and torch's BuildExtension compiles .mm with the C++ flags
    # above. The kernels themselves are Metal source, compiled at run time
    # from the strings embed_metal_sources writes into a generated header.
    # Metal and Foundation are linked by name: torch loads both already, and
    # linking them makes the extension say so rather than rely on it.
    use_mps = (
        sys.platform == "darwin"
        and os.getenv("USE_MPS", "1") == "1"
        and torch.backends.mps.is_built()
    )
    if use_mps:
        sources += glob(csrc / "mps", "*.cpp")
        sources += glob(csrc / "mps", "*.mm")
        generated = root / "build" / "generated"
        embed_metal_sources(csrc / "mps", generated / "mps_metal_sources.h")
        include_dirs.append(str(generated))
        extra_link_args.extend(["-framework", "Metal", "-framework", "Foundation"])
    elif sys.platform == "darwin":
        print("Building without the MPS backend: ops on MPS tensors will not dispatch.")

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

# The cache is torch's to use (see compiler_cache); this says whether it will.
cache = compiler_cache()
if cache is None:
    print("No compiler cache on PATH: a clean build compiles every translation unit (see ccache).")
else:
    print(f"Compiler cache: {cache}, in front of the host compiler and nvcc.")

ext_modules = get_extensions()
IncrementalBuildExtension.build_configuration = build_configuration(ext_modules[0])

# The wheel is tagged for the stable ABI only when the extension was built
# against it (see py_limited_api above).
setup(
    name=library_name,
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": IncrementalBuildExtension.with_options(use_ninja=use_ninja)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}} if py_limited_api else {},
)
