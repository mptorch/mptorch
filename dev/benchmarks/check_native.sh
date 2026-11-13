#!/usr/bin/env bash
# Compile check of every native tool in dev/benchmarks against the working
# tree's headers, so a change to mptorch/csrc that breaks a sweep or a timing
# harness is found when it is made, not when the tool is next needed (two
# kernel copies, since retired, stopped building unnoticed when the policies
# gained the carrier as their first template parameter).
#
#   bash dev/benchmarks/check_native.sh        # from anywhere in the repo
#   ARCH=sm_75 bash dev/benchmarks/check_native.sh   # no GPU visible to nvcc
#   JOBS=4 CXX=clang++ bash dev/benchmarks/check_native.sh
#
# The host half runs `-fsyntax-only` with setup.py's host flags (the fast-cast
# pairing, -ffp-contract=off and -DMPTORCH_FAST_CAST=1) on every file that has
# a host build: the five cast sweeps as C++, the CPU tile sweep, the CPU GEMM
# A/B harness (against torch's headers) and old_sr_casts.h. ~20 s.
#
# The CUDA half, skipped when nvcc is not on PATH, compiles each .cu at -O0
# into a temporary directory, with the language standard and include paths
# its header documents. ~40 s at the default JOBS=3; -O3 is what the tools
# are run at, and only changes how long nvcc takes.
#
# Prints one line per file and exits non-zero if any fails, with the first
# lines of that compiler's output. Nothing in the tree is written.
# The flag lists ($CXX $HOST $TI $NV) are split on purpose, and the xargs body
# expands in its own shell.
# shellcheck disable=SC2086,SC2016
set -u
cd "$(dirname "$0")/../.." || exit 1

PY=.venv/bin/python3
[ -x "$PY" ] || PY=python3
TORCH=$("$PY" -c 'import os, torch; print(os.path.dirname(torch.__file__))' 2>/dev/null) || TORCH=
TI=""
[ -n "$TORCH" ] && TI="-I $TORCH/include -I $TORCH/include/torch/csrc/api/include"

if [ "$(uname)" = Darwin ]; then
    CXX=${CXX:-clang++}
    OMP="-Xpreprocessor -fopenmp ${TORCH:+-I $TORCH/include}"
else
    CXX=${CXX:-g++}
    OMP="-fopenmp"
fi
HOST="-std=c++20 $OMP -ffp-contract=off -DMPTORCH_FAST_CAST=1 -I mptorch/csrc -I dev/benchmarks/sweeps"
B=dev/benchmarks
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
FAILED=0

check() { # name, command...
    local name=$1 t0 log
    shift
    log="$OUT/${name//\//_}.log"
    t0=$(date +%s)
    if "$@" >"$log" 2>&1; then
        printf 'OK    %-36s %3ds\n' "$name" $(($(date +%s) - t0))
    else
        printf 'FAIL  %-36s %3ds\n' "$name" $(($(date +%s) - t0))
        sed -n 1,15p "$log" | sed 's/^/      /'
        return 1
    fi
}

echo "host ($CXX)"
for f in cast_all_modes_sweep cast_double_sweep gemm_cast_float_arith gemm_cast_superfp_arith gemm_cast_sr_arith; do
    check "sweeps/$f.cu" $CXX $HOST -fsyntax-only -x c++ "$B/sweeps/$f.cu" || FAILED=1
done
check sweeps/old_sr_casts.h $CXX $HOST -fsyntax-only -x c++ "$B/sweeps/old_sr_casts.h" || FAILED=1
check sweeps/cpu_rne_tile_sweep.cpp $CXX $HOST -fsyntax-only "$B/sweeps/cpu_rne_tile_sweep.cpp" || FAILED=1
if [ -n "$TI" ]; then
    check timing/cpu_gemm_ab.cpp $CXX $HOST $TI -fsyntax-only "$B/timing/cpu_gemm_ab.cpp" || FAILED=1
else
    echo "SKIP  timing/cpu_gemm_ab.cpp (no torch to take headers from)"
fi

if command -v nvcc >/dev/null 2>&1; then
    echo "cuda ($(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'), -arch=${ARCH:-native})"
    NV="-O0 -Xptxas -O0 -arch=${ARCH:-native} --expt-relaxed-constexpr -I mptorch/csrc -I $B/sweeps"
    export OUT NV TI B
    export -f check
    {
        for f in cast_all_modes_sweep cast_double_sweep gemm_cast_float_arith gemm_cast_superfp_arith gemm_cast_sr_arith; do
            echo "sweeps/$f.cu c++17"
        done
        echo "timing/gemm_sr_ab.cu c++20"
        echo "timing/gemm_f64_unroll.cu c++20"
    } | xargs -P "${JOBS:-3}" -L 1 bash -c '
        f=$0 std=$1 extra=""
        [ "$std" = c++20 ] && extra="$TI"
        o="$OUT/$(basename "$f" .cu).o"
        check "$f" nvcc -std=$std $NV $extra -c "$B/$f" -o "$o"
    ' || FAILED=1
else
    echo "SKIP  the .cu builds (nvcc not on PATH)"
fi

[ "$FAILED" = 0 ] && echo "all native tools build" || echo "some native tools do not build"
exit "$FAILED"
