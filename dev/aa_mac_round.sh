#!/usr/bin/env bash
# Phase AA's second Mac round (dev/continuation_plan.md, phase AA, step 6), on the M3 Pro.
# The first one (2026-11-14) chose NEON's bodies; this one checks the choice:
#   - the odd tiles are the directed pair under FPCR (the binary32 EFT body is gone);
#   - the default product-only and fused tiles keep the row loop (tile_rows);
#   - the split keeps the new passes.
#
# Run from anywhere in the repo, on a tree that holds those changes:
#
#   bash dev/aa_mac_round.sh                 # everything, ~15-20 min
#   SKIP_SUITE=1 bash dev/aa_mac_round.sh    # without step 1's uv sync and suite
#
# Everything it prints also goes to aa_mac_round.log at the repo root. Send that
# file back. Each step runs even if an earlier one failed, and the summary at the
# end lists what failed.
#
# The steps:
#   1. uv sync, then the suite.
#   2. check_native.sh.
#   3. The NEON A/B in dev/benchmarks/timing/cpu_gemm_ab.cpp, two binaries:
#        base  the tree before AA (f2fb952)
#        new   this tree
#      run alternated, 3 rounds, 1 thread at 256^3 and every core at 1024^3, RNE
#      rows only. The hashes must agree. Expected: the default product-only and
#      fused rows equal to base's (the row loop again), the split ~1.08-1.17x
#      faster, the odd rows as FPCR's were in round 1.
#   4. The NEON sweeps, cpu_rne_tile_sweep.cpp and cpu_odd_tile_sweep.cpp: each
#      must end in PASS.
#
# Linking: the binaries link libtorch_cpu, which loads torch's libomp, and leave
# the OpenMP symbols to bind at load (-undefined dynamic_lookup), as the
# extension does (round 1 built this way).
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
LOG="$ROOT/aa_mac_round.log"
exec > >(tee "$LOG") 2>&1

BASE=f2fb952 # the tree before AA (csrc as in 8a91cbe)
W=$(mktemp -d /tmp/aa_mac.XXXXXX)
FAILED=""
fail() { FAILED="$FAILED\n  $1"; echo "FAILED: $1"; }
step() { echo; echo "==== $1 ($(date '+%H:%M:%S'))"; }

if ! grep -q tile_rows mptorch/csrc/cpu/gemm_rne_tile.cpp || grep -q MPTORCH_ODD_NEON_FPCR mptorch/csrc/cpu/gemm_odd_tile.cpp; then
    echo "this tree lacks the changes after the first Mac round (tile_rows, FPCR as NEON's odd body)"
    exit 1
fi
if ! git cat-file -e "$BASE^{commit}" 2>/dev/null; then
    echo "commit $BASE is not in this clone (git fetch?)"
    exit 1
fi
echo "tree: $ROOT ($(git rev-parse --short HEAD), $(git status --short mptorch/csrc | wc -l | tr -d ' ') changed csrc files)"
echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null), $(sysctl -n hw.ncpu) cores; $(clang++ --version | head -1)"
NCPU=$(sysctl -n hw.ncpu)

# ---- 1. the suite ---------------------------------------------------------
if [ "${SKIP_SUITE:-0}" != 1 ]; then
    step "1. uv sync and the suite"
    uv sync || fail "uv sync"
    (set -o pipefail; uv run --no-sync pytest -q tests 2>&1 | tail -30) || fail "the suite"
else
    step "1. skipped (SKIP_SUITE=1)"
fi

# ---- 2. the native tools --------------------------------------------------
step "2. check_native.sh"
bash dev/benchmarks/check_native.sh || fail "check_native.sh"

# ---- 3. the NEON A/B ------------------------------------------------------
step "3. the NEON A/B (cpu_gemm_ab.cpp: base, new)"
T=$(.venv/bin/python -c 'import os, torch; print(os.path.dirname(torch.__file__))')
F="-O3 -std=c++20 -DNDEBUG -Xpreprocessor -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 -I $T/include -I $T/include/torch/csrc/api/include"
L="-L $T/lib -ltorch_cpu -lc10 -Wl,-undefined,dynamic_lookup -Wl,-rpath,$T/lib"
mkdir -p "$W/base" && git archive "$BASE" mptorch/csrc | tar -x -C "$W/base"
built=""
for v in base new; do
    t=mptorch/csrc
    [ "$v" = base ] && t="$W/base/mptorch/csrc"
    # shellcheck disable=SC2086
    if clang++ $F -DMPTORCH_AB_ODD -I "$t" dev/benchmarks/timing/cpu_gemm_ab.cpp \
        "$t/cpu/gemm_rne_tile.cpp" "$t/cpu/gemm_odd_tile.cpp" $L -o "$W/ab_$v"; then
        built="$built $v"
    else
        fail "building ab_$v"
    fi
done
for r in 1 2 3; do
    order="$built"
    if [ "$r" = 2 ]; then # the order flipped
        order=""
        for v in $built; do order="$v $order"; done
    fi
    for v in $order; do
        echo "-- round $r, $v, 1 thread, 256^3"
        "$W/ab_$v" --threads 1 --reps 5 --only RNE || fail "ab_$v, 1 thread"
        echo "-- round $r, $v, $NCPU threads, 1024^3"
        "$W/ab_$v" --threads "$NCPU" --n 1024 --reps 3 --only RNE || fail "ab_$v, $NCPU threads"
    done
done

# ---- 4. the NEON sweeps ---------------------------------------------------
step "4. the NEON sweeps"
for s in rne odd; do
    # shellcheck disable=SC2086
    if clang++ $F -I mptorch/csrc "dev/benchmarks/sweeps/cpu_${s}_tile_sweep.cpp" \
        "mptorch/csrc/cpu/gemm_${s}_tile.cpp" $L -o "$W/${s}_sweep"; then
        echo "-- cpu_${s}_tile_sweep"
        (time "$W/${s}_sweep") || fail "cpu_${s}_tile_sweep"
    else
        fail "building cpu_${s}_tile_sweep"
    fi
done
rm -rf "$W"
step "done"
if [ -n "$FAILED" ]; then
    printf "failed:%b\n" "$FAILED"
else
    echo "every step ran without a failure"
fi
echo "log: $LOG"
