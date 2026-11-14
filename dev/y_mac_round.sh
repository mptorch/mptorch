#!/usr/bin/env bash
# Phase Y's Mac round (dev/continuation_plan.md, phase Y), on the M3 Pro: the
# odd carrier step under KAHAN, BLOCK and TREE, and the hooks it added to
# common/gemm_accumulate.h, which the Metal prelude compiles.
#
# Run from anywhere in the repo, on a tree that holds phase Y:
#
#   bash dev/y_mac_round.sh                 # everything, ~15-20 min
#   SKIP_SUITE=1 bash dev/y_mac_round.sh    # without step 2's suite
#
# Everything it prints also goes to y_mac_round.log at the repo root. Send that
# file back. Each step runs even if an earlier one failed, and the summary at
# the end lists what failed.
#
# The steps:
#   1. MPTORCH_ODD_ACCUMULATED=1 uv sync (the odd accumulated kernels are
#      opt-in at build time), then tests/test_mps.py on its own first: the
#      hooks are Metal source, so every MPS kernel recompiles, and an MSL error
#      in them would fail every MPS op. Expected: all pass, as before.
#   2. The suite, with the flag: the odd accumulators' tests run on arm64's
#      CPU kernels (their MPS ops raise, which the suite checks).
#   3. check_native.sh.
#   4. carrier_double_rounding.py bench-accumulated on the CPU (1 and every
#      core): what the odd step costs KAHAN, BLOCK and TREE on NEON's scalar
#      loop, against the default op.
set -u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
LOG="$ROOT/y_mac_round.log"
exec > >(tee "$LOG") 2>&1

FAILED=""
fail() { FAILED="$FAILED\n  $1"; echo "FAILED: $1"; }
step() { echo; echo "==== $1 ($(date '+%H:%M:%S'))"; }

if [ ! -f mptorch/csrc/common/carrier_odd_accumulate.h ] || ! grep -q carrier_rounds_to_odd mptorch/csrc/common/gemm_accumulate.h; then
    echo "this tree lacks phase Y (common/carrier_odd_accumulate.h, carrier_rounds_to_odd)"
    exit 1
fi
export MPTORCH_ODD_ACCUMULATED=1
echo "tree: $ROOT ($(git rev-parse --short HEAD), $(git status --short mptorch/csrc | wc -l | tr -d ' ') changed csrc files)"
echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null), $(sysctl -n hw.ncpu) cores; $(clang++ --version | head -1)"
NCPU=$(sysctl -n hw.ncpu)

# ---- 1. the build and the MPS tests -----------------------------------------
step "1. MPTORCH_ODD_ACCUMULATED=1 uv sync, then tests/test_mps.py"
uv sync || fail "uv sync"
uv run --no-sync python -c "
from tests.markers import HAS_MPS, HAS_ODD_ACCUMULATED
print('MPS runs:', HAS_MPS, '| odd accumulated kernels built:', HAS_ODD_ACCUMULATED)
assert HAS_ODD_ACCUMULATED, 'the build has no odd accumulated kernels'
" || fail "the build's flags"
(set -o pipefail; uv run --no-sync pytest -q tests/test_mps.py 2>&1 | tail -15) || fail "tests/test_mps.py"

# ---- 2. the suite -----------------------------------------------------------
if [ "${SKIP_SUITE:-0}" != 1 ]; then
    step "2. the suite"
    (set -o pipefail; uv run --no-sync pytest -q tests 2>&1 | tail -30) || fail "the suite"
else
    step "2. skipped (SKIP_SUITE=1)"
fi

# ---- 3. the native tools ----------------------------------------------------
step "3. check_native.sh"
bash dev/benchmarks/check_native.sh || fail "check_native.sh"

# ---- 4. the cost on NEON ----------------------------------------------------
step "4. bench-accumulated (CPU, 1 and $NCPU threads, 256^3)"
uv run --no-sync python dev/benchmarks/timing/carrier_double_rounding.py bench-accumulated \
    --cpu-threads "1,$NCPU" || fail "bench-accumulated"

step "done"
if [ -n "$FAILED" ]; then
    printf "failed:%b\n" "$FAILED"
else
    echo "every step ran without a failure"
fi
echo "log: $LOG"
