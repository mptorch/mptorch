"""CPU GEMM per-output-element tile state -- the measurement behind ``NaiveTile`` (f5b61b4).

Each 32x32 output tile used to build a `std::vector<Accumulator>` of 1024
copies of the prototype accumulator. An `Accumulator` carries its Mac policy,
so that is 224 B per output element for split binaryK (`BinaryKParams` 72 ->
multiplier 88 -> `SplitMac` 176, plus the 44 B Philox engine) to carry one
float of running sum -- a 224 KB heap allocation per tile, strided through at
224 B per inner-loop step. The Mac policy is uniform across a single-format
call and comes from the palette on the mixed one, so it does not belong there.

This script is the shipped-op measurement. Like the other cross-build
comparisons here it needs two builds of the extension and alternates them, the
order flipped each round -- see `stacked_before_after.py` for why a plain
before/after on one machine cannot resolve this:

    cp -r mptorch /tmp/base_pkg/          # the pre-change build's package dir
    ... apply the change, rebuild ...
    for r in 1 2 3; do
      for t in /tmp/base_pkg $PWD; do
        PYTHONPATH=$t python dev/benchmarks/timing/cpu_gemm_tile_state.py --threads 8
      done
    done

The shapes straddle the 32x32 tiling deliberately: a ragged edge tile
(70x33x51) exercises the partial-tile path, and the thin-batch one (32 rows)
is where the tile grid is the only source of parallel work at all.

The question was asked again after 85061e4 gave the K-loop a packed float
array of its own (each operand tile converted to float once), which is the
other thing the state split (`NaiveTile` in common/gemm_policy.h and the CPU
kernel's `bool MIXED` template axis, f5b61b4) was for. That re-measurement used
three arms in one build rather than two builds -- shipped, the shape before the
split (one Accumulator per element, no MIXED axis), and the state split without
the `bool MIXED` axis -- switched per call by MPTORCH_CPU_ARM in a throwaway
probe of cpu/custom_matmul_kernel.h. Both halves still pay: reverting the split
costs 1-14% on the split macs, and dropping the MIXED axis costs 2-15% on every
single-format call.
"""

import argparse
import json
import statistics
import time
from typing import TypedDict

import torch

from mptorch.number import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    superfp_matmul,
    superfp_matmul_fma,
)


class _BKParams(TypedDict):
    mul_K: int
    mul_P: int
    acc_K: int
    acc_P: int


class _SFPParams(TypedDict):
    mul_man_bits: int
    mul_exp_bits: int
    mul_normal_binades: int
    mul_bias: int
    acc_man_bits: int
    acc_exp_bits: int
    acc_normal_binades: int
    acc_bias: int


class _SRParams(TypedDict):
    rounding_mode: RoundMode
    mul_prng_bits: int
    acc_prng_bits: int


BK: _BKParams = dict(mul_K=8, mul_P=4, acc_K=8, acc_P=4)
SFP: _SFPParams = dict(
    mul_man_bits=3,
    mul_exp_bits=4,
    mul_normal_binades=1,
    mul_bias=7,
    acc_man_bits=3,
    acc_exp_bits=4,
    acc_normal_binades=1,
    acc_bias=7,
)
SHAPES = [(256, 256, 256), (32, 1024, 1024), (70, 33, 51), (512, 512, 128)]


def timed(fn, reps, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"min": min(ts), "median": statistics.median(ts)}


def cases(m, k, n):
    a, b = torch.randn(m, k), torch.randn(k, n)
    sr: _SRParams = dict(rounding_mode=RoundMode.SR, mul_prng_bits=8, acc_prng_bits=8)
    return {
        "binaryK split RNE": lambda: binaryK_matmul(a, b, **BK, rounding_mode=RoundMode.RNE),
        "binaryK split SR": lambda: binaryK_matmul(a, b, **BK, **sr),
        "superfp split RNE": lambda: superfp_matmul(a, b, **SFP, rounding_mode=RoundMode.RNE),
        "superfp split SR": lambda: superfp_matmul(a, b, **SFP, **sr),
        "binaryK fma RNE": lambda: binaryK_matmul_fma(
            a, b, fma_K=8, fma_P=4, rounding_mode=RoundMode.RNE
        ),
        "superfp fma RNE": lambda: superfp_matmul_fma(
            a,
            b,
            fma_man_bits=3,
            fma_exp_bits=4,
            fma_normal_binades=1,
            fma_bias=7,
            rounding_mode=RoundMode.RNE,
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    res = {}
    for m, k, n in SHAPES:
        for name, fn in cases(m, k, n).items():
            res[f"{name} {m}x{k}x{n}"] = timed(fn, args.reps, args.warmup)
    print(
        json.dumps(
            {
                "tree": __import__("mptorch").__file__,
                "threads": torch.get_num_threads(),
                "results": res,
            }
        )
    )


if __name__ == "__main__":
    main()
