"""Elementwise CPU quantizer throughput, through the installed extension.

Written to measure c53cde8, which rewrote the CPU elementwise quantizers (the
cast inlined into a parallel loop, the format's parameters built once per tensor
rather than per element): 2.4-30x at the default thread count and 1.3-7.4x at
one thread against its parent (2026-09-02, 16 threads). It times the shipped
``binaryK_quantize`` / ``superfp_quantize`` entry points per round mode and
dtype at one thread and at the default thread count, and
``fixedpoint_quantize`` beside them (added in c558a40).

    python3 dev/benchmarks/timing/cpu_quantize_elementwise.py            # 4M elements
    python3 dev/benchmarks/timing/cpu_quantize_elementwise.py 16777216   # 16M
    python3 dev/benchmarks/timing/cpu_quantize_elementwise.py 4194304 1  # pin 1 thread

Numbers are medians of 15 timed calls after a warm-up. The SR rows include the
``randint_like`` draw the stochastic path needs, which is a large part of their
cost -- the ``rng only`` row is that draw on its own, for subtraction.

The last rows of each dtype pair an op with its in-place twin
(``binaryK_quantize_`` / ``superfp_quantize_`` / ``fixedpoint_quantize_``). An
in-place call leaves the format's values behind, and the casts are not
constant-time in their input, so
timing it twice on one tensor would time a different input the second time.
Each row of a pair therefore first refills a scratch tensor from the source
(``copy_``, timed on its own in the ``copy only`` row) and then quantizes it,
out of place or in place. The difference between the two rows is the output
allocation and its first-touch page faults, which is all the in-place op saves:
the kernel reads and writes the same number of bytes either way.
"""

import statistics
import sys
import time

import torch

from mptorch import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import (
    binaryK_quantize,
    binaryK_quantize_,
    fixedpoint_quantize,
    fixedpoint_quantize_,
    superfp_quantize,
    superfp_quantize_,
)

REPS = 15


def bench(label: str, fn) -> None:
    fn()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    ms = statistics.median(ts)
    print(f"  {label:<40} {ms:8.2f} ms  {N / ms / 1e3:8.1f} Melem/s")


def run(n: int) -> None:
    g = torch.Generator().manual_seed(1)
    x32 = torch.randn(n, generator=g)
    for name, x in (("f32", x32), ("f64", x32.double())):
        for rm in RoundMode:
            bench(
                f"binaryK e4m3 {rm.name:<3} {name}",
                lambda x=x, rm=rm: binaryK_quantize(
                    x,
                    8,
                    4,
                    prng_bits=2 if rm is RoundMode.SR else 0,
                    rounding_mode=rm,
                    saturation_mode=SaturationMode.SAT_NONE,
                    subnormals_mode=SubnormalsMode.SUBNORMALS,
                ),
            )
        for rm in RoundMode:
            bench(
                f"superfp m2e5 {rm.name:<3} {name}",
                lambda x=x, rm=rm: superfp_quantize(
                    x,
                    2,
                    5,
                    1,
                    15,
                    prng_bits=2 if rm is RoundMode.SR else 0,
                    rounding_mode=rm,
                    saturation_mode=SaturationMode.SAT_NONE,
                ),
            )
        for rm in RoundMode:
            bench(
                f"fixed  q8.6  {rm.name:<3} {name}",
                lambda x=x, rm=rm: fixedpoint_quantize(
                    x, 8, 6, prng_bits=2 if rm is RoundMode.SR else 0, rounding_mode=rm
                ),
            )
        bench(
            f"rng only     ---     {name}",
            lambda x=x: torch.randint_like(x, 2**31 - 1, dtype=torch.int32),
        )
        y = torch.empty_like(x)
        bench(f"copy only    ---     {name}", lambda x=x, y=y: y.copy_(x))
        for rm in (RoundMode.RNE, RoundMode.SR):
            prng = 2 if rm is RoundMode.SR else 0
            for label, bk, sf, fx in (
                ("out of place", binaryK_quantize, superfp_quantize, fixedpoint_quantize),
                ("in place", binaryK_quantize_, superfp_quantize_, fixedpoint_quantize_),
            ):
                bench(
                    f"copy + binaryK {rm.name:<3} {label} {name}",
                    lambda x=x, y=y, rm=rm, prng=prng, bk=bk: bk(
                        y.copy_(x), 8, 4, prng_bits=prng, rounding_mode=rm
                    ),
                )
                bench(
                    f"copy + superfp {rm.name:<3} {label} {name}",
                    lambda x=x, y=y, rm=rm, prng=prng, sf=sf: sf(
                        y.copy_(x), 2, 5, 1, 15, prng_bits=prng, rounding_mode=rm
                    ),
                )
                bench(
                    f"copy + fixed   {rm.name:<3} {label} {name}",
                    lambda x=x, y=y, rm=rm, prng=prng, fx=fx: fx(
                        y.copy_(x), 8, 6, prng_bits=prng, rounding_mode=rm
                    ),
                )


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 4 * 1024 * 1024
    threads = [int(sys.argv[2])] if len(sys.argv) > 2 else [1, torch.get_num_threads()]
    for t in threads:
        torch.set_num_threads(t)
        print(f"N = {N / 1e6:.1f}M elements, torch threads = {torch.get_num_threads()}")
        run(N)
        print()
