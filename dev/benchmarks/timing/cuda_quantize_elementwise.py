"""Elementwise CUDA quantizer throughput, through the installed extension.

The GPU twin of ``cpu_quantize_elementwise.py``: the shipped ``binaryK_quantize``
/ ``superfp_quantize`` / ``fixedpoint_quantize`` entry points, timed per round
mode and dtype against the bandwidth ceiling of a trivial elementwise op. It is
the end-to-end measurement behind two results: that the deterministic path runs
at that ceiling on a tensor past L2, and what 9f286f5 gained by having the
kernels read a ``BinaryKParams`` / ``SuperfpParams`` built once on the host
instead of re-deriving the format per element (against its parent, c53cde8).

    python3 dev/benchmarks/timing/cuda_quantize_elementwise.py             # 32M elements
    python3 dev/benchmarks/timing/cuda_quantize_elementwise.py 67108864    # 64M

Keep the tensor large. One call through the op costs ~25 us of dispatch before
any kernel runs (``.contiguous()``, ``empty_like``, AT_DISPATCH, launch), so a
size whose kernel is faster than that measures the dispatch, not the kernel --
and a tensor small enough to sit in L2 is exactly that size on this device.
Such a tensor is also the only one whose cast cost is visible at all: past L2
every row here lands on the ``x * 2`` row's figure and the cast is free no
matter what it costs. Measured without the dispatch, in a standalone kernel of
the shipped shape (e4m3, float32, RTX 4060 Laptop, 2026-09), the precomputed
parameters took RNE from 581 to 1393 GB/s on 3.1M L2-resident elements (2.40x),
and on 67M elements every variant ran at 226-227 GB/s.

Each row is timed with CUDA events, median of 15 calls, and the whole set is
round-robined PASSES times with the fastest kept -- laptop GPU clocks wander
enough that a single pass ranks the rows by boost state as much as by cost.
The SR rows also read an int32 tensor of random draws and pay for filling it,
which the ``rng only`` row isolates.

The last rows of each dtype pair an op with its in-place twin
(``binaryK_quantize_`` / ``superfp_quantize_`` / ``fixedpoint_quantize_``), each
after a ``copy_`` that
refills a scratch tensor from the source (the ``copy only`` row), so that both
rows of a pair round the same input every time. Their last column is the peak
of ``torch.cuda.max_memory_allocated`` over one call, above what was allocated
going in, in units of the tensor: one for the out-of-place op, which allocates
its result, and none in place.
"""

import statistics
import sys
from collections.abc import Callable

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
PASSES = 3


def time_ms(fn: Callable[[], object]) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(REPS):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        ts.append(start.elapsed_time(stop))
    return statistics.median(ts)


def rows_for(x: torch.Tensor, name: str, n: int) -> list[tuple[str, Callable[[], object], int]]:
    rw = 2 * x.element_size() * n  # one read + one write
    rows: list[tuple[str, Callable[[], object], int]] = []
    for rm in RoundMode:
        prng = 2 if rm is RoundMode.SR else 0
        rows.append(
            (
                f"binaryK e4m3 {rm.name:<3} {name}",
                lambda x=x, rm=rm, prng=prng: binaryK_quantize(
                    x,
                    8,
                    4,
                    prng_bits=prng,
                    rounding_mode=rm,
                    saturation_mode=SaturationMode.SAT_NONE,
                    subnormals_mode=SubnormalsMode.SUBNORMALS,
                ),
                rw,
            )
        )
    for rm in RoundMode:
        prng = 2 if rm is RoundMode.SR else 0
        rows.append(
            (
                f"superfp m2e5 {rm.name:<3} {name}",
                lambda x=x, rm=rm, prng=prng: superfp_quantize(
                    x,
                    2,
                    5,
                    1,
                    15,
                    prng_bits=prng,
                    rounding_mode=rm,
                    saturation_mode=SaturationMode.SAT_NONE,
                ),
                rw,
            )
        )
    for rm in RoundMode:
        prng = 2 if rm is RoundMode.SR else 0
        rows.append(
            (
                f"fixed  q8.6  {rm.name:<3} {name}",
                lambda x=x, rm=rm, prng=prng: fixedpoint_quantize(
                    x, 8, 6, prng_bits=prng, rounding_mode=rm
                ),
                rw,
            )
        )
    rows.append((f"x * 2        ---     {name}", lambda x=x: x * 2, rw))
    rows.append(
        (
            f"rng only     ---     {name}",
            lambda x=x: torch.randint_like(x, 2**31 - 1, dtype=torch.int32),
            4 * n,
        )
    )
    y = torch.empty_like(x)
    rows.append((f"copy only    ---     {name}", lambda x=x, y=y: y.copy_(x), rw))
    for rm in (RoundMode.RNE, RoundMode.SR):
        prng = 2 if rm is RoundMode.SR else 0
        for label, bk, sf, fx in (
            ("out of place", binaryK_quantize, superfp_quantize, fixedpoint_quantize),
            ("in place", binaryK_quantize_, superfp_quantize_, fixedpoint_quantize_),
        ):
            rows.append(
                (
                    f"copy + binaryK {rm.name:<3} {label} {name}",
                    lambda x=x, y=y, rm=rm, prng=prng, bk=bk: bk(
                        y.copy_(x), 8, 4, prng_bits=prng, rounding_mode=rm
                    ),
                    2 * rw,
                )
            )
            rows.append(
                (
                    f"copy + superfp {rm.name:<3} {label} {name}",
                    lambda x=x, y=y, rm=rm, prng=prng, sf=sf: sf(
                        y.copy_(x), 2, 5, 1, 15, prng_bits=prng, rounding_mode=rm
                    ),
                    2 * rw,
                )
            )
            rows.append(
                (
                    f"copy + fixed   {rm.name:<3} {label} {name}",
                    lambda x=x, y=y, rm=rm, prng=prng, fx=fx: fx(
                        y.copy_(x), 8, 6, prng_bits=prng, rounding_mode=rm
                    ),
                    2 * rw,
                )
            )
    return rows


def peak_tensors(fn: Callable[[], object], nbytes: int) -> float:
    """Peak allocation over one call of ``fn``, above what was allocated going
    in, in units of ``nbytes``."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    del out
    return (torch.cuda.max_memory_allocated() - start) / nbytes


def run(n: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(1)
    x32 = torch.randn(n, generator=g, device="cuda")
    for name, dtype in (
        ("f32", torch.float32),
        ("f16", torch.float16),
        ("bf16", torch.bfloat16),
        ("f64", torch.float64),
    ):
        x = x32.to(dtype)
        rows = rows_for(x, name, n)
        best = [float("inf")] * len(rows)
        for _, fn, _ in rows:
            fn()
        torch.cuda.synchronize()
        for _ in range(PASSES):
            for i, (_, fn, _) in enumerate(rows):
                best[i] = min(best[i], time_ms(fn))
        for (label, fn, nbytes), ms in zip(rows, best, strict=True):
            peak = (
                f"  peak +{peak_tensors(fn, x.nbytes):.0f}x" if label.startswith("copy +") else ""
            )
            print(f"  {label:<40} {ms:8.3f} ms  {nbytes / ms / 1e6:8.1f} GB/s{peak}")
        print()


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 32 * 1024 * 1024
    print(f"{torch.cuda.get_device_name()}, N = {N / 1e6:.1f}M elements, best of {PASSES} passes")
    run(N)
