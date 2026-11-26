"""What a CPU palette GEMM costs against its single-format twin.

A palette GEMM (``binaryK_matmul_mixed``, ``binaryK_matmul_fma_mixed``) gives
each output element the palette entry its map ``prec_idx`` selects. This
times it on the CPU, through the ops, against the single-format GEMM of one
of its entries, for the maps a model meets:

    one        every element entry 0 (the map a palette layer has before
               set_palette_maps_)
    columns    a random entry per output column, [1, N] (a linear layer's
               per-neuron map)
    halves     entry 0 for the first half of the columns, 1 for the rest
    rows       a random entry per output row, [M, 1] (a conv's per-channel
               map is per row of its GEMM)
    elements   a random entry per element, [M, N]

The palette is two entries, binaryK under RNE: the split mac (Binary8p4,
Binary16p11) and (Binary16p11, Binary16p11), the fused mac Binary16p11 and
Binary8p4; the twin is entry 0's single-format GEMM. Each time is the minimum
over ``--rounds`` rounds, the variants alternated, the order reversed every
other round. ``MPTORCH_CPU_SIMD=0`` in the environment keeps the scalar loop,
for the vector tiles' share; ``--carrier-rounding RO`` times the odd macs
(``carrier_rounding=RoundMode.RO``), palette and twin alike.

Run from the repository root (check the CPU clock first)::

    .venv/bin/python dev/benchmarks/timing/cpu_palette_gemm.py --threads 1
    .venv/bin/python dev/benchmarks/timing/cpu_palette_gemm.py --threads 16
    MPTORCH_CPU_SIMD=0 .venv/bin/python dev/benchmarks/timing/cpu_palette_gemm.py --threads 1

About a minute each.

Results (Ryzen 9 7940HS at 4.7-4.8 GHz, 2026-11-26), against the twin, at
256^3 and 32 x 128 x 128:

                         one       columns   halves    rows      elements
    1 thread   split     1.0-1.1x  1.5x      1.1x      1.1x      1.5-1.6x
               fused     1.1x      1.4x      1.1x      1.2x      1.5-1.6x
    16 threads split     1.0-1.2x  1.3-1.5x  1.0-1.2x  1.1-1.3x  1.5-1.8x
               fused     1.2x      1.4-1.5x  1.2-1.3x  1.2-1.3x  1.5-1.7x
    1 thread,  split     1.0-1.1x  1.3-1.4x  1.0-1.1x  1.1-1.2x  2.2x
    RO         fused     1.1x      1.6x      1.1x      1.3x      2.4x

Before the palette tiles (the scalar loop for every palette) the nearest
step took 9.4-12.4x, and with MPTORCH_CPU_SIMD=0, which is the scalar loop
for the twin too, 1.1x; the odd macs' palettes took about 25x their twins'
vector tiles (191 ms against 7.4 at 256^3, one thread).
"""

import argparse
import time
from typing import Any

import torch

from mptorch import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_fma_mixed,
    binaryK_matmul_mixed,
)

SHAPES = [(256, 256, 256), (32, 128, 128)]
MAPS = ["one", "columns", "halves", "rows", "elements"]


def _map(kind: str, m: int, n: int, g: torch.Generator) -> torch.Tensor:
    if kind == "one":
        return torch.zeros(1, n, dtype=torch.int32)
    if kind == "columns":
        return torch.randint(0, 2, (1, n), generator=g, dtype=torch.int32)
    if kind == "halves":
        return (torch.arange(n) >= n // 2).to(torch.int32).view(1, n)
    if kind == "rows":
        return torch.randint(0, 2, (m, 1), generator=g, dtype=torch.int32)
    return torch.randint(0, 2, (m, n), generator=g, dtype=torch.int32)


def _cases(m: int, k: int, n: int, rounding: RoundMode):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(m, k, generator=g)
    b = torch.randn(k, n, generator=g)
    odd: dict[str, Any] = dict(carrier_rounding=rounding)
    split: dict[str, Any] = dict(
        mul_K=[8, 16], mul_P=[4, 11], acc_K=[16, 16], acc_P=[11, 11], **odd
    )
    fused: dict[str, Any] = dict(fma_K=[16, 8], fma_P=[11, 4], **odd)
    cases = {
        "split, twin": lambda: binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=11, **odd),
        "fused, twin": lambda: binaryK_matmul_fma(a, b, fma_K=16, fma_P=11, **odd),
    }
    for kind in MAPS:
        p = _map(kind, m, n, g)
        cases[f"split, {kind}"] = lambda p=p: binaryK_matmul_mixed(a, b, p, **split)
        cases[f"fused, {kind}"] = lambda p=p: binaryK_matmul_fma_mixed(a, b, p, **fused)
    return cases


def _time(fn, reps: int) -> float:
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--carrier-rounding", choices=["RNE", "RO"], default="RNE")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rounding = RoundMode[args.carrier_rounding]
    print(
        f"torch {torch.__version__}, {args.threads} threads, carrier_rounding "
        f"{rounding.name}; ms, and against the twin"
    )
    for m, k, n in SHAPES:
        cases = _cases(m, k, n, rounding)
        reps = max(1, int(2e7 // (m * k * n)))
        for fn in cases.values():
            fn()
        best = dict.fromkeys(cases, float("inf"))
        for r in range(args.rounds):
            names = list(cases) if r % 2 == 0 else list(reversed(cases))
            for name in names:
                best[name] = min(best[name], _time(cases[name], reps))
        print(f"\n{m} x {k} x {n}")
        for mac in ("split", "fused"):
            twin = best[f"{mac}, twin"]
            cells = [f"twin {twin * 1e3:7.3f}"]
            for kind in MAPS:
                t = best[f"{mac}, {kind}"]
                cells.append(f"{kind} {t * 1e3:7.3f} ({t / twin:4.1f}x)")
            print(f"  {mac:5}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
