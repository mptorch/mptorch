#!/usr/bin/env python3
"""What stochastic rounding costs per rounding, in each of P3109's three modes.

Commit cf69807 made every stochastic cast exact (all regions in integers on the
word) and added StochasticB and StochasticC (``RoundMode.SR_B``, ``SR_C``) to
the one stochastic instantiation each kernel has, its variant chosen from
constants the policy builds once. This times what that costs, per element for
the quantizers and per mac step for the GEMMs:

    binaryK, superfp and fixed-point quantizers   2^24 elements
    binaryK split and fused GEMMs, superfp split  CPU 256^3, CUDA 1024^3

on the CPU (one thread and five) and on CUDA, minimum over ``--rounds``. A
build before cf69807 (4164cee or older) has only ``SR``; ``bench`` prints the
modes the build has. Compare two builds by running this under each in turn,
alternated, which is how the exact casts were costed against 4164cee.

Run: python3 dev/benchmarks/timing/sr_variants.py bench [--rounds 5]
"""

import argparse
import time
from functools import partial

import torch

from mptorch import RoundMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_quantize,
    fixedpoint_quantize,
    superfp_matmul,
    superfp_quantize,
)


def _modes() -> list[RoundMode]:
    """The stochastic modes this build's kernels take. A build before cf69807
    has only SR: its GEMMs refuse SR_B and SR_C, while its quantizers took
    them for a deterministic mode, so the probe is a GEMM."""
    x = torch.ones(2, 2)
    have = []
    for m in RoundMode:
        if m.value < 6:
            continue
        try:
            binaryK_matmul(x, x, mul_K=8, mul_P=4, acc_K=16, acc_P=11, rounding_mode=m)
        except RuntimeError:
            continue
        have.append(m)
    return have


def _time(fn, device: str, rounds: int) -> float:
    best = float("inf")
    fn()
    for _ in range(rounds):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def _case(device: str, threads: int, rounds: int, modes: list[RoundMode]) -> None:
    """One device's rows: each op, its time per element or step in each mode."""
    if threads:
        torch.set_num_threads(threads)
    where = f"{device} {threads} thr" if threads else device
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1 << 24, generator=g).to(device)
    n = 1024 if device == "cuda" else 256
    a = torch.randn(n, n, generator=g).to(device)
    b = torch.randn(n, n, generator=g).to(device)

    def bk_quant(m):
        return binaryK_quantize(x, 8, 4, prng_bits=8, rounding_mode=m)

    def sf_quant(m):
        return superfp_quantize(x, 3, 4, 2, 7, prng_bits=8, rounding_mode=m)

    def fx_quant(m):
        return fixedpoint_quantize(x, 8, 5, prng_bits=8, rounding_mode=m)

    def bk_split(m):
        return binaryK_matmul(
            a,
            b,
            mul_K=8,
            mul_P=4,
            mul_prng_bits=8,
            acc_K=16,
            acc_P=11,
            acc_prng_bits=8,
            rounding_mode=m,
        )

    def bk_fused(m):
        return binaryK_matmul_fma(a, b, fma_K=16, fma_P=11, fma_prng_bits=8, rounding_mode=m)

    def sf_split(m):
        return superfp_matmul(
            a, b, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=2, mul_bias=7, mul_prng_bits=8,
            acc_man_bits=10, acc_exp_bits=5, acc_normal_binades=8, acc_bias=15, acc_prng_bits=8,
            rounding_mode=m,
        )  # fmt: skip

    ops = {
        "binaryK quantize": (x.numel(), bk_quant),
        "superfp quantize": (x.numel(), sf_quant),
        "fixedpoint quantize": (x.numel(), fx_quant),
        "binaryK split GEMM": (n**3, bk_split),
        "binaryK fused GEMM": (n**3, bk_fused),
        "superfp split GEMM": (n**3, sf_split),
    }
    per = (threads or 1) * (1e9 if device == "cpu" else 1e12)
    unit = "ns" if device == "cpu" else "ps"
    for name, (count, fn) in ops.items():
        cells = []
        for m in modes:
            t = _time(partial(fn, m), device, rounds)
            cells.append(f"{t / count * per:.3f} {unit}")
        print(f"| {where} | {name} | " + " | ".join(cells) + " |", flush=True)


def bench(args: argparse.Namespace) -> None:
    cases = [("cpu", 1), ("cpu", 5)] + ([("cuda", 0)] if torch.cuda.is_available() else [])
    modes = _modes()
    print("| device | op | " + " | ".join(m.name for m in modes) + " |")
    print("|---|---|" + "---|" * len(modes))
    for device, threads in cases:
        _case(device, threads, args.rounds, modes)
    print("(CPU times are per element or step per thread: wall time x threads / count.)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--rounds", type=int, default=5)
    args = p.parse_args()
    {"bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
