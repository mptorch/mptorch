"""
What the fixed-point GEMM ops cost per mac step, against the shipped binaryK
split GEMM, through the ops: the carrier path, the integer accumulator
(split, forced with the op's ``integer`` flag, and fused), and a wide
result; and, in ``layer`` mode, a ``QLinear`` step (forward and backward) on
fixed-point formats against binaryK formats and plain PyTorch.

The binaryK reference is ``Binary16p9`` into ``Binary16p12``, whose casts
take the fast RNE path (every value at or above 2^-125, none past
binary32). ``Binary16p8`` at bias 127, the reference the fixed-point ops
were first compared with, has its smallest value at 2^-133, below the fast
cast's floor (and its top at 2^128, which warns), so it runs the slow cast;
it is the second row. The fixed-point formats are ``(16, 8)``
into ``(16, 4)``, and ``(16, 8)`` into ``(40, 8)`` for the wide result, on
N(0, 16) operands; under the stochastic modes every format draws 4 random
bits.

Run from the repository root::

    .venv/bin/python dev/benchmarks/timing/fixedpoint_gemm.py ops --device cuda
    .venv/bin/python dev/benchmarks/timing/fixedpoint_gemm.py ops --device cpu --threads 1
    .venv/bin/python dev/benchmarks/timing/fixedpoint_gemm.py layer --device cuda
    .venv/bin/python dev/benchmarks/timing/fixedpoint_gemm.py ops --device mps    # on a Mac

Modes: ``ops`` times each row at n^3 (2048 on CUDA, 1024 on MPS, 512 on the
CPU by default), ~1 min on CUDA, ~3 min on one CPU thread; ``layer`` times a
``QLinear(1024, 1024)`` step on 4096 rows, ~1 min on CUDA. Each round runs
every row once, in an order reversed every other round, and each row keeps
its minimum.

Results (2026-11-25, RTX 4060 and the Ryzen 7940HS at 4.6-4.7 GHz; ps per
mac step; SR with the GEMM's own stochastic cast and the integer rounding
from per-launch constants, added 2026-11-25):

=========================================  ======  ======  ======  ======
row                                        CUDA    CUDA    1 thr   1 thr
                                           RNE     SR      RNE     SR
=========================================  ======  ======  ======  ======
binaryK split, Binary16p9 into Binary16p12    7.5    20.3     408   8,150
binaryK split, Binary16p8 (bias 127)         12.3    21.5   3,939   8,369
fixed point, the carrier path                 4.1    23.1     563  10,085
fixed point, integer accumulator, split      13.5    26.4   5,388   9,698
fixed point, integer accumulator, fused      20.1    23.6   9,825   6,263
fixed point, split into (40, 8), wide         8.6    21.6   1,004   8,036
=========================================  ======  ======  ======  ======

SR_C times as SR. Under the other modes (2026-11-26, the GEMM's own
deterministic layouts; ``--modes RNA RU RD RZ RO``), the carrier path on
CUDA: 10.9 (RNA), 10.9 (RU), 11.6 (RD), 8.3 (RZ), 12.0 (RO); on one CPU
thread 4,260, 5,241, 5,477, 2,764 and 4,929 (at 384^3). On 16 threads: 53
(binaryK, the vector tiles), 77 (the carrier path), 530 (integer split) and
875 (integer fused) under RNE; 1,027,
1,308, 1,233 and 830 under SR. ``layer`` on CUDA, a QLinear step: plain
2.9 ms; binaryK 98 ms (RNE) and 286 ms (SR); fixed point 54 ms (RNE) and
336 ms (SR). Before the stochastic layouts (137dc14) SR was, on CUDA, 29.8
(carrier), 37.3 (split), 28.9 (fused) and 27.3 (wide), and on one CPU thread
9,846, 13,367, 10,974 and 8,280; the layer 401 ms.

On an Apple M3 Pro (2026-11-26, macOS 26.7, at 1024^3; MPS has no wide row,
since a wide result is float64), with the layouts on Metal too (every
deterministic mode on the line, in the device's logical form, and the SR
layouts), ps per mac step:

=========================================  =====  =====  =====  =====  =====  =====  =====
row                                        RNE    RNA    RU     RD     RZ     RO     SR
=========================================  =====  =====  =====  =====  =====  =====  =====
binaryK split, Binary16p9 into Binary16p12  10.4   18.0   82.1   82.2  109.4   16.8   62.3
fixed point, the carrier path                5.0   22.5   22.1   22.1   19.4   22.6   60.7
fixed point, integer accumulator, split     36.0   46.1   41.1   36.6   39.6   40.1   81.6
fixed point, integer accumulator, fused     49.9   50.5   45.7   40.8   46.1   45.3   63.8
=========================================  =====  =====  =====  =====  =====  =====  =====

Before Metal took the layouts, the carrier path took 24.5, 48.4, 48.4, 57.8,
25.3 and 75.3 under RNA, RU, RD, RZ, RO and SR. ``layer`` there: plain 7.0
ms; binaryK 133 ms (RNE) and 783 ms (SR); fixed point 63 ms (RNE) and 842 ms
(SR; 993 before). On its CPU (one thread, 384^3, Apple
clang), the carrier path takes 1,195 (RNE), 2,981 (RU), 2,299 (RD), 2,302
(RZ) and 9,725 (SR), and with every deterministic mode but RNE on the
GEMM's line (the arm64 host's choice) 3,211 (RNA) and 3,592 (RO), against
11,095 and 12,093 through the shipped casts.
"""

import argparse
import time
from collections.abc import Callable
from typing import Any

import torch

from mptorch import FixedPoint, RoundMode
from mptorch.quant import (
    QLinear,
    Quant,
    binaryK_gemm_formats,
    binaryK_matmul,
    fixedpoint_gemm_formats,
    fixedpoint_matmul,
    fixedpoint_matmul_fma,
)
from mptorch.quant.ops.fixedpoint import _fixedpoint_spec

R = RoundMode
MODES = {m.name: m for m in (R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO, R.SR, R.SR_C)}


def _timer(device: str) -> Callable[[Callable[[], object]], float]:
    """Seconds one call of ``f`` takes, measured once."""
    if device == "cuda":

        def cuda(f: Callable[[], object]) -> float:
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            f()
            e1.record()
            torch.cuda.synchronize()
            return e0.elapsed_time(e1) / 1e3

        return cuda

    def host(f: Callable[[], object]) -> float:
        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        f()
        if device == "mps":
            torch.mps.synchronize()
        return time.perf_counter() - t0

    return host


def _alternated(
    rows: dict[str, Callable[[], object]], rounds: int, device: str
) -> dict[str, float]:
    """Each row's minimum over ``rounds``, every round running every row once,
    the order reversed every other round."""
    once = _timer(device)
    names = list(rows)
    for name in names:  # warm-up: compiles (Metal's kernels too), allocates, fills caches
        rows[name]()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    best = dict.fromkeys(names, float("inf"))
    for r in range(rounds):
        for name in names if r % 2 == 0 else reversed(names):
            best[name] = min(best[name], once(rows[name]))
    return best


def _operands(n: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    a = torch.randn(n, n, generator=g) * 4
    b = torch.randn(n, n, generator=g) * 4
    return a.to(device), b.to(device)


def _op_rows(a: torch.Tensor, b: torch.Tensor, mode: RoundMode) -> dict[str, Callable[[], object]]:
    prng = 4 if mode is not R.RNE else 0
    split: dict[str, Any] = dict(
        mul_wl=16, mul_fl=8, acc_wl=16, acc_fl=4, mul_prng_bits=prng, acc_prng_bits=prng
    )
    spec = _fixedpoint_spec(**split, rounding_mode=mode)

    def integer() -> object:  # the op's test hook: the integer accumulator on the same pair
        return spec.op(a, b, False, False, *spec.args, True)

    rows = {
        "binaryK split, Binary16p9 into Binary16p12": lambda: binaryK_matmul(
            a,
            b,
            mul_K=16,
            mul_P=9,
            acc_K=16,
            acc_P=12,
            rounding_mode=mode,
            mul_prng_bits=prng,
            acc_prng_bits=prng,
        ),  # fmt: skip
        "binaryK split, Binary16p8 (bias 127) into Binary16p12": lambda: binaryK_matmul(
            a,
            b,
            mul_K=16,
            mul_P=8,
            mul_bias=127,
            acc_K=16,
            acc_P=12,
            rounding_mode=mode,
            mul_prng_bits=prng,
            acc_prng_bits=prng,
        ),  # fmt: skip
        "fixed point, the carrier path": lambda: fixedpoint_matmul(
            a, b, **split, rounding_mode=mode
        ),
        "fixed point, integer accumulator, split": integer,
        "fixed point, integer accumulator, fused": lambda: fixedpoint_matmul_fma(
            a, b, fma_wl=16, fma_fl=4, fma_prng_bits=prng, rounding_mode=mode
        ),
        "fixed point, split into (40, 8), wide": lambda: fixedpoint_matmul(
            a,
            b,
            mul_wl=16,
            mul_fl=8,
            acc_wl=40,
            acc_fl=8,
            mul_prng_bits=prng,
            acc_prng_bits=prng,
            rounding_mode=mode,
        ),  # fmt: skip
    }
    if a.device.type == "mps":  # a wide result is float64, which MPS does not have
        del rows["fixed point, split into (40, 8), wide"]
    return rows


def ops(args: argparse.Namespace) -> None:
    n = args.n or {"cuda": 2048, "mps": 1024}.get(args.device, 512)
    a, b = _operands(n, args.device)
    for label in args.modes:
        best = _alternated(_op_rows(a, b, MODES[label]), args.rounds, args.device)
        for name, t in best.items():
            print(f"{label:5s} {name:55s} {t * 1e3:9.2f} ms {t / n**3 * 1e12:8.2f} ps/step")


def _layer_rows(
    device: str, mode: RoundMode, x: torch.Tensor, g: torch.Tensor
) -> dict[str, Callable[[], object]]:
    prng = 4 if mode is not R.RNE else 0
    fp = fixedpoint_gemm_formats(
        16, 8, acc_wl=16, acc_fl=4, mul_prng_bits=prng, acc_prng_bits=prng, rounding_mode=mode
    )
    fp.input_quant = fp.weight_quant = Quant(FixedPoint(8, 4))
    bk = binaryK_gemm_formats(
        16, 9, acc_K=16, acc_P=12, mul_prng_bits=prng, acc_prng_bits=prng, rounding_mode=mode
    )
    bk.input_quant = bk.weight_quant = Quant(FixedPoint(8, 4))
    layers: dict[str, torch.nn.Linear] = {
        "plain nn.Linear": torch.nn.Linear(1024, 1024),
        "QLinear, binaryK split (Binary16p9 into Binary16p12)": QLinear(1024, 1024, formats=bk),
        "QLinear, fixed point (16, 8) into (16, 4)": QLinear(1024, 1024, formats=fp),
    }

    # Weights the operand quantizer holds: nn.Linear's default init, within
    # +-1/32 here, would round every weight of FixedPoint(8, 4) (step 1/16)
    # to zero, and with every product zero the rows would time their
    # formats' zero arms. The same weights for every row.
    weight = torch.randn(1024, 1024, generator=torch.Generator().manual_seed(2)) * 0.5

    def step(layer: torch.nn.Linear) -> Callable[[], object]:
        with torch.no_grad():
            layer.weight.copy_(weight)
        layer.to(device)

        def run() -> object:
            layer.zero_grad(set_to_none=True)
            xg = x.detach().requires_grad_()
            layer(xg).backward(g)
            return xg.grad

        return run

    return {name: step(layer) for name, layer in layers.items()}


def layer(args: argparse.Namespace) -> None:
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(4096, 1024, generator=gen).to(args.device)
    g = torch.randn(4096, 1024, generator=gen).to(args.device)
    for label in args.modes:
        best = _alternated(_layer_rows(args.device, MODES[label], x, g), args.rounds, args.device)
        for name, t in best.items():
            print(f"{label:5s} {name:55s} {t * 1e3:9.2f} ms a step")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("mode", choices=["ops", "layer"])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--modes", nargs="+", default=["RNE", "SR"], choices=list(MODES))
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    ops(args) if args.mode == "ops" else layer(args)


if __name__ == "__main__":
    main()
