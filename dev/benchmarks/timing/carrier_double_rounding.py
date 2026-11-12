#!/usr/bin/env python3
"""How often a GEMM's carrier rounds a mac step wrongly, and what the odd step costs.

A NAIVE GEMM step rounds twice: the carrier's add or FMA to nearest, then the
format's cast. The ``custom_matmul_*_odd`` ops (``carrier_rounding=RoundMode.RO``,
since a13760a) round the carrier step to odd instead, which makes the cast
round the exact value. This script measures both sides of that:

``rates``
    The share of outputs on which the binary32 carrier's GEMM and the
    binary64 carrier's differ (M = N = 256, operands from N(0, 1) quantized
    to the format with RNE, default saturation and subnormals). With the odd
    ops built, two more columns per mode: odd binary32 against odd binary64,
    which is 0.00% for every format inside the guarantee's bounds, and the
    default binary64 op against the odd one, which is how wrong the binary64
    carrier itself is.
``bench``
    Time per mac step of each default op and its odd twin, alternated round
    by round (min of ``--rounds``): the CPU at one and five threads, CUDA in
    binary32 (1024^3) and binary64 (512^3).

Both call the ops through the specs the Python layer builds, so they run as
soon as the C++ ops exist.

Run: python3 dev/benchmarks/timing/carrier_double_rounding.py rates [--device cuda]
     python3 dev/benchmarks/timing/carrier_double_rounding.py bench [--rounds 3] [--cpu-n 256]
"""

import argparse
import time
import warnings
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch

from mptorch import RoundMode
from mptorch.number import FormatRangeWarning
from mptorch.quant import binaryK_quantize
from mptorch.quant.ops import _binaryK_fma_spec, _binaryK_spec, _GemmSpec

R = RoundMode
FORMATS = [(8, 4), (8, 3), (12, 6), (16, 8), (16, 11), (16, 12), (24, 16)]
RATE_MODES = [R.RNE, R.RZ, R.RU, R.RD, R.RO]


def _spec(mac: str, K: int, P: int, mode: RoundMode) -> _GemmSpec:
    if mac == "split":
        return _binaryK_spec(mul_K=K, mul_P=P, rounding_mode=mode)
    return _binaryK_fma_spec(fma_K=K, fma_P=P, rounding_mode=mode)


def _odd_op(spec: _GemmSpec) -> Callable | None:
    """The odd twin of a spec's op, or None before it is built."""
    name = cast(Any, spec.op)._schema.name.split("::")[1] + "_odd"
    packet = getattr(torch.ops.mptorch, name, None)
    return None if packet is None else packet.default


def _call(op: Callable, spec: _GemmSpec, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return op(a, b, False, False, *spec.args)


def _differ(x: torch.Tensor, y: torch.Tensor) -> float:
    """The share of elements that differ, NaNs counting as equal to NaNs."""
    same = (x == y) | (x.isnan() & y.isnan())
    return 100.0 * (1.0 - same.double().mean().item())


def rates(args: argparse.Namespace) -> None:
    # binary24p16's smallest value is below binary32's reach (2^-142); it is
    # measured all the same, on values far from that floor.
    warnings.simplefilter("ignore", FormatRangeWarning)
    dev = args.device
    torch.manual_seed(0)
    n = 256
    have_odd = _odd_op(_spec("fused", 8, 4, R.RNE)) is not None
    head = "| mac | format | K | " + " | ".join(m.name for m in RATE_MODES) + " |"
    if have_odd:
        head += " odd 32 vs 64 (max) | default 64 vs odd 64 (max) |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for mac in ("split", "fused"):
        for K, P in FORMATS:
            for k in (16, 256):
                a = binaryK_quantize(torch.randn(n, k, device=dev), K, P)
                b = binaryK_quantize(torch.randn(k, n, device=dev), K, P)
                a64, b64 = a.double(), b.double()
                cells, odd_cells, b64_cells = [], [], []
                for mode in RATE_MODES:
                    spec = _spec(mac, K, P, mode)
                    d32 = _call(spec.op, spec, a, b).double()
                    d64 = _call(spec.op, spec, a64, b64)
                    cells.append(f"{_differ(d32, d64):.2f}%")
                    odd = _odd_op(spec)
                    if odd is not None:
                        o32 = _call(odd, spec, a, b).double()
                        o64 = _call(odd, spec, a64, b64)
                        odd_cells.append(_differ(o32, o64))
                        b64_cells.append(_differ(d64, o64))
                row = f"| {mac} | binary{K}p{P} | {k} | " + " | ".join(cells) + " |"
                if have_odd:
                    row += f" {max(odd_cells):.2f}% | {max(b64_cells):.2f}% |"
                print(row, flush=True)


def _time(fn: Callable[[], Any], device: str) -> float:
    if device == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1e3
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def bench(args: argparse.Namespace) -> None:
    cases = []
    for thr in (1, 5):
        cases.append(("cpu", thr, torch.float32, args.cpu_n))
        cases.append(("cpu", thr, torch.float64, args.cpu_n))
    if torch.cuda.is_available():
        cases.append(("cuda", 0, torch.float32, 1024))
        cases.append(("cuda", 0, torch.float64, 512))
    print("| device | carrier | mac | format | mode | default | odd | odd / default |")
    print("|---|---|---|---|---|---|---|---|")
    for dev, thr, dtype, n in cases:
        if thr:
            torch.set_num_threads(thr)
        torch.manual_seed(0)
        unit, scale = ("ns", 1e9) if dev == "cpu" else ("ps", 1e12)
        for mac in ("fused", "split"):
            for K, P in ((8, 4), (16, 8)):
                a = binaryK_quantize(torch.randn(n, n, device=dev, dtype=dtype), K, P)
                b = binaryK_quantize(torch.randn(n, n, device=dev, dtype=dtype), K, P)
                for mode in (R.RNE, R.RU, R.SR):
                    spec = _spec(mac, K, P, mode)
                    odd = _odd_op(spec)
                    plain = partial(_call, spec.op, spec, a, b)
                    twin = None if odd is None else partial(_call, odd, spec, a, b)
                    plain()
                    if twin is not None:
                        twin()
                    ts, to = [], []
                    for _ in range(args.rounds):
                        ts.append(_time(plain, dev))
                        if twin is not None:
                            to.append(_time(twin, dev))
                    per = scale * thr if thr else scale
                    t0 = min(ts) / n**3 * per
                    where = f"{dev} {thr} thr" if thr else dev
                    carrier = "binary64" if dtype is torch.float64 else "binary32"
                    if to:
                        t1 = min(to) / n**3 * per
                        tail = f"{t1:.2f} {unit} | {t1 / t0:.2f} |"
                    else:
                        tail = "— | — |"
                    print(
                        f"| {where} | {carrier} | {mac} | binary{K}p{P} | {mode.name} | "
                        f"{t0:.2f} {unit} | {tail}",
                        flush=True,
                    )
    print("(CPU times are per step per thread: wall time × threads / n³.)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rates")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    b = sub.add_parser("bench")
    b.add_argument("--rounds", type=int, default=3)
    b.add_argument("--cpu-n", type=int, default=256)
    args = p.parse_args()
    {"rates": rates, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
