"""What the CPU GEMM's vector tiles cost on tiles of a few outputs.

The CPU kernel runs a binaryK GEMM's packed tiles in vector code
(``cpu/gemm_rne_tile.cpp``, ``cpu/gemm_odd_tile.cpp``,
``cpu/gemm_word_tile.cpp``, ``cpu/gemm_word_odd_tile.cpp``), whose K-steps
make passes over the tile's outputs. On a tile of one output or a few (a sum
into one scalar, as ``qsum`` over a whole tensor runs; a matrix-vector
product; a GEMM of a few rows and columns), the passes run one lane or a few,
and the scalar loop can be the faster. The tiles hand a tile narrower than a
cut-off back to the scalar loop whole, so this times the two on such shapes
to set it: M x N outputs with K = 2^20 / (M N), so that every shape makes
2^20 mac steps, one thread, through the ops.

The macs, binaryK, each a tile family:

    RNE            SplitMac(Binary8p4, Binary16p11), the RNE tiles
    RNE odd        the same with carrier_rounding=RO, the odd tiles
    RZ             the same under RZ, the word tiles
    RZ odd         RZ with carrier_rounding=RO, the odd word tiles
    fused RNE odd  FusedMac(Binary16p11, carrier_rounding=RO), the odd tiles

It runs the shapes in one process per ``MPTORCH_CPU_SIMD`` setting (``0``,
the scalar loop, against the default), in two rounds with the order reversed,
and prints ns a mac step and their ratio, tiles over scalar. Below the
cut-off both settings run the scalar loop, so the ratio there is about 1; the
cut-offs were set on a build whose tiles took every shape. Run from the
repository root (check the CPU clock first)::

    .venv/bin/python dev/benchmarks/timing/cpu_narrow_gemm.py

About 1 min. Ryzen 7940HS at 4.7 GHz, 2026-11-27, every shape taken by the
tiles, tiles / scalar: 1.23-1.98x at one output, 0.73-1.57x at two, 0.43-1.35x
at four (the RNE and odd tiles faster, the word tiles slower), 0.07-0.58x
from eight on; hence the cut-offs of four outputs for the RNE and odd tiles
and eight for the word tiles.
"""

import argparse
import os
import subprocess
import sys
import textwrap

_CASES = textwrap.dedent(
    """
    import sys, time, torch
    from mptorch import BinaryK, RoundMode as R
    from mptorch.quant import FusedMac, SplitMac
    from mptorch.quant.mac import spec_for_mac
    from mptorch.quant.ops._spec import _run_gemm

    torch.set_num_threads(1)
    f16, f8 = BinaryK(16, 11), BinaryK(8, 4)
    g = torch.Generator().manual_seed(0)
    macs = {
        "RNE": SplitMac(f8, f16),
        "RNE odd": SplitMac(f8, f16, carrier_rounding=R.RO),
        "RZ": SplitMac(f8, f16, rounding=R.RZ),
        "RZ odd": SplitMac(f8, f16, rounding=R.RZ, carrier_rounding=R.RO),
        "fused RNE odd": FusedMac(f16, carrier_rounding=R.RO),
    }
    shapes = [(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (1, 32), (2, 1), (4, 1), (8, 1),
              (16, 1), (32, 1), (2, 2), (4, 4), (8, 8)]
    for m, n in shapes:
        k = 2**20 // (m * n)
        a = torch.randn(m, k, generator=g)
        b = torch.randn(k, n, generator=g)
        for name, mac in macs.items():
            spec = spec_for_mac(mac)
            _run_gemm(spec, a, b, False, False)
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                _run_gemm(spec, a, b, False, False)
                ts.append(time.perf_counter() - t0)
            print(f"{m}x{n}\\t{name}\\t{min(ts) / 2**20 * 1e9:.2f}")
    """
)


def _run(simd: str | None) -> list[tuple[str, str, float]]:
    env = dict(os.environ)
    env.pop("MPTORCH_CPU_SIMD", None)
    if simd is not None:
        env["MPTORCH_CPU_SIMD"] = simd
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", _CASES],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        shape, name, ns = line.split("\t")
        rows.append((shape, name, float(ns)))
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    best: dict[tuple[str, str], dict[str, float]] = {}
    for settings in (("0", None), (None, "0")):
        for simd in settings:
            for shape, name, ns in _run(simd):
                cell = best.setdefault((shape, name), {})
                key = simd or "tiles"
                cell[key] = min(cell.get(key, float("inf")), ns)
    names = list(dict.fromkeys(name for _, name in best))
    shapes = list(dict.fromkeys(shape for shape, _ in best))
    print("ns a mac step, one thread: scalar / tiles, and tiles over scalar")
    print(f"{'M x N':6s} " + " ".join(f"{n:>22s}" for n in names))
    for shape in shapes:
        cells = []
        for name in names:
            c = best[(shape, name)]
            cells.append(f"{c['0']:6.2f} /{c['tiles']:6.2f} {c['tiles'] / c['0']:5.2f}x")
        print(f"{shape:6s} " + " ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
