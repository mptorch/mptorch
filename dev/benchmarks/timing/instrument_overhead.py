"""What ``mptorch.quant.simulate`` costs a model that it leaves in PyTorch's arithmetic.

``simulate`` intercepts every torch call a model makes, through a torch
function mode, and keeps the module stack through hooks on every module. Both
cost Python on every call, whether or not the recipe gives the call formats.
This times ``tests/models.py``'s ``MiniLlama`` (transformers' Llama, call for
call), forward under ``no_grad``, in three variants alternated round by round
(the order flipped each round, the minimum kept):

    plain           the model alone
    passing         ``simulate(Recipe(), model)``: the mode and the hooks, and
                    every call passed through
    routed          ``simulate(Recipe(gemm=GemmFormats()), model)``: every
                    product through mptorch's autograd Functions, with formats
                    that round nothing, so PyTorch's arithmetic runs inside

at two sizes: ``tiny`` (hidden 64, 2 layers, 32 tokens, batch 2), where the
~220 torch calls of a forward are most of its time, and ``small`` (hidden
512, 4 layers, 256 tokens, batch 4), a model whose arithmetic is not
negligible. The gate is ``passing`` against ``plain`` on ``small``: at most
5-10%.

Run from the repository root (check the CPU clock first: a capped clock
scales every number):

    uv run --no-sync python dev/benchmarks/timing/instrument_overhead.py --device cpu
    uv run --no-sync python dev/benchmarks/timing/instrument_overhead.py --device cuda

About a minute per device.

Results (Ryzen 9 7940HS at 4.75 GHz, 8 threads, RTX 4060 laptop GPU, torch 2.15, 2026-11-23):

    cpu   tiny   plain 0.718 ms   passing 1.350 ms (+88.0%)   routed 1.554 ms (+116.4%)
    cpu   small  plain 105.2 ms   passing 110.0 ms  (+4.5%)   routed 107.7 ms   (+2.4%)
    cuda  tiny   plain 1.589 ms   passing 2.513 ms (+58.1%)   routed 2.747 ms  (+72.8%)
    cuda  small  plain 5.331 ms   passing 5.605 ms  (+5.2%)   routed 5.647 ms   (+5.9%)

So the gate holds on both devices, and a tiny model pays the mode's Python
cost on every call, nearly doubling its forward.
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mptorch.quant import GemmFormats, Recipe, simulate  # noqa: E402
from tests.models import MiniLlama  # noqa: E402

CONFIGS = {
    "tiny": dict(hidden=64, layers=2, heads=8, kv_heads=4, seq=32, batch=2),
    "small": dict(hidden=512, layers=4, heads=8, kv_heads=4, seq=256, batch=4),
}


def _timer(device: str, fn, n: int) -> float:
    """Milliseconds per call of ``fn``, over ``n`` calls."""
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def measure(device: str, name: str, rounds: int) -> dict[str, float]:
    c = CONFIGS[name]
    torch.manual_seed(0)
    model = MiniLlama(
        vocab=1000,
        hidden=c["hidden"],
        layers=c["layers"],
        heads=c["heads"],
        kv_heads=c["kv_heads"],
        intermediate=2 * c["hidden"],
    ).to(device)
    ids = torch.randint(0, 1000, (c["batch"], c["seq"]), device=device)
    passing = simulate(Recipe(), model)
    routed = simulate(Recipe(gemm=GemmFormats()), model)

    def plain():
        model(ids)

    def under(sim):
        def run():
            with sim:
                model(ids)

        return run

    variants = {"plain": plain, "passing": under(passing), "routed": under(routed)}
    n = 20 if name == "tiny" else 3
    best = dict.fromkeys(variants, float("inf"))
    with torch.no_grad():
        for fn in variants.values():
            fn()
        for r in range(rounds):
            order = list(variants) if r % 2 == 0 else list(reversed(variants))
            for v in order:
                best[v] = min(best[v], _timer(device, variants[v], n))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    print(f"torch {torch.__version__}, {args.device}, {torch.get_num_threads()} threads")
    for name in CONFIGS:
        best = measure(args.device, name, args.rounds)
        plain = best["plain"]
        cells = [f"{v} {t:8.3f} ms ({100 * (t / plain - 1):+5.1f}%)" for v, t in best.items()]
        verdict = ""
        if name == "small":
            verdict = "  gate: PASS" if best["passing"] <= 1.10 * plain else "  gate: FAIL (>10%)"
        print(f"{name:5}  " + "  ".join(cells) + verdict)


if __name__ == "__main__":
    main()
