"""What compiling a model under ``mptorch.quant.simulate`` costs as it grows.

A recipe with formats per layer gives every layer's products their own
mptorch Functions, so a compiled model's graph, its guards and its compile
time grow with its depth. This compiles ``tests/models.py``'s ``MiniLlama``
(transformers' Llama, call for call; hidden 32, eager attention) at 4, 8, 16
and 32 layers, through ``torch.compile(sim.wrap(), backend="aot_eager",
fullgraph=True)``, with its even layers' linears in one 8-bit format and its
odd layers' in another, and records for each depth:

    compile     the first call, forward and backward: Dynamo's trace and
                AOTAutograd's
    second      a second call, forward and backward
    frames      compiled frames after both calls: 1 means no graph break and
                no recompile
    Functions   mptorch Functions in the graph (7 per layer and the head)
    guards      the guard tree's entries, and Dynamo's own measure of the
                time it takes to check them on each call

Run from the repository root:

    uv run --no-sync python dev/benchmarks/timing/compile_scale.py [--layers 4 8 16 32]

About a minute and a half for the four depths.

Results (Ryzen 9 7940HS at 4.75 GHz, torch 2.15, CPU, 8 threads, 2026-11-23):

    layers  compile      second    frames  Functions  guards  guard check
         4     4.7 s     12.5 ms       1         29    4103     579.9 us
         8     8.3 s     11.7 ms       1         57    7177    1554.6 us
        16    16.7 s     33.8 ms       1        113   13325    1688.0 us
        32    33.1 s     58.0 ms       1        225   25621    3151.4 us

One graph at every depth; compile time and guards grow linearly, about a
second and 800 guard entries per layer.
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import torch
from torch._dynamo.testing import CompileCounterWithBackend

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mptorch import BinaryK  # noqa: E402
from mptorch.quant import GemmFormats, Quant, Recipe, SplitMac, simulate  # noqa: E402
from tests.models import MiniLlama  # noqa: E402


class _Guards(logging.Handler):
    """Dynamo's ``guards`` log: the guard tree and its evaluation latency."""

    def __init__(self):
        super().__init__()
        self.entries = 0
        self.latency_us: list[float] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        self.entries += sum(1 for line in message.splitlines() if "+- " in line)
        found = re.search(r"Guard eval latency = ([0-9.]+) us", message)
        if found:
            self.latency_us.append(float(found.group(1)))


def _formats(p: int) -> GemmFormats:
    fmt = BinaryK(8, p)
    return GemmFormats.uniform(SplitMac(fmt, BinaryK(16, 11)), Quant(fmt))


def measure(layers: int) -> dict:
    torch._dynamo.reset()
    torch.manual_seed(0)
    model = MiniLlama(layers=layers)
    ids = torch.randint(0, 64, (2, 16))
    recipe = Recipe(linear=_formats(4))
    for i in range(0, layers, 2):
        recipe = recipe.override(f"model.layers.{i}.*", linear=_formats(3))
    sim = simulate(recipe, model)
    wrapped = sim.wrap()
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(wrapped, backend=counter, fullgraph=True)
    guards = _Guards()
    logger = logging.getLogger("torch._dynamo.guards.__guards")
    torch._logging.set_logs(guards=True)
    logger.handlers = [guards]
    logger.propagate = False
    try:
        times = []
        for _ in range(2):
            model.zero_grad(set_to_none=True)
            t = time.perf_counter()
            compiled(ids).sum().backward()
            times.append(time.perf_counter() - t)
    finally:
        torch._logging.set_logs()
        wrapped.close()
    functions = sum(
        1
        for graph in counter.graphs
        for node in graph.graph.nodes
        if "autograd_function_apply" in str(node.target)
    )
    return dict(
        layers=layers,
        compile_s=times[0],
        second_ms=1e3 * times[1],
        frames=counter.frame_count,
        functions=functions,
        guards=guards.entries,
        guard_us=guards.latency_us[-1] if guards.latency_us else float("nan"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()
    print(f"torch {torch.__version__}, CPU, {torch.get_num_threads()} threads")
    print("layers  compile      second    frames  Functions  guards  guard check")
    for n in args.layers:
        r = measure(n)
        print(
            f"{r['layers']:6}  {r['compile_s']:6.1f} s  {r['second_ms']:7.1f} ms  {r['frames']:6}"
            f"  {r['functions']:9}  {r['guards']:6}  {r['guard_us']:8.1f} us"
        )


if __name__ == "__main__":
    main()
