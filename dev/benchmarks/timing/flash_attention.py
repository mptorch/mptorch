"""What flash attention costs against the standard path: the tile loop, the
batched path, and optionally compiled code.

Each case is ``qattention`` over causal ``[1, 8, L, 64]`` float32 operands in
one of two recipes:

    carrier   q, k and v rounded to E4M3 by plain ``Quant`` slots, both
              products and the softmax's two sums ``FusedMac(None)``:
              mptorch's GEMMs in the carrier, so flash runs its batched path
              (every query tile a key tile reaches, in one call)
    torch     the same operands, both products and the softmax torch's own:
              flash keeps its loop, since cuBLAS's words for a taller call
              are not its words for a tile

and these variants:

    standard     the product, softmax and product over whole tensors
    flash-loop   FlashFormats(block_q=128, block_kv=128), forced onto the
                 loop: one query tile against one key tile at a time
    flash        the same formats as flash runs them (batched where they
                 allow it)
    *-c, *-cg    with ``--compile``: under torch.compile(fullgraph=True),
                 Inductor's default mode, and mode="reduce-overhead" (CUDA
                 graphs; CUDA only)

forward only, and forward and backward. Each variant is the minimum over
``--rounds`` rounds, alternating in order, after a warm-up (which compiles).
``flash`` is compared with ``flash-loop`` and a compiled variant with its
eager twin, forward and gradients: where words differ, the largest
difference relative to the result's largest magnitude is printed. The
batched path gives the loop's words; Inductor's Triton kernels contract a
multiply and an add into an FMA unless ``TRITON_DEFAULT_FP_FUSION=0``, and
compute ``exp`` and their sums their own way. On CUDA each variant's peak
memory is printed too.

Run from the repository root with the memory capped, and check the CPU clock
first: the loop is launch-bound, so a capped clock scales it. Compiling
unrolls the tile loop, so a compile's time and host memory grow with the
number of tiles (at L = 4,096, about 500 tiles of 128 x 128, it ran past 20
minutes and 10 GiB before it was stopped): keep ``--compile`` to L <= 2,048.

    systemd-run --user --scope -p MemoryMax=12G \\
        uv run --no-sync python dev/benchmarks/timing/flash_attention.py --device cuda
    systemd-run --user --scope -p MemoryMax=12G \\
        uv run --no-sync python dev/benchmarks/timing/flash_attention.py --device cuda \\
        --compile --lengths 512 1024 2048
    uv run --no-sync python dev/benchmarks/timing/flash_attention.py --device cpu --threads 16
"""

import argparse
import dataclasses
import time
from unittest import mock

import torch

from mptorch import E4M3
from mptorch.quant import (
    AttentionFormats,
    FlashFormats,
    FusedMac,
    GemmFormats,
    Quant,
    SoftmaxFormats,
    qattention,
)
from mptorch.quant.functional import _flash

LENGTHS = {"cuda": (512, 1024, 2048, 4096), "cpu": (512, 1024, 2048), "mps": (512, 1024, 2048)}


def _formats(recipe: str, flash: bool) -> AttentionFormats:
    q = Quant(E4M3)
    if recipe == "torch":
        gf = GemmFormats(input_quant=q, weight_quant=q)
        af = AttentionFormats(qk=gf, pv=gf)
    else:
        mac = FusedMac(None)
        gf = GemmFormats(input_quant=q, weight_quant=q, fwd_mac=mac, igrad_mac=mac, wgrad_mac=mac)
        af = AttentionFormats(qk=gf, pv=gf, softmax=SoftmaxFormats(acc=mac, bwd_acc=mac))
    if not flash:
        return af
    return dataclasses.replace(af, flash=FlashFormats(block_q=128, block_kv=128))


def _calls(device: str, recipe: str, compiled: bool):
    """Each variant's call, ``(q, k, v) -> out``."""

    def make(af, loop=False):
        def call(q, k, v):
            if loop:
                with mock.patch.object(_flash, "_batchable", lambda formats: (False, False)):
                    return qattention(q, k, v, af, is_causal=True)
            return qattention(q, k, v, af, is_causal=True)

        return call

    calls = {
        "standard": make(_formats(recipe, False)),
        "flash-loop": make(_formats(recipe, True), loop=True),
        "flash": make(_formats(recipe, True)),
    }
    if compiled:
        for name in ("standard", "flash"):
            calls[f"{name}-c"] = torch.compile(calls[name], fullgraph=True)
            if device == "cuda":
                calls[f"{name}-cg"] = torch.compile(
                    calls[name], fullgraph=True, mode="reduce-overhead"
                )
    return calls


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure(device, recipe, length, backward, rounds, compiled) -> None:
    g = torch.Generator().manual_seed(length)
    base = [torch.randn(1, 8, length, 64, generator=g).to(device) for _ in range(3)]
    calls = _calls(device, recipe, compiled)
    times: dict[str, float] = {n: float("inf") for n in calls}
    peaks: dict[str, int] = {}
    results = {}

    def run(name):
        q, k, v = (t.clone().requires_grad_(backward) for t in base)
        if device == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        _sync(device)
        t = time.perf_counter()
        out = calls[name](q, k, v)
        grads = ()
        if backward:
            out.square().sum().backward()
            grads = (q.grad, k.grad, v.grad)
        _sync(device)
        kept = (out.detach().clone(), *(x.clone() for x in grads if x is not None))
        return time.perf_counter() - t, kept

    for name in calls:  # warm-up: compiles, and records the results
        for _ in range(3 if "-c" in name else 1):
            _, results[name] = run(name)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            run(name)
            peaks[name] = torch.cuda.max_memory_allocated() >> 20
    for _ in range(rounds):
        for name in calls:
            dt, _ = run(name)
            times[name] = min(times[name], dt)
    twin = {"flash": "flash-loop"} | {n: n.split("-")[0] for n in calls if "-c" in n}
    words = {}
    for name in calls:
        diff = max(
            ((a - b).abs().max() / b.abs().max()).item()
            for a, b in zip(results[name], results[twin.get(name, name)], strict=True)
        )
        words[name] = "" if diff == 0 else f" [differs: {diff:.1e}]"
    what = "fwd+bwd" if backward else "fwd"
    cells = "  ".join(
        f"{n} {times[n] * 1e3:8.2f}" + (f" ({peaks[n]} MiB)" if n in peaks else "") + words[n]
        for n in calls
    )
    print(f"{device} {recipe:7s} L={length:5d} {what:7s}  {cells}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--recipes", nargs="*", choices=("carrier", "torch"), default=["carrier"])
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    ap.add_argument("--compile", action="store_true", help="add the compiled variants")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    lengths = args.lengths or LENGTHS[args.device]
    for recipe in args.recipes:
        for backward in (False, True):
            for length in lengths:
                measure(args.device, recipe, length, backward, args.rounds, args.compile)


if __name__ == "__main__":
    main()
