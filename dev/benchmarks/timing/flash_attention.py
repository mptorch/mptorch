"""What flash attention costs, eager and compiled, against the standard path.

Each case is ``qattention`` over causal ``[1, 8, L, 64]`` float32 operands,
q, k and v rounded to E4M3 by plain ``Quant`` slots (the operand quantizers of
both products), everything else in binary32, in these variants:

    standard     AttentionFormats(qk=..., pv=...): the product, softmax and
                 product over whole tensors
    flash        the same with FlashFormats(block_q=128, block_kv=128): FA2's
                 order, a Python loop over tiles
    flash-c      flash under torch.compile(fullgraph=True), Inductor's
                 default mode: the elementwise steps between the GEMMs fuse
    flash-cg     the same with mode="reduce-overhead" (CUDA graphs replay
                 the launches; CUDA only)
    standard-c   the standard path compiled, for the same comparison

forward only, and forward and backward. Each variant is the minimum over
``--rounds`` rounds, alternating in order, after a warm-up that compiles.
A compiled result is compared with its eager twin's (forward and
gradients): where it differs, the largest difference relative to the
result's largest magnitude is printed. Inductor's Triton kernels contract a
multiply and an add into an FMA unless ``TRITON_DEFAULT_FP_FUSION=0``, and
compute ``exp`` and their sums in their own way. On CUDA the peak memory of
each variant is printed too.

Run from the repository root, with its memory capped (a compile of a long
sequence can exhaust the host's), and check the CPU clock first: the tile
loop is launch-bound, so a capped clock scales the eager flash numbers:

    systemd-run --user --scope -p MemoryMax=12G \
        uv run --no-sync python dev/benchmarks/timing/flash_attention.py --device cuda
    uv run --no-sync python dev/benchmarks/timing/flash_attention.py --device cpu --threads 16
"""

import argparse
import dataclasses
import time

import torch

from mptorch import E4M3
from mptorch.quant import AttentionFormats, FlashFormats, GemmFormats, Quant, qattention

# Compiling unrolls the tile loop when it traces it, so a compile's time and
# host memory grow with the number of tiles: at L = 4,096 (about 500 tiles of
# 128 x 128) it ran past 20 minutes and 10 GiB before it was stopped.
LENGTHS = {"cuda": (512, 1024, 2048), "cpu": (512, 1024, 2048)}


def _formats(flash: bool) -> AttentionFormats:
    q = Quant(E4M3)
    qk = GemmFormats(input_quant=q, weight_quant=q)
    pv = GemmFormats(input_quant=q, weight_quant=q)
    standard = AttentionFormats(qk=qk, pv=pv)
    if not flash:
        return standard
    return dataclasses.replace(standard, flash=FlashFormats(block_q=128, block_kv=128))


def _calls(device: str):
    """Each variant's call, ``(q, k, v) -> out``."""
    calls = {}
    for name, flash in (("standard", False), ("flash", True)):
        af = _formats(flash)

        def call(q, k, v, af=af):
            return qattention(q, k, v, af, is_causal=True)

        calls[name] = call
        calls[f"{name}-c"] = torch.compile(call, fullgraph=True)
        if flash and device == "cuda":
            calls["flash-cg"] = torch.compile(call, fullgraph=True, mode="reduce-overhead")
    order = ("standard", "standard-c", "flash", "flash-c", "flash-cg")
    return {n: calls[n] for n in order if n in calls}


def _step(call, q, k, v, backward: bool):
    out = call(q, k, v)
    if not backward:
        return out, ()
    out.square().sum().backward()
    return out, (q.grad, k.grad, v.grad)


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def measure(device: str, length: int, backward: bool, rounds: int) -> None:
    g = torch.Generator().manual_seed(length)
    base = [torch.randn(1, 8, length, 64, generator=g).to(device) for _ in range(3)]
    calls = _calls(device)
    times: dict[str, float] = {n: float("inf") for n in calls}
    peaks: dict[str, int] = {}
    results = {}

    def run(name):
        q, k, v = (t.clone().requires_grad_(backward) for t in base)
        if device == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        _sync(device)
        t = time.perf_counter()
        out, grads = _step(calls[name], q, k, v, backward)
        _sync(device)
        return time.perf_counter() - t, (out.detach().clone(), *(x.clone() for x in grads))

    for name in calls:  # warm-up: compiles, and records the results
        for _ in range(3):
            _, results[name] = run(name)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            run(name)
            peaks[name] = torch.cuda.max_memory_allocated() >> 20
    for _ in range(rounds):
        for name in calls:
            dt, _ = run(name)
            times[name] = min(times[name], dt)
    words = {}
    for name in calls:
        eager = results[name.split("-")[0]]
        diff = max(
            ((a - b).abs().max() / b.abs().max()).item()
            for a, b in zip(results[name], eager, strict=True)
        )
        words[name] = "" if diff == 0 else f" [differs: {diff:.1e}]"
    what = "fwd+bwd" if backward else "fwd"
    cells = "  ".join(
        f"{n} {times[n] * 1e3:8.2f}" + (f" ({peaks[n]} MiB)" if n in peaks else "") + words[n]
        for n in calls
    )
    print(f"{device} L={length:5d} {what:7s}  {cells}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="*", default=None)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    lengths = args.lengths or LENGTHS.get(args.device, LENGTHS["cpu"])
    for backward in (False, True):
        for length in lengths:
            measure(args.device, length, backward, args.rounds)


if __name__ == "__main__":
    main()
