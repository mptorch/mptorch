"""What ``qattention`` costs against ``F.scaled_dot_product_attention``.

Each case is a causal self-attention over ``[batch, heads, length, 64]``
float32 operands, in these variants:

    sdpa      torch's SDPA, whichever kernel it picks (fused on CUDA)
    math      torch's SDPA on its math kernel: the unfused product, softmax
              and product, which is what qattention composes
    none      qattention with no formats: torch's SDPA behind its checks
    rne       both products as SplitMac(BinaryK(8, 4), BinaryK(14, 8)), the
              operands and the probabilities rounded to BinaryK(8, 4), and
              the softmax's steps too, under RNE
    sr        the same under stochastic rounding

forward only, and forward and backward (``rne`` with its gradients rounded
as its forward is). Each variant is the minimum over ``--rounds`` rounds,
alternating in order.

Run from the repository root (check the CPU clock first: a capped clock
scales every number):

    uv run --no-sync python dev/benchmarks/timing/attention_ops.py --device cpu --threads 1
    uv run --no-sync python dev/benchmarks/timing/attention_ops.py --device cpu --threads 16
    uv run --no-sync python dev/benchmarks/timing/attention_ops.py --device cuda

About a minute on CUDA, four on sixteen CPU threads and twenty on one (most
of it ``sr``).

Results (Ryzen 9 7940HS at 4.6-4.7 GHz, RTX 4060 laptop GPU, torch 2.15,
2026-11-24), against SDPA's math kernel; ``sdpa`` is torch's default:

                              sdpa    rne     sr      rne, fwd+bwd
    cuda     8x8x256          0.32x   7.7x    25x     8.6x
             4x16x1024        0.20x   8.0x    23x     10x
             1x16x4096        0.17x   7.5x    21x     10x
    cpu x16  4x8x256          0.12x   4.6x    54x     4.7x
             2x8x1024         0.09x   5.3x    64x     4.9x
    cpu x1   4x8x256          0.25x   7.5x    131x    18x
             2x8x1024         0.21x   9.0x    149x    12x

On CUDA at 4x16x1024 the two products are 72 of the 98 ms and the softmax
23: the products' cost is the simulated GEMMs' (each 4.3 GMAC with K = 64 or
1024), so attention costs what its products would cost as layers. ``none``
is SDPA's time to within a few percent. The CPU's ``sr`` runs the scalar
GEMM loop: its vector tiles take RNE only.
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from mptorch import BinaryK, RoundMode
from mptorch.quant import AttentionFormats, Quant, SplitMac, qattention

SHAPES = {
    "cpu": [(4, 8, 256), (2, 8, 1024)],
    "cuda": [(8, 8, 256), (4, 16, 1024), (1, 16, 4096)],
}


def _formats(variant: str) -> AttentionFormats | None:
    if variant == "none":
        return None
    sr = variant == "sr"
    mode = RoundMode.SR if sr else RoundMode.RNE
    fmt = BinaryK(8, 4, prng_bits=10 if sr else 0)
    mac = SplitMac(fmt, BinaryK(14, 8, prng_bits=6 if sr else 0), rounding=mode)
    return AttentionFormats.uniform(mac, Quant(fmt, mode))


def _call(variant: str):
    if variant == "sdpa":
        return lambda q, k, v: F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if variant == "math":

        def math(q, k, v):
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v, is_causal=True)

        return math
    f = _formats(variant)
    return lambda q, k, v: qattention(q, k, v, f, is_causal=True)


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


def measure(device: str, batch: int, heads: int, length: int, backward: bool, rounds: int):
    g = torch.Generator().manual_seed(0)
    q, k, v = (
        torch.randn(batch, heads, length, 64, generator=g).to(device).requires_grad_(backward)
        for _ in range(3)
    )
    variants = ["sdpa", "math", "rne"] if backward else ["sdpa", "math", "none", "rne", "sr"]
    calls = {name: _call(name) for name in variants}

    def step(call):
        def run():
            out = call(q, k, v)
            if backward:
                out.sum().backward()

        return run

    steps = {name: step(c) for name, c in calls.items()}
    n = 2 if device == "cpu" else 10
    best = dict.fromkeys(steps, float("inf"))
    for fn in steps.values():
        fn()
    for r in range(rounds):
        order = list(steps) if r % 2 == 0 else list(reversed(steps))
        for name in order:
            best[name] = min(best[name], _timer(device, steps[name], n))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    print(
        f"torch {torch.__version__}, {args.device}, {torch.get_num_threads()} threads; "
        "ms, and each variant against SDPA's math kernel"
    )
    for backward in (False, True):
        print("\nforward and backward" if backward else "\nforward")
        for batch, heads, length in SHAPES.get(args.device, SHAPES["cpu"]):
            best = measure(args.device, batch, heads, length, backward, args.rounds)
            base = best["math"]
            cells = [f"{name} {t:8.2f} ({t / base:5.2f}x)" for name, t in best.items()]
            print(f"  {batch}x{heads}x{length:<5} " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
