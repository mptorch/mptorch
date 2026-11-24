"""What ``qsum``, ``qsoftmax``, ``qlayernorm`` and ``qrmsnorm`` cost against torch's own.

Each op is timed on a ``[rows, K]`` float32 tensor, over its last dimension,
forward only, in these variants:

    torch     torch's function
    none      the mptorch op with no formats: torch's function behind its checks
    rne       every step ``BinaryK(14, 8)``, its sums on the GEMM kernels
              (``FusedMac``) under RNE
    sr        the same under stochastic rounding (8 random bits)
    kahan     ``rne`` with the sums under KAHAN

and, for the softmax and the norms, forward and backward with the backward's
steps rounded too (``bwd``) against torch's forward and backward. Each
variant is the minimum over ``--rounds`` rounds, alternating with torch's.

Run from the repository root (check the CPU clock first: a capped clock
scales every number):

    uv run --no-sync python dev/benchmarks/timing/reduction_ops.py --device cpu --threads 1
    uv run --no-sync python dev/benchmarks/timing/reduction_ops.py --device cpu --threads 16
    uv run --no-sync python dev/benchmarks/timing/reduction_ops.py --device cuda

About a minute on CUDA, three on one CPU thread.

Results (Ryzen 9 7940HS at 4.7 GHz, RTX 4060 laptop GPU, torch 2.15,
2026-11-24), forward, against torch's, under RNE:

                           sum    softmax  layernorm  rmsnorm
    cpu x1   1024x512      33x    26x      79x        2.4x
             8192x4096     35x    5.6x     10x        2.1x
    cpu x16  1024x512      2.7x   24x      42x        1.7x
             8192x4096     4.8x   4.7x     8.4x       1.6x
    cuda     8192x256      8.7x   13x      7.5x       6.6x
             8192x4096     3.9x   8.0x     15x        8.8x
             65536x2048    3.7x   8.1x     14x        8.7x

SR and KAHAN cost 3-4x RNE on CUDA; on one CPU thread SR 2-6x and KAHAN
5-17x, without the vector tile. Forward and backward together, against
torch's, are 1.8-4.7x on CUDA with torch's backward, 2.4-8.5x with the
backward rounded too.
torch's own CPU ``rms_norm`` is a composition of ops, which makes it slow
and mptorch's ratio small. A format below binary32's fast-cast floor
(``BinaryK(16, 8)``, bias 128) costs about twice these.
"""

import argparse
import time

import torch
import torch.nn.functional as F

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode
from mptorch.quant import (
    FusedMac,
    NormFormats,
    Quant,
    SoftmaxFormats,
    qlayernorm,
    qrmsnorm,
    qsoftmax,
    qsum,
)

SHAPES = {
    "cpu": [(1024, 512), (8192, 256), (8192, 4096)],
    "cuda": [(1024, 512), (8192, 256), (8192, 4096), (65536, 2048)],
}


def _formats(variant: str):
    """The sum's mac, and softmax's and the norms' formats, of a variant."""
    if variant == "none":
        return None, None, None
    sr = variant == "sr"
    fmt = BinaryK(14, 8, prng_bits=8 if sr else 0)
    mode = RoundMode.SR if sr else RoundMode.RNE
    algorithm = AccumulateAlgorithm.KAHAN if variant == "kahan" else AccumulateAlgorithm.NAIVE
    mac = FusedMac(fmt, rounding=mode, accumulate_algorithm=algorithm)
    q = Quant(fmt, mode)
    grad = q if variant == "bwd" else None
    return (
        mac,
        SoftmaxFormats.uniform(mac, q, grad=grad),
        NormFormats.uniform(mac, q, grad=grad),
    )


def _call(op: str, variant: str, w: torch.Tensor, b: torch.Tensor):
    k = w.shape[0]
    if variant == "torch":
        return {
            "sum": lambda x: x.sum(-1),
            "softmax": lambda x: F.softmax(x, -1),
            "layernorm": lambda x: F.layer_norm(x, (k,), w, b),
            "rmsnorm": lambda x: F.rms_norm(x, (k,), w, 1e-6),
        }[op]
    mac, sf, nf = _formats(variant)
    return {
        "sum": lambda x: qsum(x, -1, mac),
        "softmax": lambda x: qsoftmax(x, -1, sf),
        "layernorm": lambda x: qlayernorm(x, (k,), w, b, 1e-5, nf),
        "rmsnorm": lambda x: qrmsnorm(x, (k,), w, 1e-6, nf),
    }[op]


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


def measure(device: str, rows: int, k: int, op: str, backward: bool, rounds: int):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(rows, k, generator=g).to(device)
    w = torch.randn(k, generator=g).to(device)
    b = torch.randn(k, generator=g).to(device)
    if backward:
        variants = ["torch", "rne", "bwd"]
        x.requires_grad_(True)
        grad = torch.ones(rows, k, device=device)
    else:
        variants = ["torch", "none", "rne", "sr", "kahan"]
    calls = {v: _call(op, v, w, b) for v in variants}

    def step(call):
        def run():
            out = call(x)
            if backward:
                out.backward(grad)

        return run

    steps = {v: step(c) for v, c in calls.items()}
    n = 3 if device == "cpu" else 10
    best = dict.fromkeys(steps, float("inf"))
    for fn in steps.values():
        fn()
    for r in range(rounds):
        order = list(steps) if r % 2 == 0 else list(reversed(steps))
        for v in order:
            best[v] = min(best[v], _timer(device, steps[v], n))
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
        "ms, and each variant against torch's"
    )
    for backward in (False, True):
        print("\nforward and backward" if backward else "\nforward")
        for rows, k in SHAPES.get(args.device, SHAPES["cpu"]):
            for op in (
                ("softmax", "layernorm", "rmsnorm")
                if backward
                else ("sum", "softmax", "layernorm", "rmsnorm")
            ):
                best = measure(args.device, rows, k, op, backward, args.rounds)
                base = best["torch"]
                cells = [f"torch {base:7.3f}"] + [
                    f"{v} {t:8.3f} ({t / base:5.1f}x)" for v, t in best.items() if v != "torch"
                ]
                print(f"  {rows:>6}x{k:<5} {op:9}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
