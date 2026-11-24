"""What ``mptorch.quant.pointwise``'s ops cost against torch's own.

Each op is timed on one tensor, forward and then forward with backward, in
float32 and in bfloat16, in these variants:

    torch    torch's function (autograd's backward)
    none     the q-op with every slot None: torch's function behind its checks
    rne      every slot ``Quant(BinaryK(8, 4))``: the carrier op, then the cast
    sr       the same under stochastic rounding (8 random bits)
    f64      ``rne`` computed and rounded in binary64 (``carrier=torch.float64``)
    ro       ``rne`` rounded to odd in the carrier (add, mul, div and sqrt only)

The ops are ``qadd``, ``qmul``, ``qdiv``, ``qsqrt``, ``qexp``, ``qgelu`` and
``qsilu`` (the gated activations with their gate rounded as well). The binary
ops take a second tensor of the same shape. Each variant is the minimum over
``--rounds`` rounds, alternating with ``torch``'s.

Run from the repository root (check the CPU clock first: a capped clock
scales every number):

    uv run --no-sync python dev/benchmarks/timing/pointwise_ops.py --device cpu --threads 1
    uv run --no-sync python dev/benchmarks/timing/pointwise_ops.py --device cpu --threads 16
    uv run --no-sync python dev/benchmarks/timing/pointwise_ops.py --device cuda

About two minutes for one CPU thread, less for the others. The shape is
2048 x 2048 on the CPU and 4096 x 4096 on CUDA (``--size``).

On the CPU, the variants that widen (bfloat16 to float32, anything to
float64, and the float32 square root, computed in float64 because torch's CPU
root is not correctly rounded) allocate temporaries of 16 to 32 MB, which
glibc maps fresh from the system on every call, so their times are mostly
first-touch page faults. ``MALLOC_MMAP_THRESHOLD_=4294967296
MALLOC_TRIM_THRESHOLD_=4294967296`` keeps them in the heap: at one thread,
float32 ``qadd`` then takes 6.5 ms (6.7 without), bfloat16 9.3 (32.6),
binary64 15.4 (70.2), rounded to odd 75.7 (389.1), and the float32 root 15.8
(49.8), against torch's 1.0-1.5 ms.

Results (Ryzen 9 7940HS at 4.5 GHz, RTX 4060 laptop GPU, torch 2.15,
2026-11-23), forward, ms and against torch's, for float32:

                 torch   none   rne          sr           f64           ro
    cpu x1  add   1.27   1.0x   6.62 (5.2x)  15.6 (12x)   69.2 (55x)   371 (293x)
            exp   2.57   1.0x   8.44 (3.3x)  17.4 (6.8x)  50.4 (20x)
            sqrt  1.38   1.0x   51.6 (37x)   59.6 (43x)   52.1 (38x)   138 (100x)
            gelu  1.64   1.0x   61.4 (37x)   82.6 (50x)   147 (90x)
            silu  1.73   1.1x   14.1 (8.1x)  33.1 (19x)   61.7 (36x)
    cpu x16 add   0.95   1.1x   2.05 (2.2x)  2.97 (3.1x)  25.0 (26x)   129 (136x)
            gelu  0.66   1.0x   19.5 (30x)   21.0 (32x)   47.4 (72x)
    cuda    add   0.86   1.0x   1.45 (1.7x)  1.47 (1.7x)  5.52 (6.4x)  30.9 (36x)
            sqrt  0.60   1.0x   1.19 (2.0x)  1.19 (2.0x)  4.18 (7.0x)  10.1 (17x)
            gelu  0.60   1.0x   4.42 (7.4x)  4.43 (7.4x)  16.8 (28x)
            silu  0.60   1.0x   2.64 (4.4x)  2.64 (4.4x)  9.45 (16x)

``none`` costs nothing, ``rne`` is the op plus the quantizer's pass (and,
for the gated activations, the gate's), and the forward and backward
together are 1.3-3x torch's in float32 on CUDA, 1.5-3.3x on one CPU thread
(an addition 10x, since torch's backward of one is free). bfloat16 adds the
widening and the store: the elementary ops' forward is 6.3-7x torch's on
CUDA, whose bfloat16 ops take half the float32 ones' time.
"""

import argparse
import time
from typing import Any

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    ActivationFormats,
    Quant,
    qadd,
    qdiv,
    qexp,
    qgelu,
    qmul,
    qsilu,
    qsqrt,
)

OPS = {
    "add": (lambda x, y, *s, **k: qadd(x, y, *s, **k), lambda x, y: x + y),
    "mul": (lambda x, y, *s, **k: qmul(x, y, *s, **k), lambda x, y: x * y),
    "div": (lambda x, y, *s, **k: qdiv(x, y, *s, **k), lambda x, y: x / y),
    "sqrt": (lambda x, y, *s, **k: qsqrt(x, *s, **k), lambda x, y: torch.sqrt(x)),
    "exp": (lambda x, y, *s, **k: qexp(x, *s, **k), lambda x, y: torch.exp(x)),
    "gelu": (None, lambda x, y: torch.nn.functional.gelu(x)),
    "silu": (None, lambda x, y: torch.nn.functional.silu(x)),
}
ODD = ("add", "mul", "div", "sqrt")


def _slots(variant: str):
    if variant == "sr":
        q = Quant(BinaryK(8, 4, prng_bits=8), RoundMode.SR)
    else:
        q = Quant(BinaryK(8, 4))
    if variant == "none":
        return (None, None), {}, ActivationFormats()
    kw: dict[str, Any] = {}
    if variant == "f64":
        kw = {"carrier": torch.float64}
    elif variant == "ro":
        kw = {"carrier_rounding": RoundMode.RO}
    carrier = kw.get("carrier")
    return (q, q), kw, ActivationFormats(inner=q, out=q, grad=q, carrier=carrier)


def _call(op: str, variant: str):
    q_op, torch_op = OPS[op]
    if variant == "torch":
        return torch_op
    slots, kw, act = _slots(variant)
    if op == "gelu":
        return lambda x, y: qgelu(x, act)
    if op == "silu":
        return lambda x, y: qsilu(x, act)
    assert q_op is not None  # only the activations have none
    return lambda x, y: q_op(x, y, *slots, **kw)


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


def measure(device: str, size: int, dtype: torch.dtype, op: str, backward: bool, rounds: int):
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(size, size, generator=g) + 0.5).to(device=device, dtype=dtype)
    y = (torch.rand(size, size, generator=g) + 0.5).to(device=device, dtype=dtype)
    variants = ["torch", "none", "rne", "sr", "f64"] + (["ro"] if op in ODD else [])
    if device == "mps":
        variants = [v for v in variants if v not in ("f64", "ro")]
    calls = {v: _call(op, v) for v in variants}
    if backward:
        x.requires_grad_(True)
        y.requires_grad_(True)
        grad = torch.ones_like(x)

    def step(call):
        def run():
            out = call(x, y)
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
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    size = args.size or (2048 if args.device == "cpu" else 4096)
    print(
        f"torch {torch.__version__}, {args.device}, {torch.get_num_threads()} threads, "
        f"{size} x {size}; ms, and each variant against torch's"
    )
    for dtype in (torch.float32, torch.bfloat16):
        for backward in (False, True):
            passes = "forward and backward" if backward else "forward"
            print(f"\n{str(dtype).removeprefix('torch.')}, {passes}")
            for op in OPS:
                best = measure(args.device, size, dtype, op, backward, args.rounds)
                base = best["torch"]
                cells = [f"torch {base:7.2f}"] + [
                    f"{v} {t:7.2f} ({t / base:4.1f}x)" for v, t in best.items() if v != "torch"
                ]
                print(f"  {op:5}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
