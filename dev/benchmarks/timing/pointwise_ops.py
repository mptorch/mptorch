"""What ``mptorch.quant.pointwise``'s ops cost against torch's own.

Each op is timed on one tensor, forward and then forward with backward, in
float32 and in bfloat16, in these variants:

    torch    torch's function (autograd's backward)
    none     the q-op with every slot None: torch's function behind its checks
    rne      every slot ``Quant(BinaryK(8, 4))``: the carrier op, then the cast
    sr       the same under stochastic rounding (8 random bits)
    f64      ``rne`` computed and rounded in binary64 (``carrier=torch.float64``)
    ro       ``rne`` rounded to odd in the carrier (add, mul, div and sqrt only)
    2pass    ``rne`` as two passes, torch's op in the carrier and then the
             quantizer in place, which is what ``qadd``, ``qmul``, ``qdiv``,
             ``qsqrt`` and ``qpwl`` computed before their fused ops (forward
             only)

``qadd``, ``qmul``, ``qdiv``, ``qsqrt`` and ``qpwl`` with a ``Quant`` slot run
one fused op on the CPU and CUDA (``csrc/common/pointwise.h``), so their
``rne``, ``sr``, ``f64`` and ``ro`` rows are its; the other ops run the two
passes.

The ops are ``qadd``, ``qmul``, ``qdiv``, ``qsqrt``, ``qexp``, ``qgelu`` and
``qsilu`` (the gated activations with their gate rounded as well), and
``pwl``, ``qpwl`` over ``silu_table()`` against torch's SiLU (its ``none`` is
the table unrounded). The binary ops take a second tensor of the same shape.
Each variant is the minimum over ``--rounds`` rounds, alternating with
``torch``'s.

Run from the repository root (check the CPU clock first: a capped clock
scales every number):

    uv run --no-sync python dev/benchmarks/timing/pointwise_ops.py --device cpu --threads 1
    uv run --no-sync python dev/benchmarks/timing/pointwise_ops.py --device cpu --threads 16
    uv run --no-sync python dev/benchmarks/timing/pointwise_ops.py --device cuda

About eight minutes for one CPU thread, less for the others. The shape is
2048 x 2048 on the CPU and 4096 x 4096 on CUDA (``--size``).

On the CPU, a result or temporary of 16 to 32 MB is mapped fresh from the
system on every call by glibc, unless its dynamic threshold has risen past
it, so CPU times move by a millisecond or more from one process to the next.
``MALLOC_MMAP_THRESHOLD_=4294967296 MALLOC_TRIM_THRESHOLD_=4294967296`` keeps
them in the heap; the CPU rows below were taken so.

Results (Ryzen 9 7940HS at 4.7-4.8 GHz, RTX 4060 laptop GPU, torch 2.15,
2026-11-26), float32 forward, ms; ``2pass`` is what each op cost before its
fused op:

                 torch   rne    2pass   sr     f64    ro
    cpu x1  add   1.24   1.54   6.88   15.0   14.8   13.6
            sqrt  1.36   6.76   15.4   19.5   18.0   13.6
            pwl   2.79   16.0   58.1   24.3   21.7
    cpu x16 add   0.98   1.13   2.23   1.88   7.92   2.49
            sqrt  0.56   1.12   6.29   2.48   5.37   2.01
            pwl   0.74   3.01   22.9   3.58   5.95
    cuda    add   0.87   0.86   1.47   0.84   4.34   0.90
            sqrt  0.60   0.59   1.19   0.57   3.18   0.58
            pwl   0.60   0.57   14.2   0.57   4.36

(``pwl``'s torch is SiLU.) On CUDA the fused ops cost what torch's op costs,
stochastic rounding and rounding to odd included; in bfloat16 an add is 0.44
against the two passes' 2.77. On one CPU thread ``rne`` runs the vector
blocks (``cpu/pointwise_rne.cpp``; with ``MPTORCH_CPU_SIMD=0`` an add takes
6.4), and ``sr`` is the scalar cast's cost, as in the quantizer. ``f64``
widens the operands and narrows the result in passes of their own. The
backward passes are unchanged: torch's formula, then the gradient slot's
quantizer.
"""

import argparse
import time
from typing import Any

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    ActivationFormats,
    Quant,
    pointwise,
    pwl,
    qadd,
    qdiv,
    qexp,
    qgelu,
    qmul,
    qpwl,
    qsilu,
    qsqrt,
    silu_table,
)

TABLE = silu_table()

OPS = {
    "add": (lambda x, y, *s, **k: qadd(x, y, *s, **k), lambda x, y: x + y),
    "mul": (lambda x, y, *s, **k: qmul(x, y, *s, **k), lambda x, y: x * y),
    "div": (lambda x, y, *s, **k: qdiv(x, y, *s, **k), lambda x, y: x / y),
    "sqrt": (lambda x, y, *s, **k: qsqrt(x, *s, **k), lambda x, y: torch.sqrt(x)),
    "exp": (lambda x, y, *s, **k: qexp(x, *s, **k), lambda x, y: torch.exp(x)),
    "gelu": (None, lambda x, y: torch.nn.functional.gelu(x)),
    "silu": (None, lambda x, y: torch.nn.functional.silu(x)),
    "pwl": (
        lambda x, y, *s, **k: qpwl(x, TABLE, quant=s[0], grad_quant=s[1], **k),
        lambda x, y: torch.nn.functional.silu(x),
    ),
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


def _two_passes(op: str):
    """What the op computed before its fused op: torch's op in binary32 (the
    CPU's root through float64), the quantizer in place on that result, then
    the store."""
    q = Quant(BinaryK(8, 4), inplace=True)

    def run(x, y):
        xc = x.float()
        if op == "pwl":
            t = pwl._evaluate(TABLE, xc)
        elif op == "sqrt":
            t = pointwise._carrier_sqrt(xc)
        else:
            t = OPS[op][1](xc, y.float())
        q(t)
        return t.to(x.dtype)

    return run


def _call(op: str, variant: str):
    q_op, torch_op = OPS[op]
    if variant == "torch":
        return torch_op
    if variant == "2pass":
        return _two_passes(op)
    slots, kw, act = _slots(variant)
    if op == "gelu":
        return lambda x, y: qgelu(x, act)
    if op == "silu":
        return lambda x, y: qsilu(x, act)
    assert q_op is not None  # only the activations have none
    return lambda x, y: q_op(x, y, *slots, **kw)


def _sync(device: str) -> None:
    """Wait for the device's queued work, so that a timing covers it."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _timer(device: str, fn, n: int) -> float:
    """Milliseconds per call of ``fn``, over ``n`` calls."""
    _sync(device)
    t = time.perf_counter()
    for _ in range(n):
        fn()
    _sync(device)
    return (time.perf_counter() - t) / n * 1e3


def measure(device: str, size: int, dtype: torch.dtype, op: str, backward: bool, rounds: int):
    g = torch.Generator().manual_seed(0)
    x = (torch.rand(size, size, generator=g) + 0.5).to(device=device, dtype=dtype)
    y = (torch.rand(size, size, generator=g) + 0.5).to(device=device, dtype=dtype)
    variants = ["torch", "none", "rne", "sr", "f64"] + (["ro"] if op in ODD else [])
    if op in (*ODD, "pwl") and not backward:
        variants.append("2pass")
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
