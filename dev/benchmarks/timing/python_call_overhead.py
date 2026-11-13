"""What the Python wrappers cost per GEMM call, against the op they dispatch to.

`mptorch.quant.ops`'s matmul wrappers resolve a format before every call: the
`acc_*` fallbacks to the `mul_*` values, the default exponent biases, and what
the range checks against each carrier find. None of that depends on the
operands (what does -- which carrier's verdict to report, and holding the
stored format against a float16 or bfloat16 result -- is left to the call,
where a float32 operand pays a tuple index and one dict membership test), and a
`QAffineFormats` fixes all of it for the lifetime of the layer -- so 444d83a
split the resolution out (`_binaryK_spec` and friends) and had
`mptorch.quant.gemm`'s factories do it once, at construction, instead of on
every forward and backward pass.

This measures the three points that bracket that change, on a 1x1x1 GEMM so
that what is being timed is the call and not the arithmetic:

    torch.ops floor   the bare `torch.ops.mptorch.*` call with the format
                      already resolved -- the dispatcher, the C++ prologue,
                      the allocation and (on CUDA) the launch
    resolved once     `_run_gemm` over a spec built beforehand: what a layer
                      built by `binaryK_gemm_formats` now pays
    resolved per call `binaryK_matmul(...)`: what it paid before, and what a
                      caller passing formats by keyword still pays

The format objects of `mptorch.quant.mac` (621acea) added a fourth: a format
*object* (`SplitMac(BinaryK(8, 4), ...)`) passed per call, whose resolution is
memoized on the frozen value. An object API is a regression unless that cache
hit is cheaper than the resolve it replaces, so this table is the gate for it
-- "memoized" has to land at or under "resolved once" plus a dict lookup, not
near "per call".

and then the same three hooks inside a training step, against a plain
`nn.Linear` one -- because that is the number that says what the saving is
worth. Microseconds off a hook are real, and they are still a rounding error
next to what torch's own autograd charges for a step at these sizes.

CUDA host timing is the noisier of the two -- the launches are asynchronous
and the queue is not -- so read the CPU table for the size of the effect and
the CUDA one for its sign.

Run: python3 dev/benchmarks/timing/python_call_overhead.py [--device cuda]
"""

import argparse
import functools
import time
from collections.abc import Callable

import torch
from torch import nn

from mptorch import MXFP8_E4M3, BinaryK, SuperFP
from mptorch.quant import BlockQuant, QAffineFormats, QLinear, Quant, SplitMac
from mptorch.quant.gemm import _gemm_formats, binaryK_gemm_formats, block_gemm_formats
from mptorch.quant.mac import FusedMac, spec_for_mac
from mptorch.quant.modules._saved import quantize_operand
from mptorch.quant.ops import (
    _binaryK_fma_spec,
    _binaryK_spec,
    _GemmSpec,
    _run_gemm,
    _superfp_fma_spec,
    _superfp_spec,
    binaryK_matmul,
    binaryK_matmul_fma,
    superfp_matmul,
    superfp_matmul_fma,
)

# Each case twice over: once resolved into a spec, once as the keyword call
# that resolves on every invocation. Spelling the formats out both times is
# the point -- these are the two ways a caller can say the same thing.
Call2D = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

BK = BinaryK(8, 4)
SFP = SuperFP(3, 4, 1, 7)

# name, spec, keyword call, and the mac object that names the same format.
CASES: list[tuple[str, _GemmSpec, Call2D, object]] = [
    (
        "binaryK",
        _binaryK_spec(mul_K=8, mul_P=4),
        lambda a, b: binaryK_matmul(a, b, mul_K=8, mul_P=4),
        SplitMac(BK, BK),
    ),
    (
        "binaryK fma",
        _binaryK_fma_spec(fma_K=8, fma_P=4),
        lambda a, b: binaryK_matmul_fma(a, b, fma_K=8, fma_P=4),
        FusedMac(BK),
    ),
    (
        "superfp",
        _superfp_spec(mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7),
        lambda a, b: superfp_matmul(
            a, b, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7
        ),
        SplitMac(SFP, SFP),
    ),
    (
        "superfp fma",
        _superfp_fma_spec(fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=1, fma_bias=7),
        lambda a, b: superfp_matmul_fma(
            a, b, fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=1, fma_bias=7
        ),
        FusedMac(SFP),
    ),
]


def bench(fn: Callable[[], object], *, device: str, iters: int) -> float:
    """Microseconds per call, min of five rounds.

    No synchronize inside the loop: on CUDA these kernels are trivial and the
    queue never becomes the bottleneck, so what this measures is host time --
    which is the whole subject.
    """
    for _ in range(300):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
        if device == "cuda":
            torch.cuda.synchronize()
    return best


def _hooks(fmt: QAffineFormats) -> tuple[Callable, Callable, Callable]:
    """The three math hooks a factory set, as plain callables."""
    assert fmt.fwd_math and fmt.bwd_igrad_math and fmt.bwd_wgrad_math
    return fmt.fwd_math, fmt.bwd_igrad_math, fmt.bwd_wgrad_math


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=20000)
    args = ap.parse_args()

    a = torch.randn(1, 1, device=args.device)
    b = torch.randn(1, 1, device=args.device)
    run = functools.partial(bench, device=args.device, iters=args.iters)

    print(f"{args.device}, 1x1x1 GEMM, us/call (min of 5 rounds of {args.iters})\n")
    print(
        f"{'op':<14}{'torch.ops':>11}{'resolved once':>15}{'per call':>11}"
        f"{'memoized':>11}{'saved':>9}"
    )
    for name, spec, keyword_call, mac in CASES:
        floor = run(lambda s=spec: s.op(a, b, False, False, *s.args))
        once = run(lambda s=spec: _run_gemm(s, a, b, False, False))
        each = run(lambda c=keyword_call: c(a, b))
        memo = run(lambda m=mac: _run_gemm(spec_for_mac(m), a, b, False, False))
        print(f"{name:<14}{floor:>11.2f}{once:>15.2f}{each:>11.2f}{memo:>11.2f}{each - once:>9.2f}")

    # The same difference where it is actually paid: a Linear layer's math
    # hooks, whose format is fixed when the QAffineFormats is built.
    spec = CASES[0][1]
    once_h = _gemm_formats(lambda x, y, ta, tb: _run_gemm(spec, x, y, ta, tb))
    each_h = _gemm_formats(
        lambda x, y, ta, tb: binaryK_matmul(x, y, trans_a=ta, trans_b=tb, mul_K=8, mul_P=4)
    )
    x = torch.randn(1, 8, device=args.device)
    w = torch.randn(8, 8, device=args.device)
    bias = torch.randn(8, device=args.device)
    once_f, once_i, once_w = _hooks(once_h)
    each_f, each_i, each_w = _hooks(each_h)
    hooks: list[tuple[str, Callable[[], object], Callable[[], object]]] = [
        ("fwd", lambda: once_f(x, w, bias), lambda: each_f(x, w, bias)),
        ("bwd_igrad", lambda: once_i(x, w), lambda: each_i(x, w)),
        ("bwd_wgrad", lambda: once_w(x, x), lambda: each_w(x, x)),
    ]
    print(f"\n{'binaryK hook':<14}{'':>11}{'resolved once':>15}{'per call':>11}{'saved':>9}")
    for label, call_once, call_each in hooks:
        once = run(call_once, iters=args.iters // 4)
        each = run(call_each, iters=args.iters // 4)
        print(f"{label:<14}{'':>11}{once:>15.2f}{each:>11.2f}{each - once:>9.2f}")

    # And the scale those microseconds sit on. A step here is the forward,
    # `.sum().backward()` and `zero_grad()`; the nn.Linear row is entirely
    # torch's own, and it is the floor the other two are measured from.
    xb = torch.randn(8, 64, device=args.device, requires_grad=True)
    layers: list[tuple[str, nn.Module]] = [
        ("nn.Linear", nn.Linear(64, 64)),
        ("QLinear, no hooks", QLinear(64, 64, formats=QAffineFormats())),
        ("QLinear, binaryK", QLinear(64, 64, formats=binaryK_gemm_formats(mul_K=8, mul_P=4))),
    ]

    def step(layer: nn.Module) -> None:
        layer(xb).sum().backward()
        layer.zero_grad(set_to_none=True)

    print(f"\n{'training step':<24}{'ms':>10}")
    for label, layer in layers:
        layer.to(args.device)
        print(f"{label:<24}{run(lambda m=layer: step(m), iters=400) / 1e3:>10.3f}")

    # A forward alone, under no_grad: where a quantizer slot's own Python
    # (`_saved.quantize_operand`) and a block layer's weight path are not
    # hidden under autograd's. The bar set when `freeze_packed_` was added
    # (13f73db): the slot rows stay within noise, and a frozen layer (which
    # builds its BlockPacked per call) costs about what the packed-once
    # recipe's memo lookup does.
    with_slots = binaryK_gemm_formats(mul_K=8, mul_P=4)
    with_slots.input_quant = Quant(BK)
    with_slots.weight_quant = Quant(BK)
    packed_once = block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3)
    packed_once.weight_quant = BlockQuant(MXFP8_E4M3, axis=1).pack
    forwards: list[tuple[str, nn.Module]] = [
        ("QLinear, Quant slots", QLinear(32, 8, formats=with_slots)),
        ("block, packed once", QLinear(32, 8, formats=packed_once)),
    ]
    if hasattr(QLinear, "freeze_packed_"):  # 13f73db or later
        frozen = QLinear(32, 8, formats=block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3))
        forwards.append(("block, frozen", frozen.freeze_packed_()))
    xf = torch.randn(1, 32, device=args.device)
    print(f"\n{'forward, no_grad':<24}{'us':>10}")
    with torch.no_grad():
        for label, layer in forwards:
            layer.to(args.device)
            print(f"{label:<24}{run(lambda m=layer: m(xf), iters=args.iters // 4):>10.2f}")

    # And the slot's own Python, with the quantizer an identity: what a
    # quantized operand pays in `_saved.quantize_operand` beyond its kernel.
    def identity(t: torch.Tensor) -> torch.Tensor:
        return t

    slot = run(lambda: quantize_operand(xf, identity, None, False, True))
    print(f"{'quantize_operand':<24}{slot:>10.3f}")


if __name__ == "__main__":
    main()
