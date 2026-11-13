"""
Peak memory and step time for a quantized layer's forward+backward, split by
which gradients the layer is actually asked for.

Written to measure three changes that 9e6ea34 made to the layers' autograd
Functions and to the gemm.py fwd hooks (the tree before them is 64982e6).
Before them:

  grads   backward quantized grad_output for *both* gradient paths regardless
          of which ones were needed;
  saved   forward saved q_input, q_weight and q_bias unconditionally, though
          q_bias is never read and each of the other two feeds exactly one path;
  bias    the bias add in the gemm.py fwd hooks allocated a second output.

Three requires_grad patterns, because they are what those changes turn on:

  full    input and weight both require grad -- an interior training layer.
          Only ``bias`` touches this case.
  frozen  weight frozen, input still requires grad -- a backbone block under
          a linear probe / LoRA / partial fine-tune. Drops the saved q_input
          (activation-sized) and the wgrad quantize.
  first   input does not require grad, weight does -- the first layer of a
          network. Drops the saved q_weight and the igrad quantize.

Three memory numbers, because the changes land in different places:

  fwd     allocator high-water of the forward alone, above what was resident
          before it -- where ``bias``'s second output shows up.
  held    what is still pinned once forward has returned, minus the output
          itself: the saved tensors. This is the number that matters in a
          real network, because it is held for every layer at once until the
          backward pass unwinds. Where ``saved`` shows up.
  step    high-water of forward+backward together. Where ``grads`` shows up.

Run:  python dev/benchmarks/timing/layer_bwd_memory.py [cuda|cpu]

The layers' ``recompute_input`` / ``recompute_weight`` flags (b35b568), which
quantize an operand again in backward instead of saving its quantized copy,
have a mode of their own, CUDA only:

    python dev/benchmarks/timing/layer_bwd_memory.py cuda recompute

It runs stacks of QConv2d or QLinear layers with a ReLU after each and
``Quant(BinaryK(8, 4))`` on input, weight and both gradients, with each
layer's recompute flags set four ways (copies kept, input, weight, both) and
left at the layer's default, on PyTorch's products and on mptorch's GEMM and
conv kernels: held and peak memory of one step from a clean start, and the
step time, min over alternating rounds. The quantizers of the default mode
are ``functools.partial``s, whose results a layer keeps saved whatever the
flags (to the recompute rule they are a user's callables), so the sections
that mode prints are unchanged by the flags.
"""

import functools
import statistics
import sys
import time

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    QAffineFormats,
    QConv2d,
    QLinear,
    Quant,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_quantize,
    conv_formats,
)

DEV = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
MODE = sys.argv[2] if len(sys.argv) > 2 else "audit"
REPS = 20

# The GEMM core simulates every dot product, so it needs a far smaller shape
# than the PyTorch-matmul case to finish in comparable time.
SHAPE = (8192, 4096, 4096)
GEMM_SHAPE = (512, 1024, 1024)
STACK_DEPTH, STACK_WIDTH, STACK_BATCH = 12, 2048, 8192
MIB = 1024 * 1024

QUANT_FIELDS = (
    "weight_quant",
    "input_quant",
    "bias_quant",
    "wgrad_quant",
    "igrad_quant",
    "bgrad_quant",
)

PATTERNS = {
    "full": (True, True),
    "frozen": (True, False),
    "first": (False, True),
}


def with_quantizers(f):
    q = functools.partial(binaryK_quantize, K=8, P=4, rounding_mode=RoundMode.RNE)
    for name in QUANT_FIELDS:
        setattr(f, name, q)
    return f


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


def allocated():
    return torch.cuda.memory_allocated() if DEV == "cuda" else 0


def step(layer, x, g):
    out = layer(x)
    out.backward(g)
    layer.zero_grad(set_to_none=True)
    x.grad = None


def peak_of(fn):
    if DEV != "cuda":
        fn()
        return float("nan")
    sync()
    torch.cuda.reset_peak_memory_stats()
    base = allocated()
    fn()
    sync()
    return (torch.cuda.max_memory_allocated() - base) / MIB


def measure(make_formats, label, shape, reps=REPS):
    b, n_in, n_out = shape
    print(f"\n{label}  [{b}x{n_in} -> {n_out}]")
    print(f"  {'pattern':8s} {'fwd MiB':>9s} {'held MiB':>9s} {'step MiB':>9s} {'ms/step':>10s}")
    for name, (rg_in, rg_w) in PATTERNS.items():
        layer = QLinear(n_in, n_out, bias=True, formats=make_formats()).to(DEV)
        layer.weight.requires_grad_(rg_w)
        layer.bias.requires_grad_(rg_w)
        x = torch.randn(b, n_in, device=DEV, requires_grad=rg_in)
        g = torch.randn(b, n_out, device=DEV)

        for _ in range(3):
            step(layer, x, g)
        sync()

        held_box = {}

        def do_fwd(box=held_box, layer=layer, x=x):
            box["out"] = layer(x)

        def do_step(layer=layer, x=x, g=g):
            step(layer, x, g)

        before = allocated()
        fwd_peak = peak_of(do_fwd)
        out = held_box.pop("out")
        held = (allocated() - before - out.numel() * out.element_size()) / MIB
        out = None

        step_peak = peak_of(do_step)

        ts = []
        for _ in range(reps):
            sync()
            t0 = time.perf_counter()
            step(layer, x, g)
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)

        print(
            f"  {name:8s} {fwd_peak:9.1f} {held:9.1f} {step_peak:9.1f} "
            f"{statistics.median(ts):10.2f}"
        )

        del layer, x, g
        if DEV == "cuda":
            torch.cuda.empty_cache()


def audit_shapes():
    b, n_in, n_out = SHAPE
    print(
        f"device={DEV}  activation {b * n_in * 4 / MIB:.0f} MiB, "
        f"weight {n_in * n_out * 4 / MIB:.0f} MiB, grad_output {b * n_out * 4 / MIB:.0f} MiB"
    )

    measure(lambda: with_quantizers(QAffineFormats()), "quantizers only, PyTorch matmul", SHAPE)
    measure(
        lambda: with_quantizers(binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=10, acc_P=5)),
        "quantizers + binaryK GEMM core (the hooks' bias add)",
        GEMM_SHAPE,
        reps=5,
    )


# The saved operands are one tensor per layer; what makes them the largest
# memory item of the three changes is that a network holds every layer's at
# once until the backward pass has unwound past it.
#
# Which layers actually pay is worth being precise about. A frozen *prefix*
# never did: with neither its input nor its parameters requiring grad,
# autograd builds no node for it and the ctx is dropped as soon as forward
# returns. The case that pays is a frozen block that gradients still have to
# flow *through* -- a frozen middle under adapter/LoRA/partial fine-tuning,
# with something trainable below it. There the layer must produce an input
# gradient, so it gets a graph node, and it used to save a quantized copy of
# its whole activation for the weight gradient it will never be asked for.
def measure_stack(depth=STACK_DEPTH, width=STACK_WIDTH, batch=STACK_BATCH):
    print(f"\n{depth} stacked QLinear {batch}x{width} -> {width}, ReLU between")
    print(f"  {'trainable':12s} {'peak MiB':>10s} {'ms/step':>10s}")
    configs = (
        ("all", lambda i: True),
        ("head only", lambda i: i == depth - 1),
        ("first+head", lambda i: i in (0, depth - 1)),
    )
    for label, trainable in configs:
        layers = []
        for _ in range(depth):
            layer = QLinear(width, width, bias=True, formats=with_quantizers(QAffineFormats()))
            layers.append(layer.to(DEV))
        for i, layer in enumerate(layers):
            layer.weight.requires_grad_(trainable(i))
            layer.bias.requires_grad_(trainable(i))
        net = torch.nn.Sequential(*[m for layer in layers for m in (layer, torch.nn.ReLU())][:-1])
        x = torch.randn(batch, width, device=DEV)

        def one(net=net, x=x):
            net(x).sum().backward()
            net.zero_grad(set_to_none=True)

        one()
        one()
        peak = peak_of(one)
        ts = []
        for _ in range(5):
            sync()
            t0 = time.perf_counter()
            one()
            sync()
            ts.append((time.perf_counter() - t0) * 1e3)
        print(f"  {label:12s} {peak:10.1f} {statistics.median(ts):10.2f}")
        del layers, net, x
        if DEV == "cuda":
            torch.cuda.empty_cache()


# The recompute mode: quantizing the operands again in backward instead of
# saving them. (conv, mptorch kernels, width or channels, depth, input shape)
RECOMPUTE_STACKS = {
    "6 x conv3x3, mptorch conv kernels": (True, True, 64, 6, (16, 64, 32, 32)),
    "6 x conv3x3, PyTorch": (True, False, 64, 6, (32, 64, 56, 56)),
    "6 x Linear 1024, mptorch GEMM": (False, True, 1024, 6, (1024, 1024)),
    "6 x Linear 1024, PyTorch": (False, False, 1024, 6, (1024, 1024)),
    "3 x Linear 4096, 256 tokens, mptorch GEMM": (False, True, 4096, 3, (256, 4096)),
    "3 x Linear 4096, 256 tokens, PyTorch": (False, False, 4096, 3, (256, 4096)),
    "3 x Linear 4096, 2048 tokens, PyTorch": (False, False, 4096, 3, (2048, 4096)),
}
RECOMPUTE_FLAGS = {
    "copies": (False, False),
    "input": (True, False),
    "weight": (False, True),
    "both": (True, True),
    "default": (None, None),
}


def recompute_stack(conv, kernels, width, depth, flags, rounding):
    torch.manual_seed(0)
    q = Quant(BinaryK(8, 4, prng_bits=8), rounding)
    layers = []
    for _ in range(depth):
        if not kernels:
            f = QAffineFormats()
        elif conv:
            f = conv_formats(SplitMac(BinaryK(8, 4), BinaryK(16, 11)))
        else:
            f = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11)
        f.input_quant = f.weight_quant = f.igrad_quant = f.wgrad_quant = q
        f.recompute_input, f.recompute_weight = flags
        if conv:
            layer = QConv2d(width, width, 3, padding=1, bias=False, formats=f)
        else:
            layer = QLinear(width, width, bias=False, formats=f)
        layers += [layer, torch.nn.ReLU()]
    return torch.nn.Sequential(*layers).to(DEV)


def measure_recompute(rounds=4, steps=3):
    for label, (conv, kernels, width, depth, shape) in RECOMPUTE_STACKS.items():
        x = torch.randn(*shape, device=DEV)
        act = x.numel() * 4 / MIB
        wt = width * width * (9 if conv else 1) * 4 / MIB
        print(f"\n{label}: input {tuple(shape)}, activation {act:.1f} MiB, weight {wt:.2f} MiB")
        print(
            f"  {'flags':12s} {'held MiB':>9s} {'(acts)':>7s} {'peak MiB':>9s} "
            f"{'ms/step':>9s} {'vs copies':>9s}"
        )
        for rounding in (RoundMode.RNE, RoundMode.SR):
            names = list(RECOMPUTE_FLAGS) if rounding is RoundMode.RNE else ["copies", "both"]
            nets = {
                n: recompute_stack(conv, kernels, width, depth, RECOMPUTE_FLAGS[n], rounding)
                for n in names
            }
            mem = {}
            for n, net in nets.items():
                sync()
                torch.cuda.reset_peak_memory_stats()
                base = allocated()
                out = net(x)
                sync()
                held = (allocated() - base - out.numel() * out.element_size()) / MIB
                out.sum().backward()
                out = None
                sync()
                mem[n] = (held, (torch.cuda.max_memory_allocated() - base) / MIB)
                net.zero_grad(set_to_none=True)
            best = dict.fromkeys(nets, float("inf"))
            for _ in range(rounds):  # the flags alternate inside every round
                for n, net in nets.items():
                    for _ in range(steps):
                        sync()
                        t0 = time.perf_counter()
                        net(x).sum().backward()
                        sync()
                        best[n] = min(best[n], time.perf_counter() - t0)
                        net.zero_grad(set_to_none=True)
            for n in names:
                held, peak = mem[n]
                tag = f"{n}{' SR' if rounding is RoundMode.SR else ''}"
                print(
                    f"  {tag:12s} {held:9.1f} {held / act:7.2f} {peak:9.1f} "
                    f"{best[n] * 1e3:9.2f} {best[n] / best['copies']:9.3f}x"
                )
            del nets
            torch.cuda.empty_cache()


def main():
    if MODE == "recompute":
        if DEV != "cuda":
            sys.exit("the recompute mode reads the CUDA allocator: run it with cuda")
        measure_recompute()
    else:
        audit_shapes()
        measure_stack()


if __name__ == "__main__":
    main()
