"""
Layer-level bit-exactness capture, for changes above the op boundary.

The GEMM and elementwise gate (``stacked_before_after.py capture``) captures
op outputs; it runs no layer, so nothing there can see a change in
``CustomArithLinear`` / ``CustomArithConvNd`` -- which gradient quantizers
backward calls, what forward saves for it, how the bias is added. This runs
QLinear and QConv{1,2,3}d over a matrix of format configurations,
``requires_grad`` patterns and bias settings, and dumps every produced tensor
plus the generator state at the end of the step.

The generator state is captured on purpose. It is what proves that a
stochastic quantizer still draws from the global generator in the same
sequence -- reordering the quantize calls relative to the ``*_math`` hooks
would move every SR value in the step without moving any of the tensors this
particular capture happens to look at.

Usage::

    python dev/benchmarks/gates/layer_output_capture.py capture cpu  before_cpu.pt
    # ... change the code ...
    python dev/benchmarks/gates/layer_output_capture.py capture cpu  after_cpu.pt
    python dev/benchmarks/gates/layer_output_capture.py compare before_cpu.pt after_cpu.pt

``compare`` exits non-zero on any difference outside the one documented
exception: a stochastic quantizer that is legitimately no longer called,
because the gradient path that would have consumed it was not requested. Every
other difference -- any deterministic format, or any configuration in which
all gradients are requested -- is reported as unexpected.
"""

import collections
import functools
import sys

import torch

from mptorch.number import BinaryK, RoundMode
from mptorch.quant import (
    QAffineFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    QLinear,
    Quant,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_quantize,
    conv_formats,
    superfp_quantize,
)

QUANT_FIELDS = (
    "weight_quant",
    "input_quant",
    "bias_quant",
    "wgrad_quant",
    "igrad_quant",
    "bgrad_quant",
)

# (input, weight, bias) requires_grad. The all-True row is the ordinary
# training case and must stay bit-identical through any change here; the rest
# skip a gradient path, and with it that path's gradient quantizer and the
# operand forward would otherwise save for it.
PATTERNS = [(True, True, True), (False, True, True), (True, False, False), (False, False, True)]

LINEAR_FORMATS = ["identity", "bk_rne", "sfp_rne", "bk_sr", "bk_gemm", "bk_sr_gemm"]
# The q_* formats put mptorch's own Quant in the slots, which the layers'
# default recompute rule (recompute_input / recompute_weight left at None,
# mptorch/quant/modules/_saved.py) quantizes again in backward instead of
# saving (the functools.partial quantizers above are user callables to that
# rule, and stay saved); added with it, so a capture taken before b35b568 has
# fewer tensors.
LINEAR_FORMATS += ["q_rne", "q_sr", "q_sr_gemm"]
# The last two run the three convolution passes on the GEMM core as implicit
# GEMMs (mptorch.quant.conv_formats); added with it, so a capture taken before
# b16a3aa has 162 fewer tensors on the CPU.
CONV_FORMATS = ["identity", "bk_rne", "sfp_rne", "bk_sr", "bk_conv", "bk_sr_conv"]
CONV_FORMATS += ["q_rne", "q_sr", "q_sr_conv"]


def _bk(K, P, rm=RoundMode.RNE, prng=0):
    return functools.partial(binaryK_quantize, K=K, P=P, prng_bits=prng, rounding_mode=rm)


def _sfp(m, e, nb, bias, rm=RoundMode.RNE, prng=0):
    return functools.partial(
        superfp_quantize,
        man_bits=m,
        exp_bits=e,
        normal_binades=nb,
        bias=bias,
        prng_bits=prng,
        rounding_mode=rm,
    )


def _with(formats, q):
    for name in QUANT_FIELDS:
        setattr(formats, name, q)
    return formats


def make_formats(name):
    if name == "identity":
        return QAffineFormats()
    if name == "bk_rne":
        return _with(QAffineFormats(), _bk(8, 4))
    if name == "sfp_rne":
        return _with(QAffineFormats(), _sfp(4, 4, 1, 7))
    if name == "bk_sr":
        # prng_bits > 0, or SR degenerates to truncation and the capture is
        # blind to exactly the thing it is meant to watch.
        return _with(QAffineFormats(), _bk(8, 4, RoundMode.SR, prng=8))
    if name == "bk_gemm":
        core = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=10, acc_P=5)
        return _with(core, _bk(8, 4))
    if name == "bk_sr_gemm":
        core = binaryK_gemm_formats(
            mul_K=8,
            mul_P=4,
            mul_prng_bits=8,
            acc_K=10,
            acc_P=5,
            acc_prng_bits=8,
            rounding_mode=RoundMode.SR,
        )
        return _with(core, _bk(8, 4, RoundMode.SR, prng=8))
    if name == "bk_conv":
        return _with(conv_formats(SplitMac(BinaryK(8, 4), BinaryK(10, 5))), _bk(8, 4))
    if name == "bk_sr_conv":
        mac = SplitMac(
            BinaryK(8, 4, prng_bits=8), BinaryK(10, 5, prng_bits=8), rounding=RoundMode.SR
        )
        return _with(conv_formats(mac), _bk(8, 4, RoundMode.SR, prng=8))
    q_sr = Quant(BinaryK(8, 4, prng_bits=8), RoundMode.SR)
    if name == "q_rne":
        return _with(QAffineFormats(), Quant(BinaryK(8, 4)))
    if name == "q_sr":
        return _with(QAffineFormats(), q_sr)
    if name == "q_sr_gemm":
        core = binaryK_gemm_formats(
            mul_K=8,
            mul_P=4,
            mul_prng_bits=8,
            acc_K=10,
            acc_P=5,
            acc_prng_bits=8,
            rounding_mode=RoundMode.SR,
        )
        return _with(core, q_sr)
    if name == "q_sr_conv":
        mac = SplitMac(
            BinaryK(8, 4, prng_bits=8), BinaryK(10, 5, prng_bits=8), rounding=RoundMode.SR
        )
        return _with(conv_formats(mac), q_sr)
    raise ValueError(name)


def capture(device, out_path):
    # cuDNN's conv backward-data picks a nondeterministic algorithm at these
    # shapes; without this the baseline does not reproduce against itself.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    res = {}

    def run(tag, layer, x, rg):
        rg_in, rg_w, rg_b = rg
        x = x.detach().clone().requires_grad_(rg_in)
        layer.weight.requires_grad_(rg_w)
        if layer.bias is not None:
            layer.bias.requires_grad_(rg_b)
        layer.zero_grad(set_to_none=True)

        torch.manual_seed(4321)
        if device == "cuda":
            torch.cuda.manual_seed_all(4321)

        out = layer(x)
        res[tag + "/out"] = out.detach().cpu().clone()
        if out.requires_grad:
            g = torch.arange(out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
            out.backward((g % 17) / 16.0 - 0.5)
            if x.grad is not None:
                res[tag + "/gi"] = x.grad.detach().cpu().clone()
            if layer.weight.grad is not None:
                res[tag + "/gw"] = layer.weight.grad.detach().cpu().clone()
            if layer.bias is not None and layer.bias.grad is not None:
                res[tag + "/gb"] = layer.bias.grad.detach().cpu().clone()
        res[tag + "/rng"] = torch.get_rng_state().clone()
        if device == "cuda":
            res[tag + "/crng"] = torch.cuda.get_rng_state().clone()

    torch.manual_seed(1234)
    for use_bias in (True, False):
        b = int(use_bias)
        for fname in LINEAR_FORMATS:
            for rg in PATTERNS:
                for shape in ((13, 24), (3, 5, 24)):
                    torch.manual_seed(7)
                    lin = QLinear(24, 16, bias=use_bias, formats=make_formats(fname)).to(device)
                    run(
                        f"lin/{fname}/b{b}/{rg}/{shape}",
                        lin,
                        torch.randn(*shape, device=device),
                        rg,
                    )

        for fname in CONV_FORMATS:
            for rg in PATTERNS:
                for nd, cls, shape in (
                    (1, QConv1d, (2, 4, 9)),
                    (2, QConv2d, (2, 4, 7, 7)),
                    (3, QConv3d, (2, 4, 5, 5, 5)),
                ):
                    torch.manual_seed(7)
                    conv = cls(4, 6, 3, padding=1, bias=use_bias, formats=make_formats(fname))
                    run(
                        f"c{nd}/{fname}/b{b}/{rg}",
                        conv.to(device),
                        torch.randn(*shape, device=device),
                        rg,
                    )

    torch.save(res, out_path)
    print(f"{len(res)} tensors -> {out_path}")


def compare(before_path, after_path):
    a = torch.load(before_path, weights_only=False)
    b = torch.load(after_path, weights_only=False)
    if set(a) != set(b):
        print(f"KEY MISMATCH: {len(a)} vs {len(b)}")
        return 1

    diff = []
    for k in sorted(a):
        x, y = a[k], b[k]
        same = (
            torch.equal(x, y)
            if x.dtype == torch.uint8
            else (x.shape == y.shape and torch.equal(x.view(torch.uint8), y.view(torch.uint8)))
        )
        if not same:
            diff.append(k)

    unexpected, buckets = [], collections.Counter()
    for k in diff:
        parts = k.split("/")
        fmt, kind = parts[1], parts[-1]
        rg = next(p for p in parts if p.startswith("("))
        if "sr" not in fmt or rg == "(True, True, True)":
            unexpected.append(k)
        buckets[(fmt, kind)] += 1

    print(f"{len(a)} entries, {len(a) - len(diff)} bit-identical, {len(diff)} differ")
    for (fmt, kind), n in sorted(buckets.items()):
        print(f"   {fmt:12s} {kind:5s} {n}")
    print(f"unexpected (non-SR, or every gradient path requested): {len(unexpected)}")
    for k in unexpected[:20]:
        print("   ", k)
    return 1 if unexpected else 0


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "capture":
        capture(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
