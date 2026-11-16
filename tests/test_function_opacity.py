"""mptorch's autograd Functions are opaque to torch function modes.

A ``TorchFunctionMode`` that intercepts a model's calls (to simulate them in
other arithmetic, to profile them) must see a layer's call and none of the
torch calls the layer makes inside, or it would handle those a second time.
Every Function in ``mptorch.quant.modules`` therefore runs its forward and its
backward with torch function modes disabled. These tests hold the four
Functions to that: under a recording mode, nothing inside them is recorded,
in either pass, and the results are those without the mode; the same hooks
called directly under the mode are recorded, which is what shows that the
mode would see them otherwise.

The guard hides ``torch.set_default_device`` and ``with torch.device(...)``
too, which are modes as well, so the last test holds the layers to their
explicit-device results under a default device.
"""

import pytest
import torch
from torch.overrides import TorchFunctionMode

from mptorch import NVFP4, BinaryK, RoundMode
from mptorch.quant import (
    QConv2d,
    QLinear,
    QMatmul,
    Quant,
    Quantizer,
    SplitMac,
    binaryK_gemm_formats,
    block_gemm_formats,
    conv_formats,
)
from tests.markers import available_devices, requires_cuda

SR = RoundMode.SR
MAC = SplitMac(BinaryK(8, 4), BinaryK(16, 11))


class _Recorder(TorchFunctionMode):
    """Records every call it receives, then runs it."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.calls.append(func)
        return func(*args, **(kwargs or {}))


def _linear():
    f = binaryK_gemm_formats(8, 4, mul_prng_bits=8, acc_K=16, acc_P=11, rounding_mode=SR)
    f.input_quant = f.weight_quant = Quant(BinaryK(8, 4, prng_bits=8), SR)
    f.igrad_quant = f.wgrad_quant = f.bgrad_quant = Quant(BinaryK(8, 5))
    return QLinear(16, 4, formats=f), [(3, 16)]


def _conv():
    f = conv_formats(MAC)
    f.input_quant = f.weight_quant = Quant(BinaryK(8, 4))
    f.igrad_quant = f.wgrad_quant = Quant(BinaryK(8, 5))
    return QConv2d(2, 3, 3, padding=1, formats=f), [(1, 2, 5, 5)]


def _matmul():
    return QMatmul(MAC), [(2, 3, 8), (8, 4)]


def _quantizer():
    return Quantizer(Quant(BinaryK(8, 4, prng_bits=8), SR), BinaryK(8, 5)), [(7,)]


# CustomArithLinear, CustomArithConvNd, CustomArithMatmul and _StraightThrough.
MODELS = {"linear": _linear, "conv": _conv, "matmul": _matmul, "quantizer": _quantizer}


def _run(module, shapes, device, mode=None):
    """The output and every gradient, with ``mode`` entered around the
    forward and, separately, the backward; and what it recorded in each."""
    torch.manual_seed(5)
    operands = [torch.randn(*s, device=device, requires_grad=True) for s in shapes]
    module.zero_grad(set_to_none=True)
    torch.manual_seed(6)
    if mode is None:
        out = module(*operands)
        out.backward(torch.ones_like(out))
        fwd = bwd = []
    else:
        with mode:
            out = module(*operands)
        fwd, mode.calls = mode.calls, []
        grad = torch.ones_like(out)
        with mode:
            out.backward(grad)
        bwd = mode.calls
    grads = [t.grad for t in operands] + [p.grad for p in module.parameters()]
    return [out.detach(), *grads], fwd, bwd


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", list(MODELS))
def test_a_mode_sees_nothing_inside_a_function(name, device):
    """Under a recording mode the forward records no call at all (the layer's
    ``Function.apply`` is not one a mode receives), and the backward only the
    ``Tensor.backward`` that starts it; outputs and gradients are those
    without the mode, word for word."""
    torch.manual_seed(0)
    module, shapes = MODELS[name]()
    module.to(device)
    want, _, _ = _run(module, shapes, device)
    got, fwd, bwd = _run(module, shapes, device, _Recorder())
    assert fwd == []
    assert bwd == [torch.Tensor.backward]
    for w, g in zip(want, got, strict=True):
        assert torch.equal(w, g)


@pytest.mark.parametrize("device", available_devices)
def test_the_hooks_alone_are_seen(device):
    """The negative control: the same Linear hook and quantizer, called
    outside a Function, are recorded, so the empty record above is the
    guard's doing."""
    module, _ = _linear()
    module.to(device)
    x = torch.randn(3, 16, device=device)
    mode = _Recorder()
    with mode, torch.no_grad():
        module.formats.input_quant(x)
        module.formats.fwd_math(x, module.weight, module.bias)
    assert torch.ops.mptorch.binaryK_quant.default in mode.calls
    assert torch.ops.mptorch.custom_matmul_binaryK.default in mode.calls


@requires_cuda
@pytest.mark.parametrize("name", [*MODELS, "nvfp4"])
def test_a_default_device_does_not_reach_inside(name):
    """``with torch.device(...)`` is a mode too, and the guard hides it: every
    allocation inside the Functions follows the operands' device, or is meant
    for the CPU (an NVFP4 operand's tensor scale is computed there), so the
    results under a default device are the explicit-device ones."""
    torch.manual_seed(0)
    if name == "nvfp4":
        module, shapes = QLinear(32, 8, formats=block_gemm_formats(NVFP4, NVFP4)), [(4, 32)]
    else:
        module, shapes = MODELS[name]()
    module.to("cuda")
    want, _, _ = _run(module, shapes, "cuda")
    with torch.device("cuda"):
        got, _, _ = _run(module, shapes, "cuda")
    for w, g in zip(want, got, strict=True):
        assert torch.equal(w, g)
