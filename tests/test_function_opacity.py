"""mptorch's autograd Functions are opaque to torch function modes.

A ``TorchFunctionMode`` that intercepts a model's calls (to simulate them in
other arithmetic, to profile them) must see a layer's call and none of the
torch calls the layer makes inside, or it would handle those a second time.
Every Function in ``mptorch.quant.modules`` and ``mptorch.quant.pointwise``
therefore runs its forward and its backward with torch function modes
disabled, and so does a pointwise op's own code around its Function. These
tests hold the Functions to that: under a recording mode, nothing inside them is recorded,
in either pass, and the results are those without the mode; the same hooks
called directly under the mode are recorded, which is what shows that the
mode would see them otherwise.

The guard hides ``torch.set_default_device`` and ``with torch.device(...)``
too, which are modes as well, so a test holds the layers to their
explicit-device results under a default device. The last tests hold a mode
entered inside compiled code to routing every layer (``_opaque.py`` says why
the backward guard is off while compiling).
"""

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend
from torch.overrides import TorchFunctionMode

from mptorch import NVFP4, BinaryK, RoundMode
from mptorch.quant import (
    QGELU,
    ActivationFormats,
    FusedMac,
    GemmFormats,
    NormFormats,
    QBatchNorm2d,
    QConv2d,
    QLayerNorm,
    QLinear,
    QMatmul,
    QSoftmax,
    Quant,
    Quantizer,
    SoftmaxFormats,
    SplitMac,
    binaryK_gemm_formats,
    block_gemm_formats,
    conv_formats,
    qadd,
    qdiv,
    qexp,
    qmean,
    qmul,
    qpow,
    qrsqrt,
    qsqrt,
    qsub,
    qsum,
)
from mptorch.quant.modules.linear import CustomArithLinear
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


class _Pointwise(torch.nn.Module):
    """Every elementary pointwise op, rounded, with a number operand."""

    def forward(self, x, y):
        q, g = Quant(BinaryK(8, 4, prng_bits=8), SR), Quant(BinaryK(8, 5))
        z = qmul(qadd(x, y, q, g), 0.5, q, g)
        z = qdiv(qsub(z, y, q, g), qadd(qmul(y, y, q, g), 1, q, g), q, g)
        positive = qadd(qmul(z, z, q, g), 1, q, g)
        return qpow(qexp(qrsqrt(qsqrt(positive, q, g), q, g), q, g), 2, q, g)


def _pointwise():
    return _Pointwise(), [(3, 4), (4,)]


def _activation():
    q = Quant(BinaryK(8, 4, prng_bits=8), SR)
    return QGELU(formats=ActivationFormats(inner=q, out=q, grad=Quant(BinaryK(8, 5)))), [(3, 5)]


class _Reductions(torch.nn.Module):
    """A sum and a mean, rounded at every step and in their gradients."""

    def forward(self, x):
        mac, g = FusedMac(BinaryK(8, 4, prng_bits=8), rounding=SR), Quant(BinaryK(8, 5))
        return qadd(qsum(x, 0, mac, grad_quant=g), qmean(x, 0, mac, quant=g, grad_quant=g), g)


def _reductions():
    return _Reductions(), [(5, 4)]


def _softmax():
    q = Quant(BinaryK(8, 4, prng_bits=8), SR)
    return QSoftmax(-1, SoftmaxFormats.uniform(FusedMac(BinaryK(16, 8)), q)), [(3, 6)]


def _closed_form_norm():
    q = Quant(BinaryK(8, 4, prng_bits=8), SR)
    return QLayerNorm(6, formats=NormFormats.uniform(FusedMac(BinaryK(16, 8)), q)), [(3, 6)]


def _torchs_gradient_norm():
    q = Quant(BinaryK(8, 4, prng_bits=8), SR)
    f = NormFormats.uniform(FusedMac(BinaryK(16, 8)), q, grad=None)
    return QBatchNorm2d(4, formats=f), [(3, 4, 2, 2)]


# CustomArithLinear, CustomArithConvNd, CustomArithMatmul, _StraightThrough,
# the pointwise ops' and an activation's Functions, and those of the sums,
# softmax and the norms (their closed-form backward, and the rounded forward
# over torch's own gradient).
MODELS = {
    "linear": _linear,
    "conv": _conv,
    "matmul": _matmul,
    "quantizer": _quantizer,
    "pointwise": _pointwise,
    "activation": _activation,
    "reductions": _reductions,
    "softmax": _softmax,
    "closed-form norm": _closed_form_norm,
    "torch's-gradient norm": _torchs_gradient_norm,
}


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


# --- under torch.compile ------------------------------------------------------------


class _Router(TorchFunctionMode):
    """Routes each ``F.linear`` to ``CustomArithLinear`` over the next of its
    containers, as an intercepting mode (``simulate``) does."""

    def __init__(self, containers):
        super().__init__()
        self.containers = containers
        self.calls = 0

    def __torch_function__(self, func, types, args: tuple = (), kwargs=None):
        kwargs = kwargs or {}
        if func is torch._C._nn.linear:
            container = self.containers[self.calls % len(self.containers)]
            self.calls += 1
            return CustomArithLinear.apply(args[0], args[1], None, container)
        return func(*args, **kwargs)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, SR])
@pytest.mark.parametrize("passes", ["backward macs", "gradient quantizers"])
def test_a_compiled_mode_routes_every_function(device, rounding, passes):
    """A mode entered inside compiled code must keep seeing calls after a
    Function's backward has been traced: on torch 2.15, a backward guarded by
    ``DisableTorchFunction`` that runs an mptorch GEMM left modes disabled
    for the rest of the trace, so only the first layer was routed and the
    others compiled as plain ``F.linear``, with no graph break ("backward
    macs" fails with that guard; gradient quantizers alone did not trigger
    it). The routed layers are counted in the graph, since an unrouted layer
    whose forward has no mac computes the same values; outputs and gradients
    equal eager's."""
    torch._dynamo.reset()
    fmt = BinaryK(8, 4, prng_bits=8) if rounding is SR else BinaryK(8, 4)
    mac = SplitMac(fmt, BinaryK(16, 11, prng_bits=fmt.prng_bits), rounding=rounding)
    q = Quant(fmt, rounding)
    if passes == "backward macs":
        gf = GemmFormats.uniform(mac, q)
    else:
        gf = GemmFormats(input_quant=q, weight_quant=q, fwd_mac=mac, igrad_quant=q, wgrad_quant=q)
    torch.manual_seed(0)
    weights = [torch.nn.Parameter(torch.randn(16, 16, device=device) / 4) for _ in range(3)]
    mode = _Router([gf.affine() for _ in weights])

    def step(x):
        with mode:
            for w in weights:
                x = F.linear(x, w)
            return x

    def run(fn):
        for w in weights:
            w.grad = None
        x = torch.randn(3, 16, device=device, generator=torch.Generator(device).manual_seed(1))
        x.requires_grad_(True)
        torch.manual_seed(2)
        out = fn(x)
        out.backward(torch.ones_like(out))
        assert x.grad is not None
        return [out.detach(), x.grad, *(w.grad for w in weights)]

    want = run(step)
    counter = CompileCounterWithBackend("aot_eager")
    got = run(torch.compile(step, backend=counter, fullgraph=True))
    routed = [
        n for n in counter.graphs[0].graph.nodes if "autograd_function_apply" in str(n.target)
    ]
    assert len(routed) == len(weights)
    for w, g in zip(want, got, strict=True):
        assert torch.equal(w, g)
