"""The activations, their modules and the embedding.

Tier 1: with every slot ``None`` an activation is torch's, and with only its
result rounded its gradient is torch's. Tier 2: a real format comes out close
to float. Tier 3: an activation is torch's function in the carrier (or, with
its gate rounded, the composite over the rounded gate), then the result's
slot; its gradient torch's backward op in the carrier, then the gradient's
slot, word for word, stochastic rounding included. The modules are their
torch classes until their formats round, and then the ops; the embedding's
rows and weight gradient are rounded after the gather and after the sum.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo.testing import CompileCounterWithBackend

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    QGELU,
    ActivationFormats,
    PointwiseFormats,
    QEmbedding,
    QHardsigmoid,
    QHardswish,
    QReLU,
    QSigmoid,
    QSiLU,
    QTanh,
    Quant,
    qembedding,
    qgelu,
    qhardsigmoid,
    qhardswish,
    qrelu,
    qsigmoid,
    qsilu,
    qtanh,
)
from tests.markers import available_devices

R = RoundMode
P8 = BinaryK(8, 4)

# name: (the op, torch's function, the torch module, its mptorch module)
ACTS = {
    "gelu": (qgelu, F.gelu, nn.GELU, QGELU),
    "gelu tanh": (
        lambda x, f=None: qgelu(x, f, "tanh"),
        lambda x: F.gelu(x, approximate="tanh"),
        lambda: nn.GELU("tanh"),
        lambda formats=None: QGELU("tanh", formats),
    ),
    "silu": (qsilu, F.silu, nn.SiLU, QSiLU),
    "sigmoid": (qsigmoid, torch.sigmoid, nn.Sigmoid, QSigmoid),
    "tanh": (qtanh, torch.tanh, nn.Tanh, QTanh),
    "hardswish": (qhardswish, F.hardswish, nn.Hardswish, QHardswish),
    "hardsigmoid": (qhardsigmoid, F.hardsigmoid, nn.Hardsigmoid, QHardsigmoid),
    "relu": (qrelu, torch.relu, nn.ReLU, QReLU),
}


def _x(device, dtype=torch.float32, shape=(8, 16), seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float64) * 3).to(device=device, dtype=dtype)


def _grad(out, x, seed=1):
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.autograd.grad(out, x, g.to(out), retain_graph=True)[0]


def _same(a, b) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


def _slots(rounding):
    stochastic = rounding.is_stochastic
    return (
        Quant(BinaryK(8, 4, prng_bits=12 if stochastic else 0), rounding),
        Quant(BinaryK(10, 6, prng_bits=10 if stochastic else 0), rounding),
        Quant(BinaryK(12, 8, prng_bits=8 if stochastic else 0), rounding),
    )


# --- tier 1 ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", list(ACTS))
def test_without_slots_an_activation_is_torchs(device, dtype, name):
    op, fn, _, _ = ACTS[name]
    x = _x(device, dtype).requires_grad_()
    want = fn(x)
    for formats in (None, ActivationFormats()):
        got = op(x, formats)
        assert _same(got, want) and _same(_grad(got, x), _grad(want, x))
    # the result alone rounded: torch's gradient, its backward op in the input's dtype
    got = op(x, ActivationFormats(out=Quant(P8)))
    assert _same(_grad(got, x), _grad(want, x))


# --- tier 2 ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", list(ACTS))
def test_a_real_format_comes_out_close(device, name):
    """Within a few of the format's steps, of the result's and of the gate's:
    GELU's ``1 + gate`` cancels near -1, where the gate's error, about
    ``|x| * 2**-12``, is what remains."""
    op, fn, _, _ = ACTS[name]
    x = _x(device)
    got = op(x, ActivationFormats.uniform(BinaryK(16, 11)))
    torch.testing.assert_close(got, fn(x), rtol=2**-9, atol=2**-9)


# --- tier 3 ---------------------------------------------------------------------------

_SQRT1_2 = math.sqrt(0.5)
_BETA = math.sqrt(2.0) * (2.0 / math.sqrt(math.pi)) * 0.5


def _gated(name, x, inner):
    """The composite over a rounded gate, as the docstrings state it."""
    if name == "gelu":
        return x * 0.5 * (1 + inner(torch.erf(x * _SQRT1_2)))
    if name == "gelu tanh":
        return x * 0.5 * (1 + inner(torch.tanh(_BETA * (x + 0.044715 * (x * x * x)))))
    if name == "silu":
        return x * inner(torch.sigmoid(x))
    return x * inner(F.hardsigmoid(x))


def _backward(name, g, x):
    aten = torch.ops.aten
    return {
        "gelu": lambda: aten.gelu_backward(g, x),
        "gelu tanh": lambda: aten.gelu_backward(g, x, approximate="tanh"),
        "silu": lambda: aten.silu_backward(g, x),
        "sigmoid": lambda: aten.sigmoid_backward(g, torch.sigmoid(x)),
        "tanh": lambda: aten.tanh_backward(g, torch.tanh(x)),
        "hardswish": lambda: aten.hardswish_backward(g, x),
        "hardsigmoid": lambda: aten.hardsigmoid_backward(g, x),
        "relu": lambda: aten.threshold_backward(g, x, 0),
    }[name]()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
@pytest.mark.parametrize("name", list(ACTS))
def test_an_activation_is_its_carrier_step_then_the_slots(device, dtype, rounding, name):
    """In binary32: the gate (for the gated ones), the result and the
    gradient, each rounded by its slot, drawing in that order under SR."""
    op, fn, _, _ = ACTS[name]
    inner, out, grad = _slots(rounding)
    x = _x(device, dtype).requires_grad_()
    g = _x(device, dtype, seed=3)
    formats = ActivationFormats(inner=inner, out=out, grad=grad)
    torch.manual_seed(5)
    got = op(x, formats)
    (got_g,) = torch.autograd.grad(got, x, g)
    xc, gc = x.detach().float(), g.float()
    torch.manual_seed(5)
    if name in ("gelu", "gelu tanh", "silu", "hardswish"):
        want = out(_gated(name, xc, inner)).to(dtype)
    else:
        want = out(fn(xc)).to(dtype)
    want_g = grad(_backward(name, gc, xc)).to(dtype)
    assert _same(got, want) and _same(got_g, want_g)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", ["gelu", "silu", "hardswish"])
def test_without_an_inner_slot_the_carrier_runs_torchs_function(device, name):
    op, fn, _, _ = ACTS[name]
    x = _x(device)
    assert _same(op(x, ActivationFormats(out=Quant(P8))), Quant(P8)(fn(x)))


@pytest.mark.parametrize("device", available_devices)
def test_a_binary64_carrier(device):
    if device == "mps":
        pytest.skip("MPS has no float64")
    x = _x(device)
    formats = ActivationFormats(out=Quant(P8), carrier=torch.float64)
    assert _same(qgelu(x, formats), Quant(P8)(F.gelu(x.double())).float())


def test_the_formats_check_their_slots():
    with pytest.raises(ValueError, match="carrier argument names"):
        ActivationFormats(out=Quant(P8, carrier=torch.float64))
    with pytest.raises(TypeError, match="a module"):
        ActivationFormats(grad=nn.Identity())
    with pytest.raises(TypeError, match="ActivationFormats or None"):
        qrelu(torch.ones(2), PointwiseFormats())  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="'none' or 'tanh'"):
        qgelu(torch.ones(2), approximate="erf")
    assert ActivationFormats.uniform(P8).grad == Quant(P8)
    assert ActivationFormats.uniform(P8, grad=None).grad is None


# --- the modules ------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", list(ACTS))
def test_a_module_is_torchs_until_its_formats_round(device, name):
    op, _, torch_cls, q_cls = ACTS[name]
    x = _x(device)
    assert _same(q_cls()(x), torch_cls()(x))
    formats = ActivationFormats.uniform(P8)
    assert _same(q_cls(formats=formats)(x), op(x, formats))


@pytest.mark.parametrize("cls", [QReLU, QSiLU, QHardswish, QHardsigmoid])
def test_an_inplace_module_rounds_out_of_place(cls):
    """``inplace=True`` writes the input when nothing rounds, as torch's
    module does; a rounded result is a new tensor, and the input is kept."""
    x = _x("cpu")
    y = x.clone()
    cls(inplace=True)(y)
    assert not torch.equal(y, x)
    y = x.clone()
    out = cls(inplace=True, formats=ActivationFormats(out=Quant(P8)))(y)
    assert torch.equal(y, x) and out is not y


# --- compile ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
def test_the_activations_compile_in_one_graph(device, rounding):
    inner, out, grad = _slots(rounding)
    formats = ActivationFormats(inner=inner, out=out, grad=grad)

    def f(x):
        y = qgelu(x, formats) + qgelu(x, formats, "tanh") + qsilu(x, formats)
        y = y + qsigmoid(x, formats) + qtanh(x, formats) + qhardswish(x, formats)
        return y + qhardsigmoid(x, formats) + qrelu(x, formats)

    torch._dynamo.reset()
    x = _x(device).requires_grad_()
    torch.manual_seed(3)
    want = f(x)
    want_g = _grad(want, x)
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(f, backend=counter, fullgraph=True)
    torch.manual_seed(3)
    got = compiled(x)
    assert counter.frame_count == 1 and _same(got, want) and _same(_grad(got, x), want_g)


# --- the embedding ------------------------------------------------------------------------


def _embedding(device, dtype=torch.float32, **kwargs):
    torch.manual_seed(0)
    layer = nn.Embedding(12, 8, **kwargs).to(device=device, dtype=dtype)
    idx = torch.tensor([[1, 2, 2, 5], [0, 2, 11, 1]], device=device)
    return layer, idx


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("padding_idx", [None, 2, -1])
def test_without_slots_the_embedding_is_torchs(device, padding_idx):
    layer, idx = _embedding(device, padding_idx=padding_idx)
    for formats in (None, PointwiseFormats()):
        got = qembedding(idx, layer.weight, formats, layer.padding_idx)
        want = layer(idx)
        assert _same(got, want)
        assert _same(_grad(got, layer.weight), _grad(want, layer.weight))
    got = qembedding(idx, layer.weight, PointwiseFormats(quant=Quant(P8)), layer.padding_idx)
    assert _same(_grad(got, layer.weight), _grad(layer(idx), layer.weight))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
@pytest.mark.parametrize("by_freq", [False, True])
def test_the_rows_and_the_summed_gradient_are_rounded(device, rounding, by_freq):
    """The rows are the slot's rounding of the weight's rows, and the
    weight's gradient the gradient slot's rounding of torch's, summed over
    the repeated indices first, the padding row's left zero."""
    _, q, gq = _slots(rounding)
    layer, idx = _embedding(device, padding_idx=1, scale_grad_by_freq=by_freq)
    formats = PointwiseFormats(quant=q, grad_quant=gq)
    g = _x(device, shape=(2, 4, 8), seed=4)
    torch.manual_seed(6)
    got = qembedding(idx, layer.weight, formats, 1, scale_grad_by_freq=by_freq)
    (got_g,) = torch.autograd.grad(got, layer.weight, g)
    torch.manual_seed(6)
    want = q(layer.weight.detach()[idx])
    want_g = gq(torch.ops.aten.embedding_backward(g, idx, 12, 1, by_freq, False))
    assert _same(got, want) and _same(got_g, want_g) and not got_g[1].any()


@pytest.mark.parametrize("device", available_devices)
def test_qembedding_is_its_module_and_keeps_tied_weights(device):
    layer, idx = _embedding(device)
    formats = PointwiseFormats.uniform(P8)
    module = QEmbedding(12, 8, formats=formats).to(device)
    module.weight = layer.weight
    head = nn.Linear(8, 12, bias=False).to(device)
    head.weight = layer.weight  # tied: one Parameter, two uses
    out = head(module(idx)).sum()
    out.backward()
    assert _same(module(idx), qembedding(idx, layer.weight, formats))
    assert layer.weight.grad is not None
    plain = QEmbedding(12, 8).to(device)
    plain.weight = layer.weight
    assert _same(plain(idx), layer(idx))


def test_the_embedding_refuses_what_it_cannot_round():
    layer, idx = _embedding("cpu", sparse=True)
    with pytest.raises(ValueError, match="sparse gradient"):
        qembedding(idx, layer.weight, PointwiseFormats(grad_quant=Quant(P8)), sparse=True)
    qembedding(idx, layer.weight, PointwiseFormats(quant=Quant(P8)), sparse=True).sum().backward()
    with pytest.raises(ValueError, match="no carrier step to round to odd"):
        qembedding(idx, layer.weight, PointwiseFormats(quant=Quant(P8), carrier_rounding=R.RO))


def test_max_norm_renormalizes_as_torchs_does():
    a, idx = _embedding("cpu", max_norm=1.0)
    b, _ = _embedding("cpu", max_norm=1.0)
    want = Quant(P8)(a(idx).detach())
    got = qembedding(idx, b.weight, PointwiseFormats(quant=Quant(P8)), max_norm=1.0)
    assert _same(got.detach(), want.detach()) and torch.equal(a.weight, b.weight)
