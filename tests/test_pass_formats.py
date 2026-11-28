"""The private builders behind a layer's per-pass arithmetic and its output
rounding: ``_hooks._gemm_formats`` and ``conv._conv_formats``, which give each
of a layer's three GEMMs its own mac, and ``_hooks._with_output``, which rounds
the forward's result after the bias.

Tier 3 throughout. A layer whose passes run three macs is held, pass by pass,
to three layers that each run one of them, with the same parameters: under a
deterministic rounding a pass depends only on its own mac, so the forward of
the mixed layer is the first one's, its input gradient the second's and its
weight gradient the third's, word for word. An output rounding is held to the
same quantizer applied to the unrounded layer's output, and its gradients to
the unrounded layer's (it is straight-through).
"""

import copy
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from mptorch.number import BinaryK, RoundMode
from mptorch.quant import (
    FusedMac,
    QAffineFormats,
    QConv1d,
    QConv2d,
    QLinear,
    QMatmul,
    QMatmulFormats,
    Quant,
    SplitMac,
    conv_formats,
    matmul_formats,
)
from mptorch.quant.factories._hooks import _gemm_formats, _with_output
from mptorch.quant.factories.conv import _conv_formats, _plain_conv
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops._spec import _run_gemm
from tests.markers import available_devices

SR = RoundMode.SR
# Three macs whose results differ on the same operands.
FWD = SplitMac(BinaryK(8, 4), BinaryK(16, 11))
IGRAD = FusedMac(BinaryK(12, 7))
WGRAD = SplitMac(BinaryK(8, 5), BinaryK(10, 5))
OUT = Quant(BinaryK(8, 3))


def _linear_formats(fwd, igrad=None, wgrad=None):
    def run(mac):
        return None if mac is None else partial(_run_gemm, spec_for_mac(mac))

    return _gemm_formats(run(fwd), run(igrad), run(wgrad))


def _step(layer, *inputs, grad_seed=3):
    """The output, the inputs' gradients and the parameters', by name."""
    inputs = [x.detach().clone().requires_grad_() for x in inputs]
    layer.zero_grad(set_to_none=True)
    out = layer(*inputs)
    torch.manual_seed(grad_seed)
    # not randn_like, which draws in the output's memory order
    out.backward(torch.randn(out.shape, dtype=out.dtype).to(out.device))
    grads = {f"input{i}": x.grad for i, x in enumerate(inputs)}
    # a compiled module names its parameters under _orig_mod
    grads.update({n.removeprefix("_orig_mod."): p.grad for n, p in layer.named_parameters()})
    return out.detach(), grads


def _same_parameters(make, formats):
    torch.manual_seed(0)
    return make(formats)


@pytest.mark.parametrize("device", available_devices)
def test_linear_runs_one_mac_per_pass(device):
    def make(formats):
        return QLinear(16, 8, formats=formats).to(device)

    x = torch.randn(2, 5, 16, device=device)
    out, grads = _step(_same_parameters(make, _linear_formats(FWD, IGRAD, WGRAD)), x)
    out_f, grads_f = _step(_same_parameters(make, _linear_formats(FWD)), x)
    _, grads_i = _step(_same_parameters(make, _linear_formats(IGRAD)), x)
    _, grads_w = _step(_same_parameters(make, _linear_formats(WGRAD)), x)
    assert torch.equal(out, out_f)
    assert torch.equal(grads["input0"], grads_i["input0"])
    assert torch.equal(grads["weight"], grads_w["weight"])
    assert torch.equal(grads["bias"], grads_f["bias"])  # a sum, in no mac's arithmetic
    # the macs differ, so the comparisons above say something
    assert not torch.equal(grads_f["input0"], grads_i["input0"])
    assert not torch.equal(grads_f["weight"], grads_w["weight"])


@pytest.mark.parametrize("device", available_devices)
def test_linear_passes_default_to_the_forward_mac(device):
    """``igrad`` and ``wgrad`` left out run the forward's mac, as every
    factory's single-mac call relies on."""

    def make(formats):
        return QLinear(16, 8, formats=formats).to(device)

    x = torch.randn(7, 16, device=device)
    got = _step(_same_parameters(make, _linear_formats(FWD)), x)
    want = _step(_same_parameters(make, _linear_formats(FWD, FWD, FWD)), x)
    assert torch.equal(got[0], want[0])
    for name in want[1]:
        assert torch.equal(got[1][name], want[1][name]), name


@pytest.mark.parametrize("device", available_devices)
def test_conv_runs_one_mac_per_pass(device):
    specs = [spec_for_mac(m) for m in (FWD, IGRAD, WGRAD)]

    def make(formats):
        return QConv2d(3, 4, 3, stride=2, padding=1, formats=formats).to(device)

    x = torch.randn(2, 3, 9, 9, device=device)
    out, grads = _step(_same_parameters(make, _conv_formats(*specs)), x)
    out_f, grads_f = _step(_same_parameters(make, conv_formats(FWD)), x)
    _, grads_i = _step(_same_parameters(make, conv_formats(IGRAD)), x)
    _, grads_w = _step(_same_parameters(make, conv_formats(WGRAD)), x)
    assert torch.equal(out, out_f)
    assert torch.equal(grads["input0"], grads_i["input0"])
    assert torch.equal(grads["weight"], grads_w["weight"])
    assert torch.equal(grads["bias"], grads_f["bias"])
    assert not torch.equal(grads_f["input0"], grads_i["input0"])
    assert not torch.equal(grads_f["weight"], grads_w["weight"])


# --- the output rounding ------------------------------------------------------


def _held_to_the_unrounded(rounded, plain, inputs, out=OUT):
    """``rounded``'s output is ``out`` of ``plain``'s, and its gradients are
    ``plain``'s, for the same output gradient."""
    got, got_grads = _step(rounded, *inputs)
    want, want_grads = _step(plain, *inputs)
    assert torch.equal(got, out(want))
    assert not torch.equal(got, want)  # the rounding did something
    for name, g in want_grads.items():
        assert torch.equal(got_grads[name], g), name


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("hooks", [False, True], ids=["plain", "mac"])
def test_linear_output_rounding(device, hooks):
    def formats():
        return _linear_formats(FWD) if hooks else QAffineFormats()

    torch.manual_seed(0)
    plain = QLinear(16, 8, formats=formats()).to(device)
    rounded = copy.deepcopy(plain)
    rounded.formats = _with_output(formats(), OUT, F.linear)
    _held_to_the_unrounded(rounded, plain, [torch.randn(2, 5, 16, device=device)])


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "hooks,layout",
    [(h, layout) for h in (False, True) for layout in ("batched", "unbatched", "channels_last")],
    ids=lambda v: ("mac" if v else "plain") if isinstance(v, bool) else v,
)
def test_conv_output_rounding(device, hooks, layout):
    """A channels_last output is not contiguous, which the in-place quantizer
    refuses, so it is rounded into a copy instead."""

    def formats():
        return conv_formats(FWD) if hooks else QAffineFormats()

    torch.manual_seed(0)
    if layout == "unbatched":
        plain, x = QConv1d(3, 4, 3, padding=1, formats=formats()), torch.randn(3, 11)
    else:
        plain, x = QConv2d(3, 4, 3, padding=1, formats=formats()), torch.randn(2, 3, 7, 7)
    plain, x = plain.to(device), x.to(device)
    if layout == "channels_last":
        plain = plain.to(memory_format=torch.channels_last)
        x = x.contiguous(memory_format=torch.channels_last)
    rounded = copy.deepcopy(plain)
    rounded.formats = _with_output(formats(), OUT, _plain_conv)
    _held_to_the_unrounded(rounded, plain, [x])


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("hooks", [False, True], ids=["plain", "mac"])
def test_matmul_output_rounding(device, hooks):
    def formats():
        return matmul_formats(FWD) if hooks else QMatmulFormats()

    plain = QMatmul(formats())
    rounded = QMatmul(_with_output(formats(), OUT, torch.matmul))
    a, b = torch.randn(3, 4, 16, device=device), torch.randn(16, 5, device=device)
    _held_to_the_unrounded(rounded, plain, [a, b])


def test_a_widening_output_rounding_is_out_of_place():
    """A ``Quant`` that rounds a float32 result in binary64 cannot round it in
    place (the in-place quantizers refuse a widening carrier), and is called
    as it is."""
    out = Quant(BinaryK(8, 3), carrier=torch.float64)
    torch.manual_seed(0)
    plain = QLinear(16, 8, formats=_linear_formats(FWD))
    rounded = copy.deepcopy(plain)
    rounded.formats = _with_output(_linear_formats(FWD), out, F.linear)
    _held_to_the_unrounded(rounded, plain, [torch.randn(4, 16)], out=out)


@pytest.mark.parametrize("device", available_devices)
def test_a_stochastic_output_rounding_draws_once(device):
    """Under SR the rounding draws what one quantize call of the unrounded
    output draws, from the same generator state, and leaves it the same."""
    out = Quant(BinaryK(8, 3, prng_bits=8), SR)
    torch.manual_seed(0)
    plain = QLinear(16, 8, formats=_linear_formats(FWD)).to(device)
    rounded = copy.deepcopy(plain)
    rounded.formats = _with_output(_linear_formats(FWD), out, F.linear)
    x = torch.randn(6, 16, device=device)
    with torch.no_grad():
        torch.manual_seed(11)
        got = rounded(x)
        after = torch.get_rng_state()
        torch.manual_seed(11)
        want = out(plain(x))
    assert torch.equal(got, want)
    assert torch.equal(after, torch.get_rng_state())


# Dynamo says once that it traces through the format checks' lru_cache; the
# checks are pure, so the cache only saves time.
@pytest.mark.filterwarnings("ignore:Dynamo detected a call to a `functools.lru_cache`:UserWarning")
@pytest.mark.parametrize("backend", ["aot_eager"])
def test_an_in_place_output_rounding_compiles(backend):
    """The in-place rounding of the forward's own result is functionalized
    under compile: one graph, eager's results."""
    torch.manual_seed(0)
    plain = QLinear(16, 8, formats=_linear_formats(FWD))
    rounded = copy.deepcopy(plain)
    rounded.formats = _with_output(_linear_formats(FWD), OUT, F.linear)
    torch._dynamo.reset()
    compiled = torch.compile(copy.deepcopy(rounded), backend=backend, fullgraph=True)
    x = torch.randn(4, 16)
    got, got_grads = _step(compiled, x)
    want, want_grads = _step(rounded, x)
    assert torch.equal(got, want)
    for name, g in want_grads.items():
        assert torch.equal(got_grads[name], g), name
