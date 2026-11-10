"""
A layer that keeps an operand raw for backward and quantizes it again there.

``CustomArithLinear``, ``CustomArithConvNd`` and ``CustomArithMatmul`` need
each quantized operand again in backward. Since phase V they can keep the raw
operand instead of the quantized copy and run the quantizer a second time
(``mptorch/quant/modules/_saved.py``), which saves the copy wherever the raw
operand is alive anyway. The claims this rests on, each checked here:

- the gradients are bit-identical either way, stochastic rounding included,
  and the generators end the step where they would have;
- the graph node holds the raw operand exactly when it is recomputed, and
  ``None`` means each layer's own default (conv: both operands; Linear: the
  weight; matmul: neither);
- a quantizer mptorch does not know to be stateless keeps its result saved,
  unless the flag is ``True``, and is then replayed under the generators its
  forward call saw;
- a result that is not a tensor (a ``BlockPacked``) stays saved;
- an in-place write to a recomputed operand raises, as it does for
  ``nn.Conv2d``'s saved input;
- on CUDA, the bytes held for backward drop by exactly the quantized copies.
"""

import dataclasses
import functools

import pytest
import torch
from torch import nn

from mptorch import MXFP8_E4M3, NVFP4, BinaryK, RoundMode
from mptorch.quant import (
    BlockQuant,
    QAffineFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    QLinear,
    QMatmul,
    QMatmulFormats,
    Quant,
    Quantizer,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_quantize,
    block_gemm_formats,
    conv_formats,
    matmul_formats,
)
from mptorch.quant.modules import _saved
from tests.markers import available_devices, requires_cuda

KINDS = ["linear", "conv1d", "conv2d", "conv3d", "matmul"]
# (input or a, weight or b) requires_grad
PATTERNS = [(True, True), (True, False), (False, True)]
# what a None flag means, as (input, weight) or (a, b)
DEFAULTS = {
    "linear": (False, True),
    "conv1d": (True, True),
    "conv2d": (True, True),
    "conv3d": (True, True),
    "matmul": (False, False),
}
SR = Quant(BinaryK(8, 4, prng_bits=8), RoundMode.SR)


def _flag_names(kind):
    return (
        ("recompute_a", "recompute_b")
        if kind == "matmul"
        else ("recompute_input", "recompute_weight")
    )


def _formats(kind, q, flags=None, grad_q="same", core=None):
    """Formats with ``q`` on both operands (and, by default, both gradients)."""
    g = q if grad_q == "same" else grad_q
    if kind == "matmul":
        f = core if core is not None else QMatmulFormats()
        f.a_quant, f.b_quant, f.agrad_quant, f.bgrad_quant = q, q, g, g
    else:
        f = core if core is not None else QAffineFormats()
        f.input_quant, f.weight_quant, f.igrad_quant, f.wgrad_quant = q, q, g, g
    if flags is not None:
        for name, value in zip(_flag_names(kind), flags, strict=True):
            setattr(f, name, value)
    return f


def _layer(kind, device, formats):
    """A layer of ``kind`` over ``formats`` and its two operands (the second
    is the layer's weight, or ``b`` for a matmul)."""
    torch.manual_seed(7)
    if kind == "matmul":
        layer = QMatmul(formats)
        return layer, torch.randn(4, 8, device=device), torch.randn(8, 6, device=device)
    if kind == "linear":
        layer, shape = QLinear(8, 6, bias=False, formats=formats), (4, 8)
    elif kind == "conv1d":
        layer, shape = QConv1d(3, 5, 3, padding=1, bias=False, formats=formats), (2, 3, 7)
    elif kind == "conv2d":
        layer, shape = QConv2d(3, 5, 3, padding=1, bias=False, formats=formats), (2, 3, 6, 6)
    else:
        layer, shape = QConv3d(3, 5, 3, padding=1, bias=False, formats=formats), (2, 3, 4, 4, 4)
    layer = layer.to(device)
    return layer, torch.randn(*shape, device=device), layer.weight


def _call(kind, layer, x, w):
    return layer(x, w) if kind == "matmul" else layer(x)


def _step(kind, device, formats, rg=(True, True)):
    """Forward and backward; the output, the gradients, and one draw from the
    CPU generator and one from the device's after the step."""
    layer, x, w = _layer(kind, device, formats)
    x.requires_grad_(rg[0])
    w.requires_grad_(rg[1])
    torch.manual_seed(1234)
    out = _call(kind, layer, x, w)
    out.backward(torch.randn_like(out))
    grads = [t.grad for t in (x, w) if t.requires_grad]
    after = [torch.rand(4), torch.rand(4, device=device)]
    return [out.detach(), *grads, *after]


def _equal(a, b):
    assert len(a) == len(b)
    for s, t in zip(a, b, strict=True):
        assert torch.equal(s, t)


def _shares(saved, raw):
    return saved.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("rg", PATTERNS)
@pytest.mark.parametrize("flags", [(True, True), (True, False), (False, True)])
def test_recomputing_is_bit_identical(device, kind, rounding, rg, flags):
    """
    Recomputing either operand or both gives the output, the gradients and
    the generators' next draws of saving the quantized copies.
    """
    q = Quant(BinaryK(8, 4, prng_bits=8), rounding)
    want = _step(kind, device, _formats(kind, q, (False, False)), rg)
    _equal(_step(kind, device, _formats(kind, q, flags), rg), want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("flags", [None, (True, True), (False, False), (True, False)])
def test_the_node_holds_the_raw_operand_when_it_is_recomputed(device, kind, flags):
    """
    The graph node holds an operand's raw tensor when that operand is
    recomputed and a quantized copy otherwise; ``None`` is the layer's own
    default.
    """
    layer, x, w = _layer(kind, device, _formats(kind, Quant(BinaryK(8, 4)), flags))
    x.requires_grad_()
    w.requires_grad_()
    out = _call(kind, layer, x, w)
    want = DEFAULTS[kind] if flags is None else flags
    saved = out.grad_fn.saved_tensors
    assert [_shares(s, raw) for s, raw in zip(saved, (x, w), strict=True)] == list(want)


@pytest.mark.parametrize("kind", KINDS)
def test_formats_without_the_flags_take_the_default(kind):
    """
    A formats object without the attributes (one pickled before phase V)
    behaves as if both were ``None``.
    """
    f = _formats(kind, Quant(BinaryK(8, 4)))
    for name in _flag_names(kind):
        delattr(f, name)
    layer, x, w = _layer(kind, "cpu", f)
    x.requires_grad_()
    w.requires_grad_()
    out = _call(kind, layer, x, w)
    saved = out.grad_fn.saved_tensors
    assert [_shares(s, raw) for s, raw in zip(saved, (x, w), strict=True)] == list(DEFAULTS[kind])


class CountingQuant:
    """A user's quantizer: ``Quant`` behind a call counter, which mptorch
    cannot tell from a stateful one."""

    def __init__(self, q):
        self.q, self.calls = q, 0

    def __call__(self, x):
        self.calls += 1
        return self.q(x)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", KINDS)
def test_a_users_quantizer_is_saved_unless_the_flag_says_otherwise(device, kind):
    """
    A quantizer mptorch does not know to be stateless runs once per operand
    in the forward and never again under the default, and its result is what
    the node holds; under ``True`` it runs once more per operand in backward.
    """
    for flags, backward_calls, raw in ((None, 0, False), ((True, True), 2, True)):
        cq = CountingQuant(SR)
        layer, x, w = _layer(kind, device, _formats(kind, cq, flags, grad_q=None))
        x.requires_grad_()
        w.requires_grad_()
        out = _call(kind, layer, x, w)
        assert cq.calls == 2
        assert [_shares(s, t) for s, t in zip(out.grad_fn.saved_tensors, (x, w), strict=True)] == [
            raw
        ] * 2
        out.sum().backward()
        assert cq.calls == 2 + backward_calls


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", KINDS)
def test_a_users_quantizer_is_replayed_under_its_generators(device, kind):
    """
    A user's quantizer that draws from the generator, recomputed under
    ``True``, sees in backward the draw its forward call saw: the CPU
    generator and the device's are both set back.
    """

    def noisy(x):
        return x + torch.rand_like(x) + torch.rand(1).to(x.device)

    want = _step(kind, device, _formats(kind, noisy, (False, False), grad_q=SR))
    _equal(_step(kind, device, _formats(kind, noisy, (True, True), grad_q=SR)), want)


def test_which_quantizers_are_known_to_be_stateless():
    """
    ``Quant`` (not in place), ``BlockQuant`` called as a fake quantizer, and a
    ``Quantizer`` over one of those; nothing else.
    """
    q = Quant(BinaryK(8, 4))
    bq = BlockQuant(MXFP8_E4M3)
    assert _saved._known(q) is q
    assert _saved._known(bq) is bq
    assert _saved._known(Quantizer(BinaryK(8, 4))) == q
    assert _saved._known(Quantizer(bq, BinaryK(8, 4))) is bq
    for other in (
        Quant(BinaryK(8, 4), inplace=True),
        bq.pack,
        functools.partial(bq.pack_fresh, axis=-1),
        functools.partial(binaryK_quantize, K=8, P=4),
        Quantizer(None, BinaryK(8, 4)),
        lambda x: x,
        CountingQuant(q),
    ):
        assert _saved._known(other) is None


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_a_block_fake_quantizer_is_recomputed(device, rounding):
    """
    ``BlockQuant``'s fake quantizer returns a tensor and is stateless, so a
    Linear's default recomputes the weight with it, bit-identically under SR.
    """

    def run(flags):
        torch.manual_seed(7)
        f = QAffineFormats(
            input_quant=BlockQuant(MXFP8_E4M3, rounding),
            weight_quant=BlockQuant(MXFP8_E4M3, rounding),
        )
        if flags is not None:
            f.recompute_input, f.recompute_weight = flags
        layer = QLinear(32, 6, bias=False, formats=f).to(device)
        x = torch.randn(4, 32, device=device, requires_grad=True)
        torch.manual_seed(1234)
        out = layer(x)
        saved = out.grad_fn.saved_tensors
        out.backward(torch.randn_like(out))
        return [_shares(saved[1], layer.weight), out.detach(), x.grad, layer.weight.grad]

    got, want = run(None), run((False, False))
    assert got[0] and not want[0]
    _equal(got[1:], want[1:])


@pytest.mark.parametrize("device", available_devices)
def test_a_packed_weight_stays_saved(device):
    """
    A weight slot that returns a ``BlockPacked`` keeps it saved, on ``ctx``,
    whatever the flag; the input beside it goes through ``save_for_backward``.
    """
    w16 = BlockQuant(dataclasses.replace(NVFP4, block_rows=16), axis=1)
    results = []
    for flag in (None, True, False):
        torch.manual_seed(7)
        f = block_gemm_formats(NVFP4, w16.fmt, acc=BinaryK(16, 11))
        f.weight_quant = w16.pack
        f.recompute_weight = flag
        layer = QLinear(64, 32, bias=False, formats=f).to(device)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        out = layer(x)
        node = out.grad_fn
        assert node.saved_structs[1] is w16.pack(layer.weight)
        assert node.saved_tensors[1] is None and _shares(node.saved_tensors[0], x)
        out.sum().backward()
        results.append([out.detach(), x.grad, layer.weight.grad])
    _equal(results[1], results[0])
    _equal(results[2], results[0])


@pytest.mark.parametrize("device", available_devices)
def test_an_in_place_write_to_a_recomputed_input_raises(device):
    """
    The raw input a convolution keeps is checked by autograd's version
    counter, as ``nn.Conv2d``'s is, so writing it in place between forward and
    backward raises; with ``recompute_input=False`` the layer keeps its own
    quantized copy, and the write is harmless.
    """
    for flag in (None, False):
        f = _formats("conv2d", Quant(BinaryK(8, 4)), (flag, None))
        layer, x, _ = _layer("conv2d", device, f)
        h = x.clone().requires_grad_()
        y = h * 1.0
        out = layer(y)
        y.add_(1.0)
        if flag is None:
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                out.sum().backward()
        else:
            out.sum().backward()
            assert h.grad is not None


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["linear", "conv2d", "matmul"])
def test_recomputing_under_the_gemm_and_conv_kernels(device, kind):
    """
    With the three passes on mptorch's own GEMM or conv kernels, under SR,
    recomputing both operands is bit-identical too.
    """
    mul = BinaryK(8, 4, prng_bits=8)
    acc = BinaryK(10, 5, prng_bits=8)
    if kind == "linear":
        core = binaryK_gemm_formats(
            mul_K=8,
            mul_P=4,
            mul_prng_bits=8,
            acc_K=10,
            acc_P=5,
            acc_prng_bits=8,
            rounding_mode=RoundMode.SR,
        )
    elif kind == "conv2d":
        core = conv_formats(SplitMac(mul, acc, rounding=RoundMode.SR))
    else:
        core = matmul_formats(SplitMac(mul, acc, rounding=RoundMode.SR))

    def run(flags):
        f = _formats(kind, SR, None, core=core)
        for name, value in zip(_flag_names(kind), flags, strict=True):
            setattr(f, name, value)
        return _step(kind, device, f)

    _equal(run((True, True)), run((False, False)))


@requires_cuda
def test_held_bytes_drop_by_the_quantized_copies():
    """
    Three conv -> ReLU layers with ``Quant`` on input and weight hold, after
    the forward, exactly the quantized copies fewer under the default than
    with both flags off: one input copy per layer, and one weight copy per
    layer whose input needs a gradient (not the first: the data needs none).
    """

    def held(flags):
        torch.manual_seed(7)
        layers = []
        for _ in range(3):
            f = _formats("conv2d", Quant(BinaryK(8, 4)), flags)
            layers += [QConv2d(16, 16, 3, padding=1, bias=False, formats=f), nn.ReLU()]
        model = nn.Sequential(*layers).cuda()
        x = torch.randn(2, 16, 16, 16, device="cuda")
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        out = model(x)
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() - base - out.numel() * out.element_size()

    def block(nbytes):  # the caching allocator rounds every block up to 512 bytes
        return -(-nbytes // 512) * 512

    activation = block(2 * 16 * 16 * 16 * 4)
    weight = block(16 * 16 * 3 * 3 * 4)
    assert held((False, False)) - held(None) == 3 * activation + 2 * weight
