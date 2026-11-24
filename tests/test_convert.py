"""
Converting a model: ``convert_``, ``convert``, ``unconvert_``, ``register`` and ``from_float``.

``mptorch/quant/convert.py`` switches each registered layer's class in place
and gives it a container built from the recipe's formats. The claims checked
here:

- tier 1: formats that round nothing give exactly the original model, forward
  and backward;
- tier 3: a converted model equals the same model built by hand from mptorch
  layers and the factories, ``torch.equal``;
- a conversion in place keeps every Parameter, buffer, hook and ``state_dict``
  key, so an optimizer built before it still trains the model; a conversion
  of a copy keeps tied weights tied and leaves the original alone;
- what is kept, and why: a subclass (a parametrized module among them), a
  layer that is mptorch's already, a module the recipe gives ``None`` or
  that ``skip`` names; ``strict`` raises for the first kind;
- ``unconvert_`` undoes a conversion, and ``from_float`` makes a layer over
  another's parameters;
- a converted model compiles in one graph; torchvision's models convert.
"""

import copy
from typing import cast

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from mptorch import MXFP8_E4M3, BinaryK, RoundMode
from mptorch.quant import (
    GemmFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    QLinear,
    Quant,
    Recipe,
    SplitMac,
    binaryK_gemm_formats,
    block_gemm_formats,
    conv_formats,
    convert,
    convert_,
    presets,
    register,
    unconvert_,
)
from mptorch.quant.convert import _REGISTRY
from tests.markers import available_devices

P8 = BinaryK(8, 4)
P16 = BinaryK(16, 11)
NOTHING = Recipe(gemm=GemmFormats())


class _Net(nn.Module):
    """Every registered class, a module the recipe leaves alone, and a bias-free layer."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 4, 3, stride=2, padding=1, groups=2)
        self.c3 = nn.Conv3d(4, 2, 1)
        self.act = nn.ReLU()
        self.fc = nn.Linear(2 * 8, 8, bias=False)
        self.head = nn.Linear(8, 3)

    def forward(self, x):
        y = self.act(self.c1(x))  # [B, 4, 16]
        y = y.view(y.shape[0], 4, 4, 4)
        y = self.act(self.c2(y))  # [B, 4, 2, 2]
        y = self.c3(y.unsqueeze(-1)).flatten(1)  # [B, 8]
        y = torch.cat([y, y], dim=1)
        return self.head(self.act(self.fc(y)))


def _input(device):
    torch.manual_seed(3)
    return torch.randn(2, 3, 16, device=device)


def _step(model, x, device):
    """Forward and backward, under one seed: the output and every gradient, by name."""
    x = x.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    if device == "cuda":
        torch.cuda.manual_seed_all(11)
    out = model(x)
    out.backward(torch.ones_like(out) / out.numel())
    # a compiled model's names are its module's, under _orig_mod
    grads = {n.removeprefix("_orig_mod."): p.grad for n, p in model.named_parameters()}
    return {"out": out.detach(), "x": x.grad, **grads}


def _equal(a, b, model=None):
    """Every entry equal. With ``model``, a convolution's bias gradient is held
    to ``atol=1e-5`` instead, as ``test_qconv*d.py``'s tier 1 holds it: the
    layer sums it in another order than PyTorch's convolution backward."""
    assert a.keys() == b.keys()
    conv_biases = set()
    if model is not None:
        conv_biases = {
            f"{n}.bias" if n else "bias"
            for n, m in model.named_modules()
            if isinstance(m, nn.Conv1d | nn.Conv2d | nn.Conv3d)
        }
    for k in a:
        if k in conv_biases:
            assert torch.allclose(a[k], b[k], atol=1e-5), k
        else:
            assert torch.equal(a[k], b[k]), k


def _net(device):
    torch.manual_seed(0)
    return _Net().to(device)


# --- what a converted model computes --------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_tier1_nothing_rounded_is_the_original(device):
    model = _net(device)
    converted, report = convert(model, NOTHING)
    assert {type(m) for m in converted.modules()} >= {QConv1d, QConv2d, QConv3d, QLinear}
    assert len(report.converted) == 5 and not report.kept
    x = _input(device)
    _equal(_step(converted, x, device), _step(model, x, device), model)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_tier3_the_model_built_by_hand(device, rounding):
    fmt = BinaryK(8, 4, prng_bits=8) if rounding is RoundMode.SR else P8
    acc = BinaryK(16, 11, prng_bits=fmt.prng_bits)
    mac, q = SplitMac(fmt, acc, rounding=rounding), Quant(fmt, rounding)
    recipe = Recipe(gemm=GemmFormats.uniform(mac, q)).override("head", linear=None)
    model = _net(device)
    convert_(model, recipe)

    def hand(formats):
        formats.input_quant = formats.weight_quant = formats.igrad_quant = q
        formats.wgrad_quant = q
        return formats

    def linear():
        return binaryK_gemm_formats(
            mul_K=8,
            mul_P=4,
            mul_prng_bits=fmt.prng_bits,
            acc_K=16,
            acc_P=11,
            acc_prng_bits=acc.prng_bits,
            rounding_mode=rounding,
        )

    torch.manual_seed(0)
    built = _Net()
    built.c1 = QConv1d(3, 4, 3, padding=1, formats=hand(conv_formats(mac)))
    built.c2 = QConv2d(4, 4, 3, stride=2, padding=1, groups=2, formats=hand(conv_formats(mac)))
    built.c3 = QConv3d(4, 2, 1, formats=hand(conv_formats(mac)))
    built.fc = QLinear(16, 8, bias=False, formats=hand(linear()))
    built = built.to(device)
    built.load_state_dict(model.state_dict())
    assert type(model.head) is nn.Linear
    x = _input(device)
    _equal(_step(model, x, device), _step(built, x, device))


@pytest.mark.parametrize("device", available_devices)
def test_a_block_format_layer(device):
    """An MX layer converts, equals block_gemm_formats' layer, and freezes."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8)).to(device)
    built = copy.deepcopy(model)
    convert_(model, presets.mx(acc=P16))
    from mptorch import MXFP8_E5M2

    for i in (0, 2):
        layer = built[i]
        assert isinstance(layer, nn.Linear)
        built[i] = QLinear.from_float(
            layer, block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3, MXFP8_E5M2, acc=P16)
        )
    x = torch.randn(4, 64, device=device)
    _equal(_step(model, x, device), _step(built, x, device))
    with torch.no_grad():
        before = model(x)
        cast(QLinear, model[0]).freeze_packed_()
        assert torch.equal(model(x), before)


# --- what a conversion in place keeps -------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_parameters_hooks_and_state_are_kept(device):
    model = _net(device)
    params = list(model.parameters())
    keys = list(model.state_dict())
    calls = []
    model.fc.register_forward_hook(lambda m, i, o: calls.append(type(m).__name__))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    convert_(model, presets.uniform(P8, acc=P16))
    assert all(a is b for a, b in zip(model.parameters(), params, strict=True))
    assert list(model.state_dict()) == keys
    before = model.fc.weight.detach().clone()
    model(_input(device)).sum().backward()
    optimizer.step()
    assert calls == ["QLinear"]
    assert not torch.equal(model.fc.weight, before)


@pytest.mark.parametrize("device", available_devices)
def test_state_dicts_load_both_ways(device):
    model = _net(device)
    converted, _ = convert(model, presets.uniform(P8))
    plain = _net(device)
    plain.load_state_dict(converted.state_dict(), strict=True)
    converted.load_state_dict(model.state_dict(), strict=True)


def test_convert_copies_and_keeps_tied_weights_tied():
    first, second = nn.Linear(4, 4), nn.Linear(4, 4)
    second.weight = first.weight
    copied, _ = convert(nn.Sequential(first, second), presets.uniform(P8))
    c0, c1 = cast(nn.Sequential, copied)
    assert c0.weight is c1.weight and c0.weight is not first.weight
    assert type(first) is nn.Linear and type(c0) is QLinear


@pytest.mark.parametrize("device", available_devices)
def test_convert_equals_convert_(device):
    model = _net(device)
    copied, _ = convert(model, presets.uniform(P8, acc=P16))
    convert_(model, presets.uniform(P8, acc=P16))
    x = _input(device)
    _equal(_step(copied, x, device), _step(model, x, device))


def test_each_layer_gets_its_own_container():
    model = _net("cpu")
    convert_(model, presets.uniform(P8))
    containers = [m.formats for m in model.modules() if hasattr(m, "formats")]
    assert len({id(c) for c in containers}) == len(containers) == 5


def test_a_model_on_the_meta_device():
    with torch.device("meta"):
        model = _Net()
    report = convert_(model, presets.uniform(P8))
    assert len(report.converted) == 5 and model.fc.weight.is_meta


# --- what is kept, and why ---------------------------------------------------------------


class _MyLinear(nn.Linear):
    pass


class _Odd(nn.Module):
    def __init__(self):
        super().__init__()
        self.mine = _MyLinear(4, 4)
        self.param = nn.Linear(4, 4)
        parametrize.register_parametrization(self.param, "weight", nn.Identity())
        self.ours = QLinear(4, 4)
        self.plain = nn.Linear(4, 4)
        self.off = nn.Linear(4, 4)
        self.conv = nn.Conv1d(4, 4, 1)


def test_the_reasons_for_keeping_a_module():
    model = _Odd()
    report = convert_(model, presets.uniform(P8).override("off", linear=None), skip="conv")
    actions = {r.name: r.action for r in report.rows}
    assert actions == {
        "mine": "kept: a subclass of Linear, and only the exact class converts",
        "param": "kept: parametrized, and only the exact class converts",
        "ours": "kept: already a QLinear",
        "plain": "converted",
        "off": "kept: the recipe gives it None",
        "conv": "kept: skipped",
    }
    assert type(model.mine) is _MyLinear and type(model.plain) is QLinear
    assert "1 converted, 5 kept" in str(report)


def test_strict_refuses_a_subclass_the_recipe_covers():
    with pytest.raises(ValueError, match=r"mine \(_MyLinear, a subclass of Linear\).*param"):
        convert_(_Odd(), presets.uniform(P8), strict=True)
    model = _Odd()
    setattr(model, "mine", nn.Identity())  # noqa: B010 -- a module of another type
    setattr(model, "param", nn.Identity())  # noqa: B010
    convert_(model, presets.uniform(P8), strict=True)  # QLinear and None are fine
    report = convert_(_Odd(), presets.uniform(P8), strict=True, skip=["mine", "param"])
    assert {r.name for r in report.converted} == {"plain", "off", "conv"}


@pytest.mark.parametrize(
    "skip",
    ["conv", nn.Conv1d, (nn.Conv1d,), lambda name, m: isinstance(m, nn.Conv1d), ["x", nn.Conv1d]],
)
def test_skip_forms(skip):
    model = _Odd()
    convert_(model, presets.uniform(P8), skip=skip)
    assert type(model.conv) is nn.Conv1d and type(model.plain) is QLinear


def test_kinds():
    model = _Odd()
    report = convert_(model, presets.uniform(P8), kinds=("conv",))
    assert [r.name for r in report.converted] == ["conv"] and type(model.plain) is nn.Linear
    with pytest.raises(ValueError, match="unknown kinds"):
        convert_(_Odd(), presets.uniform(P8), kinds=("norm",))


def test_a_shared_module():
    shared = nn.Linear(4, 4)
    model = nn.ModuleDict({"a": shared, "b": shared})
    report = convert_(model, presets.uniform(P8))
    assert [r.name for r in report.rows] == ["a"] and type(shared) is QLinear
    model = nn.ModuleDict({"a": nn.Linear(4, 4)})
    model["b"] = model["a"]
    with pytest.raises(ValueError, match="different formats"):
        convert_(model, presets.uniform(P8).override("b", linear=None))


def test_register():
    class Dense(nn.Linear):
        pass

    class QDense(QLinear, Dense):
        pass

    with pytest.raises(TypeError, match="subclass"):
        register(Dense, kind="linear", qcls=QConv2d)
    with pytest.raises(ValueError, match="unknown kind"):
        register(Dense, kind="norm", qcls=QDense)
    register(Dense, kind="linear", qcls=QDense)
    try:
        layer = Dense(4, 4)
        convert_(layer, presets.uniform(P8))
        assert type(layer) is QDense and layer.formats.fwd_math is not None
        assert unconvert_(layer) == 1 and type(layer) is Dense
    finally:
        del _REGISTRY[Dense]


class _Swish(nn.SiLU):
    pass


def test_activations_and_embeddings_convert_when_the_recipe_names_them():
    """torch's activations and ``nn.Embedding`` convert, and report, only
    under a recipe that names their kind; a subclass is kept and reported,
    and ``strict`` refuses it; ``unconvert_`` turns them back; the
    ``elementwise`` kind has no module to convert."""
    model = nn.Sequential(nn.Embedding(10, 8), nn.Linear(8, 8), nn.ReLU(inplace=True), _Swish())
    report = convert_(copy.deepcopy(model), presets.uniform(P8))
    assert [r.name for r in report.rows] == ["1"]
    recipe = presets.uniform(P8, kinds=("linear", "activation", "embedding"))
    converted = copy.deepcopy(model)
    report = convert_(converted, recipe)
    assert [type(m).__name__ for m in converted] == ["QEmbedding", "QLinear", "QReLU", "_Swish"]
    assert report.kept[0].action == "kept: a subclass of SiLU, and only the exact class converts"
    with pytest.raises(ValueError, match="cannot be converted"):
        convert_(copy.deepcopy(model), recipe, strict=True)
    assert unconvert_(converted) == 3
    assert [type(m).__name__ for m in converted] == ["Embedding", "Linear", "ReLU", "_Swish"]
    with pytest.raises(ValueError, match="only simulate reaches"):
        register(nn.Softplus, kind="elementwise", qcls=nn.Softplus)


def test_a_recipe_is_required():
    with pytest.raises(TypeError, match="Recipe"):
        convert_(nn.Linear(2, 2), GemmFormats())  # ty: ignore[invalid-argument-type]


# --- turning back, and from_float --------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_unconvert_round_trip(device):
    model = _net(device)
    reference = copy.deepcopy(model)
    params = list(model.parameters())
    convert_(model, presets.uniform(P8))
    assert unconvert_(model) == 5
    assert [type(m) for m in model.modules()] == [type(m) for m in reference.modules()]
    assert not any(hasattr(m, "formats") for m in model.modules())
    assert all(a is b for a, b in zip(model.parameters(), params, strict=True))
    x = _input(device)
    _equal(_step(model, x, device), _step(reference, x, device), model)
    assert unconvert_(model) == 0


def test_unconvert_refuses_a_frozen_layer():
    model = nn.Sequential(nn.Linear(64, 8), nn.Linear(8, 4))
    convert_(model, presets.mx())
    cast(QLinear, model[0]).freeze_packed_()
    with pytest.raises(TypeError, match="unfreeze_packed_"):
        unconvert_(model)
    assert type(model[1]) is QLinear  # nothing was changed


def test_from_float():
    lin = nn.Linear(4, 3)
    q = QLinear.from_float(lin, binaryK_gemm_formats(mul_K=8, mul_P=4))
    assert type(q) is QLinear and type(lin) is nn.Linear
    assert q.weight is lin.weight and q.bias is lin.bias
    q.register_forward_hook(lambda *a: None)
    assert not lin._forward_hooks
    plain = QConv2d.from_float(nn.Conv2d(2, 3, 3))
    x = torch.randn(1, 2, 5, 5)
    assert torch.equal(plain(x), nn.functional.conv2d(x, plain.weight, plain.bias))
    with pytest.raises(TypeError, match="from_float takes an Linear"):
        QLinear.from_float(_MyLinear(2, 2))
    with pytest.raises(TypeError):
        QConv1d.from_float(nn.Conv2d(2, 2, 1))  # ty: ignore[invalid-argument-type]


# --- compile, and torchvision's models ---------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_a_converted_model_compiles(device):
    model = _net(device)
    convert_(model, presets.uniform(P8, acc=P16))
    x = _input(device)
    eager = _step(model, x, device)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    _equal(_step(compiled, x, device), eager)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", ["resnet18", "mobilenet_v3_small"])
def test_torchvision_models(device, name):
    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = getattr(models, name)(num_classes=10).to(device)
    gemm_layers = [m for m in model.modules() if type(m) in (nn.Linear, nn.Conv2d)]  # what converts
    x = torch.randn(2, 3, 32, 32, device=device)
    plain, report = convert(model, NOTHING)
    assert len(report.converted) == len(gemm_layers) and not report.kept
    _equal(_step(plain, x, device), _step(model, x, device), model)
    # scaled, so that the random weights' activations stay in an 8-bit range
    quantized, _ = convert(model, presets.fp8(acc=P16))
    out = _step(quantized, x, device)
    assert all(t.isfinite().all() for t in out.values())


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "layer,shape",
    [
        (nn.Linear(8, 4), (3, 8)),
        (nn.Linear(8, 4), (2, 3, 8)),
        (nn.Linear(8, 4), (8,)),
        (nn.Conv2d(2, 4, 3), (1, 2, 5, 5)),
        (nn.Conv2d(2, 4, 3), (2, 5, 5)),
        (nn.Conv1d(2, 4, 3), (2, 5)),
    ],
    ids=str,
)
@pytest.mark.parametrize("recipe", [NOTHING, presets.uniform(P8, acc=P16)], ids=["plain", "p8"])
def test_an_in_place_op_after_a_layer(device, layer, shape, recipe):
    """An in-place activation after a converted layer, as in torchvision's
    classifiers: the layer's output is not a view autograd refuses to modify,
    and the gradient is the one an out-of-place activation gives."""
    torch.manual_seed(0)
    layer = layer.to(device)
    x = torch.randn(*shape, device=device)
    grads = []
    for inplace in (True, False):
        model = nn.Sequential(copy.deepcopy(layer), nn.ReLU(inplace=inplace))
        convert_(model, recipe)
        grads.append(_step(model, x, device))
    _equal(grads[0], grads[1])
