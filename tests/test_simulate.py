"""
``simulate``: a model run in a recipe's arithmetic, without converting it.

The claims checked here:

- tier 1: a recipe that rounds nothing, whether it routes nothing or every
  product, gives exactly the plain model, forward and backward;
- the invariant: on every layer :func:`convert` converts, ``simulate`` gives
  the converted model's words, outputs and gradients, stochastic rounding
  included;
- what ``convert`` cannot reach, ``simulate`` runs as the equivalent mptorch
  layer would: eager attention's two products as ``QMatmul`` s at their
  sites, GPT-2's ``Conv1D`` as a ``QLinear`` over the transposed weight;
- the rules: a product with a weight is resolved under the weight's owner
  (swin's ``F.linear(x, self.qkv.weight)``), a tied ``lm_head`` under
  itself, a product with a buffer or a view of a weight is a ``matmul``;
- an mptorch layer's own calls, and calls outside the model, are left
  alone; nested simulates, an exception inside the forward, the report and
  ``strict``;
- Hugging Face's own GPT-2 and Llama classes and torchvision's models;
- the kinds past the GEMMs: an activation is the converted module's, an
  embedding too, a model's arithmetic under ``elementwise`` the q-ops written
  out by hand; a recipe that names none of them reports what it reported
  before, and installs none of their handlers;
- softmax and the norms are the converted modules', running statistics
  included; a norm written out of elementwise ops and registered with its
  forward is reached whole, its own calls passed through; ``torch.sum`` and
  ``torch.mean`` under ``reduction`` are ``qsum`` and ``qmean`` written out;
- attention: an SDPA model under the ``attention`` kind computes its eager
  twin's words under ``Recipe.attention_parts``, gradients and SR included;
  ``nn.MultiheadAttention`` is the converted ``QMultiheadAttention``, its
  projections under the ``linear`` kind, its core reported without an
  ``attention`` kind.
"""

import copy
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    QGELU,
    ActivationFormats,
    AttentionFormats,
    GemmFormats,
    NormFormats,
    PointwiseFormats,
    QLinear,
    QMatmul,
    QSiLU,
    Quant,
    Recipe,
    ReductionFormats,
    SoftmaxFormats,
    SplitMac,
    convert,
    convert_,
    qadd,
    qdiv,
    qexp,
    qmean,
    qmul,
    qpow,
    qrmsnorm,
    qrsqrt,
    qsqrt,
    qsub,
    qsum,
    register,
    simulate,
)
from mptorch.quant.convert import _REGISTRY
from mptorch.quant.simulate import _HANDLERS, OWNED, PLAIN, SIMULATED, WHOLE
from tests.markers import available_devices
from tests.models import Conv1D, GPT2Attention, LlamaAttention, LlamaRMSNorm, MiniGPT2, MiniLlama
from tests.models import attention_modules as _attention_modules
from tests.test_convert import _equal, _Net

ACC = BinaryK(16, 11)
NOTHING_ROUNDED = Recipe(gemm=GemmFormats())


def _formats(rounding: RoundMode, p: int = 4) -> GemmFormats:
    """Every pass, operand and gradient in an 8-bit binaryK format with ``p``
    bits of precision, summed in a 16-bit one."""
    sr = rounding is RoundMode.SR
    fmt = BinaryK(8, p, prng_bits=8) if sr else BinaryK(8, p)
    acc = BinaryK(16, 11, prng_bits=8) if sr else ACC
    return GemmFormats.uniform(SplitMac(fmt, acc, rounding=rounding), Quant(fmt, rounding))


def _ids(device):
    ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(3))
    # torch's MPS embedding backward sums a row's repeated ids in no fixed order; with at
    # most two per row the sum is the same either way, so the MPS gradients stay word for word
    assert torch.bincount(ids.flatten()).max() <= 2, "a token id repeats three times or more"
    return ids.to(device)


def _run(model, inputs, device, sim=None, call=None):
    """The output and every gradient, by name, of one forward and backward
    under one seed, with ``sim`` active around the forward if given. The
    forward is ``call`` if given (a compiled wrapper), else ``model``; the
    gradients are read from ``model``'s parameters either way."""
    model.zero_grad(set_to_none=True)
    inputs = [
        x.detach().clone().requires_grad_(True) if x.is_floating_point() else x for x in inputs
    ]
    torch.manual_seed(11)
    if device == "cuda":
        torch.cuda.manual_seed_all(11)
    forward = model if call is None else call
    if sim is None:
        out = forward(*inputs)
    else:
        with sim:
            out = forward(*inputs)
    out.backward(torch.ones_like(out) / out.numel())
    grads = {n: p.grad for n, p in model.named_parameters()}
    return {
        "out": out.detach(),
        **{f"input{i}": x.grad for i, x in enumerate(inputs) if x.is_floating_point()},
        **grads,
    }


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Equal word for word, a NaN equal to a NaN in the same place: a random
    resnet18's 8-bit gradients overflow, and their NaNs must agree too."""
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(a[~a.isnan()], b[~b.isnan()])


def _mini(name, device, **kwargs):
    torch.manual_seed(0)
    model = {"llama": MiniLlama, "gpt2": MiniGPT2}[name](**kwargs)
    return model.to(device), [_ids(device)]


def _net(device):
    torch.manual_seed(0)
    model = _Net().to(device)
    torch.manual_seed(3)
    return model, [torch.randn(2, 3, 16, device=device)]


# --- tier 1 ----------------------------------------------------------------------------


_TIER1 = {
    "none": Recipe(),
    "linear and conv": Recipe(linear=GemmFormats(), conv=GemmFormats()),
    "every product": NOTHING_ROUNDED,
}


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", ["llama", "gpt2", "net"])
@pytest.mark.parametrize("recipe", list(_TIER1))
def test_tier1_nothing_rounded_is_the_plain_model(device, name, recipe):
    """A recipe that names no kind routes nothing, and one whose formats
    round nothing routes products through mptorch's Functions, which then run
    PyTorch's own arithmetic. Routing nothing, or the ``nn.Linear`` and
    convolution calls, gives the plain model's words (a convolution's bias
    gradient to ``atol=1e-5``, summed in another order). Routing a product
    PyTorch computes in another layout keeps the outputs, and keeps the
    gradients to float32 noise only, because on CUDA the two layouts take
    cuBLAS kernels that round differently:

    * a ``matmul``: ``CustomArithMatmul``'s backward multiplies the operands
      as it saved them (a transposed view, ``repeat_kv``'s copy), where
      autograd's backward of ``torch.matmul`` multiplies its own reshaped
      batches, as ``QMatmul``'s own tier 1 allows for;
    * GPT-2's ``Conv1D``, run as a linear over ``W.mT``: its weight gradient
      is ``W.mT``'s, ``gᵀx`` transposed back, where ``addmm``'s backward
      computes ``xᵀg``."""
    model, inputs = _net(device) if name == "net" else _mini(name, device)
    want = _run(model, inputs, device)
    got = _run(model, inputs, device, simulate(_TIER1[recipe], model))
    if recipe == "none" or (recipe == "linear and conv" and name != "gpt2"):
        _equal(got, want, model)
        return
    assert torch.equal(got["out"], want["out"])
    for k in want:
        assert torch.allclose(got[k], want[k], rtol=1e-5, atol=1e-7), k


# --- the invariant ---------------------------------------------------------------------


class _Convs(nn.Module):
    """Every convolution layout convert takes: 1D, 2D with groups and
    ``padding="same"``, a reflected padding, 3D."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(3, 4, 3, padding=1)
        self.c2 = nn.Conv2d(4, 4, 3, padding="same", groups=2)
        self.c3 = nn.Conv2d(4, 4, 3, padding=1, padding_mode="reflect")
        self.c4 = nn.Conv3d(4, 2, 1)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        y = torch.relu(self.c1(x)).view(x.shape[0], 4, 4, 4)
        y = torch.relu(self.c3(torch.relu(self.c2(y))))
        return self.fc(self.c4(y.unsqueeze(-1)).flatten(1))


def _model(name, device):
    if name in ("llama", "llama tied"):
        return _mini("llama", device, tie=name == "llama tied")
    if name == "convs":
        torch.manual_seed(0)
        return _Convs().to(device), [torch.randn(2, 3, 16, device=device)]
    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    return models.resnet18(num_classes=10).to(device), [torch.randn(2, 3, 32, 32, device=device)]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("name", ["llama", "llama tied", "convs", "resnet18"])
def test_simulate_is_the_converted_model(device, rounding, name):
    """On the layers convert converts, simulate gives the converted model's
    outputs and gradients, word for word, stochastic rounding included; a
    later rule (the head in other formats) is honoured by both."""
    model, inputs = _model(name, device)
    head = "lm_head" if "llama" in name else "fc"
    recipe = Recipe(linear=_formats(rounding), conv=_formats(rounding)).override(
        head, linear=_formats(rounding, p=3)
    )
    converted, _ = convert(model, recipe)
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    assert want.keys() == got.keys()
    for k in want:
        assert _same(got[k], want[k]), k
    assert any(r.scope == head and r.action == SIMULATED for r in sim.report().rows)


def _site_recipe(rounding, linear=True):
    recipe = Recipe(linear=_formats(rounding)) if linear else Recipe()
    where = (LlamaAttention, GPT2Attention)
    recipe = recipe.override(where, site="matmul#0", matmul=_formats(rounding, p=3))
    return recipe.override(where, site="matmul#1", matmul=_formats(rounding, p=5))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("name", ["llama", "gpt2"])
def test_attention_sites_are_hand_built_qmatmuls(device, rounding, name):
    """Eager attention's QKᵀ and PV are ``matmul#0`` and ``matmul#1`` of the
    attention module: simulated in two different formats, they equal the
    same model with a ``QMatmul`` in each format at each place, built by
    hand (and its linears converted)."""
    model, inputs = _mini(name, device)
    recipe = _site_recipe(rounding, linear=name == "llama")
    built = copy.deepcopy(model)
    if name == "llama":
        convert_(built, Recipe(linear=_formats(rounding)))
    for attn in _attention_modules(built):
        attn.qk = QMatmul(_formats(rounding, p=3).matmul())
        attn.pv = QMatmul(_formats(rounding, p=5).matmul())
    want = _run(built, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    for k in got:
        assert _same(got[k], want[k]), k
    sites = {(r.scope, r.site) for r in sim.report().simulated if r.kind == "matmul"}
    attn_names = [
        n for n, m in model.named_modules() if isinstance(m, LlamaAttention | GPT2Attention)
    ]
    assert sites == {(n, s) for n in attn_names for s in ("matmul#0", "matmul#1")}


def _conv1d_as_qlinear(model: nn.Module, formats: GemmFormats) -> nn.Module:
    """A copy of ``model`` in which each ``Conv1D`` is a ``QLinear`` over the
    transposed weight: what simulate runs a ``Conv1D``'s ``addmm`` as."""
    model = copy.deepcopy(model)
    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, Conv1D) or type(child).__name__ == "Conv1D":
                weight, bias = cast(torch.Tensor, child.weight), cast(torch.Tensor, child.bias)
                nx, nf = weight.shape
                layer = QLinear(nx, nf, formats=formats.affine(), device=weight.device)
                with torch.no_grad():
                    layer.weight.copy_(weight.mT)
                    assert layer.bias is not None
                    layer.bias.copy_(bias)
                setattr(parent, child_name, layer)
    return model


def _conv1d_grads_match(model, built, got, want):
    """Every gradient equal, a ``Conv1D`` weight's to the transpose of its
    ``QLinear``'s."""
    built_params = dict(built.named_parameters())
    for n, _p in model.named_parameters():
        owner = model.get_submodule(n.rsplit(".", 1)[0]) if "." in n else model
        if type(owner).__name__ == "Conv1D" and n.endswith(".weight"):
            assert torch.equal(got[n], want[n].mT), n
        else:
            assert torch.equal(got[n], want[n]), n
        assert n in built_params


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_conv1d_is_a_qlinear_over_the_transposed_weight(device, rounding):
    """GPT-2's ``Conv1D`` is ``addmm(b, x, W)`` with ``W`` stored ``[in,
    out]``, which no conversion reaches: simulate runs it as ``QLinear``
    over ``W.mT`` (no copy), and that equals a ``QLinear`` holding ``Wᵀ``,
    weight gradients transposed. The tied ``lm_head`` converts either way."""
    model, inputs = _mini("gpt2", device, dropout=0.1)
    recipe = Recipe(linear=_formats(rounding))
    built = _conv1d_as_qlinear(model, _formats(rounding))
    convert_(built, recipe)
    want = _run(built, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    assert torch.equal(got["out"], want["out"])
    _conv1d_grads_match(model, built, got, want)
    owners = {r.scope for r in sim.report().simulated if r.call == "torch.addmm"}
    assert owners == {n for n, m in model.named_modules() if isinstance(m, Conv1D)}


# --- the rules -------------------------------------------------------------------------


class _Swin(nn.Module):
    """swin's attention calls ``F.linear`` on its ``qkv`` layer's weight,
    without calling the layer."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)

    def forward(self, x):
        q, k, v = F.linear(x, self.qkv.weight, self.qkv.bias).chunk(3, dim=-1)
        return self.proj(torch.softmax(q @ k.mT, dim=-1) @ v)


@pytest.mark.parametrize("device", available_devices)
def test_a_weight_is_resolved_under_its_owner(device):
    """``F.linear(x, self.qkv.weight)`` in the attention module's frame is
    resolved under ``qkv``, the weight's owner: an override there reaches it,
    and the result is ``qkv`` converted to those formats."""
    torch.manual_seed(0)
    model = nn.Sequential(_Swin()).to(device)
    x = torch.randn(2, 4, 8, device=device)
    recipe = Recipe(linear=_formats(RoundMode.RNE)).override(
        "*.qkv", linear=_formats(RoundMode.RNE, 3)
    )
    sim = simulate(recipe, model)
    with sim, torch.no_grad():
        got = model(x)
    attn = cast(_Swin, model[0])
    qkv = QLinear.from_float(copy.deepcopy(attn.qkv), _formats(RoundMode.RNE, 3).affine())
    proj = QLinear.from_float(copy.deepcopy(attn.proj), _formats(RoundMode.RNE).affine())
    with torch.no_grad():
        q, k, v = qkv(x).chunk(3, dim=-1)
        want = proj(torch.softmax(q @ k.mT, dim=-1) @ v)
    assert torch.equal(got, want)
    rows = {(r.scope, r.kind, r.call, r.action) for r in sim.report().rows}
    assert ("0.qkv", "linear", "F.linear", SIMULATED) in rows
    assert ("0", "matmul", "Tensor.matmul", PLAIN) in rows


class _Products(nn.Module):
    """A product with a buffer, and one with a view of a weight."""

    def __init__(self):
        super().__init__()
        self.register_buffer("basis", torch.randn(8, 8))
        self.w = nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        return (x @ self.basis) @ self.w.mT


@pytest.mark.parametrize("device", available_devices)
def test_a_buffer_or_a_view_of_a_weight_is_an_activation(device):
    """Only one of the model's Parameters itself is a weight: a buffer, or a
    view such as ``W.mT`` (which compiled code cannot trace back to ``W``),
    makes the product one of two activations, at the module's next site."""
    torch.manual_seed(0)
    model = _Products().to(device)
    sim = simulate(Recipe(matmul=_formats(RoundMode.RNE)), model)
    with sim, torch.no_grad():
        model(torch.randn(3, 8, device=device))
    sites = [(r.kind, r.site) for r in sim.report().simulated]
    assert sites == [("matmul", "matmul#0"), ("matmul", "matmul#1")]


class _Owned(nn.Module):
    """A module that runs in mptorch's arithmetic already, as a Q-layer does."""

    _mptorch_owned = True

    def forward(self, x):
        return x @ x.mT


@pytest.mark.parametrize("device", available_devices)
def test_an_mptorch_layer_and_a_converted_model_are_left_alone(device):
    """A converted model under simulate is the converted model: its layers'
    Functions hide their calls, and a call made inside an mptorch layer's
    own frame passes through, reported as that layer's."""
    model, inputs = _mini("llama", device)
    recipe = Recipe(linear=_formats(RoundMode.RNE))
    converted, _ = convert(model, recipe)
    want = _run(converted, inputs, device)
    sim = simulate(recipe, converted)
    got = _run(converted, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    assert not sim.report().simulated
    owned = nn.Sequential(_Owned()).to(device)
    sim = simulate(Recipe(matmul=_formats(RoundMode.RNE)), owned)
    x = torch.randn(3, 8, device=device)
    with sim:
        got = owned(x)
    assert torch.equal(got, x @ x.mT)
    assert [r.action for r in sim.report().rows] == [OWNED]


@pytest.mark.parametrize("device", available_devices)
def test_nested_simulates_inner_first(device):
    """The inner simulate handles what its recipe covers, and the outer
    gets every call the inner passes through."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
    x = torch.randn(3, 8, device=device)
    inner = Recipe(linear=None).override("0", linear=_formats(RoundMode.RNE))
    outer = Recipe(linear=_formats(RoundMode.RNE, 3))
    with simulate(outer, model), simulate(inner, model), torch.no_grad():
        got = model(x)
    want_model = copy.deepcopy(model)
    convert_(
        want_model,
        Recipe(linear=_formats(RoundMode.RNE, 3)).override("0", linear=_formats(RoundMode.RNE)),
    )
    with torch.no_grad():
        assert torch.equal(got, want_model(x))


class _Raises(nn.Module):
    def forward(self, x):
        raise RuntimeError("boom")


@pytest.mark.parametrize("device", available_devices)
def test_restored_after_an_exception_and_outside_the_model(device):
    """A forward that raises leaves no hook, no frame on the stack and no
    mode behind; and a call outside the model's modules is never
    intercepted, during the simulate or after it."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), _Raises()).to(device)
    sim = simulate(Recipe(gemm=_formats(RoundMode.RNE)), model)
    x = torch.randn(3, 8, device=device)
    with pytest.raises(RuntimeError, match="boom"), sim:
        model(x)
    assert sim._state.stack == []
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    a, b = torch.randn(4, 4, device=device), torch.randn(4, 4, device=device)
    with sim:
        inside = a @ b
    assert torch.equal(inside, a @ b)
    assert all(r.scope == "0" for r in sim.report().rows)  # only the Linear ran inside


class _Uncovered(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        y = torch.einsum("ij,jk->ik", self.fc(x), x.mT)
        q = y[None, None]
        return F.scaled_dot_product_attention(q, q, q)


@pytest.mark.parametrize("device", available_devices)
def test_the_report_and_strict(device):
    """What simulate cannot run (an ``einsum``, SDPA) runs in PyTorch's
    arithmetic and is reported, with why; ``strict`` raises for it in a
    module the recipe covers, and not in one it gives ``None``."""
    torch.manual_seed(0)
    model = nn.Sequential(_Uncovered()).to(device)
    x = torch.randn(8, 8, device=device)
    recipe = Recipe(gemm=_formats(RoundMode.RNE))
    sim = simulate(recipe, model)
    with sim, torch.no_grad():
        model(x)
    report = sim.report()
    assert {r.call for r in report.not_covered} == {
        "torch.einsum",
        "F.scaled_dot_product_attention",
    }
    assert [r.scope for r in report.simulated] == ["0.fc"]
    assert "not covered" in str(report) and "0.fc" in str(report)
    with pytest.raises(RuntimeError, match="torch.einsum in 0 is not simulated"):
        with simulate(recipe, model, strict=True), torch.no_grad():
            model(x)
    lenient = recipe.override("0", gemm=None)
    with simulate(lenient, model, strict=True), torch.no_grad():
        model(x)


def test_one_module_under_two_names_must_agree():
    """A module reachable under two names is resolved under both, which must
    give it the same formats, as convert_ requires."""
    shared = nn.Linear(4, 4)
    model = nn.ModuleDict({"a": shared, "b": shared})
    simulate(Recipe(linear=_formats(RoundMode.RNE)), model)
    with pytest.raises(ValueError, match="also at 'b'"):
        simulate(Recipe(linear=_formats(RoundMode.RNE)).override("b", linear=None), model)


# --- the kinds past the GEMMs --------------------------------------------------------


def _act_formats(rounding, p=4):
    sr = rounding is RoundMode.SR
    fmt = BinaryK(8, p, prng_bits=8) if sr else BinaryK(8, p)
    return ActivationFormats.uniform(Quant(fmt, rounding))


def _pw_formats(rounding, p=11):
    sr = rounding is RoundMode.SR
    fmt = BinaryK(16, p, prng_bits=8) if sr else BinaryK(16, p)
    return PointwiseFormats.uniform(Quant(fmt, rounding))


class _Acts(nn.Module):
    """Every activation convert converts, the in-place ones in place."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.acts = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Hardswish(inplace=True),
            nn.SiLU(),
            nn.GELU(),
            nn.GELU("tanh"),
            nn.Tanh(),
            nn.Hardsigmoid(),
            nn.Sigmoid(),
        )
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc2(self.acts(self.fc1(x)))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_an_activation_is_the_converted_modules(device, rounding):
    """convert turns torch's activation modules into mptorch's, simulate
    rounds their functional calls, and the two agree word for word, an
    in-place module (computed out of place by both) included."""
    torch.manual_seed(0)
    model = _Acts().to(device)
    inputs = [torch.randn(4, 8, device=device)]
    recipe = Recipe(linear=_formats(rounding), activation=_act_formats(rounding))
    recipe = recipe.override("acts.3", activation=_act_formats(rounding, p=3))
    converted, report = convert(model, recipe)
    assert len(report.converted) == 10
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    acts = [r for r in sim.report().rows if r.kind == "activation"]
    assert len(acts) == 8 and all(r.action == SIMULATED for r in acts)


def _swapped(model: nn.Module, name: str, cls, formats) -> nn.Module:
    """A copy of ``model`` with each module named ``name`` replaced by ``cls``
    holding ``formats``."""
    built = copy.deepcopy(model)
    for parent in list(built.modules()):
        child = getattr(parent, name, None)
        if isinstance(child, nn.Module):
            setattr(parent, name, cls(formats=formats))
    return built


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_an_hf_activation_is_a_hand_built_module(device, rounding):
    """Hugging Face's activation classes are no torch class convert knows;
    simulate rounds the ``F.silu`` and ``F.gelu`` they call as a ``QSiLU``
    and a ``QGELU`` built in their place would."""
    model, inputs = _mini("llama", device)
    formats = _act_formats(rounding)
    built = _swapped(model, "act_fn", QSiLU, formats)
    want = _run(built, inputs, device)
    got = _run(model, inputs, device, simulate(Recipe(activation=formats), model))
    for k in want:
        assert _same(got[k], want[k]), k
    model, inputs = _mini("gpt2", device, act="gelu_pytorch_tanh")
    built = _swapped(model, "act", lambda formats: QGELU("tanh", formats), formats)
    want = _run(built, inputs, device)
    got = _run(model, inputs, device, simulate(Recipe(activation=formats), model))
    for k in want:
        assert _same(got[k], want[k]), k


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_an_embedding_is_the_converted_modules(device, rounding):
    """GPT-2's two embeddings, the token one tied to the head: convert's
    ``QEmbedding`` s and simulate's ``F.embedding`` agree, each resolved
    under the embedding that owns the weight."""
    model, inputs = _mini("gpt2", device)
    recipe = Recipe(linear=_formats(rounding), embedding=_pw_formats(rounding))
    recipe = recipe.override("*.wpe", embedding=_pw_formats(rounding, p=8))
    recipe = recipe.override("*.h.*", linear=None)  # Conv1D: no layer to convert
    converted, _ = convert(model, recipe)
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    rows = {(r.scope, r.kind) for r in sim.report().simulated}
    assert {("transformer.wte", "embedding"), ("transformer.wpe", "embedding")} <= rows
    assert ("lm_head", "linear") in rows


class _Glue(nn.Module):
    def forward(self, x, y):
        return torch.sqrt((x * y + x) / 2.0 - y.exp().rsqrt()) ** 2 + (1 - x) / y


def _glue_by_hand(x, y, f: PointwiseFormats):
    s = (f.quant, f.grad_quant)
    left = qsub(qdiv(qadd(qmul(x, y, *s), x, *s), 2.0, *s), qrsqrt(qexp(y, *s), *s), *s)
    return qadd(qpow(qsqrt(left, *s), 2, *s), qdiv(qsub(1, x, *s), y, *s), *s)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_elementwise_arithmetic_is_the_q_ops_written_out(device, rounding):
    """Under ``elementwise`` a module's operators and functions run as the
    q-ops, in the order Python evaluates them, gradients and draws included;
    ``1 - x`` (``__rsub__``) and a number divisor too."""
    model = nn.Sequential()
    model.add_module("glue", _Glue())
    torch.manual_seed(0)
    x = (torch.rand(6, 5, device=device) + 1).requires_grad_()
    y = (torch.rand(5, device=device) + 1).requires_grad_()
    f = _pw_formats(rounding)
    torch.manual_seed(4)
    want = _glue_by_hand(x, y, f)
    want_g = torch.autograd.grad(want.sum(), (x, y))
    sim = simulate(Recipe(elementwise=f), model)
    torch.manual_seed(4)
    with sim:
        got = model[0](x, y)
    got_g = torch.autograd.grad(got.sum(), (x, y))
    assert _same(got, want) and all(_same(a, b) for a, b in zip(got_g, want_g, strict=True))
    calls = {r.call for r in sim.report().simulated}
    assert {"Tensor.mul", "Tensor.add", "Tensor.div", "Tensor.__rsub__", "torch.sqrt"} <= calls


@pytest.mark.parametrize("device", available_devices)
def test_elementwise_in_a_llama(device):
    """``elementwise`` on the decoder layers rounds their norms' ``pow``,
    ``rsqrt`` and ``*``, RoPE's and attention's ``*`` and ``+``, the MLP's
    product and the residuals, and nothing outside them: the rotary
    frequencies and the final norm are left plain, their cosines and the
    norms' ``mean`` are no arithmetic it rounds."""
    model, inputs = _mini("llama", device)
    recipe = Recipe(elementwise=None).override("*.layers.*", elementwise=_pw_formats(RoundMode.RNE))
    sim = simulate(recipe, model)
    _run(model, inputs, device, sim)
    rows = sim.report().rows
    simulated = {(r.scope, r.call) for r in rows if r.action == SIMULATED}
    layer = "model.layers.0"
    for scope, call in [
        (f"{layer}.input_layernorm", "Tensor.pow"),
        (f"{layer}.input_layernorm", "torch.rsqrt"),
        (f"{layer}.input_layernorm", "Tensor.mul"),
        (f"{layer}.self_attn", "Tensor.mul"),
        (f"{layer}.self_attn", "Tensor.add"),
        (f"{layer}.mlp", "Tensor.mul"),
        (layer, "Tensor.add"),
    ]:
        assert (scope, call) in simulated, (scope, call)
    plain = {(r.scope, r.call) for r in rows if r.action == PLAIN}
    assert ("model.rotary_emb", "Tensor.mul") in plain and ("model.norm", "torch.rsqrt") in plain
    assert not {r.call for r in rows} & {"Tensor.cos", "Tensor.mean"}


@pytest.mark.parametrize("device", available_devices)
def test_a_recipe_for_the_gemms_reaches_nothing_else(device):
    """A recipe that names none of the kinds past the GEMMs installs none of
    their handlers and reports only its products, as it did before them."""
    model, inputs = _mini("llama", device)
    sim = simulate(Recipe(linear=_formats(RoundMode.RNE)), model)
    _run(model, inputs, device, sim)
    assert sim._state.handlers.keys() == _HANDLERS.keys()
    assert {r.kind for r in sim.report().rows} == {"linear", "matmul"}


class _InPlace(nn.Module):
    def forward(self, x):
        y = x * 2
        y.add_(1)
        torch.arange(3, device=x.device).add_(1)  # an index counter: not arithmetic it rounds
        return y.relu_()


def test_in_place_arithmetic_is_reported_and_strict_raises():
    model = nn.Sequential()
    model.add_module("inner", _InPlace())
    recipe = Recipe(elementwise=_pw_formats(RoundMode.RNE), activation=_act_formats(RoundMode.RNE))
    sim = simulate(recipe, model)
    with sim, torch.no_grad():
        model(torch.randn(4))
    rows = {(r.call, r.action) for r in sim.report().rows}
    assert ("Tensor.mul", SIMULATED) in rows
    assert ("Tensor.add_", "not covered: it writes its operand in place") in rows
    assert ("Tensor.relu_", "not covered: it writes its operand in place") in rows
    assert len(rows) == 3
    with pytest.raises(RuntimeError, match="Tensor.add_ in inner is not simulated"):
        with simulate(recipe, model, strict=True), torch.no_grad():
            model(torch.randn(4))
    lenient = recipe.override("inner", elementwise=None, activation=None)
    with simulate(lenient, model, strict=True), torch.no_grad():
        model(torch.randn(4))


# --- softmax, the norms and the sums ----------------------------------------------------


def _norm_formats(rounding, p=8, grad=True):
    sr = rounding is RoundMode.SR
    fmt = BinaryK(16, p, prng_bits=8) if sr else BinaryK(16, p)
    q = Quant(fmt, rounding)
    return NormFormats.uniform(SplitMac(fmt, ACC, rounding=rounding), q, grad=q if grad else None)


def _softmax_formats(rounding, p=8):
    sr = rounding is RoundMode.SR
    fmt = BinaryK(16, p, prng_bits=8) if sr else BinaryK(16, p)
    return SoftmaxFormats.uniform(SplitMac(fmt, ACC, rounding=rounding), Quant(fmt, rounding))


class _Norms(nn.Module):
    """Each of torch's norm modules, and a softmax module."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.gn = nn.GroupNorm(4, 8)
        self.bn1 = nn.BatchNorm1d(8)
        self.ln = nn.LayerNorm(8)
        self.rms = nn.RMSNorm(8)
        self.fc = nn.Linear(8, 6)
        self.sm = nn.Softmax(-1)

    def forward(self, x):
        h = self.gn(self.bn(self.conv(x)))
        h = h.flatten(2).amax(-1)
        return self.sm(self.fc(self.rms(self.ln(self.bn1(h)))))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("grad", [True, False], ids=["closed-form", "torch's gradient"])
def test_norms_and_softmax_are_the_converted_modules(device, rounding, grad):
    """convert turns torch's norm and softmax modules into mptorch's, simulate
    rounds their functional calls, and the two agree word for word, the
    batch norms' running statistics included."""
    torch.manual_seed(0)
    model = _Norms().to(device).train()
    inputs = [torch.randn(4, 3, 5, 5, device=device)]
    recipe = Recipe(
        gemm=_formats(rounding),
        norm=_norm_formats(rounding, grad=grad),
        softmax=_softmax_formats(rounding),
    ).override("ln", norm=_norm_formats(rounding, p=6, grad=grad))
    converted, report = convert(model, recipe)
    assert len(report.converted) == 8
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    for name in ("bn", "bn1"):
        for buf in ("running_mean", "running_var", "num_batches_tracked"):
            a, b = getattr(model, name), getattr(converted, name)
            assert torch.equal(getattr(a, buf), getattr(b, buf)), (name, buf)
    rows = [r for r in sim.report().rows if r.kind in ("norm", "softmax")]
    assert len(rows) == 6 and all(r.action == SIMULATED for r in rows)


def _llama_rms(module, formats, x):
    return qrmsnorm(x, module.weight.shape, module.weight, module.variance_epsilon, formats)


@pytest.fixture
def registered_llama_norm():
    register(LlamaRMSNorm, kind="norm", forward=_llama_rms)
    yield
    del _REGISTRY[LlamaRMSNorm]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.usefixtures("registered_llama_norm")
def test_a_registered_norm_is_the_converted_module(device, rounding):
    """Llama's RMSNorm, written out of elementwise ops, is registered with the
    forward it simulates: convert switches it to a class that runs it, and
    simulate runs it in place of the module's own forward."""
    model, inputs = _mini("llama", device)
    recipe = Recipe(linear=_formats(rounding), norm=_norm_formats(rounding))
    converted, report = convert(model, recipe)
    assert sum(r.type == "LlamaRMSNorm" for r in report.converted) == 5
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    norms = [r for r in sim.report().rows if r.kind == "norm"]
    assert len(norms) == 5 and {r.call for r in norms} == {"LlamaRMSNorm.forward"}
    model, inputs = _mini("gpt2", device)  # nn.LayerNorm, through the registry
    recipe = Recipe(norm=_norm_formats(rounding))
    converted, _ = convert(model, recipe)
    want = _run(converted, inputs, device)
    got = _run(model, inputs, device, simulate(recipe, model))
    for k in want:
        assert _same(got[k], want[k]), k


@pytest.mark.usefixtures("registered_llama_norm")
def test_a_whole_module_passes_its_own_calls_through():
    """Inside a norm simulated whole, the elementwise kind rounds nothing, even
    under strict: its calls are reported as inside it."""
    model, inputs = _mini("llama", "cpu")
    recipe = Recipe(norm=_norm_formats(RoundMode.RNE), elementwise=_pw_formats(RoundMode.RNE))
    sim = simulate(recipe, model, strict=True)
    _run(model, inputs, "cpu", sim)
    norm = "model.layers.0.input_layernorm"
    rows = {(r.call, r.action) for r in sim.report().rows if r.scope == norm}
    assert ("LlamaRMSNorm.forward", SIMULATED) in rows
    assert ("Tensor.pow", WHOLE) in rows and ("torch.rsqrt", WHOLE) in rows
    assert not any(a == SIMULATED for c, a in rows if c != "LlamaRMSNorm.forward")


class _Sums(nn.Module):
    def forward(self, x, ids):
        s = x.sum(axis=1)
        m = torch.mean(x, (0, 2), keepdim=True)
        return s, m, ids.sum()  # the integers' sum is counting


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_sums_and_means_are_qsum_and_qmean(device, rounding):
    sr = rounding is RoundMode.SR
    fmt = BinaryK(16, 8, prng_bits=8) if sr else BinaryK(16, 8)
    f = ReductionFormats.uniform(SplitMac(fmt, ACC, rounding=rounding), Quant(fmt, rounding))
    sums = _Sums()
    model = nn.Sequential()
    model.add_module("sums", sums)
    x = torch.randn(3, 4, 5, device=device, requires_grad=True)
    ids = torch.arange(6, device=device)
    sim = simulate(Recipe(reduction=f), model)
    torch.manual_seed(2)
    with sim:
        s, m, total = sums(x, ids)
    torch.manual_seed(2)
    ws = qsum(x, 1, f.mac, grad_quant=f.grad)
    wm = qmean(x, (0, 2), f.mac, keepdim=True, quant=f.div, grad_quant=f.grad)
    assert _same(s, ws) and _same(m, wm) and total.item() == 15
    torch.manual_seed(4)
    gs = torch.autograd.grad((s.sum() + m.sum()), x)[0]
    torch.manual_seed(4)
    gw = torch.autograd.grad((ws.sum() + wm.sum()), x)[0]
    assert _same(gs, gw)
    rows = sim.report().rows
    assert {(r.call, r.action) for r in rows} == {
        ("Tensor.sum", SIMULATED),
        ("torch.mean", SIMULATED),
    }


def test_reduction_strict():
    class _Converting(nn.Module):
        def forward(self, x):
            return x.sum(dtype=torch.float64)

    model = nn.Sequential()
    model.add_module("inner", _Converting())
    f = ReductionFormats.uniform(BinaryK(16, 8), Quant(BinaryK(16, 8)))
    sim = simulate(Recipe(reduction=f), model)
    with sim:
        model(torch.randn(4))
    assert [r.action for r in sim.report().rows] == [
        "not covered: dtype= converts the operand first"
    ]
    with pytest.raises(RuntimeError, match="Tensor.sum in inner is not simulated"):
        with simulate(Recipe(reduction=f), model, strict=True):
            model(torch.randn(4))


@pytest.mark.parametrize("device", available_devices)
def test_a_llama_mean_under_reduction(device):
    """Without the registration, a Llama norm's ``mean`` is a reduction."""
    model, inputs = _mini("llama", device)
    f = ReductionFormats.uniform(BinaryK(16, 8), Quant(BinaryK(16, 8)))
    sim = simulate(Recipe(reduction=f), model)
    _run(model, inputs, device, sim)
    rows = {(r.scope, r.call, r.action) for r in sim.report().rows}
    assert ("model.layers.0.input_layernorm", "Tensor.mean", SIMULATED) in rows
    assert {r.kind for r in sim.report().simulated} == {"reduction"}


# --- Hugging Face's own classes --------------------------------------------------------


def _hf_llama(attn):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=32,
        bos_token_id=0,
        eos_token_id=0,
        attn_implementation=attn,
        use_cache=False,
    )
    return transformers.LlamaForCausalLM(config)


def _hf_gpt2(attn="eager"):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        n_layer=2,
        n_head=2,
        n_embd=32,
        vocab_size=64,
        n_positions=32,
        bos_token_id=0,
        eos_token_id=0,
        attn_implementation=attn,
        use_cache=False,
    )
    return transformers.GPT2LMHeadModel(config)


class _Logits(nn.Module):
    """An HF model's logits, as a module of token ids."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids):
        return self.model(input_ids=ids).logits


@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_hf_llama(rounding):
    """transformers' ``LlamaForCausalLM``, eager: simulate is the converted
    model on its linears, and its attention products are reported at the
    attention modules' two sites."""
    model = _Logits(_hf_llama("eager"))
    inputs = [_ids("cpu")]
    recipe = Recipe(linear=_formats(rounding))
    converted, _ = convert(model, recipe)
    want = _run(converted, inputs, "cpu")
    sim = simulate(recipe, model)
    got = _run(model, inputs, "cpu", sim)
    for k in want:
        assert _same(got[k], want[k]), k
    sites = {(r.scope, r.site) for r in sim.report().rows if r.kind == "matmul"}
    assert sites == {
        (f"model.model.layers.{i}.self_attn", s) for i in range(2) for s in ("matmul#0", "matmul#1")
    }


def test_hf_activations():
    """transformers' Llama calls ``F.silu`` in its ``SiLUActivation``, and
    GPT-2's default ``NewGELUActivation`` writes its GELU out: its ``tanh`` is
    an activation and its products, sums and power are elementwise."""
    model = _Logits(_hf_llama("eager"))
    sim = simulate(Recipe(activation=_act_formats(RoundMode.RNE)), model)
    _run(model, [_ids("cpu")], "cpu", sim)
    acts = {(r.scope, r.call) for r in sim.report().simulated}
    assert acts == {(f"model.model.layers.{i}.mlp.act_fn", "F.silu") for i in range(2)}
    model = _Logits(_hf_gpt2())
    recipe = Recipe(activation=_act_formats(RoundMode.RNE)).override(
        "*.mlp.act", elementwise=_pw_formats(RoundMode.RNE)
    )
    sim = simulate(recipe, model)
    _run(model, [_ids("cpu")], "cpu", sim)
    act = "model.transformer.h.0.mlp.act"
    calls = {(r.kind, r.call) for r in sim.report().simulated if r.scope == act}
    assert ("activation", "torch.tanh") in calls
    assert {("elementwise", c) for c in ("Tensor.mul", "Tensor.add", "torch.pow")} <= calls


@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_hf_gpt2_conv1d(rounding):
    """transformers' ``GPT2LMHeadModel``: each ``Conv1D``'s ``addmm`` is
    simulated under its owner, as a ``QLinear`` over the transposed weight."""
    model = _Logits(_hf_gpt2())
    inputs = [_ids("cpu")]
    recipe = Recipe(linear=_formats(rounding))
    built = _conv1d_as_qlinear(model, _formats(rounding))
    convert_(built, recipe)
    want = _run(built, inputs, "cpu")
    sim = simulate(recipe, model)
    got = _run(model, inputs, "cpu", sim)
    assert torch.equal(got["out"], want["out"])
    _conv1d_grads_match(model, built, got, want)
    owners = {r.scope for r in sim.report().simulated if r.call == "torch.addmm"}
    assert owners == {n for n, m in model.named_modules() if type(m).__name__ == "Conv1D"}


def test_hf_sdpa_is_reported():
    """HF's default attention is SDPA, one call that simulate does not run
    yet: it is reported, and ``strict`` raises for it."""
    model = _Logits(_hf_llama("sdpa"))
    recipe = Recipe(gemm=_formats(RoundMode.RNE))
    sim = simulate(recipe, model)
    with sim, torch.no_grad():
        model(_ids("cpu"))
    assert {r.call for r in sim.report().not_covered} == {"F.scaled_dot_product_attention"}
    with pytest.raises(RuntimeError, match="scaled_dot_product_attention"):
        with simulate(recipe, model, strict=True), torch.no_grad():
            model(_ids("cpu"))


# --- torchvision -----------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_torchvision_vit(device):
    """A small ViT under a recipe for the GEMMs: its patch projection, MLPs,
    head and ``nn.MultiheadAttention``'s projections are the converted
    model's, and the attention's core, which the recipe gives no formats, is
    reported and left to PyTorch, as the converted layer leaves it."""
    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = models.VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=10,
    ).to(device)
    inputs = [torch.randn(2, 3, 32, 32, device=device)]
    recipe = Recipe(linear=_formats(RoundMode.RNE), conv=_formats(RoundMode.RNE))
    converted, _ = convert(model, recipe)
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model)
    got = _run(model, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    assert {r.call for r in sim.report().not_covered} == {"F.multi_head_attention_forward"}


@pytest.mark.parametrize("device", available_devices)
def test_torchvision_swin(device):
    """A small Swin transformer: its attention's ``F.linear`` calls on the
    ``qkv`` and ``proj`` weights are resolved under those layers, which
    convert would convert but swin never calls, and its two attention
    products are sites of the attention module."""
    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = models.SwinTransformer(
        patch_size=[4, 4],
        embed_dim=16,
        depths=[1, 1],
        num_heads=[2, 2],
        window_size=[4, 4],
        num_classes=10,
    ).to(device)
    recipe = Recipe(linear=_formats(RoundMode.RNE), matmul=_formats(RoundMode.RNE))
    sim = simulate(recipe, model)
    with sim:
        out = model(torch.randn(2, 3, 32, 32, device=device))
    out.sum().backward()
    simulated = {(r.scope.rsplit(".", 1)[-1], r.kind, r.site) for r in sim.report().simulated}
    assert {("qkv", "linear", None), ("proj", "linear", None)} <= simulated
    assert {("attn", "matmul", "matmul#0"), ("attn", "matmul", "matmul#1")} <= simulated


@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_hf_norms(rounding):
    """transformers' ``LlamaRMSNorm`` registered with its forward, and GPT-2's
    ``nn.LayerNorm``: simulate is the converted model, and eager attention's
    ``F.softmax`` (with ``dtype=float32`` in Llama) is reported."""
    pytest.importorskip("transformers")
    from transformers.models.llama.modeling_llama import LlamaRMSNorm as HFRMSNorm

    register(HFRMSNorm, kind="norm", forward=_llama_rms)
    try:
        model = _Logits(_hf_llama("eager"))
        inputs = [_ids("cpu")]
        recipe = Recipe(linear=_formats(rounding), norm=_norm_formats(rounding))
        converted, report = convert(model, recipe)
        assert sum(r.type == "LlamaRMSNorm" for r in report.converted) == 5
        want = _run(converted, inputs, "cpu")
        got = _run(model, inputs, "cpu", simulate(recipe, model))
        for k in want:
            assert _same(got[k], want[k]), k
    finally:
        del _REGISTRY[HFRMSNorm]
    model = _Logits(_hf_gpt2())
    recipe = Recipe(norm=_norm_formats(rounding), softmax=_softmax_formats(rounding))
    converted, _ = convert(model, Recipe(norm=_norm_formats(rounding)))
    sim = simulate(recipe, model)
    _run(model, [_ids("cpu")], "cpu", sim)
    calls = {(r.kind, r.call) for r in sim.report().simulated}
    assert calls == {("norm", "F.layer_norm"), ("softmax", "F.softmax")}


# --- attention ----------------------------------------------------------------------------


def _attention_formats(rounding: RoundMode, acc: int = 11) -> AttentionFormats:
    """Both products and the softmax in an 8-bit format, summed in a 16-bit one
    with ``acc`` bits of precision."""
    sr = rounding is RoundMode.SR
    fmt = BinaryK(8, 4, prng_bits=8) if sr else BinaryK(8, 4)
    total = BinaryK(16, acc, prng_bits=8) if sr else BinaryK(16, acc)
    mac = SplitMac(fmt, total, rounding=rounding)
    return AttentionFormats.uniform(mac, Quant(fmt, rounding))


def _twins(name, device, dtype=torch.float32, **kwargs):
    """The same model twice, by seed: once calling SDPA, once eager attention."""
    models = []
    for attn in ("sdpa", "eager"):
        model, inputs = _mini(name, device, attn=attn, **kwargs)
        models.append(model.to(dtype))
    return models, inputs


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize(
    "name,dtype",
    [("llama", torch.float32), ("gpt2", torch.float32), ("llama", torch.bfloat16)],
)
def test_sdpa_is_its_eager_twin(device, rounding, name, dtype):
    """An SDPA model under the ``attention`` kind and its eager twin under the
    same recipe's ``attention_parts`` compute the same words, outputs and
    gradients, Llama's GQA (SDPA's ``enable_gqa`` against ``repeat_kv``)
    included. Llama's softmax is the same in bfloat16 too: its eager one
    computes in float32 as ``qattention`` does."""
    if device == "mps" and dtype is torch.bfloat16:
        pytest.skip("bfloat16 GEMMs are tested on the CPU and CUDA")
    (sdpa, eager), inputs = _twins(name, device, dtype)
    acc = 8 if dtype is torch.bfloat16 else 11
    gf = _formats(rounding) if dtype is torch.float32 else GemmFormats()
    recipe = Recipe(linear=gf).attention_parts("*", _attention_formats(rounding, acc))
    sim_sdpa = simulate(recipe, sdpa)
    want = _run(eager, inputs, device, simulate(recipe, eager))
    got = _run(sdpa, inputs, device, sim_sdpa)
    for k in want:
        assert _same(got[k], want[k]), k
    calls = {(r.kind, r.call) for r in sim_sdpa.report().simulated}
    assert ("attention", "F.scaled_dot_product_attention") in calls


@pytest.mark.parametrize("device", available_devices)
def test_sdpa_dropout_is_its_eager_twins(device):
    """In training, dropout draws the same mask in both spellings."""
    (sdpa, eager), inputs = _twins("gpt2", device, dropout=0.25)
    sdpa.train()
    eager.train()
    recipe = Recipe().attention_parts("*", _attention_formats(RoundMode.RNE))
    want = _run(eager, inputs, device, simulate(recipe, eager))
    got = _run(sdpa, inputs, device, simulate(recipe, sdpa))
    for k in want:
        assert _same(got[k], want[k]), k


@pytest.mark.parametrize("device", available_devices)
def test_the_attention_kind_and_strict(device):
    """A recipe for the GEMMs reports SDPA and the MHA core, and ``strict``
    raises for both; naming the kind with ``None`` leaves them to PyTorch,
    reported as plain; naming it with formats simulates them."""
    torch.manual_seed(0)
    model = nn.Sequential(_Encoder()).to(device)
    x = torch.randn(2, 5, 16, device=device)
    gemm = Recipe(gemm=_formats(RoundMode.RNE))
    sim = simulate(gemm, model)
    with sim, torch.no_grad():
        model(x)
    report = sim.report()
    assert {r.call for r in report.not_covered} == {
        "F.multi_head_attention_forward",
        "F.scaled_dot_product_attention",
    }
    assert "F.multi_head_attention_forward" in {r.call for r in report.simulated}
    with pytest.raises(RuntimeError, match="multi_head_attention_forward in 0.mha"):
        with simulate(gemm, model, strict=True), torch.no_grad():
            model(x)
    plain = gemm.override("*", attention=None)
    sim = simulate(plain, model, strict=True)
    with sim, torch.no_grad():
        model(x)
    assert not sim.report().not_covered
    assert {r.action for r in sim.report().rows if r.kind == "attention"} == {SIMULATED, PLAIN}
    full = gemm.override("*", attention=_attention_formats(RoundMode.RNE))
    sim = simulate(full, model, strict=True)
    with sim, torch.no_grad():
        model(x)
    assert {r.action for r in sim.report().rows if r.kind == "attention"} == {SIMULATED}


class _Encoder(nn.Module):
    """An ``nn.MultiheadAttention`` and an SDPA call side by side."""

    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(16, 4, batch_first=True)
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        h = self.mha(x, x, x, need_weights=False)[0]
        q = self.fc(h).view(2, 5, 4, 4).transpose(1, 2)
        return F.scaled_dot_product_attention(q, q, q, is_causal=True).transpose(1, 2).flatten(2)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_torchvision_vit_with_an_attention_kind(device, rounding):
    models = pytest.importorskip("torchvision.models")
    torch.manual_seed(0)
    model = models.VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=10,
    ).to(device)
    inputs = [torch.randn(2, 3, 32, 32, device=device)]
    recipe = Recipe(gemm=_formats(rounding), attention=_attention_formats(rounding))
    converted, _ = convert(model, recipe, strict=True)
    want = _run(converted, inputs, device)
    sim = simulate(recipe, model, strict=True)
    got = _run(model, inputs, device, sim)
    for k in want:
        assert _same(got[k], want[k]), k
    assert not sim.report().not_covered


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_a_transformer_encoder_layer(device, rounding):
    """``nn.TransformerEncoderLayer`` with a causal mask and a padding mask:
    simulate is the converted model, its fast path kept off by the mode."""
    torch.manual_seed(0)
    model = nn.TransformerEncoderLayer(16, 4, 32, dropout=0.0, batch_first=True).to(device)
    x = torch.randn(2, 5, 16, device=device)
    mask = torch.ones(5, 5, dtype=torch.bool, device=device).triu(1)  # True: masked
    pad = torch.zeros(2, 5, dtype=torch.bool, device=device)
    pad[1, -2:] = True
    recipe = Recipe(gemm=_formats(rounding), attention=_attention_formats(rounding))
    converted, report = convert(model, recipe, strict=True)
    assert [r.name for r in report.rows] == ["self_attn", "linear1", "linear2"]

    class _Masked(nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, x):
            return self.layer(x, src_mask=mask, src_key_padding_mask=pad)

    want = _run(_Masked(converted), [x], device)
    got = _run(_Masked(model), [x], device, simulate(recipe, _Masked(model)))
    for k in want:
        assert _same(got[k], want[k]), k


@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("name", ["llama", "gpt2"])
def test_hf_sdpa_is_its_eager_twin(rounding, name):
    """transformers' own models: SDPA under the ``attention`` kind computes the
    eager implementation's words under ``attention_parts``."""
    build = {"llama": _hf_llama, "gpt2": _hf_gpt2}[name]
    sdpa, eager = _Logits(build("sdpa")), _Logits(build("eager"))
    recipe = Recipe(linear=_formats(rounding)).attention_parts("*", _attention_formats(rounding))
    inputs = [_ids("cpu")]
    want = _run(eager, inputs, "cpu", simulate(recipe, eager))
    sim = simulate(recipe, sdpa)
    got = _run(sdpa, inputs, "cpu", sim)
    for k in want:
        assert _same(got[k], want[k]), k
    assert "F.scaled_dot_product_attention" in {r.call for r in sim.report().simulated}
