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
- Hugging Face's own GPT-2 and Llama classes and torchvision's models.
"""

import copy
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    GemmFormats,
    QLinear,
    QMatmul,
    Quant,
    Recipe,
    SplitMac,
    convert,
    convert_,
    simulate,
)
from mptorch.quant.simulate import OWNED, PLAIN, SIMULATED
from tests.markers import available_devices
from tests.models import Conv1D, GPT2Attention, LlamaAttention, MiniGPT2, MiniLlama
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
    return torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(3)).to(device)


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
    """A small ViT: its patch projection, MLPs and head are the converted
    model's, and ``nn.MultiheadAttention``, one call to
    ``multi_head_attention_forward``, is reported and left to PyTorch, as
    convert leaves it."""
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
