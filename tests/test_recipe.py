"""
Recipes: ``GemmFormats``, ``Rule``, ``Recipe`` and the presets (``mptorch/quant/recipe.py``,
``presets.py``).

``GemmFormats`` builds the containers a layer reads, one mac per pass. Tier 3
holds each container to the one the factories build by hand
(``binaryK_gemm_formats``, ``conv_formats``, ``matmul_formats``,
``block_gemm_formats``, ``block_matmul_formats``), forward and both gradients,
``torch.equal``, with a different mac in each pass and a pass left to
PyTorch. Then: what a ``GemmFormats`` accepts and normalizes, and what it
refuses; how a ``Recipe``'s rules match and which one wins; and what the
presets contain.
"""

import dataclasses

import pytest
import torch
from torch import nn

from mptorch import (
    E4M3,
    E5M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    AccumulateAlgorithm,
    BinaryK,
    Domain,
    RoundMode,
    SaturationMode,
    SuperFP,
)
from mptorch.quant import (
    BlockMac,
    BlockQuant,
    FusedMac,
    GemmFormats,
    NormFormats,
    Palette,
    QAffineFormats,
    QConv2d,
    QLinear,
    QMatmul,
    QMatmulFormats,
    Quant,
    Quantizer,
    Recipe,
    ReductionFormats,
    Rule,
    ScaledQuant,
    SoftmaxFormats,
    SplitMac,
    binaryK_gemm_formats,
    binaryK_gemm_formats_fma,
    block_gemm_formats,
    block_matmul_formats,
    conv_formats,
    matmul_formats,
    presets,
)
from tests.markers import available_devices

P8 = BinaryK(8, 4)
P8_SR = BinaryK(8, 4, prng_bits=8)
P16 = BinaryK(16, 11)
P12 = BinaryK(12, 7)
SR = RoundMode.SR
QUANT_FIELDS = ("input_quant", "weight_quant", "bias_quant", "igrad_quant", "wgrad_quant")


def _with_quants(formats, q, gq=None, fields=QUANT_FIELDS):
    """``formats`` with ``q`` on the operands and ``gq`` (default ``q``) on the gradients."""
    gq = q if gq is None else gq
    for name in fields:
        setattr(formats, name, gq if "grad" in name else q)
    return formats


def _seeded(device):
    torch.manual_seed(321)
    if device == "cuda":
        torch.cuda.manual_seed_all(321)


def _linear_step(formats, device, bias=True):
    torch.manual_seed(7)
    layer = QLinear(48, 16, bias=bias, formats=formats).to(device)
    x = torch.randn(3, 5, 48, device=device, requires_grad=True)
    _seeded(device)
    out = layer(x)
    out.backward(torch.randn_like(out))
    grads = [x.grad, layer.weight.grad] + ([layer.bias.grad] if bias else [])
    return [out.detach(), *grads, torch.rand(3), torch.rand(3, device=device)]


def _conv_step(formats, device):
    torch.manual_seed(7)
    layer = QConv2d(3, 6, 3, stride=2, padding=1, formats=formats).to(device)
    x = torch.randn(2, 3, 9, 9, device=device, requires_grad=True)
    _seeded(device)
    out = layer(x)
    out.backward(torch.randn_like(out))
    assert layer.bias is not None
    return [out.detach(), x.grad, layer.weight.grad, layer.bias.grad]


def _matmul_step(formats, device):
    torch.manual_seed(7)
    a = torch.randn(4, 6, 64, device=device, requires_grad=True)
    b = torch.randn(64, 32, device=device, requires_grad=True)
    _seeded(device)
    out = QMatmul(formats)(a, b)
    out.backward(torch.randn_like(out))
    return [out.detach(), a.grad, b.grad]


def _equal(a, b):
    assert len(a) == len(b)
    for s, t in zip(a, b, strict=True):
        assert torch.equal(s, t)


# --- the containers, against the factories ----------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, SR])
def test_tier3_affine_is_the_factorys(device, rounding):
    fmt, acc = (P8_SR, BinaryK(16, 11, prng_bits=8)) if rounding is SR else (P8, P16)
    q = Quant(fmt, rounding)
    gf = GemmFormats.uniform(SplitMac(fmt, acc, rounding=rounding), q)
    gf = dataclasses.replace(gf, bias_quant=q)
    hand = binaryK_gemm_formats(
        mul_K=8,
        mul_P=4,
        mul_prng_bits=fmt.prng_bits,
        acc_K=16,
        acc_P=11,
        acc_prng_bits=acc.prng_bits,
        rounding_mode=rounding,
    )
    _equal(_linear_step(gf.affine(), device), _linear_step(_with_quants(hand, q), device))


@pytest.mark.parametrize("device", available_devices)
def test_tier3_one_mac_per_pass(device):
    """Each pass in its own mac, the weight gradient in PyTorch's own product."""
    gf = GemmFormats(fwd_mac=SplitMac(P8, P16), igrad_mac=FusedMac(P12), wgrad_mac=None)
    split = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11)
    fused = binaryK_gemm_formats_fma(fma_K=12, fma_P=7)
    hand = QAffineFormats(fwd_math=split.fwd_math, bwd_igrad_math=fused.bwd_igrad_math)
    _equal(_linear_step(gf.affine(), device), _linear_step(hand, device))


@pytest.mark.parametrize("device", available_devices)
def test_tier3_block_passes(device):
    acc = BinaryK(16, 11)
    gf = GemmFormats(
        fwd_mac=BlockMac(MXFP8_E4M3, MXFP8_E4M3, acc),
        igrad_mac=BlockMac(MXFP8_E5M2, MXFP8_E4M3, acc),
        wgrad_mac=BlockMac(MXFP8_E5M2, MXFP8_E4M3, acc),
    )
    hand = block_gemm_formats(MXFP8_E4M3, MXFP8_E4M3, MXFP8_E5M2, acc=acc)
    built = gf.affine()
    assert built._block_weight == hand._block_weight
    _equal(_linear_step(built, device), _linear_step(hand, device))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, SR])
def test_tier3_conv_is_the_factorys(device, rounding):
    fmt = P8_SR if rounding is SR else P8
    mac = SplitMac(fmt, BinaryK(16, 11, prng_bits=fmt.prng_bits), rounding=rounding)
    q = Quant(fmt, rounding)
    gf = GemmFormats.uniform(mac, q)
    hand = _with_quants(conv_formats(mac), q, fields=QUANT_FIELDS[:2] + QUANT_FIELDS[3:])
    _equal(_conv_step(gf.conv(), device), _conv_step(hand, device))
    # a pass left to PyTorch
    partial = GemmFormats(fwd_mac=mac, wgrad_mac=mac)
    hand = conv_formats(mac)
    hand.bwd_igrad_math = None
    _equal(_conv_step(partial.conv(), device), _conv_step(hand, device))


@pytest.mark.parametrize("device", available_devices)
def test_tier3_matmul_is_the_factorys(device):
    mac = SplitMac(P8, P16)
    gf = GemmFormats.uniform(mac, Quant(P8), grad_quant=Quant(E5M2))
    hand = matmul_formats(mac)
    hand.a_quant = hand.b_quant = Quant(P8)
    hand.agrad_quant = hand.bgrad_quant = Quant(E5M2)
    _equal(_matmul_step(gf.matmul(), device), _matmul_step(hand, device))
    block = BlockMac(dataclasses.replace(MXFP8_E4M3, block_rows=32), acc=P16)
    gf = GemmFormats.uniform(block)
    _equal(_matmul_step(gf.matmul(), device), _matmul_step(block_matmul_formats(block), device))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["affine", "conv", "matmul"])
def test_the_output_rounding(device, kind):
    """``output_quant`` rounds the forward result as ``gemm._with_output`` does."""
    gf = GemmFormats(fwd_mac=SplitMac(P8, P16), output_quant=Quant(P8))
    plain = GemmFormats(fwd_mac=SplitMac(P8, P16))
    step = {"affine": _linear_step, "conv": _conv_step, "matmul": _matmul_step}[kind]
    rounded, unrounded = step(getattr(gf, kind)(), device), step(getattr(plain, kind)(), device)
    assert torch.equal(rounded[0], Quant(P8)(unrounded[0]))
    _equal(rounded[1:], unrounded[1:])  # straight through


def test_containers_are_fresh_and_carry_their_source():
    gf = GemmFormats.uniform(SplitMac(P8, P16), Quant(P8))
    a, b = gf.affine(), gf.affine()
    assert a is not b and a.fwd_math is not b.fwd_math
    assert a._source is gf and gf.conv()._source is gf and gf.matmul()._source is gf
    assert QAffineFormats()._source is None and QMatmulFormats()._source is None
    assert a.recompute_input is None and gf.matmul().recompute_a is None
    flags = dataclasses.replace(gf, recompute_input=True, recompute_weight=False)
    m = flags.matmul()
    assert (m.recompute_a, m.recompute_b) == (True, False)
    assert flags.affine().recompute_input is True


# --- what a GemmFormats takes ---------------------------------------------------------


def test_formats_are_normalized():
    gf = GemmFormats(input_quant=P8, weight_quant=MXFP8_E4M3, fwd_mac=P8, igrad_mac=MXFP8_E4M3)
    assert gf.input_quant == Quant(P8)
    assert gf.weight_quant == BlockQuant(MXFP8_E4M3)
    assert gf.fwd_mac == SplitMac(P8, P8)
    assert gf.igrad_mac == BlockMac(MXFP8_E4M3)
    assert GemmFormats(fwd_mac=P8) == GemmFormats(fwd_mac=SplitMac(P8, P8))


def test_uniform():
    q, g = Quant(P8), ScaledQuant(E5M2)
    gf = GemmFormats.uniform(SplitMac(P8, P16), q, grad_mac=FusedMac(P16), grad_quant=g)
    assert (gf.input_quant, gf.weight_quant, gf.igrad_quant, gf.wgrad_quant) == (q, q, g, g)
    assert gf.fwd_mac == SplitMac(P8, P16)
    assert gf.igrad_mac == gf.wgrad_mac == FusedMac(P16)
    assert gf.bias_quant is gf.output_quant is gf.bgrad_quant is None
    same = GemmFormats.uniform(SplitMac(P8, P16), q)
    assert same.igrad_mac == same.fwd_mac and same.igrad_quant == q


@pytest.mark.parametrize(
    "kwargs,error,match",
    [
        (dict(input_quant=Quantizer(P8)), TypeError, "module"),
        (dict(input_quant=nn.Identity()), TypeError, "module"),
        (dict(input_quant=3), TypeError, "takes a format"),
        (dict(fwd_mac=Quant(P8)), TypeError, "takes a SplitMac"),
        (dict(fwd_mac=SplitMac(Palette([P8, BinaryK(8, 3)]))), ValueError, "palette"),
        (dict(recompute_input=1), TypeError, "True, False or None"),
        (
            dict(fwd_mac=SplitMac(P8, block_size=4, accumulate_algorithm=AccumulateAlgorithm.TREE)),
            None,
            None,
        ),
    ],
)
def test_what_gemm_formats_refuses(kwargs, error, match):
    if error is None:
        GemmFormats(**kwargs)
        return
    with pytest.raises(error, match=match):
        GemmFormats(**kwargs)


def test_a_block_pass_has_no_convolution():
    gf = GemmFormats(fwd_mac=BlockMac(MXFP8_E4M3))
    with pytest.raises(TypeError, match="convolution"):
        gf.conv()


def test_gemm_formats_is_frozen_and_keyword_only():
    gf = GemmFormats()
    with pytest.raises(dataclasses.FrozenInstanceError):
        gf.fwd_mac = SplitMac(P8)  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError):
        GemmFormats(Quant(P8))  # ty: ignore[too-many-positional-arguments]


def test_the_label():
    assert GemmFormats().label() == "mac torch"
    gf = GemmFormats.uniform(SplitMac(P8, P16), Quant(P8))
    assert gf.label() == (
        "mac SplitMac(BinaryK(K=8, P=4, bias=8), BinaryK(K=16, P=11, bias=16)), "
        "quant Quant(BinaryK(K=8, P=4, bias=8), RNE)"
    )
    mixed = GemmFormats(fwd_mac=BlockMac(MXFP8_E4M3), input_quant=Quant(P8))
    assert mixed.label() == "fwd BlockMac(MXFP8_E4M3, MXFP8_E4M3), igrad torch, wgrad torch, " + (
        "input Quant(BinaryK(K=8, P=4, bias=8), RNE)"
    )


# --- rules and recipes --------------------------------------------------------------------


A = GemmFormats(fwd_mac=SplitMac(P8))
B = GemmFormats(fwd_mac=SplitMac(P16))
C = GemmFormats(fwd_mac=FusedMac(P12))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Conv1d(4, 4, 1))


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])
        self.head = nn.Linear(4, 2)


def _resolved(recipe, model, kind):
    return {n: recipe.resolve(n, m, kind) for n, m in model.named_modules()}


def test_later_rules_win_and_omitted_kinds_inherit():
    model = _Model()
    recipe = (
        Recipe(linear=A, conv=A)
        .override("layers.*", linear=B)
        .override("layers.1.*", conv=C)
        .override("head", linear=None)
    )
    linear, conv = _resolved(recipe, model, "linear"), _resolved(recipe, model, "conv")
    assert linear["layers.0.attn"] is B and linear["layers.1.mlp.0"] is B
    assert linear["head"] is None and linear[""] is A
    assert conv["layers.0.mlp.2"] is A and conv["layers.1.mlp.2"] is C
    assert recipe.resolve("head", model.head, "matmul") is None  # never named


def test_globs_types_tuples_and_predicates():
    model = _Model()
    conv, relu = model.get_submodule("layers.0.mlp.2"), model.get_submodule("layers.0.mlp.1")
    by_type = Recipe(linear=A).override(nn.Conv1d, linear=B)
    assert by_type.resolve("layers.0.mlp.2", conv, "linear") is B
    by_tuple = Recipe().override((nn.Conv1d, nn.ReLU), conv=C)
    assert by_tuple.resolve("layers.0.mlp.1", relu, "conv") is C
    pred = Recipe().override(lambda name, m: name.endswith("attn"), linear=C)
    assert pred.resolve("layers.1.attn", model.get_submodule("layers.1.attn"), "linear") is C
    assert pred.resolve("head", model.get_submodule("head"), "linear") is None
    # a glob matches the whole name, `*` crossing dots, and "*" matches the root
    glob = Recipe().override("layers.*.mlp*", linear=B)
    assert glob.resolve("layers.0.mlp.0", model, "linear") is B
    assert glob.resolve("layers.0.attn", model, "linear") is None
    assert Recipe().override("*", linear=A).resolve("", model, "linear") is A
    assert Recipe().override("layers", linear=A).resolve("layers.0", model, "linear") is None


def test_sites():
    recipe = Recipe(matmul=A).override("attn", site="matmul#1", matmul=B)
    m = nn.Linear(2, 2)
    assert recipe.resolve("attn", m, "matmul", site="matmul#1") is B
    assert recipe.resolve("attn", m, "matmul", site="matmul#0") is A
    assert recipe.resolve("attn", m, "matmul") is A


def test_gemm_sets_the_three_kinds_and_an_explicit_kind_wins():
    rule = Rule("*", {"gemm": A, "conv": B})
    assert dict(rule.kinds) == {"linear": A, "conv": B, "matmul": A}
    recipe = Recipe(gemm=A).override("x", gemm=None)
    assert recipe.resolve("x", nn.Linear(1, 1), "conv") is None


@pytest.mark.parametrize(
    "build,error,match",
    [
        (lambda: Recipe(dropout=A), TypeError, "no kind 'dropout'"),
        (lambda: Recipe(linear=Quant(P8)), TypeError, "GemmFormats or None"),
        (lambda: Recipe(conv=GemmFormats(fwd_mac=BlockMac(MXFP8_E4M3))), ValueError, "linear="),
        (lambda: Recipe(gemm=GemmFormats(fwd_mac=BlockMac(MXFP8_E4M3))), ValueError, "linear="),
        (lambda: Recipe([A]), TypeError, "Rule"),  # ty: ignore[invalid-argument-type]
        (lambda: Rule(3, {}), TypeError, "where"),  # ty: ignore[invalid-argument-type]
        (lambda: Recipe().resolve("", nn.Linear(1, 1), "dropout"), ValueError, "no kind"),
    ],
)
def test_what_a_recipe_refuses(build, error, match):
    with pytest.raises(error, match=match):
        build()


def test_a_recipe_is_an_immutable_value():
    r = Recipe(linear=A)
    assert r.override("x", linear=B) is not r and len(r.rules) == 1
    with pytest.raises(AttributeError):
        r.rules = ()
    assert Recipe(linear=A) == Recipe(linear=A) and hash(Recipe(linear=A)) == hash(Recipe(linear=A))
    assert Recipe(linear=A) != Recipe(linear=B)
    rules = (Rule("*", {"linear": A}), Rule("head", {"linear": None}))
    assert Recipe(rules) == Recipe(linear=A).override("head", linear=None)


def test_plan():
    model = _Model()
    report = Recipe(gemm=A).override("head", linear=None).plan(model)
    assert report.dry_run
    assert [r.name for r in report.converted] == [
        "layers.0.attn",
        "layers.0.mlp.0",
        "layers.0.mlp.2",
        "layers.1.attn",
        "layers.1.mlp.0",
        "layers.1.mlp.2",
    ]
    assert [(r.name, r.action) for r in report.kept] == [("head", "kept: the recipe gives it None")]
    assert type(model.head) is nn.Linear  # nothing converted
    assert "6 to convert, 1 kept" in str(report)


# --- presets ------------------------------------------------------------------------------


def _formats_of(recipe):
    """{kind: value} of a recipe made of one rule on every module."""
    (rule,) = recipe.rules
    assert rule.where == "*"
    return dict(rule.kinds)


def test_uniform_preset():
    kinds = _formats_of(presets.uniform(P8, acc=P16, rounding=RoundMode.RZ))
    assert set(kinds) == {"linear", "conv"}
    gf = kinds["linear"]
    assert gf == GemmFormats.uniform(
        SplitMac(P8, P16, rounding=RoundMode.RZ), Quant(P8, RoundMode.RZ)
    )
    assert set(_formats_of(presets.uniform(P8, kinds=("matmul",)))) == {"matmul"}
    with pytest.raises(ValueError, match="unknown kinds"):
        presets.uniform(P8, kinds=("dropout",))
    # softmax, the norms and the sums: every step the Quant, every sum the GEMMs' mac
    mac, q = SplitMac(P8, P16, rounding=RoundMode.RZ), Quant(P8, RoundMode.RZ)
    kinds = _formats_of(
        presets.uniform(P8, acc=P16, rounding=RoundMode.RZ, kinds=("softmax", "norm", "reduction"))
    )
    assert kinds == {
        "softmax": SoftmaxFormats.uniform(mac, q),
        "norm": NormFormats.uniform(mac, q),
        "reduction": ReductionFormats.uniform(mac, q),
    }
    with pytest.raises(TypeError, match="the norm kind takes a NormFormats or None"):
        Recipe(norm=GemmFormats())


def test_fp8_preset():
    kinds = _formats_of(presets.fp8(acc=P16))
    assert set(kinds) == {"linear", "conv"}
    gf = kinds["linear"]
    fwd, bwd = gf.input_quant, gf.igrad_quant
    assert isinstance(fwd, ScaledQuant) and isinstance(bwd, ScaledQuant)
    assert (fwd.target, bwd.target) == (448.0, 57344.0)
    for q, layout in ((fwd, E4M3), (bwd, E5M2)):
        assert q.fmt.saturation is SaturationMode.SAT_FINITE and q.fmt.domain is Domain.FINITE
        assert (q.fmt.K, q.fmt.P, q.fmt.bias) == (layout.K, layout.P, layout.bias)
    assert gf.weight_quant == fwd and gf.wgrad_quant == bwd
    assert gf.fwd_mac == gf.igrad_mac == gf.wgrad_mac == FusedMac(P16)
    assert gf.bias_quant is None and gf.output_quant is None
    # another format is scaled to its own largest value and keeps its saturation
    sfp = SuperFP(3, 4, 2, 7)
    other = _formats_of(presets.fp8(fwd=sfp))["linear"].input_quant
    assert other == ScaledQuant(sfp) and other.target == sfp.max_finite


def test_mx_preset():
    kinds = _formats_of(presets.mx(acc=P16))
    assert set(kinds) == {"linear"}
    gf = kinds["linear"]
    assert gf.fwd_mac == BlockMac(MXFP8_E4M3, MXFP8_E4M3, P16)
    assert gf.igrad_mac == gf.wgrad_mac == BlockMac(MXFP8_E5M2, MXFP8_E4M3, P16)


@pytest.mark.parametrize("recipe", [presets.uniform(P8), presets.fp8(), presets.mx()])
def test_no_preset_sets_matmul(recipe):
    assert "matmul" not in _formats_of(recipe)
