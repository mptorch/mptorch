"""``simulate(..., events=True)``: a model's numeric events by site, pass and role.

A counting ``simulate`` gives each site a container of its own, whose GEMMs
run their ops' counted twins and whose quantizer slots recount their
``Quant``'s casts (`mptorch.quant.events`). What is held here:

- the arithmetic is the uncounted ``simulate``'s, word for word, forward and
  gradients, under SR too;
- a site's counts are those of the counted ops called by hand on the same
  operands, and of the recount of the same quantizers;
- an operand a layer quantizes again in backward (recompute) is counted once;
- the parts not counted are listed with their reasons, and the refusals.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK, RoundMode, SuperFP
from mptorch.quant import (
    ActivationFormats,
    AttentionFormats,
    BlockMac,
    GemmFormats,
    NormFormats,
    PointwiseFormats,
    Quant,
    Recipe,
    ReductionFormats,
    ScaledQuant,
    SoftmaxFormats,
    SplitMac,
    simulate,
)
from mptorch.quant.events import _recount
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops._counted import _counted_twin

from .markers import cuda_devices, requires_counting

E4M3, E5M2, B16 = BinaryK(8, 4), BinaryK(8, 3), BinaryK(16, 11)
devices = ["cpu", *cuda_devices]


def _recipe(mode=RoundMode.RNE, **extra):
    mac = SplitMac(E4M3, B16, rounding=mode)
    return Recipe(
        linear=GemmFormats(
            fwd_mac=mac,
            igrad_mac=mac,
            wgrad_mac=mac,
            input_quant=Quant(E4M3, mode),
            weight_quant=Quant(E4M3, mode),
            igrad_quant=Quant(E5M2, mode),
            wgrad_quant=Quant(E5M2, mode),
            **extra,
        )
    )


def _model(device):
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3)).to(device)


def _run(recipe, model, x, events):
    model.zero_grad()
    x = x.clone().requires_grad_()
    torch.manual_seed(7)
    with simulate(recipe, model, events=events) as sim:
        y = model(x)
        y.square().sum().backward()
    grads = [p.grad.clone() for p in model.parameters()]
    return sim, y, x.grad, grads


@requires_counting
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("device", devices)
def test_counting_changes_no_word(device, mode):
    """The counting simulate computes what the plain one does, gradients and
    stochastic rounding included."""
    recipe, model = _recipe(mode), _model(device)
    x = torch.randn(4, 8, device=device) * 20
    _, y0, gx0, g0 = _run(recipe, model, x, events=False)
    sim, y1, gx1, g1 = _run(recipe, model, x, events=True)
    assert torch.equal(y0, y1) and torch.equal(gx0, gx1)
    assert all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))
    assert sim.events().rows


@requires_counting
@pytest.mark.parametrize("device", devices)
def test_a_sites_counts_are_its_ops(device):
    """The first layer's forward: the quantizers' recounts of its input and
    weight, and the counted GEMM's counts on the quantized operands."""
    recipe, model = _recipe(), _model(device)
    x = torch.randn(4, 8, device=device) * 20
    sim, *_ = _run(recipe, model, x, events=True)
    rows = {(r.pass_name, r.role): r.counts for r in sim.events().rows if r.module == "0"}
    q = Quant(E4M3)
    qx, qw = q(x), q(model[0].weight.detach())
    assert list(rows[("fwd", "input_quant")]) == _recount(x, qx, E4M3).tolist()
    assert list(rows[("fwd", "weight_quant")]) == _recount(model[0].weight, qw, E4M3).tolist()
    op, args = _counted_twin(spec_for_mac(SplitMac(E4M3, B16)))  # ty: ignore[not-iterable]
    _, n = op(qx, qw, False, True, *args)
    assert list(rows[("fwd", "mul")]) == n[0].tolist()
    assert list(rows[("fwd", "acc")]) == n[1].tolist()
    # the input gradient [4, 8] and the weight gradient [16, 8], 4 x 16 x 8 steps each
    assert rows[("igrad", "mul")].casts == rows[("wgrad", "mul")].casts == 4 * 16 * 8


@requires_counting
@pytest.mark.parametrize("device", devices)
def test_recompute_counts_once(device):
    """A layer that quantizes its input again in backward replays the
    forward's casts, which are not counted again."""
    model = _model(device)
    x = torch.randn(4, 8, device=device)
    saved, *_ = _run(_recipe(), model, x, events=True)
    again, *_ = _run(_recipe(recompute_input=True, recompute_weight=True), model, x, events=True)
    assert str(saved.events()) == str(again.events())


@requires_counting
def test_counts_accumulate_over_calls():
    recipe, model = _recipe(), _model("cpu")
    x = torch.randn(4, 8)
    with simulate(recipe, model, events=True) as sim:
        model(x)
        once = {(r.module, r.pass_name, r.role): r.counts.casts for r in sim.events().rows}
        model(x)
        twice = {(r.module, r.pass_name, r.role): r.counts.casts for r in sim.events().rows}
    assert twice == {k: 2 * v for k, v in once.items()}


@requires_counting
def test_what_is_not_counted():
    """A palette mac, a block GEMM, a cross-family mac and a quantizer that is
    not a Quant are listed, with their reasons; the rest of their sites is
    counted."""
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 32), nn.Linear(32, 4))
    from mptorch.number import MXFP8_E4M3
    from mptorch.quant import Rule

    plain = SplitMac(E4M3, B16)
    recipe = Recipe(
        [
            Rule(
                "0", {"linear": GemmFormats(fwd_mac=SplitMac([E4M3, E5M2], B16), igrad_mac=plain)}
            ),
            Rule("1", {"linear": GemmFormats(fwd_mac=BlockMac(MXFP8_E4M3))}),
            Rule("2", {"linear": GemmFormats(fwd_mac=SplitMac(E4M3, SuperFP(5, 5, 16, 15)))}),
            Rule("3", {"linear": GemmFormats(fwd_mac=plain, input_quant=ScaledQuant(E4M3))}),
        ]
    )
    with simulate(recipe, model, events=True) as sim:
        model(torch.randn(2, 32, requires_grad=True)).sum().backward()
    report = sim.events()
    reasons = {(m, p, r): why for m, _, _, p, r, why in report.not_counted}
    assert reasons[("0", "fwd", "mac")] == "a palette mac"
    assert reasons[("1", "fwd", "mac")] == "a block GEMM"
    assert reasons[("2", "fwd", "mac")] == "a cross-family mac"
    assert reasons[("3", "fwd", "input_quant")] == "a ScaledQuant quantizer"
    counted = {(r.module, r.pass_name, r.role) for r in report.rows}
    assert ("0", "igrad", "mul") in counted and ("3", "fwd", "mul") in counted
    assert "not counted: 1 linear fwd mac: a block GEMM" in str(report)


def test_refusals():
    model = _model("cpu")
    with simulate(_recipe(), model) as sim:
        model(torch.randn(2, 8))
    with pytest.raises(RuntimeError, match="events=True"):
        sim.events()
    with pytest.raises(RuntimeError, match="eager calls only"):
        simulate(_recipe(), model, events=True).wrap()


class _Block(nn.Module):
    """A convolution, a linear layer and a product of two activations."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3, padding=1)
        self.proj = nn.Linear(36, 6)

    def forward(self, x):
        h = self.proj(self.conv(x).flatten(2))  # [B, 4, 6]
        return h @ h.transpose(1, 2)


@requires_counting
@pytest.mark.parametrize("device", devices)
def test_every_gemm_kind(device):
    """A conv layer, a linear one and a product of two activations, each
    counted at its own site, every pass of each."""
    mac = SplitMac(E4M3, B16)
    gf = GemmFormats(fwd_mac=mac, igrad_mac=mac, wgrad_mac=mac, input_quant=Quant(E4M3))
    recipe = Recipe(gemm=gf)
    torch.manual_seed(0)
    model = _Block().to(device)
    x = torch.randn(3, 2, 6, 6, device=device, requires_grad=True)
    with simulate(recipe, model, events=True) as sim:
        model(x).sum().backward()
    sites = {(r.module, r.kind, r.site, r.pass_name) for r in sim.events().rows if r.role == "mul"}
    for pass_ in ("fwd", "igrad", "wgrad"):
        assert ("conv", "conv", None, pass_) in sites
        assert ("proj", "linear", None, pass_) in sites
        assert ("", "matmul", None, pass_) in sites  # no rule names its site
    rows = {(r.module, r.pass_name, r.role): r.counts for r in sim.events().rows}
    # the forward conv: [3 * 1 groups] GEMMs of [4, 2 * 9] by [2 * 9, 36]
    assert rows[("conv", "fwd", "mul")].casts == 3 * 4 * 18 * 36


@requires_counting
def test_profile_counts_a_converted_model_as_simulate_does():
    """profile(events=True) counts a converted layer through a counting
    container rebuilt from its formats, swapped in for the pass and put back:
    the same events as the simulated model's."""
    from mptorch.quant import convert, profile

    model = _model("cpu")
    x = torch.randn(4, 8) * 30
    simulated = profile(model, x, recipe=_recipe(), events=True).events
    converted = convert(model, _recipe())[0]
    before = converted.get_submodule("0").formats
    counted = profile(converted, x, events=True).events
    assert simulated is not None and counted is not None
    assert str(simulated) == str(counted) and simulated.rows
    assert converted.get_submodule("0").formats is before
    with pytest.raises(RuntimeError, match="run=True"):
        profile(model, x, recipe=_recipe(), events=True, run=False)


# --- the opt-in kinds ------------------------------------------------------------


class _Transformerish(nn.Module):
    """An embedding, a norm, an attention, an activation, a residual sum, a
    softmax and a mean: every opt-in kind."""

    def __init__(self, d=16):
        super().__init__()
        self.emb = nn.Embedding(20, d)
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.ff = nn.Linear(d, d)

    def forward(self, ids):
        x = self.emb(ids)
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        h = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None]).squeeze(1)
        x = x + F.gelu(self.ff(h))
        return F.softmax(x, -1).sum(-1).mean()


def _opt_in_recipe(mode=RoundMode.RNE, carrier_rounding=RoundMode.RNE):
    mac = SplitMac(E4M3, B16, rounding=mode)
    q = Quant(BinaryK(8, 4, prng_bits=8), mode)
    return Recipe(
        linear=GemmFormats.uniform(mac, q),
        activation=ActivationFormats.uniform(q),
        elementwise=PointwiseFormats.uniform(q, carrier_rounding=carrier_rounding),
        embedding=PointwiseFormats.uniform(q),
        norm=NormFormats.uniform(mac, q),
        softmax=SoftmaxFormats.uniform(mac, q),
        reduction=ReductionFormats.uniform(mac, q),
        attention=AttentionFormats.uniform(mac, q),
    )


def _run_ids(recipe, model, ids, events):
    model.zero_grad()
    torch.manual_seed(1)
    with simulate(recipe, model, events=events) as sim:
        y = model(ids)
        y.backward()
    return sim, y, [p.grad.clone() for p in model.parameters()]


@requires_counting
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR], ids=lambda m: m.name)
@pytest.mark.parametrize("device", devices)
def test_every_opt_in_kind(device, mode):
    """Every opt-in kind is counted, forward and backward, its sums through
    the counted GEMMs and its slots recounted, and the words are the
    uncounted simulate's: the fused pointwise ops' two passes give them."""
    torch.manual_seed(0)
    model = _Transformerish().to(device)
    ids = torch.randint(0, 20, (2, 6), device=device)
    recipe = _opt_in_recipe(mode)
    _, y0, g0 = _run_ids(recipe, model, ids, events=False)
    sim, y1, g1 = _run_ids(recipe, model, ids, events=True)
    assert torch.equal(y0, y1) and all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))
    report = sim.events()
    assert not report.not_counted
    rows = {(r.kind, r.pass_name, r.role): r.counts.casts for r in report.rows}
    n = 2 * 6 * 16  # the activations' elements
    for kind, role in (("embedding", "quant"), ("norm", "out"), ("softmax", "exp")):
        assert rows[(kind, "fwd", role)] == n, (kind, role)
    assert rows[("activation", "fwd", "out")] == n and rows[("activation", "bwd", "grad")] == n
    assert (
        rows[("elementwise", "fwd", "quant")] == n and ("elementwise", "bwd", "grad_quant") in rows
    )
    # its mean and its variance: a product with 1 per element each
    assert rows[("norm", "fwd", "acc.mul")] == 2 * n
    assert rows[("norm", "bwd", "bwd_acc.mul")] > 0
    assert rows[("reduction", "fwd", "div")] == 1  # the mean of 12 sums
    for part in ("qk", "pv"):
        for pass_ in ("fwd", "igrad", "wgrad"):
            assert rows[("attention", pass_, f"{part}.mul")] > 0
    assert rows[("attention", "fwd", "softmax.exp")] == 2 * 6 * 6


@requires_counting
def test_an_elementwise_slot_is_recounted():
    """``a * 3 + b`` under one PointwiseFormats: its slot's row is the recount
    of each op's cast, on the carrier value before the rounding."""

    class M(nn.Module):
        def forward(self, a, b):
            return a * 3.0 + b

    q = Quant(E4M3)
    recipe = Recipe(elementwise=PointwiseFormats(quant=q))
    a, b = torch.randn(64) * 40, torch.randn(64)
    model = M()
    with simulate(recipe, model, events=True) as sim:
        model(a, b)
    (row,) = [r for r in sim.events().rows if r.role == "quant"]
    m = q(a * 3.0)
    expected = _recount(a * 3.0, m, E4M3) + _recount(m + b, q(m + b), E4M3)
    assert list(row.counts) == expected.tolist()


@requires_counting
def test_round_to_odd_counts_through_the_two_passes():
    """Rounded to odd, a counted pointwise op runs the two passes, whose
    words are the fused op's."""
    torch.manual_seed(0)
    model = _Transformerish()
    ids = torch.randint(0, 20, (2, 6))
    recipe = _opt_in_recipe(carrier_rounding=RoundMode.RO)
    _, y0, g0 = _run_ids(recipe, model, ids, events=False)
    sim, y1, g1 = _run_ids(recipe, model, ids, events=True)
    assert torch.equal(y0, y1) and all(torch.equal(a, b) for a, b in zip(g0, g1, strict=True))
    assert any(r.kind == "elementwise" for r in sim.events().rows)


@requires_counting
def test_profile_counts_converted_opt_in_layers():
    from mptorch.quant import convert, profile

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.GELU(), nn.Softmax(-1))
    x = torch.randn(4, 16)
    recipe = _opt_in_recipe()
    simulated = profile(model, x, recipe=recipe, events=True).events
    converted = convert(model, recipe)[0]
    counted = profile(converted, x, events=True).events
    assert simulated is not None and counted is not None
    assert str(simulated) == str(counted)
    assert {r.kind for r in counted.rows} == {"linear", "norm", "activation", "softmax"}
