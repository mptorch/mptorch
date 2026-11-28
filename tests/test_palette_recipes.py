"""
Palettes in recipes: a ``GemmFormats`` whose ``fwd_mac`` holds a palette
computes each output neuron (channel, column) in the entry of its map.

``mptorch/quant/instrument/palettes.py`` holds the palette layers' containers, whose
forward reads the layer's map at call time, and ``set_palette_maps_`` /
``palette_maps``. These tests hold a converted palette layer to the
hand-built model ``examples/mp_mlp_paper`` builds (``MPLinear``'s static
path, rewritten here), to ``matmul_formats`` and ``conv_formats`` over the
same palette and map for each kind, and to ``simulate``; then they check an
unset map (entry 0, word for word), the maps' life (devices, ``state_dict``,
copies, saving, compile), what is refused, and ``profile``'s split.
"""

import copy
import io
from typing import Any

import pytest
import torch
from torch import nn

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    ActivationFormats,
    AttentionFormats,
    FusedMac,
    GemmFormats,
    Palette,
    QAffineFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    QMatmul,
    Quant,
    Recipe,
    SplitMac,
    conv_formats,
    convert,
    matmul_formats,
    palette_maps,
    profile,
    qmatmul,
    qrelu,
    set_palette_maps_,
    simulate,
)
from mptorch.quant.modules.linear import CustomArithLinear
from tests.markers import available_devices

SR = RoundMode.SR


def _formats(rounding: RoundMode):
    """An 8-bit low format and a 16-bit high one, as the examples' E4M3 and FP16."""
    bits = 8 if rounding is SR else 0
    low = BinaryK(8, 4, prng_bits=bits)
    high = BinaryK(16, 11, prng_bits=bits)
    return low, high


def _palette_gf(rounding: RoundMode, fused: bool = True, outputs: bool = True) -> GemmFormats:
    low, high = _formats(rounding)
    if fused:
        mac = FusedMac(Palette([low, high]), rounding=rounding)
    else:
        acc = BinaryK(16, 11, prng_bits=low.prng_bits)
        mac = SplitMac(Palette([low, high]), acc, rounding=rounding)
    out = (Quant(low, rounding), Quant(high, rounding)) if outputs else None
    return GemmFormats(
        weight_quant=Quant(low, rounding),
        bias_quant=Quant(low, rounding),
        fwd_mac=mac,
        output_quant=out,
    )


# --- the examples' static workflow, word for word ---------------------------------------


class _StaticMPLinear(nn.Linear):
    """``examples/mp_mlp_paper``'s ``MPLinear`` in its static mode, rewritten:
    the forward over ``matmul_formats``' palette op with the map, the bias,
    then each neuron rounded to its entry's format; the activation, then the
    low format; float32 gradients, straight through every rounding."""

    def __init__(self, n_in, n_out, rounding, high_mask, activation):
        super().__init__(n_in, n_out)
        low, high = _formats(rounding)
        q_low, q_high = Quant(low, rounding), Quant(high, rounding)
        mac = FusedMac(Palette([low, high]), rounding=rounding)
        self.high = high_mask
        self.activation = activation
        self.q_low = q_low
        formats = matmul_formats(mac, prec_idx=high_mask.to(torch.int32).reshape(1, -1))

        def fwd(q_input, q_weight, q_bias):
            v = qmatmul(q_input, q_weight.T, formats) + q_bias
            return torch.where(self.high, q_high(v), q_low(v))

        self.static_formats = QAffineFormats(weight_quant=q_low, bias_quant=q_low, fwd_math=fwd)

    def forward(self, input):
        v = CustomArithLinear.apply(input, self.weight, self.bias, self.static_formats)
        if not self.activation:
            return v
        return qrelu(v, ActivationFormats(out=self.q_low))


def _run(model, x):
    x = x.detach().clone().requires_grad_(True)
    torch.manual_seed(5)
    out = model(x)
    out.backward(torch.ones_like(out))
    return [out.detach(), x.grad, *(p.grad for p in model.parameters())]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, SR], ids=["RNE", "SR"])
def test_the_examples_static_mplinear(device, rounding):
    """A converted MLP with per-neuron maps equals the examples' hand-built
    static MPLinear model: outputs and every gradient, under SR too (each
    neuron's two output roundings drawn in the reference's order)."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(12, 10), nn.ReLU(), nn.Linear(10, 6)).to(device)
    maps = {
        "0": torch.rand(10, generator=torch.Generator().manual_seed(1)) < 0.4,
        "2": torch.rand(6, generator=torch.Generator().manual_seed(2)) < 0.5,
    }
    low, _ = _formats(rounding)
    recipe = Recipe(
        linear=_palette_gf(rounding), activation=ActivationFormats(out=Quant(low, rounding))
    )
    converted, _ = convert(model, recipe)
    set_palette_maps_(converted, maps)

    reference = nn.Sequential(
        _StaticMPLinear(12, 10, rounding, maps["0"].to(device), activation=True),
        _StaticMPLinear(10, 6, rounding, maps["2"].to(device), activation=False),
    ).to(device)
    with torch.no_grad():
        for ref, layer in zip(reference, (model[0], model[2]), strict=True):
            assert isinstance(ref, nn.Linear) and isinstance(layer, nn.Linear)
            ref.weight.copy_(layer.weight)
            ref.bias.copy_(layer.bias)
    x = torch.randn(7, 12, device=device)
    got, want = _run(converted, x), _run(reference, x)
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w)


# --- each kind against the hand-built containers ----------------------------------------


def _outputs(gf, mask, v):
    """``gf``'s output rounding by entry, as a palette layer selects it."""
    q_low, q_high = gf.output_quant
    return torch.where(mask, q_high(v), q_low(v))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, SR], ids=["RNE", "SR"])
@pytest.mark.parametrize("fused", [True, False], ids=["fused", "split"])
@pytest.mark.parametrize("shape", [(5, 16), (2, 3, 16), (16,)], ids=str)
@torch.no_grad()
def test_a_linear_layer(device, rounding, fused, shape):
    gf = _palette_gf(rounding, fused)
    torch.manual_seed(0)
    layer = nn.Linear(16, 8).to(device)
    converted, _ = convert(nn.Sequential(layer), Recipe(linear=gf))
    high = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0], dtype=torch.bool, device=device)
    set_palette_maps_(converted, {"0": high})
    ref = matmul_formats(gf.fwd_mac, prec_idx=high.to(torch.int32).reshape(1, -1))
    x = torch.randn(*shape, device=device)
    torch.manual_seed(3)
    got = converted(x)
    torch.manual_seed(3)  # the weight, then the bias, as the layer rounds them
    q_w, q_b = gf.weight_quant(layer.weight), gf.bias_quant(layer.bias)
    want = _outputs(gf, high, qmatmul(x, q_w.T, ref) + q_b)
    assert got.shape == want.shape and torch.equal(got, want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "cls,groups,size",
    [(QConv1d, 2, (9,)), (QConv2d, 2, (6, 5)), (QConv2d, 4, (5, 5)), (QConv3d, 1, (4, 3, 3))],
    ids=str,
)
@torch.no_grad()
def test_a_convolution(device, cls, groups, size):
    """Per output channel, against conv_formats with the same map, across groups."""
    gf = _palette_gf(RoundMode.RNE, outputs=False)
    torch.manual_seed(0)
    nd = len(size)
    plain = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[nd](4, 8, 3, padding=1, groups=groups)
    plain = plain.to(device)
    converted, _ = convert(nn.Sequential(plain), Recipe(conv=gf))
    channels = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], device=device)
    set_palette_maps_(converted, {"0": channels})
    ref_formats = conv_formats(gf.fwd_mac, prec_idx=channels.to(torch.int32).reshape(-1, 1))
    ref_formats.weight_quant, ref_formats.bias_quant = gf.weight_quant, gf.bias_quant
    ref = cls(4, 8, 3, padding=1, groups=groups, formats=ref_formats).to(device)
    ref.load_state_dict(plain.state_dict())
    x = torch.randn(2, 4, *size, device=device)
    assert torch.equal(converted(x), ref(x))
    assert torch.equal(converted(x[0]), ref(x[0]))  # unbatched


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("a_shape", [(5, 7), (3, 5, 7), (2, 3, 5, 7)], ids=str)
@torch.no_grad()
def test_a_matmul(device, a_shape):
    """Per output column, against matmul_formats with the same map, which
    follows the module to its device."""
    low, high = _formats(RoundMode.RNE)
    gf = GemmFormats(fwd_mac=FusedMac(Palette([low, high])), output_quant=Palette([low, high]))
    qk = QMatmul(gf.matmul())
    holder = nn.Sequential()
    holder.add_module("qk", qk)
    columns = torch.tensor([1, 0, 0, 1, 1, 0], device=device)
    set_palette_maps_(holder, {"qk": columns})
    holder.to(device)
    a, b = torch.randn(*a_shape, device=device), torch.randn(7, 6, device=device)
    ref = matmul_formats(gf.fwd_mac, prec_idx=columns.to(torch.int32).reshape(1, -1))
    assert torch.equal(qk(a, b), _outputs(gf, columns.bool(), qmatmul(a, b, ref)))
    with pytest.raises(ValueError, match="width"):
        qk(a, torch.randn(7, 5, device=device))


# --- an unset map --------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.RZ, SR], ids=str)
@torch.no_grad()
def test_an_unset_map_is_entry_0(device, rounding):
    """An unset map computes every neuron in entry 0, by entry 0's own kernel:
    the words of a map of zeros (one more rounding draws under SR, so there
    the unset layer equals entry 0's single-format layer), and no map memory."""
    gf = _palette_gf(rounding)
    torch.manual_seed(0)
    layer = nn.Linear(16, 8).to(device)
    unset, _ = convert(nn.Sequential(layer), Recipe(linear=gf))
    formats = unset.get_submodule("0").formats
    assert isinstance(formats, nn.Module) and formats.get_buffer("prec_idx") is None
    zeros, _ = convert(nn.Sequential(layer), Recipe(linear=gf))
    set_palette_maps_(zeros, {"0": torch.zeros(8, dtype=torch.int64)})
    low, _ = _formats(rounding)
    single = GemmFormats(
        weight_quant=gf.weight_quant,
        bias_quant=gf.bias_quant,
        fwd_mac=FusedMac(low, rounding=rounding),
        output_quant=gf.output_quant[0],
    )
    entry0, _ = convert(nn.Sequential(layer), Recipe(linear=single))
    x = torch.randn(5, 16, device=device)
    torch.manual_seed(3)
    got = unset(x)
    torch.manual_seed(3)
    assert torch.equal(got, entry0(x))
    if rounding is not SR:
        assert torch.equal(got, zeros(x))


def test_an_unset_map_keeps_the_palettes_checks():
    """An unset layer refuses what the palette op refuses: a bfloat16 result
    can hold the low entry's values and not the high one's."""
    gf = GemmFormats(fwd_mac=FusedMac(Palette([BinaryK(8, 4), BinaryK(16, 11)])))
    layer, _ = convert(nn.Sequential(nn.Linear(4, 4)), Recipe(linear=gf))
    layer = layer.to(torch.bfloat16)
    with torch.no_grad(), pytest.raises(ValueError, match="bfloat16"):
        layer(torch.randn(2, 4, dtype=torch.bfloat16))


def test_a_layer_without_a_palette_carries_nothing():
    gf = GemmFormats(fwd_mac=FusedMac(BinaryK(8, 4)))
    converted, _ = convert(nn.Sequential(nn.Linear(4, 4)), Recipe(linear=gf))
    formats = converted.get_submodule("0").formats
    assert type(formats) is QAffineFormats and "prec_idx" not in formats._buffers


# --- convert and simulate ------------------------------------------------------------------


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 6, 3, padding=1)
        self.fc = nn.Linear(6, 6)

    def forward(self, x):
        h = self.fc(self.conv(x).transpose(1, 2))
        return torch.matmul(h, h.transpose(1, 2))


@pytest.mark.parametrize("device", available_devices)
def test_simulate_is_convert(device):
    """The same maps in a converted model and a simulate give the same words,
    under SR, outputs and gradients; a matmul site has a map of its own."""
    gf = _palette_gf(SR)
    recipe = Recipe(linear=gf, conv=gf).override(_Block, site="matmul#0", matmul=gf)
    torch.manual_seed(0)
    model = _Block().to(device)
    maps = {
        "conv": torch.tensor([1, 0, 1, 0, 0, 1]),
        "fc": torch.tensor([0, 1, 1, 0, 1, 0]),
    }
    converted, _ = convert(model, recipe)
    set_palette_maps_(converted, maps)
    sim = simulate(recipe, model)
    set_palette_maps_(sim, {**maps, "#matmul#0": torch.tensor([1, 0, 0, 1, 1])})
    # the converted model has no QMatmul at the site: compare its layers only
    x = torch.randn(2, 4, 5, device=device)
    torch.manual_seed(7)
    fc, conv = converted.get_submodule("fc"), converted.get_submodule("conv")
    h_conv = fc(conv(x).transpose(1, 2))
    torch.manual_seed(7)
    with sim:
        h_sim = model.fc(model.conv(x).transpose(1, 2))
    assert torch.equal(h_conv, h_sim)
    with sim:
        model(x)
    report = sim.report()
    assert ("#matmul#0", True) in report.palette_maps and ("fc", True) in report.palette_maps
    assert set(palette_maps(sim)) == {"conv", "fc", "#matmul#0"}


# --- the maps ------------------------------------------------------------------------------


def _converted(gf=None):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 5), nn.ReLU(), nn.Linear(5, 3))
    converted, _ = convert(model, Recipe(linear=gf or _palette_gf(RoundMode.RNE)))
    return model, converted


def test_maps_follow_the_layer_and_stay_out_of_the_state_dict():
    model, converted = _converted()
    set_palette_maps_(converted, {"0": torch.tensor([1, 0, 1, 1, 0])})
    assert list(converted.state_dict()) == list(model.state_dict())
    converted.load_state_dict(model.state_dict(), strict=True)
    model.load_state_dict(converted.state_dict(), strict=True)
    converted.half()
    assert converted[0].formats.prec_idx.dtype is torch.int32
    if torch.cuda.is_available():
        converted.cuda()
        assert converted[0].formats.prec_idx.device.type == "cuda"


def test_a_copy_has_its_own_map():
    _, converted = _converted()
    set_palette_maps_(converted, {"0": torch.tensor([1, 0, 1, 1, 0])})
    copied = copy.deepcopy(converted)
    set_palette_maps_(copied, {"0": torch.zeros(5, dtype=torch.int64)})
    assert palette_maps(converted)["0"].tolist() == [1, 0, 1, 1, 0]
    x = torch.randn(2, 6)
    with torch.no_grad():
        assert not torch.equal(copied(x), converted(x))


def test_maps_are_saved_and_restored():
    _, converted = _converted()
    maps = {"0": torch.tensor([1, 0, 1, 1, 0]), "2": torch.tensor([0, 1, 1])}
    set_palette_maps_(converted, maps)
    buffer = io.BytesIO()
    torch.save(palette_maps(converted), buffer)
    _, other = _converted()
    buffer.seek(0)
    set_palette_maps_(other, torch.load(buffer))
    x = torch.randn(2, 6)
    with torch.no_grad():
        assert torch.equal(other(x), converted(x))
    set_palette_maps_(other, {"2": None})
    assert set(palette_maps(other)) == {"0"}


@pytest.mark.parametrize(
    "maps,error,match",
    [
        ({"0": torch.tensor([1, 0, 1])}, ValueError, "has 5 outputs"),
        ({"0": torch.tensor([1, 0, 1, 2, 0])}, ValueError, "palette has 2 entries"),
        ({"0": torch.rand(5)}, TypeError, "integer tensor"),
        ({"1": torch.zeros(5, dtype=torch.int64)}, KeyError, "not a palette layer"),
        ({"9": torch.zeros(5, dtype=torch.int64)}, KeyError, "not a palette layer"),
    ],
    ids=["width", "range", "float", "relu", "unknown"],
)
def test_what_set_palette_maps_refuses(maps, error, match):
    _, converted = _converted()
    with pytest.raises(error, match=match):
        set_palette_maps_(converted, maps)


def test_a_compiled_model_keeps_its_graph_across_maps():
    """Setting an unset map recompiles once; a map of the same shape is
    copied into the buffer, and the graph stays."""
    _, converted = _converted()
    frames = []

    def backend(gm, example_inputs):
        frames.append(gm)
        return gm

    torch._dynamo.reset()
    compiled = torch.compile(converted, backend=backend, fullgraph=True)
    x = torch.randn(2, 6)
    with torch.no_grad():
        compiled(x)
        set_palette_maps_(converted, {"0": torch.tensor([1, 0, 1, 1, 0])})
        first = compiled(x)
        assert len(frames) == 2
        set_palette_maps_(converted, {"0": torch.tensor([0, 1, 1, 0, 0])})
        second = compiled(x)
        assert len(frames) == 2
        assert torch.equal(second, converted(x)) and not torch.equal(first, second)


# --- what is refused -----------------------------------------------------------------------

P8, P16 = BinaryK(8, 4), BinaryK(16, 11)
PALETTE: Any = FusedMac(Palette([P8, P16]))


def test_what_a_palette_gemm_formats_refuses():
    with pytest.raises(ValueError, match="grad_mac="):
        GemmFormats.uniform(PALETTE)
    with pytest.raises(ValueError, match="one rounding per entry"):
        GemmFormats(fwd_mac=PALETTE, output_quant=(Quant(P8),) * 3)
    GemmFormats.uniform(PALETTE, grad_mac=SplitMac(P8, P16))  # the gradients single
    gf = GemmFormats(fwd_mac=PALETTE, output_quant=Palette([P8, P16]))
    assert gf.output_quant == (Quant(P8), Quant(P16))


def test_palettes_are_refused_in_attention():
    gf = GemmFormats(fwd_mac=PALETTE)
    with pytest.raises(ValueError, match="builds its products' containers once"):
        AttentionFormats(qk=gf)
    model = nn.Sequential(nn.MultiheadAttention(8, 2))
    with pytest.raises(ValueError, match="packs the query's"):
        convert(model, Recipe(linear=gf))
    with pytest.raises(ValueError, match="packs the query's"):
        simulate(Recipe(linear=gf), model)


# --- profile -------------------------------------------------------------------------------


def _entry_macs(p):
    out: dict[Any, int] = {}
    for r in p.records:
        if r.step == "fwd":
            out[r.key] = out.get(r.key, 0) + r.macs
    return out


def test_profile_splits_a_palette_layer_by_its_map():
    """Each entry's MACs are its neurons' (channels', columns'); an unset map
    counts all in entry 0; run=False and a recipe's palette_maps= count alike."""
    low, high = FusedMac(P8), FusedMac(P16)
    gf = GemmFormats(fwd_mac=PALETTE)
    model = _Block()
    recipe = Recipe(linear=gf, conv=gf)
    converted, _ = convert(model, recipe)
    x = torch.randn(2, 4, 5)
    unset = _entry_macs(profile(converted, x))
    conv_macs = 2 * 6 * 5 * 4 * 3  # B, Cout, positions, Cin * taps
    fc_macs = 10 * 6 * 6
    assert unset[low] == conv_macs + fc_macs and high not in unset
    maps = {"conv": torch.tensor([1, 0, 1, 0, 0, 1]), "fc": torch.tensor([0, 1, 1, 1, 1, 0])}
    set_palette_maps_(converted, maps)
    split = _entry_macs(profile(converted, x))
    assert split[high] == conv_macs * 3 // 6 + fc_macs * 4 // 6
    assert split[low] == conv_macs * 3 // 6 + fc_macs * 2 // 6
    assert _entry_macs(profile(model, x, recipe=recipe, palette_maps=maps)) == split
    assert _entry_macs(profile(model, x, recipe=recipe, palette_maps=maps, run=False)) == split
    with torch.device("meta"):
        meta = _Block()
    meta_converted, _ = convert(meta, recipe)
    set_palette_maps_(meta_converted, maps)
    meta_x = torch.empty(2, 4, 5, device="meta")
    assert _entry_macs(profile(meta_converted, meta_x, run=False)) == split
    with pytest.raises(TypeError, match="no recipe"):
        profile(model, x, palette_maps=maps)
