"""Hadamard rotations before quantization: ``Hadamard``, ``hadamard_rotate``
and ``GemmFormats.rotation``.

The transform is held to the explicit Walsh-Hadamard matrix and to its own
inverse, and to the same words on every device. A rotated layer follows the
three tiers: with nothing quantized it computes the unrotated function
(tier 1); with MX block macs on data with outlier channels it is closer to
the float64 reference than the unrotated layer (tier 2); and it is, word for
word, the unrotated layer over rotated operands with each gradient rotated
back, and its forward the block GEMM of the rotated operands built from the
public functions (tier 3). Around it: palettes, recompute, frozen weights,
``simulate`` against ``convert``, attention, compile and ``profile``.
"""

import copy
import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mptorch import MXFP4_E2M1, MXFP8_E4M3, MXFP8_E5M2, BinaryK, RoundMode
from mptorch.quant import (
    AttentionFormats,
    BlockMac,
    GemmFormats,
    Hadamard,
    Palette,
    QLinear,
    QMatmul,
    Recipe,
    SplitMac,
    block_matmul,
    block_pack,
    block_quantize,
    convert,
    freeze_packed_,
    hadamard_rotate,
    presets,
    profile,
    qattention,
    set_palette_maps_,
    simulate,
)
from mptorch.quant.modules._saved import _ROTATION_ONLY, _known
from mptorch.quant.rotation import _RotatedQuant, _signs
from tests.markers import accelerator_devices, available_devices, float64_devices
from tests.models import MiniLlama


def _sylvester(r: int) -> torch.Tensor:
    """The orthonormal Walsh-Hadamard matrix of order ``r``, in float64,
    built by Sylvester's recursion."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < r:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(r)


def _linear_formats(rotation, rounding=RoundMode.RNE, fwd=MXFP4_E2M1) -> GemmFormats:
    acc = BinaryK(16, 8) if rounding.is_stochastic else None
    recipe = presets.mx(fwd, MXFP8_E5M2, acc, rounding=rounding, rotation=rotation)
    return dict(recipe.rules[0].kinds)["linear"]


def _grad_of(t: torch.Tensor) -> torch.Tensor:
    assert t.grad is not None
    return t.grad


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double(), b.detach().double()
    return float((a - b).norm() / b.norm())


# --- the transform -------------------------------------------------------------------------


@pytest.mark.parametrize("r", [2, 4, 32, 128])
def test_the_transform_is_the_hadamard_matrix(r):
    """Unsigned, a rotation is ``x`` times the block-diagonal Walsh-Hadamard
    matrix of order ``r``: in float64 to its rounding, in float32 to its."""
    x = torch.randn(3, 5, 256, dtype=torch.float64)
    want = (x.reshape(3, 5, 256 // r, r) @ _sylvester(r)).reshape(x.shape)
    rot = Hadamard(r, signed=False)
    torch.testing.assert_close(hadamard_rotate(x, rot), want, rtol=0, atol=1e-13)
    got = hadamard_rotate(x.float(), rot)
    assert got.dtype is torch.float32
    torch.testing.assert_close(got.double(), want, rtol=0, atol=1e-5)


def test_signed_is_the_signs_then_the_transform():
    """``D H`` is the signs, an exact product, then the unsigned transform."""
    x = torch.randn(7, 192)
    signs = _signs(192, 5, x.device, torch.float32)
    want = hadamard_rotate(x * signs, Hadamard(64, signed=False))
    assert torch.equal(hadamard_rotate(x, Hadamard(64, seed=5)), want)
    back = hadamard_rotate(want, Hadamard(64, signed=False), inverse=True) * signs
    assert torch.equal(hadamard_rotate(want, Hadamard(64, seed=5), inverse=True), back)


@pytest.mark.parametrize(
    ("dtype", "tol"),
    [(torch.float64, 1e-14), (torch.float32, 1e-6), (torch.bfloat16, 2**-6)],
)
@pytest.mark.parametrize("signed", [False, True])
def test_the_inverse_undoes_the_rotation(dtype, tol, signed):
    x = torch.randn(4, 9, 128).to(dtype)
    rot = Hadamard(64, signed=signed)
    y = hadamard_rotate(x, rot)
    assert y.dtype is dtype and y.shape == x.shape
    back = hadamard_rotate(y, rot, inverse=True)
    torch.testing.assert_close(back.double(), x.double(), rtol=tol, atol=tol)


def test_a_middle_axis_is_rotated_as_the_last():
    x = torch.randn(3, 64, 10)
    rot = Hadamard(32)
    want = hadamard_rotate(x.transpose(1, 2), rot).transpose(1, 2)
    assert torch.equal(hadamard_rotate(x, rot, 1), want)
    assert torch.equal(hadamard_rotate(x, rot, -2), want)


def test_a_ragged_axis_takes_the_largest_power_of_two_dividing_it():
    """R-e: 784 = 16 x 49 under 128 rotates in blocks of 16; an odd axis is
    returned as it is."""
    rot = Hadamard(128)
    assert [rot.block(k) for k in (784, 96, 768, 4096, 7)] == [16, 32, 128, 128, 1]
    x = torch.randn(5, 784)
    assert torch.equal(hadamard_rotate(x, rot), hadamard_rotate(x, Hadamard(16)))
    odd = torch.randn(5, 7)
    assert hadamard_rotate(odd, rot) is odd


def test_the_signs():
    """+1 and -1, about half each, a channel's sign independent of the axis
    length, and different seeds different signs."""
    s = _signs(4096, 0, torch.device("cpu"), torch.float32)
    assert set(s.unique().tolist()) == {-1.0, 1.0}
    assert abs(float(s.sum())) < 4 * math.sqrt(4096)
    assert torch.equal(_signs(64, 0, s.device, s.dtype), s[:64])
    assert not torch.equal(_signs(4096, 1, s.device, s.dtype), s)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"size": 1}, ValueError, "power of two"),
        ({"size": 96}, ValueError, "power of two"),
        ({"size": True}, TypeError, "an int"),
        ({"size": 32.0}, TypeError, "an int"),
        ({"size": 32, "signed": 1}, TypeError, "True or False"),
        ({"size": 32, "seed": -1}, ValueError, r"\[0, 2\*\*31\)"),
        ({"size": 32, "seed": 2**31}, ValueError, r"\[0, 2\*\*31\)"),
        ({"size": 32, "carrier": "float64"}, TypeError, "torch.dtype"),
        ({"size": 32, "carrier": torch.float16}, ValueError, "carrier must be"),
    ],
)
def test_a_rotation_refuses(kwargs, error, match):
    with pytest.raises(error, match=match):
        Hadamard(**kwargs)


def test_gemm_formats_refuse_a_rotation_of_the_wrong_type():
    with pytest.raises(TypeError, match="rotation is a Hadamard or None"):
        GemmFormats(rotation=128)  # ty: ignore[invalid-argument-type]


def test_the_carrier():
    """``carrier=torch.float64`` computes in binary64 and rounds once to the
    tensor's dtype; a carrier narrower than the tensor raises."""
    x = torch.randn(6, 64)
    rot = Hadamard(64)
    want = hadamard_rotate(x.double(), rot).float()
    assert torch.equal(hadamard_rotate(x, rot, carrier=torch.float64), want)
    assert torch.equal(hadamard_rotate(x, dataclasses.replace(rot, carrier=torch.float64)), want)
    with pytest.raises(ValueError, match="narrower than float64"):
        hadamard_rotate(x.double(), rot, carrier=torch.float32)


@pytest.mark.parametrize("device", accelerator_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_every_device_rotates_as_the_cpu(device, dtype):
    """The butterfly's additions are correctly rounded everywhere, and the
    signs are integer arithmetic, so every device gives the CPU's words (on
    normal inputs: the Apple GPU flushes subnormal sums)."""
    x = torch.randn(5, 3, 256).to(dtype)
    for rot in (Hadamard(128), Hadamard(16, signed=False), Hadamard(64, seed=9)):
        for inverse in (False, True):
            want = hadamard_rotate(x, rot, inverse=inverse)
            got = hadamard_rotate(x.to(device), rot, inverse=inverse).cpu()
            assert torch.equal(got, want)


# --- tier 1: nothing quantized -------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tier1_a_rotated_linear_is_the_linear(device, dtype):
    """With PyTorch's arithmetic and no quantizers, the rotated layer is
    ``nn.Linear``, forward and every gradient, to the rotation's rounding."""
    tol = 1e-5 if dtype is torch.float32 else 1e-13
    lin = nn.Linear(64, 24).to(device, dtype)
    q = QLinear(64, 24, formats=GemmFormats(rotation=Hadamard(32)).affine()).to(device, dtype)
    q.load_state_dict(lin.state_dict())
    x = torch.randn(3, 5, 64, device=device, dtype=dtype, requires_grad=True)
    x2 = x.detach().clone().requires_grad_()
    y, y2 = lin(x), q(x2)
    g = torch.randn_like(y)
    y.backward(g)
    y2.backward(g)
    torch.testing.assert_close(y2, y, rtol=tol, atol=tol)
    torch.testing.assert_close(x2.grad, x.grad, rtol=tol, atol=tol)
    torch.testing.assert_close(q.weight.grad, lin.weight.grad, rtol=tol, atol=tol)
    assert torch.equal(_grad_of(q.bias), _grad_of(lin.bias))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("b_shape", [(64, 16), (2, 1, 64, 16), (64,)])
def test_tier1_a_rotated_matmul_is_the_matmul(device, b_shape):
    """``a`` rotates along its last axis and ``b`` along K, its axis -2 (its
    only axis when 1D), so the product is ``torch.matmul``'s, broadcast
    included, and so are both gradients."""
    a = torch.randn(2, 3, 8, 64, device=device, requires_grad=True)
    b = torch.randn(*b_shape, device=device, requires_grad=True)
    a2, b2 = a.detach().clone().requires_grad_(), b.detach().clone().requires_grad_()
    m = QMatmul(GemmFormats(rotation=Hadamard(16)).matmul())
    out, out2 = torch.matmul(a, b), m(a2, b2)
    g = torch.randn_like(out)
    out.backward(g)
    out2.backward(g)
    for got, want in ((out2, out), (a2.grad, a.grad), (b2.grad, b.grad)):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


# --- tier 2: outliers ----------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_tier2_rotation_helps_mxfp4_on_outlier_channels(device):
    """On an input with three channels 30 times the rest, MXFP4's forward
    error falls by a quarter or more under rotation, and every result stays
    within MXFP4's tolerance of the float64 reference."""
    x = torch.randn(64, 256)
    x[:, [5, 77, 200]] *= 30
    lin = nn.Linear(256, 128)
    g = torch.randn(64, 128)
    xr = x.double().requires_grad_()
    w = lin.weight.detach().double()
    want = F.linear(xr, w, lin.bias.detach().double())
    want.backward(g.double())
    errors = {}
    for rot in (None, Hadamard(32), Hadamard(128)):
        q = QLinear(256, 128, formats=_linear_formats(rot).affine())
        q.load_state_dict(lin.state_dict())
        q = q.to(device)
        xa = x.to(device, copy=True).requires_grad_()
        out = q(xa)
        out.backward(g.to(device))
        errors[rot] = (
            _rel(out.cpu(), want.detach()),
            _rel(_grad_of(xa).cpu(), _grad_of(xr)),
            _rel(_grad_of(q.weight).cpu(), g.double().T @ x.double()),
        )
        assert all(e < 0.3 for e in errors[rot]), (rot, errors[rot])
    for rot in (Hadamard(32), Hadamard(128)):
        assert errors[rot][0] < 0.75 * errors[None][0], errors


# --- tier 3: by hand -------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_tier3_a_rotated_layer_is_the_layer_over_rotated_operands(device, rounding):
    """Word for word, SR included: the rotated layer's output is the
    unrotated layer's over ``x R`` and ``W R``, and its gradients are that
    layer's rotated back along K."""
    rot = Hadamard(64)
    gf = _linear_formats(rot, rounding)
    rotated = QLinear(256, 96, formats=gf.affine()).to(device)
    plain = QLinear(256, 96, formats=dataclasses.replace(gf, rotation=None).affine()).to(device)
    with torch.no_grad():
        plain.weight.copy_(hadamard_rotate(rotated.weight, rot))
        plain.bias.copy_(rotated.bias)
    x = torch.randn(4, 7, 256, device=device)
    x[..., 3] *= 50
    xa = x.clone().requires_grad_()
    xb = hadamard_rotate(x, rot).requires_grad_()
    g = torch.randn(4, 7, 96, device=device)
    torch.manual_seed(5)
    out = rotated(xa)
    out.backward(g)
    torch.manual_seed(5)
    want = plain(xb)
    want.backward(g)
    assert torch.equal(out, want)
    assert torch.equal(_grad_of(xa), hadamard_rotate(_grad_of(xb), rot, inverse=True))
    assert torch.equal(
        _grad_of(rotated.weight), hadamard_rotate(_grad_of(plain.weight), rot, inverse=True)
    )
    assert torch.equal(_grad_of(rotated.bias), _grad_of(plain.bias))


@pytest.mark.parametrize("device", available_devices)
def test_tier3_the_forward_is_the_block_gemm_of_the_rotated_operands(device):
    """The forward from the public functions: both operands rotated, packed
    along K, multiplied by ``block_matmul``, the bias added."""
    rot = Hadamard(128)
    layer = QLinear(256, 48, formats=_linear_formats(rot, fwd=MXFP8_E4M3).affine()).to(device)
    x = torch.randn(10, 256, device=device)
    with torch.no_grad():
        xq = block_pack(hadamard_rotate(x, rot), MXFP8_E4M3)
        wq = block_pack(hadamard_rotate(layer.weight, rot), MXFP8_E4M3, axis=1)
        want = block_matmul(xq, wq.mT) + layer.bias
        assert torch.equal(layer(x), want)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_tier3_a_rotated_matmul_is_the_matmul_over_rotated_operands(device, rounding):
    """For a matmul, ``a R`` and ``R^T b``, the gradients rotated back along
    ``a``'s last axis and ``b``'s axis -2."""
    acc = BinaryK(16, 8) if rounding.is_stochastic else None
    mac = BlockMac(MXFP4_E2M1, MXFP4_E2M1, acc, rounding=rounding)
    gf = GemmFormats(fwd_mac=mac, igrad_mac=mac, wgrad_mac=mac, rotation=Hadamard(32))
    rotated = QMatmul(gf.matmul())
    plain = QMatmul(dataclasses.replace(gf, rotation=None).matmul())
    a = torch.randn(2, 16, 64, device=device)
    b = torch.randn(2, 64, 32, device=device)
    rot = gf.rotation
    assert rot is not None
    aa, ba = a.clone().requires_grad_(), b.clone().requires_grad_()
    ab, bb = hadamard_rotate(a, rot).requires_grad_(), hadamard_rotate(b, rot, -2).requires_grad_()
    g = torch.randn(2, 16, 32, device=device)
    torch.manual_seed(8)
    out = rotated(aa, ba)
    out.backward(g)
    torch.manual_seed(8)
    want = plain(ab, bb)
    want.backward(g)
    assert torch.equal(out, want)
    assert torch.equal(_grad_of(aa), hadamard_rotate(_grad_of(ab), rot, inverse=True))
    assert torch.equal(_grad_of(ba), hadamard_rotate(_grad_of(bb), rot, -2, inverse=True))


# --- around it -----------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_a_palette_layer_rotates_as_a_single_format_one(device):
    """A palette layer's operands rotate too: its output is the unrotated
    palette layer's over the rotated operands, under the same map."""
    p8, p16 = BinaryK(8, 4), BinaryK(16, 11)
    pal = SplitMac(Palette([p8, p16]), p16)
    rot = Hadamard(32)
    model = nn.Sequential(nn.Linear(64, 12)).to(device)
    rotated, _ = convert(model, Recipe(linear=GemmFormats(fwd_mac=pal, rotation=rot)))
    plain, _ = convert(model, Recipe(linear=GemmFormats(fwd_mac=pal)))
    with torch.no_grad():
        weight = model.get_parameter("0.weight")
        plain.get_parameter("0.weight").copy_(hadamard_rotate(weight, rot))
    mask = torch.arange(12) % 3 == 0
    set_palette_maps_(rotated, {"0": mask})
    set_palette_maps_(plain, {"0": mask})
    x = torch.randn(5, 64, device=device)
    with torch.no_grad():
        assert torch.equal(rotated(x), plain(hadamard_rotate(x, rot)))


def test_a_rotated_slot_is_known_stateless():
    """``_known`` looks through a rotation: a rotation alone draws nothing,
    and a rotated quantizer is known when its quantizer is."""
    q = _linear_formats(Hadamard(32)).affine()
    assert isinstance(q.input_quant, _RotatedQuant)
    assert _known(q.input_quant) is _ROTATION_ONLY
    gf = GemmFormats(input_quant=MXFP4_E2M1.elem, rotation=Hadamard(32))
    slot = gf.affine().input_quant
    assert isinstance(slot, _RotatedQuant)
    assert _known(slot) is slot.quant
    assert _known(_RotatedQuant(Hadamard(32), lambda t: t, -1)) is None


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("recompute", [False, True])
def test_recompute_gives_the_saved_copies_words(device, recompute):
    """A raw operand rotated and quantized again in backward gives what the
    saved copy gives, SR included."""
    rot = Hadamard(32)
    gf = dataclasses.replace(
        _linear_formats(rot, RoundMode.SR),
        input_quant=BinaryK(8, 4, prng_bits=8),
        weight_quant=BinaryK(8, 4, prng_bits=8),
    )
    results = []
    for flag in (not recompute, recompute):
        f = dataclasses.replace(gf, recompute_input=flag, recompute_weight=flag)
        torch.manual_seed(0)
        layer = QLinear(64, 32, formats=f.affine()).to(device)
        x = torch.randn(6, 64, device=device, requires_grad=True)
        torch.manual_seed(3)
        out = layer(x)
        out.backward(torch.ones_like(out))
        results.append((out, x.grad, layer.weight.grad))
    for got, want in zip(*results, strict=True):
        assert torch.equal(got, want)


def _frozen_model(rotation, device="cpu"):
    torch.manual_seed(0)
    gf = GemmFormats(fwd_mac=BlockMac(MXFP4_E2M1), rotation=rotation)
    return nn.Sequential(QLinear(128, 64, formats=gf.affine())).to(device)


@pytest.mark.parametrize("device", available_devices)
def test_a_frozen_rotated_layer_is_the_unfrozen_one(device):
    """Its codes are the rotated weight's, a frozen forward rotates only the
    input, and unfreezing rotates the decoded weight back."""
    rot = Hadamard(64)
    model = _frozen_model(rot, device)
    x = torch.randn(9, 128, device=device)
    with torch.no_grad():
        want = model(x)
        weight = model[0].weight.detach().clone()
        assert freeze_packed_(model) == 1
        assert torch.equal(model(x), want)
        model[0].unfreeze_packed_()
        decoded = hadamard_rotate(weight, rot)
        expect = hadamard_rotate(block_quantize(decoded, MXFP4_E2M1), rot, inverse=True)
        assert torch.equal(model[0].weight, expect)
        assert torch.equal(model(x), want)


def test_a_frozen_rotated_checkpoint_records_its_rotation():
    """Layout 2 holds the rotation and loads into a layer with the same one;
    another rotation, or none, refuses it; an unrotated layer still writes
    layout 1."""
    model = _frozen_model(Hadamard(64))
    x = torch.randn(3, 128)
    with torch.no_grad():
        want = model(x)
    freeze_packed_(model)
    state = model.state_dict()
    extra = state["0._extra_state"]
    assert extra["layout"] == 2
    assert extra["rotation"] == {"size": 64, "signed": True, "seed": 0, "carrier": None}
    same = _frozen_model(Hadamard(64))
    freeze_packed_(same, pack=False)
    same.load_state_dict(state)
    with torch.no_grad():
        assert torch.equal(same(x), want)
    for other in (Hadamard(32), Hadamard(64, seed=1), None):
        layer = _frozen_model(other)
        freeze_packed_(layer, pack=False)
        with pytest.raises(ValueError, match="has rotation"):
            layer.load_state_dict(state)
    plain = _frozen_model(None)
    freeze_packed_(plain)
    assert plain.state_dict()["0._extra_state"]["layout"] == 1
    assert "rotation" not in plain.state_dict()["0._extra_state"]


def _run(model, ids, sim=None):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(11)
    if ids.device.type == "cuda":
        torch.cuda.manual_seed_all(11)
    if sim is None:
        out = model(ids)
    else:
        with sim:
            out = model(ids)
    out.backward(torch.ones_like(out) / out.numel())
    return {"out": out.detach(), **{n: p.grad for n, p in model.named_parameters()}}


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
def test_simulate_is_the_converted_model(device, rounding):
    """Llama's projections and its tied head, rotated (K of 32 and 64, so
    blocks of 32 and 64 under 128): simulate gives the converted model's
    output and gradients word for word."""
    torch.manual_seed(0)
    model = MiniLlama(tie=True).to(device)
    ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(3)).to(device)
    acc = BinaryK(16, 8) if rounding.is_stochastic else None
    recipe = presets.mx(MXFP4_E2M1, acc=acc, rounding=rounding, rotation=Hadamard(128))
    converted, _ = convert(model, recipe)
    want = _run(converted, ids)
    got = _run(model, ids, simulate(recipe, model))
    assert want.keys() == got.keys()
    for k in want:
        assert torch.equal(got[k], want[k]), k


@pytest.mark.parametrize("device", float64_devices)
def test_attention_rotates_q_and_k_along_the_head_dimension(device):
    """``AttentionFormats.qk`` with a rotation is the unrotated attention over
    ``q R`` and ``k R``, and its gradients rotated back."""
    mac = SplitMac(BinaryK(8, 4), BinaryK(16, 11))
    qk = GemmFormats(fwd_mac=mac, igrad_mac=mac, wgrad_mac=mac, rotation=Hadamard(16))
    rot = qk.rotation
    assert rot is not None
    rotated = AttentionFormats(qk=qk)
    plain = AttentionFormats(qk=dataclasses.replace(qk, rotation=None))
    q, k, v = (torch.randn(2, 3, 8, 32, device=device) for _ in range(3))
    qa, ka = q.clone().requires_grad_(), k.clone().requires_grad_()
    qb, kb = hadamard_rotate(q, rot).requires_grad_(), hadamard_rotate(k, rot).requires_grad_()
    out = qattention(qa, ka, v, rotated, scale=1 / math.sqrt(32))
    want = qattention(qb, kb, v, plain, scale=1 / math.sqrt(32))
    g = torch.randn_like(out)
    out.backward(g)
    want.backward(g)
    assert torch.equal(out, want)
    assert torch.equal(_grad_of(qa), hadamard_rotate(_grad_of(qb), rot, inverse=True))
    assert torch.equal(_grad_of(ka), hadamard_rotate(_grad_of(kb), rot, inverse=True))


# The block packer looks its format's fields up through an lru_cache, which
# Dynamo says it traces through; the lookup is pure (as in test_freeze_packed).
@pytest.mark.filterwarnings("ignore:Dynamo detected a call to a `functools.lru_cache`:UserWarning")
@pytest.mark.parametrize("device", available_devices)
def test_a_rotated_layer_compiles_to_eagers_words(device):
    torch._dynamo.reset()
    torch.manual_seed(0)
    layer = QLinear(128, 32, formats=_linear_formats(Hadamard(64), fwd=MXFP8_E4M3).affine())
    layer = layer.to(device)
    x = torch.randn(4, 128, device=device, requires_grad=True)
    out = layer(x)
    out.sum().backward()
    want = (out.detach(), _grad_of(x), _grad_of(layer.weight))
    layer.zero_grad(set_to_none=True)
    x2 = x.detach().clone().requires_grad_()
    out2 = torch.compile(layer, backend="aot_eager", fullgraph=True)(x2)
    out2.sum().backward()
    for got, w in zip((out2.detach(), _grad_of(x2), _grad_of(layer.weight)), want, strict=True):
        assert torch.equal(got, w)


def test_profile_records_each_rotation():
    """One element record per rotation: both operands forward, and each
    gradient that backward rotates back; a 784-feature layer at the block
    size 784 allows."""
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
    recipe = presets.mx(MXFP4_E2M1, rotation=Hadamard(64))
    p = profile(model, torch.randn(4, 784), recipe=recipe, backward=True)
    rotations = [e for e in p.elements if e.kind == "rotation"]
    first = [e for e in rotations if e.name == "0"]
    second = [e for e in rotations if e.name == "2"]
    # the first layer's input needs no gradient: x and W forward, W's gradient
    assert [e.elements for e in first] == [4 * 784, 128 * 784, 128 * 784]
    assert {e.formats for e in first} == {Hadamard(16)}
    assert [e.elements for e in second] == [4 * 128, 10 * 128, 4 * 128, 10 * 128]
    assert {e.formats for e in second} == {Hadamard(64)}


def test_a_convolution_refuses_a_rotation():
    with pytest.raises(TypeError, match="a convolution takes no rotation"):
        GemmFormats(rotation=Hadamard(32)).conv()


def test_mx_is_unrotated_by_default():
    """R-f: ``presets.mx()`` is the recipe it was, and a rotation is named in
    the formats' label."""
    plain = dict(presets.mx().rules[0].kinds)["linear"]
    assert plain.rotation is None
    assert plain == dataclasses.replace(
        dict(presets.mx(rotation=Hadamard(32)).rules[0].kinds)["linear"], rotation=None
    )
    rotated = dict(presets.mx(rotation=Hadamard(128)).rules[0].kinds)["linear"]
    assert rotated.label().endswith("rotation DH128")
    assert "rotation" not in plain.label()
    assert copy.deepcopy(rotated) == rotated
