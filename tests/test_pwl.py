"""Piecewise-linear tables: the fits, the evaluation, and the activations that take them.

The fits are held to what they promise: the chord is the function at the
knots, the balanced fit is LoFloat's formula written out here, and the
minimax fit's largest error on its grid is the smallest any entries reach
(bisection certifies it, and moving one entry does not lower it). The
evaluation is held to the documented steps, written out here, on every
device. An activation with a table follows the three tiers: close to the
function (tier 1), and word for word with the table's steps, its rounding
and the activation's own rounded derivative (tier 3).
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import (
    ActivationFormats,
    PWLTable,
    QPWLActivation,
    Quant,
    Recipe,
    convert,
    gelu_table,
    qgelu,
    qpwl,
    qsilu,
    silu_table,
    simulate,
)
from mptorch.quant.functional.pwl import _evaluate, _fit_grid, _grid_error, _minimax_entries
from tests.markers import accelerator_devices, available_devices


def _silu64(x):
    return x * torch.sigmoid(x)


def _dense_error(table: PWLTable, fn, relative=False, points=512) -> float:
    """The table's largest error on ``points`` points a segment, by hand."""
    x = torch.linspace(table.lo, table.hi, table.segments * points + 1, dtype=torch.float64)
    u = (x - table.lo) * (table.segments / (table.hi - table.lo))
    i = u.floor().clamp(0, table.segments - 1)
    e = torch.tensor(table.entries, dtype=torch.float64)
    y = e[i.long()] + (u - i) * (e[i.long() + 1] - e[i.long()])
    r = y - fn(x)
    return float((r / fn(x).abs() if relative else r).abs().max())


# --- the fits ------------------------------------------------------------------------------


def test_the_chord_is_the_function_at_the_knots():
    t = PWLTable.fit(_silu64, -8, 8, 16, fit="chord")
    knots = -8 + torch.arange(17, dtype=torch.float64)
    assert t.entries == tuple(_silu64(knots).tolist())


def test_the_balanced_fit_is_lofloats():
    """LoFloat's init_lut (its 'minimax'): each entry moved by its segments'
    midpoint errors, halved at the ends and quartered between."""
    t = PWLTable.fit(_silu64, -8, 8, 16, fit="balanced")
    knots = -8 + torch.arange(17, dtype=torch.float64)
    f = _silu64(knots)
    mids = _silu64((knots[:-1] + knots[1:]) / 2)
    delta = (mids - (f[:-1] + f[1:]) / 2).tolist()
    want = list(f.tolist())
    want[0] += delta[0] / 2
    want[-1] += delta[-1] / 2
    for k in range(1, 16):
        want[k] += (delta[k - 1] + delta[k]) / 4
    assert t.entries == pytest.approx(want, rel=0, abs=1e-15)


@pytest.mark.parametrize(
    ("fn", "lo", "hi", "error"),
    [
        (_silu64, -8.0, 8.0, "absolute"),
        (torch.tanh, -3.0, 3.0, "absolute"),
        (lambda v: 1 / v.sqrt(), 1.0, 4.0, "relative"),
    ],
)
@pytest.mark.parametrize("segments", [4, 16, 64])
def test_minimax_beats_balanced_beats_chord(fn, lo, hi, error, segments):
    rel = error == "relative"
    errors = {
        fit: _dense_error(PWLTable.fit(fn, lo, hi, segments, fit=fit, error=error), fn, rel)
        for fit in ("minimax", "balanced", "chord")
    }
    assert errors["minimax"] <= errors["balanced"] <= errors["chord"], errors


@pytest.mark.parametrize("segments", [3, 16, 40])
def test_minimax_is_optimal_on_its_grid(segments):
    """Bisection proves no entries beat the fit's error by more than its
    tolerance, and moving any one entry either way raises the error."""
    t, y, y_end = _fit_grid(_silu64, -8.0, 8.0, segments, 64)
    entries, proven = _minimax_entries(_silu64, -8.0, 8.0, segments, 64, False)
    best = _grid_error(t, y, y_end, entries, False)
    assert proven <= best <= proven * (1 + 1e-9)
    delta = best * 1e-4
    for k in range(segments + 1):
        for step in (delta, -delta):
            moved = entries.clone()
            moved[k] += step
            assert _grid_error(t, y, y_end, moved, False) >= best * (1 - 1e-12)


def test_entry_quant_rounds_the_entries():
    q = Quant(BinaryK(8, 4))
    t = silu_table(entry_quant=lambda e: q(e))
    assert all(q(torch.tensor([e], dtype=torch.float64)).item() == e for e in t.entries)
    plain = silu_table().fit_error
    assert t.fit_error is not None and plain is not None and t.fit_error > plain


def test_the_refusals():
    with pytest.raises(ValueError, match="segments is an int"):
        PWLTable.fit(_silu64, -1, 1, 0)
    with pytest.raises(ValueError, match="fit is one of"):
        PWLTable.fit(_silu64, -1, 1, 4, fit="remez")
    with pytest.raises(ValueError, match="lo < hi"):
        PWLTable(1.0, 1.0, (0.0, 1.0))
    with pytest.raises(TypeError, match="tails is"):
        PWLTable(0.0, 1.0, (0.0, 1.0), tails="wrap")
    with pytest.raises(ValueError, match="divides by the function's values"):
        PWLTable.fit(_silu64, -1, 1, 4, error="relative")


# --- evaluation ----------------------------------------------------------------------------


def _by_hand(table: PWLTable, x: torch.Tensor) -> torch.Tensor:
    """The documented steps, in ``x``'s dtype."""
    e = torch.tensor(table.entries, dtype=x.dtype)
    xi = x.clamp(table.lo, table.hi) if table.tails == "clamp" else x
    u = (xi - table.lo) * (table.segments / (table.hi - table.lo))
    i = torch.floor(torch.nan_to_num(u, nan=0.0)).clamp(0, table.segments - 1)
    y = e[i.long()] + (u - i) * (e[i.long() + 1] - e[i.long()])
    if callable(table.tails):
        y = torch.where((x < table.lo) | (x > table.hi), table.tails(x), y)
    return y


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("tails", ["extend", "clamp", torch.relu])
def test_evaluation_is_the_documented_steps(dtype, tails):
    t = PWLTable.fit(_silu64, -4, 4, 12, tails=tails)
    x = torch.cat([torch.linspace(-9, 9, 1001), torch.tensor([float("nan"), -4.0, 4.0])]).to(dtype)
    got, want = _evaluate(t, x), _by_hand(t, x)
    assert torch.equal(got.isnan(), want.isnan())
    assert torch.equal(got[~got.isnan()], want[~want.isnan()])
    if tails == "clamp":
        assert got[0].item() == pytest.approx(t.entries[0])
    if tails == "extend":  # beyond hi, the last segment's line
        e = t.entries
        slope = (e[-1] - e[-2]) / (8 / 12)
        assert got[-4].item() == pytest.approx(e[-1] + slope * 5, rel=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_rsqrt_table_reduces_exactly(dtype):
    """``v = m * 4**k`` scales exactly: the table at ``m * 4**k`` is its value
    at ``m`` times ``2**-k``, and its relative error holds over 60 decades."""
    t = PWLTable.rsqrt()
    m = torch.linspace(1, 4, 999, dtype=dtype)[:-1]
    at_m = _evaluate(t, m)
    for k in (-20, -3, 0, 1, 7, 20):
        assert torch.equal(_evaluate(t, m * 4.0**k), at_m * 2.0**-k)
    v = torch.logspace(-30, 30, 10001, dtype=dtype)
    rel = (_evaluate(t, v).double() * v.double().sqrt() - 1).abs().max().item()
    assert t.fit_error is not None
    assert rel <= t.fit_error * 1.01 + (4 * 2**-24 if dtype is torch.float32 else 1e-14)


@pytest.mark.parametrize("device", accelerator_devices)
def test_every_device_evaluates_as_the_cpu(device):
    for t in (silu_table(), PWLTable.fit(torch.tanh, -3, 3, 20, tails="clamp"), PWLTable.rsqrt()):
        lo = 1e-3 if t.reduction else -10.0
        x = torch.linspace(lo, 10, 4001)
        assert torch.equal(_evaluate(t, x.to(device)).cpu(), _evaluate(t, x))


# --- activations ----------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_tier1_a_fine_table_is_the_activation(device):
    """A fine table is the activation to its own fit error, plus the float32
    evaluation's roundings."""
    x = torch.linspace(-12, 12, 2001, device=device)
    for table, op, ref in (
        (silu_table(1024, -12.0, 12.0), qsilu, F.silu),
        (gelu_table(1024, -12.0, 12.0), qgelu, F.gelu),
    ):
        assert table.fit_error is not None and table.fit_error < 3e-5
        bound = table.fit_error * 1.01 + 2e-6
        torch.testing.assert_close(
            op(x, ActivationFormats(table=table)), ref(x), rtol=0, atol=bound
        )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("act", ["silu", "gelu"])
def test_tier3_a_table_activation_by_hand(device, act):
    """The table's steps in the carrier, rounded by ``out``; the gradient the
    activation's own derivative, rounded by ``grad``."""
    table = (silu_table if act == "silu" else gelu_table)(16)
    out, grad = Quant(BinaryK(14, 8)), Quant(BinaryK(12, 6))
    f = ActivationFormats(table=table, out=out, grad=grad)
    op = qsilu if act == "silu" else qgelu
    x = torch.linspace(-10, 10, 401, device=device).requires_grad_()
    g = torch.randn(401, generator=torch.Generator().manual_seed(1)).to(device)
    y = op(x, f)
    (gx,) = torch.autograd.grad(y, x, g)
    with torch.no_grad():
        assert torch.equal(y, out(_by_hand(table, x.cpu()).to(device)))
        backward = torch.ops.aten.silu_backward if act == "silu" else torch.ops.aten.gelu_backward
        assert torch.equal(gx, grad(backward(g, x)))


def test_an_activation_refuses_another_ones_table():
    x = torch.randn(4)
    with pytest.raises(ValueError, match="a table of silu"):
        qgelu(x, ActivationFormats(table=silu_table()))
    with pytest.raises(ValueError, match="no gate to round"):
        ActivationFormats(table=silu_table(), inner=BinaryK(8, 4))
    with pytest.raises(ValueError, match="no reduction"):
        ActivationFormats(table=PWLTable.rsqrt())
    with pytest.raises(TypeError, match="a PWLTable or None"):
        ActivationFormats(table=(0.0, 1.0))
    mine = PWLTable.fit(_silu64, -8, 8, 16)  # no name: allowed anywhere
    assert qgelu(x, ActivationFormats(table=mine)).shape == x.shape
    assert ActivationFormats(table=silu_table()).label().startswith("silu PWL[16")


@pytest.mark.parametrize("device", available_devices)
def test_qpwl_on_a_function_of_your_own(device):
    """The backward is the derivative of the table's function, by autograd
    or as given; the module is the op."""
    t = PWLTable.fit(torch.tanh, -4, 4, 32, tails="clamp")
    x = torch.linspace(-5, 5, 101, device=device, requires_grad=True)
    y = qpwl(x, t)
    (gx,) = torch.autograd.grad(y.sum(), x)
    torch.testing.assert_close(gx, 1 - torch.tanh(x.detach()) ** 2)
    (gd,) = torch.autograd.grad(qpwl(x, t, derivative=lambda g, v: 2 * g).sum(), x)
    assert torch.equal(gd, torch.full_like(gd, 2.0))
    m = QPWLActivation(t, quant=Quant(BinaryK(14, 8)))
    assert torch.equal(m(x), qpwl(x, t, quant=Quant(BinaryK(14, 8))))
    bare = PWLTable(-1.0, 1.0, (-1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="neither fn nor derivative"):
        qpwl(x, bare).sum().backward()
    with pytest.raises(TypeError, match="pass Quant"):
        qpwl(x, t, quant=BinaryK(8, 4))


@pytest.mark.parametrize("device", available_devices)
def test_convert_and_simulate_take_a_table(device):
    model = nn.Sequential(nn.Linear(8, 8), nn.SiLU()).to(device)
    af = ActivationFormats(table=silu_table(), out=Quant(BinaryK(14, 8)))
    recipe = Recipe(activation=af)
    x = torch.randn(3, 8, device=device)
    converted, _ = convert(model, recipe)
    want = qsilu(model[0](x), af)
    assert torch.equal(converted(x), want)
    with simulate(recipe, model):
        assert torch.equal(model(x), want)


@pytest.mark.parametrize("device", available_devices)
def test_a_table_activation_compiles_to_eagers_words(device):
    torch._dynamo.reset()
    f = ActivationFormats(table=silu_table(), out=Quant(BinaryK(14, 8)), grad=Quant(BinaryK(12, 6)))
    x = torch.linspace(-10, 10, 64, device=device, requires_grad=True)
    y = qsilu(x, f)
    (gx,) = torch.autograd.grad(y.sum(), x)
    x2 = x.detach().clone().requires_grad_()
    y2 = torch.compile(lambda t: qsilu(t, f), backend="aot_eager", fullgraph=True)(x2)
    (gx2,) = torch.autograd.grad(y2.sum(), x2)
    assert torch.equal(y2, y)
    assert torch.equal(gx2, gx)


def test_the_presets_are_lofloats_layout():
    t = silu_table()
    assert (t.lo, t.hi, t.segments, t.tails, t.name) == (-8.0, 8.0, 16, torch.relu, "silu")
    assert math.isclose(t.fit_error or 0, t.max_error(points=64), rel_tol=1e-12)
