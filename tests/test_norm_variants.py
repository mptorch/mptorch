"""Norm variants: ``NormFormats(deviation=...)`` and ``NormFormats(inverse=...)``.

The L1 and L-infinity deviations are held to a spelling written here in
torch's ops: with every slot unset (tier 1), forward and gradients; with
every slot set to a precision of its own (tier 3), forward word for word
with the steps by hand; and the closed-form backward, which a backward slot
selects, to autograd of that spelling, ties included. FISR is held word for
word to a bit-level FISR written here, in both carriers, and the table
inverse to its relative error. The approximate inverses take the exact
gradient straight through, and batch norm takes them but no deviation.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from mptorch import BinaryK
from mptorch.quant import (
    FISR,
    FusedMac,
    NormFormats,
    PWLTable,
    Quant,
    profile,
    qbatchnorm,
    qgroupnorm,
    qlayernorm,
    qrmsnorm,
)
from mptorch.quant.modules import QLayerNorm
from tests.markers import accelerator_devices, available_devices, float64_devices

EPS = 1e-5


def _fmt(p: int) -> BinaryK:
    return BinaryK(p + 6, p)


def _reference(x, deviation, rms, dims, eps=EPS):
    """The variant before its affine part, in torch's ops."""
    c = x if rms else x - x.mean(dims, keepdim=True)
    a = c.abs()
    d = a.mean(dims, keepdim=True) if deviation == "l1" else a.amax(dims, keepdim=True)
    kappa = math.sqrt(2 / math.pi) if deviation == "l1" else 1.0
    return c * (kappa / (d + eps))


def _op(kind, x, w, b, f):
    if kind == "layer":
        return qlayernorm(x, x.shape[-1], w, b, EPS, f)
    if kind == "rms":
        return qrmsnorm(x, x.shape[-1], w, EPS, f)
    return qgroupnorm(x, 2, w, b, EPS, f)


def _ref_op(kind, x, w, b, deviation):
    if kind == "group":
        t = _reference(x.reshape(x.shape[0], 2, -1), deviation, False, (2,)).reshape(x.shape)
        view = (1, -1, 1)
        return t * w.view(view) + b.view(view)
    t = _reference(x, deviation, kind == "rms", (x.dim() - 1,))
    return t * w if kind == "rms" else t * w + b


def _inputs(kind, device, dtype=torch.float32):
    gen = torch.Generator().manual_seed(2)
    shape = (3, 4, 10)
    x = torch.randn(*shape, generator=gen, dtype=dtype).to(device).requires_grad_()
    n = shape[1] if kind == "group" else shape[-1]
    w = (1 + torch.randn(n, generator=gen, dtype=dtype)).to(device).requires_grad_()
    b = torch.randn(n, generator=gen, dtype=dtype).to(device).requires_grad_()
    g = torch.randn(*shape, generator=gen, dtype=dtype).to(device)
    return x, w, b, g


def _grads(y, kind, x, w, b, g):
    inputs = (x, w) if kind == "rms" else (x, w, b)
    return torch.autograd.grad(y, inputs, g)


# --- the deviations ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["layer", "rms", "group"])
@pytest.mark.parametrize("deviation", ["l1", "linf"])
def test_tier1_a_deviation_is_its_spelling(device, kind, deviation):
    x, w, b, g = _inputs(kind, device)
    y = _op(kind, x, w, b, NormFormats(deviation=deviation))
    want = _ref_op(kind, x, w, b, deviation)
    torch.testing.assert_close(y, want, rtol=1e-6, atol=1e-6)
    for got, ref in zip(_grads(y, kind, x, w, b, g), _grads(want, kind, x, w, b, g), strict=True):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", float64_devices)
@pytest.mark.parametrize("kind", ["layer", "rms", "group"])
@pytest.mark.parametrize("deviation", ["l1", "linf"])
def test_the_closed_form_is_autograds(device, kind, deviation):
    """A backward slot selects the closed form, which in binary64 with a
    53-bit rounding is autograd's gradient of the spelling."""
    x, w, b, g = _inputs(kind, device, torch.float64)
    wide = Quant(BinaryK(60, 53))
    y = _op(kind, x, w, b, NormFormats(deviation=deviation, bwd_mul=wide))
    want = _ref_op(kind, x, w, b, deviation)
    for got, ref in zip(_grads(y, kind, x, w, b, g), _grads(want, kind, x, w, b, g), strict=True):
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("closed", [False, True])
def test_linf_splits_a_tie_as_amax_does(closed):
    x = torch.tensor([[1.0, -3.0, 3.0, 0.5, -1.0, 3.0]], dtype=torch.float64, requires_grad=True)
    g = torch.tensor([[0.3, -1.0, 2.0, 0.7, 0.1, -0.4]], dtype=torch.float64)
    f = NormFormats(deviation="linf", bwd_mul=Quant(BinaryK(60, 53)) if closed else None)
    (got,) = torch.autograd.grad(qrmsnorm(x, 6, None, EPS, f), x, g)
    (want,) = torch.autograd.grad(_reference(x, "linf", True, (1,)), x, g)
    torch.testing.assert_close(got, want, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("deviation", ["l1", "linf"])
def test_tier3_a_deviation_by_hand(device, deviation):
    """Every slot a precision of its own: the forward is the steps written
    out, each rounded by its slot, word for word."""
    q = {
        name: Quant(_fmt(p))
        for name, p in (
            ("div", 10),
            ("sub", 9),
            ("rsqrt", 11),
            ("norm", 8),
            ("mul", 12),
            ("add", 7),
        )
    }
    acc = FusedMac(_fmt(13))
    f = NormFormats(
        acc=acc,
        deviation=deviation,
        div=q["div"],
        sub=q["sub"],
        rsqrt=q["rsqrt"],
        norm=q["norm"],
        mul=q["mul"],
        add=q["add"],
    )
    x, w, b, _ = _inputs("layer", device)
    with torch.no_grad():
        y = qlayernorm(x, 10, w, b, EPS, f)
        n = x.shape[-1]
        mean = q["div"](_qsum(x, acc) / n)
        c = q["sub"](x - mean)
        a = c.abs()
        d = q["div"](_qsum(a, acc) / n) if deviation == "l1" else a.amax(-1, keepdim=True)
        kappa = math.sqrt(2 / math.pi) if deviation == "l1" else 1.0
        inv = q["rsqrt"](torch.full((), kappa, device=device) / (d + EPS))
        want = q["add"](q["mul"](q["norm"](c * inv) * w) + b)
    assert torch.equal(y, want)


def _qsum(t, mac):
    from mptorch.quant import qsum

    return qsum(t, -1, mac, keepdim=True)


def test_batch_norm_refuses_a_deviation():
    x = torch.randn(4, 3, 5)
    with pytest.raises(ValueError, match="takes no deviation 'l1'"):
        qbatchnorm(
            x, torch.zeros(3), torch.ones(3), training=True, formats=NormFormats(deviation="l1")
        )


def test_the_formats_refuse():
    with pytest.raises(ValueError, match="deviation is one of"):
        NormFormats(deviation="mad")
    with pytest.raises(ValueError, match="set one of them"):
        NormFormats(deviation="l1", inverse=FISR())
    with pytest.raises(ValueError, match="no reduction"):
        NormFormats(inverse=PWLTable.fit(torch.tanh, -1, 1, 4))
    with pytest.raises(TypeError, match="a FISR or a PWLTable"):
        NormFormats(inverse="fisr")
    with pytest.raises(ValueError, match="steps is at least 0"):
        FISR(-1)
    assert NormFormats(deviation="l1").label() == "deviation l1"
    assert "inverse FISR(1)" in NormFormats(inverse=FISR(), rsqrt=Quant(_fmt(8))).label()


# --- the inverses -----------------------------------------------------------------------------


def _fisr_by_hand(v: torch.Tensor, steps: int, q) -> torch.Tensor:
    """FISR on the word, written from the classic code."""
    if v.dtype is torch.float32:
        y = (0x5F3759DF - (v.view(torch.int32) >> 1)).view(torch.float32)
    else:
        y = (0x5FE6EB50C7B537A9 - (v.view(torch.int64) >> 1)).view(torch.float64)
    if steps == 0:
        return q(y)
    for _ in range(steps):
        y = q(y * (1.5 - (v * 0.5) * y * y))
    return y


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("steps", [0, 1, 2])
def test_fisr_is_the_bit_trick_by_hand(device, dtype, steps):
    """A layer norm under FISR rounds ``1 / sqrt(var + eps)`` as the bit
    trick and its Newton steps do, each step by the rsqrt slot; one step's
    error is the classic's 1.75e-3, two steps' about 5e-6."""
    rq = Quant(_fmt(12))
    x = torch.randn(64, 32, generator=torch.Generator().manual_seed(3), dtype=dtype).to(device)
    f = NormFormats(inverse=FISR(steps), rsqrt=rq)
    with torch.no_grad():
        y = qlayernorm(x, 32, formats=f)
        c = x - x.mean(-1, keepdim=True)
        var = (c * c).mean(-1, keepdim=True)
        want = c * _fisr_by_hand(var + EPS, steps, rq)
    torch.testing.assert_close(y, want, rtol=1e-6, atol=1e-6)
    v = torch.linspace(1, 4, 100001, dtype=dtype).to(device)
    approx = _fisr_by_hand(v, steps, lambda t: t).cpu().double()  # MPS has no float64
    err = (approx * v.cpu().double().sqrt() - 1).abs().max().item()
    assert err <= {0: 3.5e-2, 1: 1.76e-3, 2: 4.8e-6}[steps]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["layer", "rms", "group"])
@pytest.mark.parametrize("inverse", [FISR(1), FISR(2), "table"])
def test_an_inverse_takes_the_exact_gradient(device, kind, inverse):
    """Without a backward slot, the gradient is torch's norm's, word for word:
    the approximation is straight through."""
    inv = PWLTable.rsqrt() if inverse == "table" else inverse
    x, w, b, g = _inputs(kind, device)
    y = _op(kind, x, w, b, NormFormats(inverse=inv))
    plain = {
        "layer": lambda: F.layer_norm(x, (10,), w, b, EPS),
        "rms": lambda: F.rms_norm(x, (10,), w, EPS),
        "group": lambda: F.group_norm(x, 2, w, b, EPS),
    }[kind]()
    rel = ((y - plain).abs().max() / plain.abs().max()).item()
    assert rel < {"table": 1e-3}.get(inverse, 2e-3) if isinstance(inverse, str) else rel < 2e-3
    for got, ref in zip(_grads(y, kind, x, w, b, g), _grads(plain, kind, x, w, b, g), strict=True):
        assert torch.equal(got, ref)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("training", [True, False])
def test_batch_norm_takes_an_inverse(device, training):
    """Under FISR, a batch norm's running statistics move as the exact
    norm's, and its output is within FISR's error."""
    x = torch.randn(8, 3, 6, generator=torch.Generator().manual_seed(4)).to(device)
    stats = [torch.zeros(3, device=device), torch.ones(3, device=device)]
    exact = [t.clone() for t in stats]
    y = qbatchnorm(x, *stats, training=training, formats=NormFormats(inverse=FISR(2)))
    want = qbatchnorm(
        x, *exact, training=training, formats=NormFormats(rsqrt=Quant(BinaryK(30, 24)))
    )
    torch.testing.assert_close(y, want, rtol=1e-5, atol=1e-5)
    for a, b in zip(stats, exact, strict=True):
        assert torch.equal(a, b)


_ACC = FusedMac(_fmt(20))


@pytest.mark.parametrize("device", accelerator_devices)
@pytest.mark.parametrize(
    "formats",
    [
        NormFormats(acc=_ACC, deviation="l1", div=Quant(_fmt(10))),
        NormFormats(acc=_ACC, deviation="linf"),
        NormFormats(acc=_ACC, inverse=FISR()),
        NormFormats(acc=_ACC, inverse=PWLTable.rsqrt()),
    ],
)
def test_every_device_normalizes_as_the_cpu(device, formats):
    """With the sums on mptorch's GEMM core (``acc``), whose order is fixed,
    every device gives the CPU's words: the maximum, FISR's arithmetic on the
    word and the table are the same everywhere. (torch's own sum is not:
    its order is each device's.)"""
    x = torch.randn(5, 7, 24, generator=torch.Generator().manual_seed(5))
    got = qlayernorm(x.to(device), 24, formats=formats).cpu()
    assert torch.equal(got, qlayernorm(x, 24, formats=formats))


def test_profile_counts_a_deviations_sums():
    """An L1 deviation sums the absolute deviations where the standard one
    sums the squares; an L-infinity deviation is a maximum and sums nothing."""
    x = torch.randn(4, 16, requires_grad=True)

    def sums(f):
        p = profile(QLayerNorm(16, formats=f), x, backward=True)
        return [(r.step, r.site) for r in p.records]

    assert sums(NormFormats(deviation="l1")) == [
        ("fwd", "mean"),
        ("fwd", "mean absolute deviation"),
        ("igrad", "input"),
        ("igrad", "input"),
        ("wgrad", "weight"),
        ("wgrad", "bias"),
    ]
    assert sums(NormFormats(deviation="linf"))[:2] == [("fwd", "mean"), ("igrad", "input")]
