"""Softmax and the norms, each step rounded as their formats say.

Tier 1: with every slot ``None`` each op is torch's function, forward and
backward, word for word, and with forward slots only its gradients are
autograd's of torch's function, word for word. Tier 2: a real format comes out
close to float. Tier 3: the forward is the composition of its steps, each
rounded by its own slot and the sums ``qsum``'s, and with backward slots the
gradient is the closed form, each step rounded, word for word. Every slot
holds a format of its own precision, so a step wired to another slot's
rounding changes the words. Then the modules and their running statistics,
compile and MPS.
"""

import math
from collections.abc import Callable
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo.testing import CompileCounterWithBackend

from mptorch import BinaryK, RoundMode
from mptorch.number import MXFP8_E4M3
from mptorch.quant import (
    FusedMac,
    NormFormats,
    QBatchNorm1d,
    QBatchNorm2d,
    QGroupNorm,
    QLayerNorm,
    QRMSNorm,
    QSoftmax,
    Quant,
    SoftmaxFormats,
    SplitMac,
    qbatchnorm,
    qgroupnorm,
    qlayernorm,
    qrmsnorm,
    qsoftmax,
    qsum,
)
from tests.markers import available_devices, requires_mps

R = RoundMode
P8 = BinaryK(8, 4)
P16 = BinaryK(16, 11)


def _q(P):
    return Quant(BinaryK(P + 5, P))


# every slot its own precision
FWD: dict[str, Any] = dict(
    acc=FusedMac(BinaryK(16, 10)),
    div=_q(9),
    sub=_q(8),
    sq=_q(7),
    rsqrt=_q(6),
    norm=_q(10),
    mul=_q(5),
    add=_q(11),
    out=_q(4),
)
BWD: dict[str, Any] = dict(bwd_acc=FusedMac(BinaryK(16, 9)), bwd_mul=_q(7), bwd_out=_q(6))
SFWD: dict[str, Any] = dict(sub=_q(9), exp=_q(8), acc=FusedMac(BinaryK(16, 10)), div=_q(6))
SBWD: dict[str, Any] = dict(
    bwd_mul=_q(7), bwd_acc=FusedMac(BinaryK(16, 9)), bwd_sub=_q(10), bwd_out=_q(5)
)
EPS = 1e-5


def _x(device, shape, dtype=torch.float32, seed=0, positive=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    x = x.abs() + 0.5 if positive else x
    return x.to(device=device, dtype=dtype)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


def _grads(out, inputs, seed=1):
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.autograd.grad(out, inputs, g.to(out), allow_unused=True, retain_graph=True)


def _case(
    name, device, dtype=torch.float32
) -> tuple[tuple[torch.Tensor, ...], Callable[..., Any], Callable[..., Any]]:
    """An op's inputs, torch's call over them and mptorch's, taking formats."""
    if name == "softmax":
        x = _x(device, (6, 16), dtype)
        return (x,), (lambda x: F.softmax(x, -1)), (lambda f, x: qsoftmax(x, -1, f))
    if name in ("layer_norm", "rms_norm"):
        x = _x(device, (2, 3, 16), dtype)
        w, b = _x(device, (16,), dtype, 1), _x(device, (16,), dtype, 2)
        if name == "layer_norm":
            return (
                (x, w, b),
                lambda x, w, b: F.layer_norm(x, (16,), w, b, EPS),
                lambda f, x, w, b: qlayernorm(x, (16,), w, b, EPS, f),
            )
        return (
            (x, w),
            lambda x, w: F.rms_norm(x, (16,), w, EPS),
            lambda f, x, w: qrmsnorm(x, 16, w, EPS, f),
        )
    x = _x(device, (4, 8, 3, 3), dtype)
    w, b = _x(device, (8,), dtype, 1), _x(device, (8,), dtype, 2)
    if name == "group_norm":
        return (
            (x, w, b),
            lambda x, w, b: F.group_norm(x, 4, w, b, EPS),
            lambda f, x, w, b: qgroupnorm(x, 4, w, b, EPS, f),
        )
    if name == "batch_norm":
        return (
            (x, w, b),
            lambda x, w, b: F.batch_norm(x, None, None, w, b, True, 0.1, EPS),
            lambda f, x, w, b: qbatchnorm(x, None, None, w, b, True, 0.1, EPS, f),
        )
    rm, rv = _x(device, (8,), dtype, 3), _x(device, (8,), dtype, 4, positive=True)
    return (
        (x, w, b),
        lambda x, w, b: F.batch_norm(x, rm, rv, w, b, False, 0.1, EPS),
        lambda f, x, w, b: qbatchnorm(x, rm, rv, w, b, False, 0.1, EPS, f),
    )


OPS = ["softmax", "layer_norm", "rms_norm", "group_norm", "batch_norm", "batch_norm_eval"]


def _formats(name, forward=True, backward=False):
    if name == "softmax":
        return SoftmaxFormats(**(SFWD if forward else {}), **(SBWD if backward else {}))
    return NormFormats(**(FWD if forward else {}), **(BWD if backward else {}))


# --- tier 1 ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("name", OPS)
def test_without_slots_each_is_torchs(device, dtype, name):
    if device == "mps" and dtype is torch.float64:
        pytest.skip("MPS has no float64")
    inputs, plain, ours = _case(name, device, dtype)
    inputs = tuple(t.requires_grad_() for t in inputs)
    want = plain(*inputs)
    for f in (None, _formats(name, forward=False)):
        got = ours(f, *inputs)
        assert _same(got, want)
        for g, w in zip(_grads(got, inputs), _grads(want, inputs), strict=True):
            assert _same(g, w)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", OPS)
def test_forward_slots_alone_keep_autograds_gradients(device, dtype, name):
    """The gradient goes to a forward of torch's function, so it is autograd's."""
    inputs, plain, ours = _case(name, device, dtype)
    inputs = tuple(t.requires_grad_() for t in inputs)
    got, want = ours(_formats(name), *inputs), plain(*inputs)
    assert not _same(got, want)
    for g, w in zip(_grads(got, inputs), _grads(want, inputs), strict=True):
        assert _same(g, w)


# --- tier 2 ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", OPS)
def test_a_real_format_comes_out_close(device, name):
    inputs, plain, ours = _case(name, device)
    mac = SplitMac(P16, P16)
    f = (SoftmaxFormats if name == "softmax" else NormFormats).uniform(mac, Quant(P16))
    inputs = tuple(t.requires_grad_() for t in inputs)
    got, want = ours(f, *inputs), plain(*inputs)
    assert torch.allclose(got, want, rtol=0.02, atol=0.02)
    for g, w in zip(_grads(got, inputs), _grads(want, inputs), strict=True):
        assert torch.allclose(g, w, rtol=0.05, atol=0.05)


# --- tier 3: the forward --------------------------------------------------------------------


def _r(q, t):
    return t if q is None else q(t)


def _over(t, n):
    return t / torch.full((), n, dtype=t.dtype, device=t.device)


def _statistics(xc, dims, f, rms):
    """The forward's steps up to x_hat, written out."""
    n = math.prod(xc.shape[d] for d in dims)

    def total(t):
        return qsum(t, dims, f.acc, keepdim=True)

    if rms:
        centred = xc
        var = _r(f.div, _over(total(_r(f.sq, xc * xc)), n))
    else:
        mean = _r(f.div, _over(total(xc), n))
        centred = _r(f.sub, xc - mean)
        var = _r(f.div, _over(total(_r(f.sq, centred * centred)), n))
    return _r(f.norm, centred * _r(f.rsqrt, torch.rsqrt(var + EPS)))


def _forward_by_hand(name, f, x, w=None, b=None):
    xc = x.float()
    view = (1, -1, 1, 1)
    if name == "softmax":
        e = _r(f.exp, torch.exp(_r(f.sub, xc - xc.amax(-1, keepdim=True))))
        return _r(f.div, e / qsum(e, -1, f.acc, keepdim=True)).to(x.dtype)
    assert w is not None
    if name in ("layer_norm", "rms_norm"):
        t = _statistics(xc, (2,), f, name == "rms_norm")
        t = _r(f.mul, t * w.float())
        if b is not None:
            t = _r(f.add, t + b.float())
        return _r(f.out, t).to(x.dtype)
    if name == "group_norm":
        t = _statistics(xc.reshape(4, 4, -1), (2,), f, False).reshape(x.shape)
    elif name == "batch_norm":
        t = _statistics(xc, (0, 2, 3), f, False)
    else:
        rm, rv = _x(x.device, (8,), x.dtype, 3), _x(x.device, (8,), x.dtype, 4, positive=True)
        centred = _r(f.sub, xc - rm.float().view(view))
        t = _r(f.norm, centred * _r(f.rsqrt, torch.rsqrt(rv.float().view(view) + EPS)))
    assert b is not None
    t = _r(f.add, _r(f.mul, t * w.float().view(view)) + b.float().view(view))
    return _r(f.out, t).to(x.dtype)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", OPS)
def test_tier3_the_forward_is_its_steps(device, dtype, name):
    inputs, _, ours = _case(name, device, dtype)
    f = _formats(name)
    assert _same(ours(f, *inputs), _forward_by_hand(name, f, *inputs))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", ["softmax", "layer_norm", "rms_norm", "group_norm"])
def test_tier3_only_the_result_rounded_is_torchs_function(device, name):
    """With every slot before the result's ``None``, the op is torch's
    function in the carrier, rounded once."""
    inputs, plain, ours = _case(name, device, torch.bfloat16)
    q = _q(6)
    f = SoftmaxFormats(div=q) if name == "softmax" else NormFormats(out=q)
    want = q(plain(*[t.float() for t in inputs])).bfloat16()
    assert _same(ours(f, *inputs), want)


# --- tier 3: the closed-form backward ---------------------------------------------------------


def _norm_grads_by_hand(name, f, g, x, w, b):
    """The closed forms, from the unrounded statistics, each step rounded."""
    xc, gc = x.float(), g.float()
    view = (1, -1, 1, 1)
    rms = name == "rms_norm"
    if name in ("layer_norm", "rms_norm"):
        dims, batch, layout, pview = (2,), (0, 1), (lambda t: t), None
    elif name == "group_norm":
        dims, batch, layout, pview = (2,), (0, 2, 3), (lambda t: t.reshape(4, 4, -1)), view
    else:
        dims, batch, layout, pview = (0, 2, 3), (0, 2, 3), (lambda t: t), view
    xl = layout(xc)
    if name == "batch_norm_eval":
        rm, rv = _x(x.device, (8,), x.dtype, 3), _x(x.device, (8,), x.dtype, 4, positive=True)
        rstd = torch.rsqrt(rv.float().view(view) + EPS)
        xhat = (xc - rm.float().view(view)) * rstd
    else:
        if rms:
            var, centred = (xl * xl).mean(dims, keepdim=True), xl
        else:
            mean = xl.mean(dims, keepdim=True)
            centred = xl - mean
            var = (centred * centred).mean(dims, keepdim=True)
        rstd = torch.rsqrt(var + EPS)
        xhat = centred * rstd
    wc = w.float() if pview is None else w.float().view(pview)
    dyhat = _r(f.bwd_mul, gc * wc)
    if name == "batch_norm_eval":
        gx = dyhat * rstd
        xhat_x = xhat
    else:
        dl, n = layout(dyhat), math.prod(xl.shape[d] for d in dims)
        bsum = _over(qsum(_r(f.bwd_mul, dl * xhat), dims, f.bwd_acc, keepdim=True), n)
        centred_g = dl - xhat * bsum
        if not rms:
            centred_g = centred_g - _over(qsum(dl, dims, f.bwd_acc, keepdim=True), n)
        gx = (rstd * centred_g).reshape(x.shape)
        xhat_x = xhat.reshape(x.shape)
    gw = _r(
        f.bwd_out, qsum(_r(f.bwd_mul, gc * xhat_x), batch, f.bwd_acc, keepdim=True).reshape(w.shape)
    )
    gb = (
        None
        if b is None
        else _r(f.bwd_out, qsum(gc, batch, f.bwd_acc, keepdim=True).reshape(b.shape))
    )
    return (_r(f.bwd_out, gx).to(x.dtype), gw.to(w.dtype), None if gb is None else gb.to(b.dtype))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("name", OPS)
def test_tier3_the_backward_is_the_closed_form(device, name):
    inputs, _, ours = _case(name, device)
    inputs = tuple(t.requires_grad_() for t in inputs)
    f = _formats(name, backward=True)
    out = ours(f, *inputs)
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(1)).to(device)
    got = torch.autograd.grad(out, inputs, g)
    x = inputs[0].detach()
    if name == "softmax":
        y = torch.softmax(x, -1)
        s = qsum(_r(f.bwd_mul, g * y), -1, f.bwd_acc, keepdim=True)
        want = (_r(f.bwd_out, y * _r(f.bwd_sub, g - s)),)
    else:
        params = [t.detach() for t in inputs[1:]]
        w, b = params[0], params[1] if len(params) > 1 else None
        want = _norm_grads_by_hand(name, f, g, x, w, b)[: len(inputs)]
    for gg, ww in zip(got, want, strict=True):
        assert _same(gg, ww), name


# --- softmax's dtype, the modules ---------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_softmax_converts_first(device):
    x = _x(device, (4, 8), torch.bfloat16)
    f = SoftmaxFormats(**SFWD)
    got = qsoftmax(x, -1, f, dtype=torch.float32)
    assert got.dtype is torch.float32 and _same(got, qsoftmax(x.float(), -1, f))
    assert _same(qsoftmax(x, -1, dtype=torch.float32), F.softmax(x, -1, dtype=torch.float32))


def _modules():
    """torch's module, mptorch's built with the same arguments (both start from
    torch's parameters), formats and an input shape."""
    return [
        (
            lambda: nn.Softmax(-1),
            lambda f: QSoftmax(-1, formats=f),
            SoftmaxFormats(**SFWD),
            (5, 16),
        ),
        (
            lambda: nn.LayerNorm(16),
            lambda f: QLayerNorm(16, formats=f),
            NormFormats(**FWD),
            (5, 16),
        ),
        (lambda: nn.RMSNorm(16), lambda f: QRMSNorm(16, formats=f), NormFormats(**FWD), (5, 16)),
        (
            lambda: nn.GroupNorm(4, 8),
            lambda f: QGroupNorm(4, 8, formats=f),
            NormFormats(**FWD),
            (4, 8, 3, 3),
        ),
        (
            lambda: nn.BatchNorm2d(8),
            lambda f: QBatchNorm2d(8, formats=f),
            NormFormats(**FWD),
            (4, 8, 3, 3),
        ),
        (
            lambda: nn.BatchNorm1d(8),
            lambda f: QBatchNorm1d(8, formats=f),
            NormFormats(**FWD),
            (4, 8, 3),
        ),
    ]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("i", range(6))
def test_modules_without_formats_are_torchs(device, i):
    torchs, ours, _, shape = _modules()[i]
    ref, q = torchs().to(device), ours(None).to(device)
    x = _x(device, shape).requires_grad_()
    got, want = q(x), ref(x)
    assert _same(got, want) and _same(_grads(got, (x,))[0], _grads(want, (x,))[0])
    assert list(q.state_dict()) == list(ref.state_dict())
    ours_state, torchs_state = q.state_dict().values(), ref.state_dict().values()
    assert all(torch.equal(a, b) for a, b in zip(ours_state, torchs_state, strict=True))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("i", range(6))
def test_modules_run_their_op(device, i):
    torchs, ours, f, shape = _modules()[i]
    ref, q = torchs().to(device), ours(f).to(device)
    x = _x(device, shape)
    got = q(x)
    if isinstance(ref, nn.Softmax):
        want = qsoftmax(x, -1, f)
    elif isinstance(ref, nn.LayerNorm):
        want = qlayernorm(x, ref.normalized_shape, ref.weight, ref.bias, ref.eps, f)
    elif isinstance(ref, nn.RMSNorm):
        want = qrmsnorm(x, ref.normalized_shape, ref.weight, ref.eps, f)
    elif isinstance(ref, nn.GroupNorm):
        want = qgroupnorm(x, 4, ref.weight, ref.bias, ref.eps, f)
    else:
        rm, rv = ref.running_mean, ref.running_var
        want = qbatchnorm(x, rm, rv, ref.weight, ref.bias, True, 0.1, ref.eps, f)
        assert torch.equal(q.running_mean, rm) and torch.equal(q.running_var, rv)
        assert int(q.num_batches_tracked) == 1
    assert _same(got, want)


@pytest.mark.parametrize("device", available_devices)
def test_batch_norm_running_statistics(device):
    """In training the buffers follow torch's formula from the rounded mean and
    variance; momentum None is a cumulative average; in eval they are used."""
    f = NormFormats(**FWD)
    bn = QBatchNorm2d(8, momentum=None, formats=f).to(device)
    ref = nn.BatchNorm2d(8, momentum=None).to(device)
    for seed in range(3):
        x = _x(device, (4, 8, 3, 3), seed=seed)
        bn(x)
        ref(x)
    assert bn.num_batches_tracked is not None and int(bn.num_batches_tracked) == 3
    rm, rv, ref_rm, ref_rv = bn.running_mean, bn.running_var, ref.running_mean, ref.running_var
    assert rm is not None and rv is not None and ref_rm is not None and ref_rv is not None
    assert torch.allclose(rm, ref_rm, atol=0.05) and torch.allclose(rv, ref_rv, rtol=0.1)
    bn.eval()
    x = _x(device, (4, 8, 3, 3), seed=9)
    want = qbatchnorm(x, bn.running_mean, bn.running_var, bn.weight, bn.bias, False, 0.0, bn.eps, f)
    assert _same(bn(x), want)
    untracked = QBatchNorm2d(8, track_running_stats=False, formats=f).to(device).eval()
    assert untracked.running_mean is None and untracked(x).shape == x.shape


# --- what is refused ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call,error,match",
    [
        (
            lambda x: qlayernorm(x, 16, None, None, EPS, SoftmaxFormats()),  # ty: ignore[invalid-argument-type]
            TypeError,
            "NormFormats",
        ),
        (lambda x: NormFormats(acc=FusedMac(P8, carrier=torch.float64)), ValueError, "carrier"),
        (lambda x: NormFormats(out=Quant(P8, carrier=torch.float64)), ValueError, "carrier"),
        (lambda x: NormFormats(acc=MXFP8_E4M3), TypeError, "block format"),
        (
            lambda x: qgroupnorm(x.reshape(4, 4, 4), 3, None, None, EPS, NormFormats(**FWD)),
            ValueError,
            "groups",
        ),
        (
            lambda x: qbatchnorm(
                x.reshape(4, 4, 4), None, None, None, None, False, 0.1, EPS, NormFormats(**FWD)
            ),
            ValueError,
            "running",
        ),
        (
            lambda x: qbatchnorm(
                x[:1, :4].reshape(1, 4), None, None, None, None, True, 0.1, EPS, NormFormats(**FWD)
            ),
            ValueError,
            "more than 1 value",
        ),
    ],
)
def test_what_is_refused(call, error, match):
    with pytest.raises(error, match=match):
        call(_x("cpu", (4, 16)))


def test_labels_and_uniform():
    f = NormFormats.uniform(P16, Quant(P8))
    assert f.acc == FusedMac(P16) and f.out == Quant(P8) and f.bwd_acc == FusedMac(P16)
    assert f.label().startswith("mac FusedMac") and NormFormats().label() == "torch"
    g = NormFormats.uniform(P16, Quant(P8), grad=None)
    assert g.bwd_acc is None and g.bwd_mul is None and not g._backward
    s = SoftmaxFormats.uniform(P16, Quant(P8))
    assert s._forward and s._backward and s.acc == FusedMac(P16)


# --- compile, and MPS ----------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("name", OPS)
def test_compiles_in_one_graph(device, backward, name):
    if device == "cuda" and name == "batch_norm" and not backward:
        pytest.skip("torch's own CUDA batch norm gradient differs once compiled (decomposed)")
    inputs, _, ours = _case(name, device)
    inputs = tuple(t.requires_grad_() for t in inputs)
    f = _formats(name, backward=backward)

    def fn(*ts):
        return ours(f, *ts)

    want = fn(*inputs)
    gw = _grads(want, inputs)
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    got = torch.compile(fn, backend=counter, fullgraph=True)(*inputs)
    assert counter.frame_count == 1
    assert _same(got, want)
    for g, w in zip(_grads(got, inputs), gw, strict=True):
        assert _same(g, w)


def test_inductor():
    inputs, _, ours = _case("layer_norm", "cpu")
    f = _formats("layer_norm")
    torch._dynamo.reset()
    got = torch.compile(lambda *t: ours(f, *t), backend="inductor", fullgraph=True)(*inputs)
    assert torch.allclose(got, ours(f, *inputs), atol=0.1)


@requires_mps
@pytest.mark.parametrize("name", OPS)
def test_mps_is_the_cpu(name):
    cpu, _, ours_cpu = _case(name, "cpu")
    mps, _, ours_mps = _case(name, "mps")  # its own closure: batch_norm_eval's running stats
    f = _formats(name)
    assert _same(ours_mps(f, *mps).cpu(), ours_cpu(f, *cpu))
