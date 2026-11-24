"""Attention, each part rounded as its formats say: ``qattention`` and
``QMultiheadAttention``.

Tier 1: without formats, or with every part ``None``, ``qattention`` is
``F.scaled_dot_product_attention`` and ``QMultiheadAttention`` is
``nn.MultiheadAttention``, word for word. Tier 2: real formats come out close
to them. Tier 3: ``qattention`` is the composition of its steps, each rounded
by its own part (every slot holds a format of its own precision, so a step
wired to another slot's rounding changes the words), forward and backward,
under SR too; ``enable_gqa`` is ``repeat_kv``; and ``QMultiheadAttention``'s
projections are ``QLinear``'s products around PyTorch's core. Then masks,
the formats values, compile and MPS.
"""

import math
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch._dynamo.testing import CompileCounterWithBackend

from mptorch import BinaryK, RoundMode
from mptorch.number import MXFP8_E4M3
from mptorch.quant import (
    AttentionFormats,
    BlockMac,
    FusedMac,
    GemmFormats,
    PointwiseFormats,
    QAffineFormats,
    QAttentionFormats,
    QMatmul,
    QMultiheadAttention,
    Quant,
    Recipe,
    SoftmaxFormats,
    SplitMac,
    qattention,
    qmul,
    qsoftmax,
)
from mptorch.quant.modules.linear import CustomArithLinear
from tests.markers import available_devices, float64_devices, requires_mps
from tests.models import repeat_kv

R = RoundMode


def _q(P: int, rounding: RoundMode = R.RNE) -> Quant:
    sr = rounding.is_stochastic
    return Quant(BinaryK(P + 5, P, prng_bits=20 - P if sr else 0), rounding)


def _mac(P: int, acc: int, rounding: RoundMode = R.RNE) -> SplitMac:
    sr = rounding.is_stochastic
    mul = BinaryK(P + 5, P, prng_bits=20 - P if sr else 0)
    return SplitMac(mul, BinaryK(16, acc, prng_bits=21 - acc if sr else 0), rounding=rounding)


def _formats(rounding: RoundMode = R.RNE) -> AttentionFormats:
    """Every slot a format of its own precision."""
    r = rounding
    qk = GemmFormats(
        input_quant=_q(9, r),
        weight_quant=_q(8, r),
        output_quant=_q(10, r),
        fwd_mac=_mac(7, 10, r),
        igrad_mac=_mac(6, 9, r),
        wgrad_mac=_mac(8, 11, r),
        igrad_quant=_q(7, r),
        wgrad_quant=_q(6, r),
    )
    pv = GemmFormats(
        input_quant=_q(7, r),
        weight_quant=_q(10, r),
        output_quant=_q(8, r),
        fwd_mac=_mac(9, 11, r),
        igrad_mac=_mac(5, 8, r),
        wgrad_mac=_mac(10, 9, r),
        igrad_quant=_q(9, r),
        wgrad_quant=_q(4, r),
    )
    sr = r.is_stochastic
    softmax = SoftmaxFormats(
        sub=_q(9, r),
        exp=_q(8, r),
        acc=FusedMac(BinaryK(16, 10, prng_bits=11 if sr else 0), rounding=r),
        div=_q(6, r),
        bwd_mul=_q(7, r),
        bwd_acc=FusedMac(BinaryK(16, 9, prng_bits=12 if sr else 0), rounding=r),
        bwd_sub=_q(10, r),
        bwd_out=_q(5, r),
    )
    scores = PointwiseFormats(quant=_q(11, r), grad_quant=_q(5, r))
    return AttentionFormats(qk=qk, scores=scores, softmax=softmax, pv=pv)


def _x(device, shape, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)


def _qkv(device, dtype=torch.float32, heads=4, kv_heads=4, L=5, S=7, E=8, seed=0):
    q = _x(device, (2, heads, L, E), dtype, seed)
    k = _x(device, (2, kv_heads, S, E), dtype, seed + 1)
    v = _x(device, (2, kv_heads, S, E), dtype, seed + 2)
    return [t.requires_grad_() for t in (q, k, v)]


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


def _grads(out, inputs, seed=1):
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.autograd.grad(out, inputs, g.to(out), allow_unused=True)


def _masks(device, L=5, S=7) -> dict[str, dict[str, Any]]:
    keep = torch.rand(L, S, generator=torch.Generator().manual_seed(4)) > 0.3
    keep[:, 0] = True  # no row left without a key
    bias = torch.zeros(L, S).masked_fill(~keep, float("-inf")) + _x("cpu", (L, S), seed=9)
    return {
        "none": {},
        "bool": {"attn_mask": keep.to(device)},
        "float": {"attn_mask": bias.to(device)},
        "causal": {"is_causal": True},
    }


def _seeded(fn, seed=3):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return fn()


# --- tier 1 ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("mask", ["none", "bool", "float", "causal"])
@pytest.mark.parametrize("gqa", [False, True])
def test_without_formats_it_is_sdpa(device, dtype, mask, gqa):
    q, k, v = _qkv(device, dtype, kv_heads=2 if gqa else 4)
    kwargs = {**_masks(device)[mask], **({"enable_gqa": True} if gqa else {})}
    if mask == "float":
        kwargs["attn_mask"] = kwargs["attn_mask"].to(dtype)
    want = F.scaled_dot_product_attention(q, k, v, **kwargs)
    gw = _grads(want, (q, k, v))
    for formats in (None, AttentionFormats()):
        got = qattention(q, k, v, formats, **kwargs)
        assert _same(got, want)
        for g, w in zip(_grads(got, (q, k, v)), gw, strict=True):
            assert _same(g, w)


# --- tier 3 ------------------------------------------------------------------------------


def _by_hand(q, k, v, f: AttentionFormats, attn_mask=None, is_causal=False, dropout_p=0.0):
    """qattention's steps, one call each, as the docstring states them."""
    assert f.qk is not None and f.scores is not None and f.pv is not None
    s = QMatmul(f.qk.matmul())(q, k.mT)
    scale = 1.0 / math.sqrt(q.size(-1))
    s = qmul(s, scale, f.scores.quant, f.scores.grad_quant)
    if is_causal:
        attn_mask = torch.ones(s.shape[-2:], dtype=torch.bool, device=s.device).tril()
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros(attn_mask.shape, dtype=s.dtype, device=s.device)
            attn_mask = bias.masked_fill(~attn_mask, float("-inf"))
        s = s + attn_mask.to(s.dtype)
    narrow = s.dtype in (torch.float16, torch.bfloat16)
    p = qsoftmax(s, -1, f.softmax, dtype=torch.float32 if narrow else None)
    p = p.to(s.dtype)
    if dropout_p > 0:
        p = F.dropout(p, dropout_p)
    return QMatmul(f.pv.matmul())(p, v)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
@pytest.mark.parametrize("mask", ["none", "bool", "float", "causal"])
def test_tier3_the_steps(device, dtype, rounding, mask):
    q, k, v = _qkv(device, dtype)
    f = _formats(rounding)
    kwargs = _masks(device)[mask]
    want = _seeded(lambda: _by_hand(q, k, v, f, **kwargs))
    gw = _seeded(lambda: _grads(want, (q, k, v)), 5)
    got = _seeded(lambda: qattention(q, k, v, f, **kwargs))
    assert _same(got, want)
    for g, w in zip(_seeded(lambda: _grads(got, (q, k, v)), 5), gw, strict=True):
        assert _same(g, w)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
def test_tier3_in_bfloat16(device, rounding):
    """bfloat16 holds 8 bits, fewer than most of the formats above: formats it
    can store, and the softmax computed in float32 then stored in bfloat16."""
    q, k, v = _qkv(device, torch.bfloat16)
    f = AttentionFormats.uniform(_mac(4, 8, rounding), _q(4, rounding), grad_quant=_q(3, rounding))
    f = AttentionFormats(
        qk=f.qk, scores=PointwiseFormats(quant=_q(7, rounding)), softmax=f.softmax, pv=f.pv
    )
    want = _seeded(lambda: _by_hand(q, k, v, f, is_causal=True))
    gw = _seeded(lambda: _grads(want, (q, k, v)), 5)
    got = _seeded(lambda: qattention(q, k, v, f, is_causal=True))
    assert _same(got, want)
    for g, w in zip(_seeded(lambda: _grads(got, (q, k, v)), 5), gw, strict=True):
        assert _same(g, w)


@pytest.mark.parametrize("device", float64_devices)
def test_tier3_in_float64(device):
    q, k, v = _qkv(device, torch.float64)
    f = _formats()
    want = _by_hand(q, k, v, f, is_causal=True)
    got = qattention(q, k, v, f, is_causal=True)
    assert _same(got, want)
    for g, w in zip(_grads(got, (q, k, v)), _grads(want, (q, k, v)), strict=True):
        assert _same(g, w)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
def test_gqa_is_repeat_kv(device, rounding):
    """``enable_gqa`` repeats ``k`` and ``v`` as ``repeat_kv`` does, so the two
    spellings give the same words, the gradients summed over each group."""
    q, k, v = _qkv(device, kv_heads=2)
    f = _formats(rounding)
    want = _seeded(lambda: qattention(q, repeat_kv(k, 2), repeat_kv(v, 2), f, is_causal=True))
    gw = _seeded(lambda: _grads(want, (q, k, v)), 5)
    got = _seeded(lambda: qattention(q, k, v, f, is_causal=True, enable_gqa=True))
    assert _same(got, want)
    for g, w in zip(_seeded(lambda: _grads(got, (q, k, v)), 5), gw, strict=True):
        assert _same(g, w)


@pytest.mark.parametrize("device", available_devices)
def test_dropout_is_the_steps_under_one_seed(device):
    q, k, v = _qkv(device)
    f = _formats()
    want = _seeded(lambda: _by_hand(q, k, v, f, dropout_p=0.25))
    got = _seeded(lambda: qattention(q, k, v, f, dropout_p=0.25))
    assert _same(got, want)
    assert not _same(got, _seeded(lambda: qattention(q, k, v, f)))


# --- tier 2, and masks -----------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_real_formats_come_out_close(device):
    q, k, v = _qkv(device)
    p8 = BinaryK(8, 4)
    f = AttentionFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
    got = qattention(q, k, v, f, is_causal=True)
    want = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(got, want, atol=0.5)  # 4 bits: 0.22 at most here, for |values| < 3
    block = AttentionFormats.uniform(BlockMac(MXFP8_E4M3))
    assert block.softmax is None
    assert torch.allclose(qattention(q, k, v, block, is_causal=True), want, atol=0.5)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("formats", [None, "rounded"])
def test_a_fully_masked_row_is_zeros(device, formats):
    """A query the mask leaves no key attends to nothing, as in PyTorch's
    kernels, and its gradients are zero, not NaN, in every operand."""
    q, k, v = _qkv(device)
    keep = torch.ones(5, 7, dtype=torch.bool, device=device)
    keep[2] = False
    f = _formats() if formats else None
    out = qattention(q, k, v, f, attn_mask=keep)
    assert (out[:, :, 2] == 0).all()
    grads = _grads(out, (q, k, v))
    assert all(g.isfinite().all() for g in grads)
    assert (grads[0][:, :, 2] == 0).all()


@pytest.mark.parametrize("device", available_devices)
def test_a_boolean_mask_is_its_float_spelling(device):
    q, k, v = _qkv(device)
    keep = _masks(device)["bool"]["attn_mask"]
    bias = torch.zeros(keep.shape, device=device).masked_fill(~keep, float("-inf"))
    f = _formats()
    assert _same(qattention(q, k, v, f, attn_mask=keep), qattention(q, k, v, f, attn_mask=bias))


@pytest.mark.parametrize("device", available_devices)
def test_a_wider_float_mask_keeps_the_scores_dtype(device):
    q, k, v = _qkv(device, torch.float16)
    mask = _masks(device)["float"]["attn_mask"]  # float32
    out = qattention(q, k, v, _formats(), attn_mask=mask)
    assert out.dtype == torch.float16
    assert _same(out, qattention(q, k, v, _formats(), attn_mask=mask.half()))


def test_what_is_refused():
    q, k, v = _qkv("cpu", kv_heads=3)
    with pytest.raises(RuntimeError, match="must divide the number of query heads"):
        qattention(q, k, v, _formats(), enable_gqa=True)
    q, k, v = _qkv("cpu")
    with pytest.raises(RuntimeError, match="both attn_mask and is_causal"):
        qattention(
            q, k, v, _formats(), attn_mask=torch.ones(5, 7, dtype=torch.bool), is_causal=True
        )
    with pytest.raises(TypeError, match="AttentionFormats or None"):
        qattention(q, k, v, SoftmaxFormats())  # ty: ignore[invalid-argument-type]


# --- the formats values ------------------------------------------------------------------------


def test_the_formats_values():
    p8 = BinaryK(8, 4)
    mac = SplitMac(p8, BinaryK(16, 11))
    f = AttentionFormats.uniform(mac, Quant(p8), grad_quant=Quant(BinaryK(8, 5)))
    assert f.qk == f.pv == GemmFormats.uniform(mac, Quant(p8), grad_quant=Quant(BinaryK(8, 5)))
    assert f.softmax == SoftmaxFormats.uniform(mac, Quant(p8), grad=Quant(BinaryK(8, 5)))
    assert f.scores is None
    assert f == AttentionFormats.uniform(mac, Quant(p8), grad_quant=Quant(BinaryK(8, 5)))
    assert hash(f) == hash(
        AttentionFormats.uniform(mac, Quant(p8), grad_quant=Quant(BinaryK(8, 5)))
    )
    assert f.label().startswith("qk [mac SplitMac(")
    assert AttentionFormats().label() == "torch"
    with pytest.raises(TypeError, match="qk is a GemmFormats or None"):
        AttentionFormats(qk=mac)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="softmax is a SoftmaxFormats or None"):
        AttentionFormats(softmax=GemmFormats())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="AttentionFormats or None"):
        QAttentionFormats(attention=GemmFormats())


def test_attention_parts():
    f = _formats()
    with pytest.raises(ValueError, match="scores are rounded"):
        Recipe().attention_parts("*.attn", f)
    f = AttentionFormats(qk=f.qk, softmax=f.softmax, pv=f.pv)
    recipe = Recipe().attention_parts("*.attn", f)
    m = nn.Module()
    assert recipe.resolve("a.attn", m, "attention") is f
    assert recipe.resolve("a.attn", m, "matmul", "matmul#0") is f.qk
    assert recipe.resolve("a.attn", m, "matmul", "matmul#1") is f.pv
    assert recipe.resolve("a.attn", m, "matmul", "matmul#2") is None
    assert recipe.resolve("a.attn", m, "softmax") is f.softmax
    assert recipe.resolve("a.mlp", m, "attention") is None
    with pytest.raises(TypeError, match="takes an AttentionFormats"):
        Recipe().attention_parts("*", cast(Any, f.qk))


# --- QMultiheadAttention -----------------------------------------------------------------------

E, HEADS = 16, 4


def _mha(device, **kwargs):
    torch.manual_seed(0)
    plain = nn.MultiheadAttention(E, HEADS, **kwargs).to(device)
    ours = QMultiheadAttention(E, HEADS, **kwargs).to(device)
    ours.load_state_dict(plain.state_dict())
    return plain, ours


def _mha_inputs(device, batch_first, cross="self", L=5, S=6, kdim=E, vdim=E):
    shape = (lambda n, d: (2, n, d)) if batch_first else (lambda n, d: (n, 2, d))
    x = _x(device, shape(L, E), seed=1).requires_grad_()
    if cross == "self":
        return x, x, x
    k = _x(device, shape(S, kdim), seed=2).requires_grad_()
    if cross == "kv":
        return x, k, k
    return x, k, _x(device, shape(S, vdim), seed=3).requires_grad_()


# name: (constructor arguments, input arguments, call arguments)
MHA_CASES: dict[str, tuple[dict, dict, dict]] = {
    "self": ({}, {}, {}),
    "batch first": ({"batch_first": True}, {}, {}),
    "weights per head": ({}, {}, {"need_weights": True, "average_attn_weights": False}),
    "no weights": ({}, {}, {"need_weights": False}),
    "cross kv": ({}, {"cross": "kv"}, {}),
    "cross": ({}, {"cross": "all"}, {"need_weights": False}),
    "kdim vdim": ({"kdim": 6, "vdim": 10}, {"cross": "all", "kdim": 6, "vdim": 10}, {}),
    "bias kv": ({"add_bias_kv": True}, {}, {}),
    "zero attn": ({"add_zero_attn": True}, {}, {"need_weights": False}),
    "no bias": ({"bias": False}, {}, {}),
}


def _mha_masks(device, name, L=5):
    if name == "padding":
        pad = torch.zeros(2, L, dtype=torch.bool, device=device)
        pad[1, -2:] = True
        return {"key_padding_mask": pad}
    if name == "causal":
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
        return {"attn_mask": mask, "is_causal": True}
    if name == "both":
        pad = torch.zeros(2, L, dtype=torch.bool, device=device)
        pad[0, -1] = True
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), 1)
        return {"key_padding_mask": pad, "attn_mask": mask}
    return {}


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("train", [True, False])
def test_mha_without_formats_is_torchs(device, train):
    """With ``formats=None`` or every part ``None`` the layer is torch's,
    fast path included (eval, no gradient, batch first)."""
    plain, ours = _mha(device, batch_first=True)
    x = _x(device, (2, 5, E))
    plain.train(train)
    ours.train(train)
    with torch.set_grad_enabled(train):
        want = _seeded(lambda: plain(x, x, x))
        for formats in (None, QAttentionFormats(attention=AttentionFormats())):
            ours.formats = formats
            got = _seeded(lambda: ours(x, x, x))
            assert _same(got[0], want[0]) and _same(got[1], want[1])


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("case", list(MHA_CASES))
@pytest.mark.parametrize("masks", ["none", "padding", "causal", "both"])
def test_mha_is_torchs_through_its_port(device, case, masks):
    """Projections over containers with nothing set, and torch's core: the
    port of ``multi_head_attention_forward`` gives torch's words forward, and
    its gradients to float noise (a ``QLinear``'s backward multiplies as it
    does, not as autograd's ``F.linear``)."""
    init, inputs, call = MHA_CASES[case]
    if masks != "none" and inputs.get("cross") is not None:
        inputs = {**inputs, "S": 5}
    plain, ours = _mha(device, **init)
    ours.formats = QAttentionFormats(QAffineFormats(), QAffineFormats())
    args = _mha_inputs(device, init.get("batch_first", False), **inputs)
    kwargs = {**call, **_mha_masks(device, masks)}
    want = plain(*args, **kwargs)
    got = ours(*args, **kwargs)
    assert _same(got[0], want[0]) and _same(got[1], want[1])
    params = [p for p in plain.parameters()]
    gw = torch.autograd.grad(want[0].sum(), params)
    gg = torch.autograd.grad(got[0].sum(), list(ours.parameters()))
    for g, w in zip(gg, gw, strict=True):
        torch.testing.assert_close(g, w, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("case", ["self", "batch first", "weights per head", "kdim vdim"])
@pytest.mark.parametrize("masks", ["none", "padding", "causal"])
def test_mha_core_is_qattentions(device, case, masks):
    """With attention formats, the core is ``qattention``'s steps: with nothing
    rounded, close to torch's; with a real format, close too."""
    init, inputs, call = MHA_CASES[case]
    if masks != "none" and inputs.get("cross") is not None:
        inputs = {**inputs, "S": 5}
    plain, ours = _mha(device, **init)
    args = _mha_inputs(device, init.get("batch_first", False), **inputs)
    kwargs = {**call, **_mha_masks(device, masks)}
    want = plain(*args, **kwargs)
    ours.formats = QAttentionFormats(attention=AttentionFormats(qk=GemmFormats()))
    got = ours(*args, **kwargs)
    torch.testing.assert_close(got[0], want[0], rtol=1e-5, atol=1e-5)
    if want[1] is not None:
        torch.testing.assert_close(got[1], want[1], rtol=1e-5, atol=1e-5)
    p8 = BinaryK(8, 4)
    ours.formats = QAttentionFormats(
        attention=AttentionFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
    )
    assert torch.allclose(ours(*args, **kwargs)[0], want[0], atol=0.25)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("need_weights", [False, True])
def test_mha_by_hand(device, need_weights):
    """Self-attention, batch first: one ``QLinear`` product over the packed
    in-projection, ``qattention`` over the heads with the scale
    ``1/sqrt(head_dim)``, and the output projection's product, word for
    word, forward and backward, under SR."""
    _, ours = _mha(device, batch_first=True)
    gf = GemmFormats.uniform(_mac(4, 11, R.SR), _q(4, R.SR))
    af = _formats(R.SR)
    af = AttentionFormats(qk=af.qk, scores=af.scores, softmax=af.softmax, pv=af.pv)
    ours.formats = QAttentionFormats(gf.affine(), gf.affine(), af)
    x = _x(device, (2, 5, E)).requires_grad_()

    def by_hand():
        f = ours.formats
        xs = x.transpose(0, 1)
        proj = CustomArithLinear.apply(xs, ours.in_proj_weight, ours.in_proj_bias, f.in_proj)
        proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        L, N, D = 5, 2, E // HEADS
        q, k, v = (t.view(L, N * HEADS, D).transpose(0, 1).view(N, HEADS, L, D) for t in proj)
        o = qattention(q, k, v, af)
        o = o.permute(2, 0, 1, 3).reshape(N * L, E)
        o = CustomArithLinear.apply(o, ours.out_proj.weight, ours.out_proj.bias, f.out_proj)
        return o.view(L, N, E).transpose(0, 1)

    want = _seeded(by_hand)
    got = _seeded(lambda: ours(x, x, x, need_weights=need_weights))
    assert _same(got[0], want)
    assert (got[1] is not None) == need_weights
    params = [x, *ours.parameters()]
    gw = _seeded(lambda: torch.autograd.grad(want.sum(), params))
    for g, w in zip(_seeded(lambda: torch.autograd.grad(got[0].sum(), params)), gw, strict=True):
        assert _same(g, w)


def test_mha_state_dict_is_torchs():
    plain, ours = _mha("cpu", kdim=6, vdim=10, add_bias_kv=True)
    assert list(ours.state_dict()) == list(plain.state_dict())
    ours.formats = QAttentionFormats(attention=_formats())
    assert list(ours.state_dict()) == list(plain.state_dict())
    plain.load_state_dict(ours.state_dict())


# --- compile, and MPS ------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("gqa", [False, True])
def test_compiles_in_one_graph(device, gqa):
    q, k, v = _qkv(device, kv_heads=2 if gqa else 4)
    f = _formats()

    def fn(q, k, v):
        return qattention(q, k, v, f, is_causal=True, enable_gqa=gqa)

    want = fn(q, k, v)
    gw = _grads(want, (q, k, v))
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    got = torch.compile(fn, backend=counter, fullgraph=True)(q, k, v)
    assert counter.frame_count == 1
    assert _same(got, want)
    for g, w in zip(_grads(got, (q, k, v)), gw, strict=True):
        assert _same(g, w)


def test_inductor():
    q, k, v = _qkv("cpu")
    f = AttentionFormats.uniform(SplitMac(BinaryK(8, 4), BinaryK(16, 11)), Quant(BinaryK(8, 4)))
    torch._dynamo.reset()
    got = torch.compile(lambda *t: qattention(*t, f), backend="inductor", fullgraph=True)(q, k, v)
    assert torch.allclose(got, qattention(q, k, v, f), atol=0.1)


@pytest.mark.parametrize("device", available_devices)
def test_mha_compiles_in_one_graph(device):
    _, ours = _mha(device, batch_first=True)
    gf = GemmFormats.uniform(_mac(4, 11), _q(4))
    ours.formats = QAttentionFormats(gf.affine(), gf.affine(), _formats())
    x = _x(device, (2, 5, E))
    want = ours(x, x, x, need_weights=False)[0]
    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    got = torch.compile(ours, backend=counter, fullgraph=True)(x, x, x, need_weights=False)[0]
    assert counter.frame_count == 1
    assert _same(got, want)


@requires_mps
@pytest.mark.parametrize("mask", ["none", "bool", "causal"])
def test_mps_is_the_cpu(mask):
    f = _formats()
    got = qattention(*_qkv("mps"), f, **_masks("mps")[mask])
    want = qattention(*_qkv("cpu"), f, **_masks("cpu")[mask])
    assert _same(got.cpu(), want)
