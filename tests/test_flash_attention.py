"""Flash attention: ``AttentionFormats(flash=FlashFormats(...))``.

Tier 1: with nothing rounded, flash is ``F.scaled_dot_product_attention``,
forward and every gradient, to the carrier's precision, over masks, GQA,
ragged tiles and rows the mask leaves empty. Tier 2: an FP8 recipe is as
close to the float64 attention as the standard path is. Tier 3: an
independent FlashAttention-2, written here from Algorithms 1 and 2 of
arXiv:2307.08691 out of public functions (``Quant``, ``qmatmul``, ``qsum``),
gives flash's words, forward and backward, every slot rounding to a
precision of its own so that a step wired to another slot's rounding
changes the words. Around it: per-tile quantizers, causal skipping, the
rotation, ``simulate``, ``QMultiheadAttention``, the refusals, compile and
``profile``.
"""

import dataclasses
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from mptorch import E4M3, BinaryK, RoundMode
from mptorch.quant import (
    AttentionFormats,
    FlashFormats,
    FusedMac,
    GemmFormats,
    Hadamard,
    PointwiseFormats,
    QAttentionFormats,
    QMultiheadAttention,
    Quant,
    Recipe,
    ScaledQuant,
    SoftmaxFormats,
    SplitMac,
    hadamard_rotate,
    profile,
    qattention,
    qmatmul,
    qsum,
    simulate,
)
from tests.markers import available_devices, float64_devices


def _fmt(p: int, prng_bits: int = 0) -> BinaryK:
    """A P-bit format with 6 exponent bits, range enough for attention."""
    return BinaryK(p + 6, p, prng_bits=prng_bits)


def _q(p: int) -> Quant:
    return Quant(_fmt(p))


def _mac(mul: int, acc: int, rounding: RoundMode) -> SplitMac:
    bits = 8 if rounding.is_stochastic else 0
    return SplitMac(_fmt(mul, bits), _fmt(acc, bits), rounding=rounding)


def _formats(rounding: RoundMode = RoundMode.RNE, block_q: int = 16, block_kv: int = 8):
    """Every slot set, each to a precision of its own."""
    qk = GemmFormats(
        input_quant=_q(4),
        weight_quant=_q(3),
        output_quant=_q(9),
        fwd_mac=_mac(7, 11, rounding),
        igrad_mac=_mac(6, 10, rounding),
        wgrad_mac=_mac(8, 12, rounding),
        igrad_quant=_q(5),
        wgrad_quant=_q(7),
    )
    pv = GemmFormats(
        input_quant=_q(4),
        weight_quant=_q(5),
        output_quant=_q(8),
        fwd_mac=_mac(7, 11, rounding),
        igrad_mac=_mac(6, 9, rounding),
        wgrad_mac=_mac(8, 13, rounding),
        igrad_quant=_q(6),
        wgrad_quant=_q(8),
    )
    return AttentionFormats(
        qk=qk,
        scores=PointwiseFormats(quant=_q(10), grad_quant=_q(8)),
        softmax=SoftmaxFormats(
            sub=_q(9),
            exp=_q(7),
            acc=FusedMac(_fmt(12)),
            div=_q(10),
            bwd_mul=_q(9),
            bwd_acc=FusedMac(_fmt(11)),
            bwd_sub=_q(10),
            bwd_out=_q(8),
        ),
        pv=pv,
        flash=FlashFormats(
            block_q=block_q, block_kv=block_kv, rescale=_q(9), combine=_q(13), lse=_q(12)
        ),
    )


def _grads(out, inputs, g):
    return torch.autograd.grad(out, inputs, g, allow_unused=True)


def _qkv(device, L=40, S=36, E=16, Ev=12, heads=2, dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)
    shapes = ((2, heads, L, E), (2, heads, S, E), (2, heads, S, Ev))
    return [torch.randn(*s, generator=gen, dtype=dtype).to(device).requires_grad_() for s in shapes]


# --- an independent FlashAttention-2 ---------------------------------------------------------


def _reference(q, k, v, af, g, *, scale, causal=False, mask=None):
    """FA2 by hand: Algorithm 1 for the output, Algorithm 2 for the gradients,
    each step rounded by the slot the table of ``kernels/attention.rst``
    gives it, each product a ``qmatmul`` over its pass's mac, each row sum a
    ``qsum``. The calls are made in the order flash makes them, so that a
    stochastic rounding draws the same bits."""
    fl, qk, sc, sm, pv = af.flash, af.qk, af.scores, af.softmax, af.pv
    L, S = q.size(-2), k.size(-2)
    rows = [(a, min(a + fl.block_q, L)) for a in range(0, L, fl.block_q)]
    cols = [(a, min(a + fl.block_kv, S)) for a in range(0, S, fl.block_kv)]

    def skipped(r, c):
        return causal and c[0] > r[1] - 1

    def bias(s, r, c):
        if mask is not None:
            m = mask[..., r[0] : r[1], c[0] : c[1]]
            if m.dtype == torch.bool:
                return torch.zeros_like(s).masked_fill(~m, float("-inf"))
            return m
        if causal and c[1] - 1 > r[0]:
            ri = torch.arange(r[0], r[1], device=s.device)[:, None]
            ci = torch.arange(c[0], c[1], device=s.device)[None, :]
            return torch.zeros_like(s).masked_fill(ri < ci, float("-inf"))
        return None

    with torch.no_grad():
        qq = [qk.input_quant(q[..., a:b, :]) for a, b in rows]
        kq = [qk.weight_quant(k[..., a:b, :].mT) for a, b in cols]
        vq = [pv.weight_quant(v[..., a:b, :]) for a, b in cols]
        outs, lses, scores = [], [], {}
        for i, r in enumerate(rows):
            m = torch.full((*q.shape[:-2], r[1] - r[0], 1), float("-inf"), device=q.device)
            total = torch.zeros_like(m)
            o = torch.zeros((*q.shape[:-2], r[1] - r[0], v.size(-1)), device=q.device)
            for j, c in enumerate(cols):
                if skipped(r, c):
                    break
                s = qk.output_quant(qmatmul(qq[i], kq[j], qk.fwd_mac))
                s = sc.quant(s * scale)
                b = bias(s, r, c)
                if b is not None:
                    s = s + b
                scores[(i, j)] = s
                m_new = torch.maximum(m, s.amax(-1, keepdim=True))
                zero = torch.where(m_new.isneginf(), 0.0, m_new)
                p = sm.exp(torch.exp(sm.sub(s - zero)))
                alpha = fl.rescale(torch.exp(m - zero))
                total = fl.combine(alpha * total + qsum(p, -1, sm.acc, keepdim=True))
                o = fl.combine(alpha * o + qmatmul(pv.input_quant(p), vq[j], pv.fwd_mac))
                m = m_new
            dead = total == 0
            o = pv.output_quant(sm.div(o / torch.where(dead, 1.0, total)))
            zero = torch.where(m.isneginf(), 0.0, m)
            lses.append(fl.lse(zero + torch.log(total)).masked_fill(dead, float("-inf")))
            outs.append(o)
        out = torch.cat(outs, -2)

        dout = [g[..., a:b, :] for a, b in rows]
        big_d = [
            qsum(sm.bwd_mul(dout[i] * out[..., a:b, :]), -1, sm.bwd_acc, keepdim=True)
            for i, (a, b) in enumerate(rows)
        ]
        dout_p = [pv.igrad_quant(t) for t in dout]
        dout_v = [pv.wgrad_quant(t) for t in dout]
        dq = [torch.zeros_like(q[..., a:b, :]) for a, b in rows]
        dk = [torch.zeros_like(k[..., a:b, :]) for a, b in cols]
        dv = [torch.zeros_like(v[..., a:b, :]) for a, b in cols]
        for j, c in enumerate(cols):
            for i, r in enumerate(rows):
                if skipped(r, c):
                    continue
                lse = lses[i]
                p = torch.exp(scores[(i, j)] - torch.where(lse.isneginf(), 0.0, lse))
                dv[j] = fl.combine(dv[j] + qmatmul(pv.input_quant(p).mT, dout_v[i], pv.wgrad_mac))
                dp = qmatmul(dout_p[i], vq[j].mT, pv.igrad_mac)
                ds = sm.bwd_out(p * sm.bwd_sub(dp - big_d[i]))
                ds = sc.grad_quant(ds * scale)
                dq[i] = fl.combine(dq[i] + qmatmul(qk.igrad_quant(ds), kq[j].mT, qk.igrad_mac))
                dk_ij = qmatmul(qq[i].mT, qk.wgrad_quant(ds), qk.wgrad_mac).mT
                dk[j] = fl.combine(dk[j] + dk_ij)
    return out, torch.cat(dq, -2), torch.cat(dk, -2), torch.cat(dv, -2)


def _masks(device, name, L, S):
    if name == "none":
        return None, False
    if name == "causal":
        return None, True
    gen = torch.Generator().manual_seed(4)
    if name == "bool":
        m = torch.rand(2, 1, L, S, generator=gen) > 0.3
        m[:, :, 3] = False  # a row with nothing to attend
        return m.to(device), False
    return (torch.randn(1, 2, L, S, generator=gen) * 2).to(device), False


# --- tier 1 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mask", ["none", "causal", "bool", "float"])
@pytest.mark.parametrize(("L", "S"), [(40, 36), (33, 70), (64, 64)])
def test_tier1_flash_is_sdpa(device, mask, L, S):
    """With nothing rounded, flash is SDPA to float32's precision, forward
    and every gradient, whatever the tiles' fit; a row the mask leaves
    empty is zeros, with zero gradients, as the standard path gives."""
    q, k, v = _qkv(device, L=L, S=S)
    attn_mask, causal = _masks(device, mask, L, S)
    af = AttentionFormats(flash=FlashFormats(block_q=16, block_kv=8))
    out = qattention(q, k, v, af, attn_mask=attn_mask, is_causal=causal)
    want = qattention(q, k, v, AttentionFormats(), attn_mask=attn_mask, is_causal=causal)
    g = torch.randn_like(out)
    torch.testing.assert_close(out, want, rtol=1e-5, atol=1e-5)
    for got, w in zip(_grads(out, (q, k, v), g), _grads(want, (q, k, v), g), strict=True):
        torch.testing.assert_close(got, w, rtol=1e-4, atol=1e-5)
    if mask == "bool":
        assert torch.equal(out[:, :, 3], torch.zeros_like(out[:, :, 3]))


@pytest.mark.parametrize("device", float64_devices)
def test_tier1_float64_and_gqa(device):
    """In binary64 flash is SDPA to binary64's precision, with grouped keys
    and values; float32 operands under a float64 carrier are the float64
    attention of their widened values, rounded once to float32."""
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(2, 4, 30, 8, generator=gen, dtype=torch.float64).to(device).requires_grad_()
    k = torch.randn(2, 2, 25, 8, generator=gen, dtype=torch.float64).to(device).requires_grad_()
    v = torch.randn(2, 2, 25, 8, generator=gen, dtype=torch.float64).to(device).requires_grad_()
    af = AttentionFormats(flash=FlashFormats(block_q=8, block_kv=16))
    out = qattention(q, k, v, af, enable_gqa=True)
    want = F.scaled_dot_product_attention(q, k, v, enable_gqa=True)
    g = torch.randn_like(out)
    torch.testing.assert_close(out, want, rtol=1e-12, atol=1e-12)
    for got, w in zip(_grads(out, (q, k, v), g), _grads(want, (q, k, v), g), strict=True):
        torch.testing.assert_close(got, w, rtol=1e-11, atol=1e-12)
    wide = AttentionFormats(flash=FlashFormats(block_q=8, block_kv=16, carrier=torch.float64))
    qf, kf, vf = (t.detach().float() for t in (q, k, v))
    widened = qattention(qf.double(), kf.double(), vf.double(), af, enable_gqa=True)
    assert torch.equal(qattention(qf, kf, vf, wide, enable_gqa=True), widened.float())


# --- tier 2 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_tier2_an_fp8_recipe_is_as_close_as_the_standard_path(device):
    """E4M3 operands and probabilities, everything else in binary32: flash is
    as close to the float64 attention as the standard path is."""
    e4 = Quant(E4M3)
    part = GemmFormats(input_quant=e4, weight_quant=e4)
    std = AttentionFormats(qk=part, pv=part)
    flash = dataclasses.replace(std, flash=FlashFormats(block_q=16, block_kv=16))
    q, k, v = _qkv(device, L=64, S=64)
    want = F.scaled_dot_product_attention(*(t.detach().cpu().double() for t in (q, k, v)))

    def error(af):
        out = qattention(q, k, v, af).detach().cpu().double()
        return float((out - want).norm() / want.norm())

    e_flash, e_std = error(flash), error(std)
    assert e_flash < 0.1
    assert e_flash < 1.5 * e_std + 1e-3, (e_flash, e_std)


# --- tier 3 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [RoundMode.RNE, RoundMode.SR])
@pytest.mark.parametrize("mask", ["none", "causal", "bool", "float"])
def test_tier3_flash_is_fa2_by_hand(device, rounding, mask):
    """Word for word with the FA2 written above, forward and backward, every
    slot set; under SR the scores are kept for backward, and the draws come
    in the reference's order."""
    q, k, v = _qkv(device)
    attn_mask, causal = _masks(device, mask, q.size(-2), k.size(-2))
    af = _formats(rounding)
    g = torch.randn(2, 2, 40, 12, generator=torch.Generator().manual_seed(9)).to(device)
    scale = 1 / math.sqrt(16)
    torch.manual_seed(3)
    out = qattention(q, k, v, af, attn_mask=attn_mask, is_causal=causal)
    grads = _grads(out, (q, k, v), g)
    torch.manual_seed(3)
    want = _reference(q, k, v, af, g, scale=scale, causal=causal, mask=attn_mask)
    assert torch.equal(out, want[0])
    for got, w in zip(grads, want[1:], strict=True):
        assert torch.equal(got, w)


@pytest.mark.parametrize("device", available_devices)
def test_causal_skipping_is_the_masked_tiles(device):
    """Skipping the tiles above the diagonal gives the words of computing
    them under the same mask, written as a boolean one: a masked tile adds
    exactly nothing to the running state."""
    q, k, v = _qkv(device, L=40, S=40)
    af = _formats()
    tril = torch.ones(40, 40, dtype=torch.bool, device=device).tril()
    g = torch.randn(2, 2, 40, 12, generator=torch.Generator().manual_seed(9)).to(device)
    out = qattention(q, k, v, af, is_causal=True)
    want = qattention(q, k, v, af, attn_mask=tril)
    assert torch.equal(out, want)
    for got, w in zip(_grads(out, (q, k, v), g), _grads(want, (q, k, v), g), strict=True):
        assert torch.equal(got, w)


@pytest.mark.parametrize("device", available_devices)
def test_operand_quantizers_run_per_tile(device):
    """A ScaledQuant on q scales each query tile by its own maximum, so
    one tile's output does not move when another tile grows a thousandfold."""
    sq = ScaledQuant(E4M3)
    af = AttentionFormats(
        qk=GemmFormats(input_quant=sq, weight_quant=sq),
        flash=FlashFormats(block_q=16, block_kv=16),
    )
    q, k, v = (t.detach() for t in _qkv(device, L=32, S=32))
    loud = q.clone()
    loud[..., 16:, :] *= 1000
    out, out_loud = qattention(q, k, v, af), qattention(loud, k, v, af)
    assert torch.equal(out[..., :16, :], out_loud[..., :16, :])
    per_tensor = AttentionFormats(qk=GemmFormats(input_quant=sq, weight_quant=sq))
    first = qattention(q, k, v, per_tensor)[..., :16, :]
    assert not torch.equal(first, qattention(loud, k, v, per_tensor)[..., :16, :])


@pytest.mark.parametrize("device", available_devices)
def test_a_rotated_qk_is_flash_over_rotated_q_and_k(device):
    rot = Hadamard(8)
    af = _formats()
    assert af.qk is not None
    rotated = dataclasses.replace(af, qk=dataclasses.replace(af.qk, rotation=rot))
    q, k, v = _qkv(device)
    qr = hadamard_rotate(q.detach(), rot).requires_grad_()
    kr = hadamard_rotate(k.detach(), rot).requires_grad_()
    g = torch.randn(2, 2, 40, 12, generator=torch.Generator().manual_seed(9)).to(device)
    out = qattention(q, k, v, rotated)
    want = qattention(qr, kr, v, af)
    assert torch.equal(out, want)
    dq, dk, dv = _grads(out, (q, k, v), g)
    wq, wk, wv = _grads(want, (qr, kr, v), g)
    assert torch.equal(dq, hadamard_rotate(wq, rot, inverse=True))
    assert torch.equal(dk, hadamard_rotate(wk, rot, inverse=True))
    assert torch.equal(dv, wv)


# --- around it -----------------------------------------------------------------------------


class _Sdpa(nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)


@pytest.mark.parametrize("device", available_devices)
def test_simulate_runs_an_sdpa_model_under_flash(device):
    af = _formats()
    model = _Sdpa()
    q, k, v = _qkv(device)
    with simulate(Recipe(attention=af), model):
        out = model(q, k, v)
    assert torch.equal(out, qattention(q, k, v, af, is_causal=True))


@pytest.mark.parametrize("device", available_devices)
def test_multihead_attention_under_flash(device):
    """With nothing rounded, a QMultiheadAttention under flash is torch's,
    and it refuses to return weights it never forms."""
    torch.manual_seed(0)
    plain = nn.MultiheadAttention(16, 4).to(device)
    flash = AttentionFormats(flash=FlashFormats(block_q=4, block_kv=4))
    ours = QMultiheadAttention(16, 4, formats=QAttentionFormats(attention=flash)).to(device)
    ours.load_state_dict(plain.state_dict())
    x = torch.randn(9, 2, 16, device=device)
    want, _ = plain(x, x, x, need_weights=False)
    got, weights = ours(x, x, x, need_weights=False)
    assert weights is None
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="need_weights=False"):
        ours(x, x, x)


def test_the_refusals():
    q, k, v = (t.detach() for t in _qkv("cpu"))
    af = AttentionFormats(flash=FlashFormats())
    with pytest.raises(ValueError, match="takes no dropout"):
        qattention(q, k, v, af, dropout_p=0.1)
    mask = torch.zeros(40, 36, requires_grad=True)
    with pytest.raises(ValueError, match="gives attn_mask no gradient"):
        qattention(q, k, v, af, attn_mask=mask)
    with pytest.raises(RuntimeError, match="both attn_mask and is_causal"):
        qattention(q, k, v, af, attn_mask=mask.detach(), is_causal=True)
    with pytest.raises(ValueError, match="flash part"):
        Recipe().attention_parts("*", af)
    with pytest.raises(TypeError, match="block_kv is an int"):
        FlashFormats(block_kv=8.0)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="block_q is at least 1"):
        FlashFormats(block_q=0)
    with pytest.raises(TypeError, match="flash is a FlashFormats"):
        AttentionFormats(flash=SoftmaxFormats())  # ty: ignore[invalid-argument-type]
    assert "flash [tiles 64x64]" in af.label()


@pytest.mark.parametrize("device", available_devices)
def test_flash_compiles_to_eagers_words(device):
    torch._dynamo.reset()
    af = _formats()
    q, k, v = _qkv(device)
    g = torch.randn(2, 2, 40, 12, generator=torch.Generator().manual_seed(9)).to(device)
    out = qattention(q, k, v, af, is_causal=True)
    want = (out.detach(), *_grads(out, (q, k, v), g))

    def call(q, k, v):
        return qattention(q, k, v, af, is_causal=True)

    out2 = torch.compile(call, backend="aot_eager", fullgraph=True)(q, k, v)
    got = (out2.detach(), *_grads(out2, (q, k, v), g))
    for a, b in zip(got, want, strict=True):
        assert torch.equal(a, b)


def test_profile_counts_the_tiles_flash_runs():
    """A causal flash attention runs the tiles on and below the diagonal: 10
    of 16 for four tiles a side, so qk's forward MACs are 10/16 of the full
    product's, and the backward recomputes the scores when they draw nothing."""
    af = AttentionFormats(flash=FlashFormats(block_q=16, block_kv=16))
    q, k, v = _qkv("cpu", L=64, S=64)
    p = profile(_Sdpa(), q, k, v, recipe=Recipe(attention=af), backward=True)
    qk_fwd = [r for r in p.records if r.site == "qk" and r.step == "fwd"]
    full = 2 * 2 * 64 * 64 * 16
    assert sum([r.macs for r in qk_fwd]) == 2 * full * 10 // 16  # forward and recompute
    assert all(r.M == 16 and r.N == 16 for r in qk_fwd)
