"""FlashAttention-2's order, tile by tile: the flash path of :func:`~mptorch.quant.qattention`.

``AttentionFormats(flash=FlashFormats(...))`` computes attention key tile by
key tile with a running maximum and sum, each of the four parts keeping its
role per tile and ``FlashFormats``' own slots rounding the rescale, the
running updates and the saved log-sum-exp. Two spellings give the same
words: the loop (``_flash_forward``, ``_flash_backward``), and the batched
path, which runs every query tile a key tile reaches in one call wherever
``_batchable`` allows it. ``_FlashAttention`` is the autograd Function that
chooses between them. ``_repeated`` (the repeated keys and values of grouped
queries) and ``_bias`` (a mask as an additive bias) serve the eager path of
``attention.py`` too. ``docs/source/kernels/attention.rst`` is the user's
account.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from .._opaque import _opaque_backward
from ..events import _uncounted
from ..mac import BlockMac, BlockQuant, FusedMac, Quant, SplitMac
from ..modules.format import QMatmulFormats
from ..modules.matmul import CustomArithMatmul
from ..ops._spec import _round_result, _stored
from ..rotation import _Rotation
from ..scaled import ScaledQuant
from .norm import _r, _total
from .pointwise import _carrier_dtype, qmul

if TYPE_CHECKING:
    from .attention import AttentionFormats


def _repeated(q: torch.Tensor, kv: torch.Tensor, name: str) -> torch.Tensor:
    """``kv``'s heads repeated over each group of query heads, as
    transformers' ``repeat_kv`` repeats them: head ``j`` becomes heads
    ``j*G`` to ``j*G + G - 1``."""
    heads, kv_heads = q.size(-3), kv.size(-3)
    if kv_heads == heads:
        return kv
    if kv_heads == 0 or heads % kv_heads:
        raise RuntimeError(
            f"qattention with enable_gqa shares each of {name}'s heads among a group of query "
            f"heads, and {name} has {kv_heads} heads for {heads} query heads: the number of "
            f"{name}'s heads must divide the number of query heads"
        )
    groups = heads // kv_heads
    shape = kv.shape
    expanded = kv[..., None, :, :].expand(*shape[:-2], groups, *shape[-2:])
    return expanded.reshape(*shape[:-3], heads, *shape[-2:])


def _bias(mask: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """The mask as the scores' summand, in their dtype: a boolean mask as 0
    where it attends and -inf where it does not, a float mask as it is."""
    if mask.dtype == torch.bool:
        bias = torch.zeros(mask.shape, dtype=s.dtype, device=s.device)
        return bias.masked_fill_(mask.logical_not(), float("-inf"))
    return mask.to(s.dtype)


# --- flash attention ---------------------------------------------------------------------


def _flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    formats: AttentionFormats,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> torch.Tensor:
    """:func:`qattention` under ``formats.flash``. The operands are broadcast
    to one batch shape and the heads repeated here, outside the Function,
    so that autograd sums their gradients back. Called with torch function
    modes disabled."""
    if attn_mask is not None and is_causal:
        raise RuntimeError(
            "qattention was given both attn_mask and is_causal=True: is_causal is a mask of "
            "its own -- pass one of them, as F.scaled_dot_product_attention requires"
        )
    if dropout_p > 0.0:
        raise ValueError(
            f"qattention under FlashFormats takes no dropout, got dropout_p={dropout_p}: flash "
            "would draw its dropout tile by tile, which cannot match the standard path's "
            "draws. Pass dropout_p=0.0 (as an eval model does), or leave flash unset"
        )
    if attn_mask is not None and attn_mask.requires_grad and torch.is_grad_enabled():
        raise ValueError(
            "qattention under FlashFormats gives attn_mask no gradient, and this one requires "
            "grad: detach it, or leave flash unset for the standard path, which does"
        )
    if enable_gqa:
        k, v = _repeated(q, k, "k"), _repeated(q, v, "v")
    batch = torch.broadcast_shapes(q.shape[:-2], k.shape[:-2], v.shape[:-2])
    q = q.expand(*batch, *q.shape[-2:])
    k = k.expand(*batch, *k.shape[-2:])
    v = v.expand(*batch, *v.shape[-2:])
    mask = None if attn_mask is None else attn_mask.expand(*batch, q.size(-2), k.size(-2))
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale
    rotation = None if formats.qk is None else formats.qk.rotation
    if rotation is not None:
        # along the head dimension, which the tiles do not cut: q and k rotated
        # whole, so that their gradients are summed in the rotated basis over
        # the tiles and rotated back once
        q, k = _Rotation.apply(q, rotation, -1), _Rotation.apply(k, rotation, -1)
    return _FlashAttention.apply(q, k, v, mask, formats, scale, is_causal)


def _tiles(n: int, size: int) -> list[tuple[int, int]]:
    """``[0, n)`` cut into tiles of ``size``, the last one shorter."""
    return [(i, min(i + size, n)) for i in range(0, n, size)]


def _skipped(causal: bool, rows: tuple[int, int], cols: tuple[int, int]) -> bool:
    """Whether a tile lies wholly above the diagonal of a causal mask (aligned
    to the top left), so that every one of its scores is masked."""
    return causal and cols[0] > rows[1] - 1


def _may_draw(step: Any) -> bool:
    """Whether a step may draw random bits: a stochastic rounding, or a
    callable mptorch does not know."""
    if step is None:
        return False
    step = _uncounted(step)  # a slot simulate(events=True) counts reads as its Quant
    if isinstance(step, Quant | ScaledQuant | BlockQuant | SplitMac | FusedMac | BlockMac):
        return step.rounding.is_stochastic
    return True


def _keeps_scores(formats: AttentionFormats) -> bool:
    """Whether flash keeps each tile's scores for backward instead of
    recomputing them: the operands are saved quantized, so a recompute
    gives the forward's words unless the product, its rounding or the
    scale's may draw random bits."""
    qk, scores = formats.qk, formats.scores
    steps = [] if qk is None else [qk.fwd_mac, qk.output_quant]
    steps.append(None if scores is None else scores.quant)
    return any([_may_draw(step) for step in steps])


def _slot(container: Any, name: str) -> Any:
    return None if container is None else getattr(container, name)


def _quantized(container: QMatmulFormats | None, name: str, t: torch.Tensor) -> Any:
    """``t`` through a container's operand slot, or as it is without one."""
    q = _slot(container, name)
    return t if q is None else q(t)


def _hook(container: QMatmulFormats | None, name: str, default: Any) -> Any:
    hook = _slot(container, name)
    return default if hook is None else hook


def _step(formats: Any, name: str, t: torch.Tensor) -> torch.Tensor:
    """``t``, fresh and in the carrier, rounded by a formats value's slot."""
    return _r(_slot(formats, name), t)


def _flash_scores(
    formats: AttentionFormats,
    qi: Any,
    kj: Any,
    scale: float,
    mask: torch.Tensor | None,
    causal: bool,
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> torch.Tensor:
    """One tile of scores, masked, in the carrier: ``qk``'s product of the
    quantized tiles (rounded by its ``output_quant``), times the scale as
    ``scores`` rounds it, plus the tile's mask."""
    fwd = _hook(formats._qk_tile, "fwd_math", CustomArithMatmul._default_fwd)
    s = fwd(qi, kj)
    sc = formats.scores
    if sc is None or sc.quant is None:
        s = s * scale
    else:
        s = qmul(s, scale, sc.quant, None, **sc._keywords)
    if mask is not None:
        s = s + _bias(mask[..., rows[0] : rows[1], cols[0] : cols[1]], s)
    elif causal and cols[1] - 1 > rows[0]:  # the tile crosses the diagonal
        r = torch.arange(rows[0], rows[1], device=s.device)[:, None]
        c = torch.arange(cols[0], cols[1], device=s.device)[None, :]
        s = s + _bias(r >= c, s)
    return s


def _scores_grad(formats: AttentionFormats, ds: torch.Tensor, scale: float) -> torch.Tensor:
    """The gradient of the scaled scores taken back through the scale, as
    :func:`~mptorch.quant.qmul`'s backward computes it: in the scores'
    carrier, rounded by their ``grad_quant``."""
    sc = formats.scores
    if sc is None or sc.grad_quant is None:
        return ds * scale
    c = _carrier_dtype(sc.carrier, ds.dtype)
    return _round_result(sc.grad_quant, ds.to(c) * scale, ds.dtype)


def _flash_forward(formats, q, k, v, mask, scale, causal):
    """FA2's forward (Algorithm 1): the output, and what backward reads."""
    fl, sm = formats.flash, formats.softmax
    dtype = q.dtype
    c = _carrier_dtype(fl.carrier, dtype)
    rows, cols = _tiles(q.size(-2), fl.block_q), _tiles(k.size(-2), fl.block_kv)
    qk, pv = formats._qk_tile, formats._pv_tile
    # each tile quantized once, in the carrier: q per query tile, k
    # (transposed, as the product's right operand) and v per key tile
    qq = [_quantized(qk, "a_quant", q[..., a:b, :].to(c)) for a, b in rows]
    kq = [_quantized(qk, "b_quant", k[..., a:b, :].to(c).mT) for a, b in cols]
    vq = [_quantized(pv, "b_quant", v[..., a:b, :].to(c)) for a, b in cols]
    pv_fwd = _hook(pv, "fwd_math", CustomArithMatmul._default_fwd)
    acc = None if sm is None else sm._specs[0]
    out_quant = None if formats.pv is None else formats.pv.output_quant
    keep = _keeps_scores(formats)
    batch, ev = q.shape[:-2], v.size(-1)
    outs, lses, kept = [], [], []
    for i, rr in enumerate(rows):
        n = rr[1] - rr[0]
        m = torch.full((*batch, n, 1), float("-inf"), dtype=c, device=q.device)
        total = torch.zeros((*batch, n, 1), dtype=c, device=q.device)
        o = torch.zeros((*batch, n, ev), dtype=c, device=q.device)
        for j, cc in enumerate(cols):
            if _skipped(causal, rr, cc):
                break  # every later tile of this row of tiles is above the diagonal too
            s = _flash_scores(formats, qq[i], kq[j], scale, mask, causal, rr, cc)
            if keep:
                kept.append(s)
            m_new = torch.maximum(m, s.amax(-1, keepdim=True))
            # a row with no score yet subtracts 0, as torch's _safe_softmax does:
            # its P is then exp(-inf) = 0 and its alpha exp(-inf) = 0, not NaN
            safe = m_new.masked_fill(m_new.isneginf(), 0.0)
            p = _step(sm, "exp", torch.exp(_step(sm, "sub", s - safe)))
            alpha = _r(fl.rescale, torch.exp(m - safe))
            total = _r(fl.combine, alpha * total + _total(acc, p, (p.dim() - 1,)))
            o = _r(fl.combine, alpha * o + pv_fwd(_quantized(pv, "a_quant", p), vq[j]))
            m = m_new
        dead = total == 0  # a row the mask leaves without a score: zeros, as the standard path
        o = _step(sm, "div", o / total.masked_fill(dead, 1.0))
        if out_quant is not None:
            o = out_quant(o)
        safe = m.masked_fill(m.isneginf(), 0.0)
        lse = _r(fl.lse, safe + torch.log(total)).masked_fill(dead, float("-inf"))
        outs.append(o)
        lses.append(lse)
    out = _stored(torch.cat(outs, -2), dtype)
    return out, (qq, kq, vq, lses, kept)


def _flash_backward(formats, saved, out, grad, mask, scale, causal, dtypes, dims, needs):
    """FA2's backward (Algorithm 2): the gradients of q, k and v that
    ``needs`` asks for, ``None`` for the others. ``dims`` is ``(L, S, E)``:
    the tiles may be packed, and carry no float shape."""
    need_q, need_k, need_v = needs
    need_s = need_q or need_k  # dP and dS feed only q's and k's gradients
    fl, sm = formats.flash, formats.softmax
    qq, kq, vq, lses, kept = saved
    c = lses[0].dtype
    n_queries, n_keys, e = dims
    rows, cols = _tiles(n_queries, fl.block_q), _tiles(n_keys, fl.block_kv)
    qk, pv = formats._qk_tile, formats._pv_tile
    qk_agrad = _hook(qk, "bwd_agrad_math", CustomArithMatmul._default_agrad)
    qk_bgrad = _hook(qk, "bwd_bgrad_math", CustomArithMatmul._default_bgrad)
    pv_agrad = _hook(pv, "bwd_agrad_math", CustomArithMatmul._default_agrad)
    pv_bgrad = _hook(pv, "bwd_bgrad_math", CustomArithMatmul._default_bgrad)
    bwd_acc = None if sm is None else sm._specs[1]
    g, o = grad.to(c), out.to(c)
    dout = [g[..., a:b, :] for a, b in rows]
    # D_i = rowsum(dO_i * O_i), once per query tile; and the output gradient
    # quantized once per query tile for each of pv's two gradients
    big_d = [
        _total(bwd_acc, _step(sm, "bwd_mul", dout[i] * o[..., a:b, :]), (o.dim() - 1,))
        if need_s
        else None
        for i, (a, b) in enumerate(rows)
    ]
    dout_p = [_quantized(pv, "agrad_quant", t) if need_s else None for t in dout]
    dout_v = [_quantized(pv, "bgrad_quant", t) if need_v else None for t in dout]
    batch, ev = out.shape[:-2], out.size(-1)
    dq = [torch.zeros((*batch, b - a, e), dtype=c, device=g.device) for a, b in rows]
    dk = [torch.zeros((*batch, b - a, e), dtype=c, device=g.device) for a, b in cols]
    dv = [torch.zeros((*batch, b - a, ev), dtype=c, device=g.device) for a, b in cols]
    tiles = {}
    if kept:
        keys = [
            (i, j)
            for i, rr in enumerate(rows)
            for j, cc in enumerate(cols)
            if not _skipped(causal, rr, cc)
        ]
        tiles = dict(zip(keys, kept, strict=True))
    for j, cc in enumerate(cols):
        for i, rr in enumerate(rows):
            if _skipped(causal, rr, cc):
                continue
            s = (
                tiles[(i, j)]
                if tiles
                else _flash_scores(formats, qq[i], kq[j], scale, mask, causal, rr, cc)
            )
            # P from the saved logsumexp, unrounded, as the standard softmax's
            # backward takes the unrounded softmax; a dead row's scores are all
            # -inf, so its P is 0 whatever its L
            lse = lses[i]
            p = torch.exp(s - lse.masked_fill(lse.isneginf(), 0.0))
            if need_v:
                pv_p = _quantized(pv, "a_quant", p)
                dv[j] = _r(fl.combine, dv[j] + pv_bgrad(dout_v[i], pv_p))
            if not need_s:
                continue
            dp = pv_agrad(dout_p[i], vq[j])
            ds = _step(sm, "bwd_out", p * _step(sm, "bwd_sub", dp - big_d[i]))
            ds = _scores_grad(formats, ds, scale)
            if need_q:
                dq_ij = qk_agrad(_quantized(qk, "agrad_quant", ds), kq[j])
                dq[i] = _r(fl.combine, dq[i] + dq_ij)
            if need_k:
                dk_ij = qk_bgrad(_quantized(qk, "bgrad_quant", ds), qq[i]).mT
                dk[j] = _r(fl.combine, dk[j] + dk_ij)
    q_dtype, k_dtype, v_dtype = dtypes
    return (
        _stored(torch.cat(dq, -2), q_dtype) if need_q else None,
        _stored(torch.cat(dk, -2), k_dtype) if need_k else None,
        _stored(torch.cat(dv, -2), v_dtype) if need_v else None,
    )


# --- the batched path: every query tile a key tile reaches, in one call ------------------
#
# The loop above runs one query tile against one key tile at a time, about
# twenty kernels a tile, and on a GPU the launches dominate. Each row's state
# (m, total, o) depends only on that row's scores, and its updates run in the
# order of the key tiles, so the query tiles a key tile reaches, a suffix of
# them under a causal mask, can take each step in one call, in the same order
# for every row. Each element's operations are then the loop's, word for word,
# where every step is elementwise or within a row: a Quant's rounding, torch's
# elementwise operations, and mptorch's products and sums, whose kernels
# compute each element of a call apart, in a fixed order. The backward's
# gradients of k and v add each query tile's product in turn, each rounded,
# so those products are computed in one call with the tiles as a batch
# dimension and then added in the loop's order.
#
# Not batched (_batchable): a product or a softmax sum that is torch's own (a
# pass without a mac): cuBLAS's words for a taller call are not its words for
# a tile (FusedMac(None), a product in the carrier, is mptorch's and
# batches); stochastic rounding, whose draws are keyed on an element's index
# in its call; a ScaledQuant, which scales each tile by its own amax; a block
# format's quantizer or mac; and any callable mptorch does not know. The
# forward and the backward decide apart, from the steps each runs (the
# backward recomputes the forward's scores, so it needs the forward's
# steps too): a recipe whose gradients are torch's still batches its
# forward. Either path's forward saves what either backward reads.


def _elementwise_step(step: Any) -> bool:
    """Whether a slot's rounding, or a mac's, is the same for an element
    whatever else its call holds."""
    if step is None:
        return True
    step = _uncounted(step)
    if isinstance(step, Quant):
        return not step.rounding.is_stochastic
    if isinstance(step, SplitMac | FusedMac):
        return not step.rounding.is_stochastic
    return False


# Each pass's steps: the products' slots, the softmax's slots and sums, the
# scores' rounding. A mac named here must be mptorch's (not None).
_FWD_GEMM_QUANTS = ("input_quant", "weight_quant", "output_quant")
_BWD_GEMM_QUANTS = ("igrad_quant", "wgrad_quant")
_FWD_GEMM_MACS, _BWD_GEMM_MACS = ("fwd_mac",), ("igrad_mac", "wgrad_mac")
_FWD_SOFTMAX = (("sub", "exp", "div"), ("acc",))
_BWD_SOFTMAX = (("bwd_mul", "bwd_sub", "bwd_out"), ("bwd_acc",))


def _pass_batchable(formats: AttentionFormats, gemm_quants, gemm_macs, softmax, score) -> bool:
    """Whether one pass's steps may run batched (see above)."""
    sm = formats.softmax
    if formats.qk is None or formats.pv is None or sm is None:
        return False  # a product or a sum torch's own
    quants, macs = [], []
    for gf in (formats.qk, formats.pv):
        quants += [getattr(gf, n) for n in gemm_quants]
        macs += [getattr(gf, n) for n in gemm_macs]
    quants += [getattr(sm, n) for n in softmax[0]]
    macs += [getattr(sm, n) for n in softmax[1]]
    if not all([isinstance(m, SplitMac | FusedMac) for m in macs]):
        return False
    if formats.scores is not None:
        quants.append(getattr(formats.scores, score))
    return all([_elementwise_step(step) for step in (*quants, *macs)])


def _batchable(formats: AttentionFormats) -> tuple[bool, bool]:
    """Whether flash's query tiles may run together in the forward, and in
    the backward (see above)."""
    fl = formats.flash
    assert fl is not None
    forward = _pass_batchable(
        formats, _FWD_GEMM_QUANTS, _FWD_GEMM_MACS, _FWD_SOFTMAX, "quant"
    ) and all([_elementwise_step(step) for step in (fl.rescale, fl.combine, fl.lse)])
    backward = forward and _pass_batchable(
        formats, _BWD_GEMM_QUANTS, _BWD_GEMM_MACS, _BWD_SOFTMAX, "grad_quant"
    )
    return forward, backward


def _reached(rows: list[tuple[int, int]], causal: bool, cols: tuple[int, int]) -> int | None:
    """The first query tile a key tile reaches (the later ones all do), or
    ``None`` when none does."""
    for i, rr in enumerate(rows):
        if not _skipped(causal, rr, cols):
            return i
    return None


def _batched_scores(formats, qa, kj, scale, mask, causal, rows, first, cols):
    """The scores of query tiles ``rows[first:]`` against one key tile, as
    `_flash_scores` computes each tile's: its rows are those of ``qa``, all of
    q's quantized tiles; the causal mask is added to the rows of the tiles
    that cross the diagonal only, as the loop adds it."""
    start, end = rows[first][0], rows[-1][1]
    fwd = _hook(formats._qk_tile, "fwd_math", CustomArithMatmul._default_fwd)
    s = fwd(qa[..., start:end, :], kj)
    sc = formats.scores
    if sc is None or sc.quant is None:
        s = s * scale
    else:
        s = qmul(s, scale, sc.quant, None, **sc._keywords)
    if mask is not None:
        s = s + _bias(mask[..., start:end, cols[0] : cols[1]], s)
    elif causal:
        crossing = [rr for rr in rows[first:] if cols[1] - 1 > rr[0]]
        if crossing:
            n = crossing[-1][1] - start
            r = torch.arange(start, start + n, device=s.device)[:, None]
            c = torch.arange(cols[0], cols[1], device=s.device)[None, :]
            s[..., :n, :].add_(_bias(r >= c, s))  # in place: `+=` on a slice writes it back
    return s


def _retire(full: torch.Tensor, active: torch.Tensor, lo: int, start: int) -> torch.Tensor:
    """Copy rows ``[lo, start)`` of a running state into ``full``, which holds
    every row, and return the rows from ``start`` on.

    ``active`` holds rows ``lo`` onwards. Under a causal mask the key tiles
    reach fewer query tiles as they go, so the rows they no longer reach are
    final: each is copied out once, and the steps update the rest out of
    place, as the loop does, instead of writing the whole suffix back at
    every key tile."""
    full[..., lo:start, :].copy_(active[..., : start - lo, :])
    return active[..., start - lo :, :]


def _flash_forward_batched(formats, q, k, v, mask, scale, causal):
    """`_flash_forward`, one call per key tile (see above)."""
    fl, sm = formats.flash, formats.softmax
    dtype = q.dtype
    c = _carrier_dtype(fl.carrier, dtype)
    nq = q.size(-2)
    rows, cols = _tiles(nq, fl.block_q), _tiles(k.size(-2), fl.block_kv)
    qk, pv = formats._qk_tile, formats._pv_tile
    # elementwise slots: q quantized whole has each tile's words
    qa = _quantized(qk, "a_quant", q.to(c))
    kq = [_quantized(qk, "b_quant", k[..., a:b, :].to(c).mT) for a, b in cols]
    vq = [_quantized(pv, "b_quant", v[..., a:b, :].to(c)) for a, b in cols]
    pv_fwd = _hook(pv, "fwd_math", CustomArithMatmul._default_fwd)
    acc = None if sm is None else sm._specs[0]
    out_quant = None if formats.pv is None else formats.pv.output_quant
    batch, ev = q.shape[:-2], v.size(-1)
    shapes = ((*batch, nq, 1), (*batch, nq, 1), (*batch, nq, ev))
    full = [torch.empty(shape, dtype=c, device=q.device) for shape in shapes]
    m = torch.full(shapes[0], float("-inf"), dtype=c, device=q.device)
    total = torch.zeros(shapes[1], dtype=c, device=q.device)
    o = torch.zeros(shapes[2], dtype=c, device=q.device)
    lo = 0  # the first row m, total and o hold
    for j, cc in enumerate(cols):
        first = _reached(rows, causal, cc)
        if first is None:
            break  # every later key tile is above the diagonal of every query tile
        start = rows[first][0]
        if start > lo:
            m, total, o = [
                _retire(f, t, lo, start) for f, t in zip(full, (m, total, o), strict=True)
            ]
            lo = start
        s = _batched_scores(formats, qa, kq[j], scale, mask, causal, rows, first, cc)
        m_new = torch.maximum(m, s.amax(-1, keepdim=True))
        safe = m_new.masked_fill(m_new.isneginf(), 0.0)
        p = _step(sm, "exp", torch.exp(_step(sm, "sub", s - safe)))
        alpha = _r(fl.rescale, torch.exp(m - safe))
        total = _r(fl.combine, alpha * total + _total(acc, p, (p.dim() - 1,)))
        o = _r(fl.combine, alpha * o + pv_fwd(_quantized(pv, "a_quant", p), vq[j]))
        m = m_new
    for f, t in zip(full, (m, total, o), strict=True):
        f[..., lo:, :].copy_(t)
    m, total, o = full
    dead = total == 0
    o = _step(sm, "div", o / total.masked_fill(dead, 1.0))
    if out_quant is not None:
        o = out_quant(o)
    safe = m.masked_fill(m.isneginf(), 0.0)
    lse = _r(fl.lse, safe + torch.log(total)).masked_fill(dead, float("-inf"))
    out = _stored(o, dtype)
    qq = [qa[..., a:b, :] for a, b in rows]
    lses = [lse[..., a:b, :] for a, b in rows]
    return out, (qq, kq, vq, lses, [])


def _per_tile(hook, a: torch.Tensor, b: torch.Tensor, rows, first: int, size: int) -> list:
    """``hook(a_i, b_i)`` for each query tile ``i >= first``, where ``a`` and
    ``b`` hold those tiles' rows: the whole tiles in one call, with the tiles
    as a batch dimension, and a shorter last tile in a call of its own."""
    whole = sum(1 for a0, b0 in rows[first:] if b0 - a0 == size)
    out: list = []
    if whole:
        n = whole * size
        prods = hook(
            a[..., :n, :].unflatten(-2, (whole, size)), b[..., :n, :].unflatten(-2, (whole, size))
        )
        out += list(prods.unbind(-3))
    if whole < len(rows) - first:
        n = whole * size
        out.append(hook(a[..., n:, :], b[..., n:, :]))
    return out


def _flash_backward_batched(formats, saved, out, grad, mask, scale, causal, dtypes, dims, needs):
    """`_flash_backward`, one call per key tile (see above)."""
    need_q, need_k, need_v = needs
    need_s = need_q or need_k
    fl, sm = formats.flash, formats.softmax
    qq, kq, vq, lses, _ = saved
    c = lses[0].dtype
    n_queries, n_keys, e = dims
    rows, cols = _tiles(n_queries, fl.block_q), _tiles(n_keys, fl.block_kv)
    qk, pv = formats._qk_tile, formats._pv_tile
    qk_agrad = _hook(qk, "bwd_agrad_math", CustomArithMatmul._default_agrad)
    qk_bgrad = _hook(qk, "bwd_bgrad_math", CustomArithMatmul._default_bgrad)
    pv_agrad = _hook(pv, "bwd_agrad_math", CustomArithMatmul._default_agrad)
    pv_bgrad = _hook(pv, "bwd_bgrad_math", CustomArithMatmul._default_bgrad)
    bwd_acc = None if sm is None else sm._specs[1]
    g, o = grad.to(c), out.to(c)
    qa, lse = torch.cat(qq, -2), torch.cat(lses, -2)
    big_d = _total(bwd_acc, _step(sm, "bwd_mul", g * o), (o.dim() - 1,)) if need_s else None
    dout_p = _quantized(pv, "agrad_quant", g) if need_s else None
    dout_v = _quantized(pv, "bgrad_quant", g) if need_v else None
    batch = out.shape[:-2]
    dq_full = torch.empty((*batch, n_queries, e), dtype=c, device=g.device)
    dq = torch.zeros((*batch, n_queries, e), dtype=c, device=g.device)
    lo = 0  # the first row dq holds
    dk = [torch.zeros((*batch, b - a, e), dtype=c, device=g.device) for a, b in cols]
    dv = [torch.zeros((*batch, b - a, out.size(-1)), dtype=c, device=g.device) for a, b in cols]
    for j, cc in enumerate(cols):
        first = _reached(rows, causal, cc)
        if first is None:
            break
        start = rows[first][0]
        if start > lo:
            dq = _retire(dq_full, dq, lo, start)
            lo = start
        s = _batched_scores(formats, qa, kq[j], scale, mask, causal, rows, first, cc)
        lse_r = lse[..., start:, :]
        p = torch.exp(s - lse_r.masked_fill(lse_r.isneginf(), 0.0))
        if need_v:
            assert dout_v is not None
            pv_p = _quantized(pv, "a_quant", p)
            for prod in _per_tile(pv_bgrad, dout_v[..., start:, :], pv_p, rows, first, fl.block_q):
                dv[j] = _r(fl.combine, dv[j] + prod)
        if not need_s:
            continue
        assert dout_p is not None and big_d is not None
        dp = pv_agrad(dout_p[..., start:, :], vq[j])
        ds = _step(sm, "bwd_out", p * _step(sm, "bwd_sub", dp - big_d[..., start:, :]))
        ds = _scores_grad(formats, ds, scale)
        if need_q:
            dq_j = qk_agrad(_quantized(qk, "agrad_quant", ds), kq[j])
            dq = _r(fl.combine, dq + dq_j)
        if need_k:
            ds_k = _quantized(qk, "bgrad_quant", ds)
            for prod in _per_tile(qk_bgrad, ds_k, qa[..., start:, :], rows, first, fl.block_q):
                dk[j] = _r(fl.combine, dk[j] + prod.mT)
    dq_full[..., lo:, :].copy_(dq)
    q_dtype, k_dtype, v_dtype = dtypes
    return (
        _stored(dq_full, q_dtype) if need_q else None,
        _stored(torch.cat(dk, -2), k_dtype) if need_k else None,
        _stored(torch.cat(dv, -2), v_dtype) if need_v else None,
    )


class _FlashAttention(torch.autograd.Function):
    """FA2's forward and backward over the parts of an ``AttentionFormats``
    with a ``flash`` part. Its saved tensors are the quantized tiles of q, k
    and v, the output, the logsumexp of each query tile, and, when the scores
    may draw random bits, each tile's scores (``_keeps_scores``)."""

    @staticmethod
    def forward(ctx, q, k, v, mask, formats, scale, causal):
        with torch._C.DisableTorchFunction():
            batched, batched_backward = _batchable(formats)
            run = _flash_forward_batched if batched else _flash_forward
            out, saved = run(formats, q, k, v, mask, scale, causal)
            qq, kq, vq, lses, kept = saved
            counts = (len(qq), len(kq), len(vq), len(lses), len(kept))
            items = [*qq, *kq, *vq, *lses, *kept, out, mask]
            ctx.save_for_backward(*[t if isinstance(t, torch.Tensor) else None for t in items])
            ctx.structs = [None if isinstance(t, torch.Tensor) else t for t in items]
            ctx.counts = counts
            ctx.formats, ctx.scale, ctx.causal = formats, scale, causal
            ctx.batched = batched_backward
            ctx.dtypes = (q.dtype, k.dtype, v.dtype)
            ctx.dims = (q.size(-2), k.size(-2), q.size(-1))
            return out

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():  # as in forward, except under compile (_opaque.py)
            items = [
                s if t is None else t for t, s in zip(ctx.saved_tensors, ctx.structs, strict=True)
            ]
            n_q, n_k, n_v, n_l, n_s = ctx.counts
            at = [0, n_q, n_q + n_k, n_q + n_k + n_v, n_q + n_k + n_v + n_l]
            saved = (
                items[: at[1]],
                items[at[1] : at[2]],
                items[at[2] : at[3]],
                items[at[3] : at[4]],
                items[at[4] : at[4] + n_s],
            )
            out, mask = items[-2], items[-1]
            needs = tuple(ctx.needs_input_grad[:3])
            run = _flash_backward_batched if ctx.batched else _flash_backward
            dq, dk, dv = run(
                ctx.formats,
                saved,
                out,
                grads[0],
                mask,
                ctx.scale,
                ctx.causal,
                ctx.dtypes,
                ctx.dims,
                needs,
            )
            return dq, dk, dv, None, None, None, None
