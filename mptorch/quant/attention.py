"""Attention, each of its parts rounded as its formats say.

:func:`qattention` is ``F.scaled_dot_product_attention`` written out of the
pieces mptorch already has, each with its own formats and its own gradient:
the product ``q @ k^T`` (a :class:`~mptorch.quant.QMatmul`'s Function), the
scale (:func:`~mptorch.quant.qmul`), the mask, the softmax
(:func:`~mptorch.quant.qsoftmax`), dropout and the product with ``v``. It is
the arithmetic of transformers' eager attention, step for step, so a model
that calls SDPA computes the words its eager twin computes under
:meth:`~mptorch.quant.Recipe.attention_parts`, gradients included.

``docs/source/kernels/attention.rst`` is the user's account.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from mptorch.number import BlockFormat

# Module references rather than names: `norm` and `pointwise` import
# `modules`, whose QMultiheadAttention imports this module, while they are
# still loading.
from . import norm, pointwise
from .mac import BlockMac
from .modules.format import QMatmulFormats
from .modules.linear import CustomArithLinear
from .modules.matmul import CustomArithMatmul

if TYPE_CHECKING:
    from .recipe import GemmFormats

__all__ = ["AttentionFormats", "qattention"]

_NARROW = (torch.float16, torch.bfloat16)


# The default of `AttentionFormats.uniform`'s gradient arguments: the forward's.
_SAME: Any = object()

_PARTS = ("qk", "scores", "softmax", "pv")


def _part_types() -> dict[str, type]:
    # imported here: recipe imports this module, and modules (which pointwise
    # imports) imports it too
    from .recipe import GemmFormats

    return {
        "qk": GemmFormats,
        "scores": pointwise.PointwiseFormats,
        "softmax": norm.SoftmaxFormats,
        "pv": GemmFormats,
    }


@dataclass(frozen=True, kw_only=True)
class AttentionFormats:
    """The formats of an attention block's core: its two products, the
    scores and the softmax.

    :func:`~mptorch.quant.qattention` computes ``softmax(q @ k^T * scale +
    mask) @ v`` from these parts, each a value that already exists for a
    computation of its own. A part left ``None`` is PyTorch's own step, and
    with every part ``None`` the attention is
    ``F.scaled_dot_product_attention``. It is the value of a recipe's
    ``attention`` kind, which ``F.scaled_dot_product_attention`` and
    ``nn.MultiheadAttention``'s core take, and
    :meth:`~mptorch.quant.Recipe.attention_parts` gives the same parts to an eager
    attention's two products and its softmax.

    Args:
        qk (GemmFormats, optional): the product ``q @ k^T``, as a
            ``matmul`` kind's formats: ``input_quant`` rounds ``q``,
            ``weight_quant`` rounds ``k``, ``output_quant`` rounds the
            product; ``igrad_mac`` computes ``q``'s gradient and ``wgrad_mac``
            ``k``'s, and ``igrad_quant`` and ``wgrad_quant`` round the scores'
            gradient for each. Default: ``None``, PyTorch's product
        scores (PointwiseFormats, optional): the product times the scale, as
            an ``elementwise`` kind's formats (its rounding, its gradient's,
            its carrier). Default: ``None``, PyTorch's multiply
        softmax (SoftmaxFormats, optional): the softmax of the masked scores.
            Default: ``None``, PyTorch's softmax
        pv (GemmFormats, optional): the product of the probabilities and
            ``v``: ``input_quant`` rounds the probabilities (after dropout),
            ``weight_quant`` rounds ``v``, ``output_quant`` rounds the result;
            ``igrad_mac`` computes the probabilities' gradient and
            ``wgrad_mac`` ``v``'s, and the gradient quantizers round the
            output's gradient for each. Default: ``None``, PyTorch's product

    Raises:
        TypeError: for a part that is not its type or ``None``.

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import AttentionFormats, Quant, SplitMac
        >>> p8 = BinaryK(8, 4)
        >>> af = AttentionFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
    """

    qk: "GemmFormats | None" = None
    scores: "pointwise.PointwiseFormats | None" = None
    softmax: "norm.SoftmaxFormats | None" = None
    pv: "GemmFormats | None" = None
    # The two products' containers, built once here so that a call (functional,
    # or simulate's, perhaps while Dynamo traces) builds none: a container built
    # during a trace gave wrong gradients. A converted layer builds its own.
    _qk: QMatmulFormats | None = field(init=False, repr=False, compare=False, default=None)
    _pv: QMatmulFormats | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        for name, cls in _part_types().items():
            value = getattr(self, name)
            if value is not None and not isinstance(value, cls):
                raise TypeError(
                    f"AttentionFormats' {name} is a {cls.__name__} or None, "
                    f"got {type(value).__name__}"
                )
        object.__setattr__(self, "_qk", None if self.qk is None else self.qk.matmul())
        object.__setattr__(self, "_pv", None if self.pv is None else self.pv.matmul())

    @classmethod
    def uniform(
        cls, mac: Any, quant: Any = None, *, grad_mac: Any = _SAME, grad_quant: Any = _SAME
    ) -> "AttentionFormats":
        """Both products as ``GemmFormats.uniform(mac, quant, ...)``, and the
        softmax as ``SoftmaxFormats.uniform(mac, quant, ...)`` in ``mac``'s
        carrier; the scores and the result are left unrounded.

        ``grad_mac`` and ``grad_quant`` give the gradients their own mac and
        rounding, as :meth:`~mptorch.quant.GemmFormats.uniform`'s do; ``grad_quant`` also
        replaces the softmax's backward roundings. A ``BlockMac`` (or a
        block format) multiplies the two products only, and leaves the
        softmax PyTorch's, since a block format has no sum of its own.
        """
        from .recipe import GemmFormats  # imported here: recipe imports this module

        grad_mac = mac if grad_mac is _SAME else grad_mac
        grad_quant = quant if grad_quant is _SAME else grad_quant
        gemm = GemmFormats.uniform(mac, quant, grad_mac=grad_mac, grad_quant=grad_quant)
        softmax = None
        if not isinstance(mac, BlockMac | BlockFormat):
            softmax = norm.SoftmaxFormats.uniform(
                mac, quant, grad=grad_quant, carrier=getattr(mac, "carrier", None)
            )
        return cls(qk=gemm, softmax=softmax, pv=gemm)

    @property
    def _plain(self) -> bool:
        """Whether every part is ``None``: the attention is PyTorch's."""
        return self.qk is None and self.scores is None and self.softmax is None and self.pv is None

    def label(self) -> str:
        """A one-line summary of the formats, for reports."""
        parts = [
            f"{name} [{getattr(self, name).label()}]"
            for name in _PARTS
            if getattr(self, name) is not None
        ]
        return "; ".join(parts) or "torch"


def qattention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    formats: AttentionFormats | None = None,
    *,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """``F.scaled_dot_product_attention(q, k, v, ...)``, each part rounded as
    ``formats`` says.

    It computes, in this order, each step a call of its own:

    1. with ``enable_gqa``, ``k`` and ``v`` repeated over each group of query
       heads, as transformers' ``repeat_kv`` repeats them (query head ``h``
       reads key head ``h // G``);
    2. ``s = q @ k^T``, with ``formats.qk``;
    3. ``s * scale``, with ``formats.scores``, the scale ``1/sqrt(E)`` by
       default. The product is scaled after the GEMM, as transformers' eager
       attention scales it, where PyTorch's math kernel scales ``q`` and
       ``k`` by ``sqrt(scale)`` each;
    4. the mask added: a boolean mask (``True`` attends), or
       ``is_causal``'s, as ``0`` and ``-inf``, a float mask as it is, both in
       the scores' dtype;
    5. the softmax, with ``formats.softmax``, in float32 for float16 and
       bfloat16 scores and then stored in their dtype, as Llama's eager
       attention computes it. A row the mask leaves without a score is all
       zeros, as in PyTorch's kernels, and its gradient is zero;
    6. PyTorch's dropout, when ``dropout_p > 0``, in eval as in training, as
       SDPA applies it;
    7. ``p @ v``, with ``formats.pv``.

    Each step's result is stored in the operands' dtype, as the eager
    spelling stores it, and each part computes and rounds in its own
    carrier (a mac's, the scores' and the softmax's formats say which). With
    ``formats=None``, or every part ``None``, this is PyTorch's SDPA itself.
    The backward is each step's own: the closed forms of the two products'
    and the softmax's, each rounded as its formats say, and PyTorch's for
    the rest.

    An SDPA model and its eager twin under
    :meth:`~mptorch.quant.Recipe.attention_parts` compute the same words,
    except in four cases: an ``elementwise`` kind on the attention module,
    which rounds the eager spelling's ``* scaling`` and ``+ mask`` but none
    of this function's steps; a ``softmax.sub`` format that, under a directed
    rounding, rounds the eager mask's ``finfo.min`` to a finite value where
    this function's ``-inf`` stays infinite; GPT-2's softmax, which has no
    ``dtype``, on float16 or bfloat16 scores; and a float64 Llama, whose
    eager softmax computes in float32.

    Args:
        q, k, v (Tensor): ``[..., L, E]``, ``[..., S, E]`` and ``[..., S, Ev]``,
            of one float dtype; with ``enable_gqa``, ``[..., H, L, E]`` with
            ``k`` and ``v`` of ``Hkv`` heads, ``Hkv`` dividing ``H``.
        formats (AttentionFormats, optional): the parts' formats.
            Default: ``None``, PyTorch's SDPA
        attn_mask (Tensor, optional): a boolean mask (``True`` attends) or a
            float one added to the scores, broadcast to ``[..., L, S]``.
            Default: ``None``
        dropout_p (float): the probability of dropping a probability.
            Default: ``0.0``
        is_causal (bool): mask the scores above the diagonal, aligned to the
            top left. Default: ``False``
        scale (float, optional): the scores' scale. Default: ``1/sqrt(E)``
        enable_gqa (bool): share each of ``k`` and ``v``'s heads among a group
            of query heads. Default: ``False``

    Returns:
        Tensor: ``[..., L, Ev]``, in the operands' dtype.

    Raises:
        TypeError: if ``formats`` is not an ``AttentionFormats``.
        RuntimeError: for ``attn_mask`` with ``is_causal``, as SDPA raises,
            or with ``enable_gqa`` for head counts that do not divide.
    """
    with torch._C.DisableTorchFunction():  # the op, not the calls it makes
        if formats is not None and not isinstance(formats, AttentionFormats):
            raise TypeError(
                f"qattention's formats is an AttentionFormats or None, got {type(formats).__name__}"
            )
        if formats is None or formats._plain:
            gqa = {"enable_gqa": True} if enable_gqa else {}
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask, dropout_p, is_causal, scale=scale, **gqa
            )
        out, _ = _attention(
            q,
            k,
            v,
            formats._qk,
            formats.scores,
            formats.softmax,
            formats._pv,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
        )
        return out


def _attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk: QMatmulFormats | None,
    scores: Any,
    softmax: Any,
    pv: QMatmulFormats | None,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`qattention`'s steps, given the two products' containers:
    ``(output, probabilities)``, the probabilities after dropout. Called with
    torch function modes disabled."""
    if attn_mask is not None and is_causal:
        raise RuntimeError(
            "qattention was given both attn_mask and is_causal=True: is_causal is a mask of "
            "its own -- pass one of them, as F.scaled_dot_product_attention requires"
        )
    if enable_gqa:
        k, v = _repeated(q, k, "k"), _repeated(q, v, "v")
    s = torch.matmul(q, k.mT) if qk is None else CustomArithMatmul.apply(q, k.mT, qk)
    scale = 1.0 / math.sqrt(q.size(-1)) if scale is None else scale
    if scores is None:
        s = s * scale
    else:
        s = pointwise.qmul(s, scale, scores.quant, scores.grad_quant, **scores._keywords)
    dead = None
    if attn_mask is not None:
        s = s + _bias(attn_mask, s)
        # A row the mask leaves without a score: zeros, as PyTorch's kernels give.
        # Its scores are replaced before the softmax as well as after it,
        # since a row of -inf would leave NaN in the softmax's graph, and its
        # backward would carry NaN into every gradient of q and k.
        dead = s.isneginf().all(-1, keepdim=True)
        s = s.masked_fill(dead, 0.0)
    elif is_causal:
        allowed = torch.ones(s.size(-2), s.size(-1), dtype=torch.bool, device=s.device).tril()
        s = s + _bias(allowed, s)
    p = _softmax(s, softmax)
    if dead is not None:
        p = p.masked_fill(dead, 0.0)
    if dropout_p > 0.0:
        p = F.dropout(p, dropout_p)
    out = torch.matmul(p, v) if pv is None else CustomArithMatmul.apply(p, v, pv)
    return out, p


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


def _softmax(s: torch.Tensor, formats: Any) -> torch.Tensor:
    """The scores' softmax over their last dimension, as Llama's eager
    attention spells it: in float32 for float16 and bfloat16 scores, then
    stored in their dtype."""
    narrow = s.dtype in _NARROW
    dtype = torch.float32 if narrow else None
    p = (
        F.softmax(s, -1, dtype=dtype)
        if formats is None
        else norm.qsoftmax(s, -1, formats, dtype=dtype)
    )
    return p.to(s.dtype) if narrow else p


# --- nn.MultiheadAttention ---------------------------------------------------------------


def _linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None, formats: Any):
    """A projection: ``F.linear``, or a ``QLinear``'s Function over ``formats``."""
    return F.linear(x, w, b) if formats is None else CustomArithLinear.apply(x, w, b, formats)


def _in_projection_packed(q, k, v, w, b, formats):
    """``F._in_projection_packed`` with each projection over ``formats``: one
    product over the packed weight when ``q is k is v``, a split one when
    ``k is v``, three otherwise, as PyTorch's."""
    E = q.size(-1)
    if k is v:
        if q is k:
            proj = _linear(q, w, b, formats)
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
            proj = proj.contiguous()
            return proj[0], proj[1], proj[2]
        w_q, w_kv = w.split([E, E * 2])
        b_q, b_kv = (None, None) if b is None else b.split([E, E * 2])
        q_proj = _linear(q, w_q, b_q, formats)
        kv_proj = _linear(k, w_kv, b_kv, formats)
        kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).squeeze(-2)
        kv_proj = kv_proj.contiguous()
        return q_proj, kv_proj[0], kv_proj[1]
    w_q, w_k, w_v = w.chunk(3)
    b_q, b_k, b_v = (None, None, None) if b is None else b.chunk(3)
    return (
        _linear(q, w_q, b_q, formats),
        _linear(k, w_k, b_k, formats),
        _linear(v, w_v, b_v, formats),
    )


def _qmha_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: torch.Tensor | None,
    in_proj_bias: torch.Tensor | None,
    bias_k: torch.Tensor | None,
    bias_v: torch.Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    training: bool = True,
    key_padding_mask: torch.Tensor | None = None,
    need_weights: bool = True,
    attn_mask: torch.Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: torch.Tensor | None = None,
    k_proj_weight: torch.Tensor | None = None,
    v_proj_weight: torch.Tensor | None = None,
    static_k: torch.Tensor | None = None,
    static_v: torch.Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    *,
    in_proj: Any = None,
    out_proj: Any = None,
    attention: AttentionFormats | None = None,
    qk: QMatmulFormats | None = None,
    pv: QMatmulFormats | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``F.multi_head_attention_forward``, its projections and its core in
    mptorch's arithmetic.

    PyTorch's function, line for line, except where it computes: each
    projection is ``F.linear`` without formats and a ``QLinear``'s Function
    over them with (``in_proj`` for the in-projections, one container, or one
    per projection with separate weights; ``out_proj`` for the output's),
    and the core is PyTorch's own without ``attention`` and
    :func:`qattention`'s steps with it, over ``qk``'s and ``pv``'s
    containers, with the scale ``1/sqrt(head_dim)``. Called with torch
    function modes disabled.
    """
    # torch's own check (functional.pyi does not declare it)
    is_batched = F._mha_shape_check(  # ty: ignore[missing-attribute]
        query, key, value, key_padding_mask, attn_mask, num_heads
    )
    if not is_batched:
        query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)
    tgt_len, bsz, embed_dim = query.shape
    src_len = key.shape[0]
    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )
    if is_causal and attn_mask is None:
        raise RuntimeError(
            "is_causal=True is a hint that attn_mask is the causal mask, and no attn_mask was "
            "given: pass the mask too, as nn.MultiheadAttention requires"
        )
    if is_causal and key_padding_mask is None and not need_weights:
        attn_mask = None
    else:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            is_causal = False
    if embed_dim != embed_dim_to_check:
        raise AssertionError(
            f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
        )
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise AssertionError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

    if not use_separate_proj_weight:
        if in_proj_weight is None:
            raise AssertionError("use_separate_proj_weight is False but in_proj_weight is None")
        formats = in_proj[0] if isinstance(in_proj, tuple) else in_proj
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias, formats)
    else:
        if q_proj_weight is None or k_proj_weight is None or v_proj_weight is None:
            raise AssertionError("use_separate_proj_weight is True but a projection weight is None")
        b_q, b_k, b_v = (None, None, None) if in_proj_bias is None else in_proj_bias.chunk(3)
        f_q, f_k, f_v = in_proj if isinstance(in_proj, tuple) else (in_proj,) * 3
        q = _linear(query, q_proj_weight, b_q, f_q)
        k = _linear(key, k_proj_weight, b_k, f_k)
        v = _linear(value, v_proj_weight, b_v, f_v)

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            if attn_mask.shape != (tgt_len, src_len):
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be "
                    f"{(tgt_len, src_len)}."
                )
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            if attn_mask.shape != (bsz * num_heads, tgt_len, src_len):
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be "
                    f"{(bsz * num_heads, tgt_len, src_len)}."
                )
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if bias_k is not None and bias_v is not None:
        if static_k is not None or static_v is not None:
            raise AssertionError("bias cannot be added to a static key or value.")
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    elif bias_k is not None or bias_v is not None:
        raise AssertionError("bias_k and bias_v are set together or not at all")

    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        v = static_v
    if add_zero_attn:
        zeros = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zeros, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zeros, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    src_len = k.size(1)
    if key_padding_mask is not None:
        key_padding_mask = (
            key_padding_mask.view(bsz, 1, 1, src_len)
            .expand(-1, num_heads, -1, -1)
            .reshape(bsz * num_heads, 1, src_len)
        )
        attn_mask = key_padding_mask if attn_mask is None else attn_mask + key_padding_mask
    if not training:
        dropout_p = 0.0

    weights: torch.Tensor | None = None
    if attention is not None and not attention._plain:
        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(0)
                if attn_mask.size(0) == 1
                else attn_mask.view(bsz, num_heads, -1, src_len)
            )
        out, p = _attention(
            q.view(bsz, num_heads, tgt_len, head_dim),
            k.view(bsz, num_heads, src_len, head_dim),
            v.view(bsz, num_heads, src_len, head_dim),
            qk,
            attention.scores,
            attention.softmax,
            pv,
            attn_mask,
            dropout_p,
            is_causal and attn_mask is None,
            None,
            False,
        )
        out = out.permute(2, 0, 1, 3).reshape(bsz * tgt_len, embed_dim)
        if need_weights:
            weights = p.mean(dim=1) if average_attn_weights else p
    elif need_weights:  # PyTorch's own core, as multi_head_attention_forward computes it
        q_scaled = q * math.sqrt(1.0 / float(q.shape[-1]))
        if is_causal and attn_mask is None:
            raise AssertionError("FIXME: is_causal not implemented for need_weights")
        if attn_mask is not None:
            p = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            p = torch.bmm(q_scaled, k.transpose(-2, -1))
        p = F.softmax(p, dim=-1)
        if dropout_p > 0.0:
            p = F.dropout(p, p=dropout_p)
        out = torch.bmm(p, v).transpose(0, 1).reshape(tgt_len * bsz, embed_dim)
        p = p.view(bsz, num_heads, tgt_len, src_len)
        weights = p.mean(dim=1) if average_attn_weights else p
    else:
        if attn_mask is not None:
            attn_mask = (
                attn_mask.unsqueeze(0)
                if attn_mask.size(0) == 1 and attn_mask.dim() == 3
                else attn_mask.view(bsz, num_heads, -1, src_len)
            )
        out = F.scaled_dot_product_attention(
            q.view(bsz, num_heads, tgt_len, head_dim),
            k.view(bsz, num_heads, src_len, head_dim),
            v.view(bsz, num_heads, src_len, head_dim),
            attn_mask,
            dropout_p,
            is_causal,
        )
        out = out.permute(2, 0, 1, 3).reshape(bsz * tgt_len, embed_dim)
    out = _linear(out, out_proj_weight, out_proj_bias, out_proj)
    out = out.view(tgt_len, bsz, out.size(1))
    if not is_batched:
        out = out.squeeze(1)
        if weights is not None:
            weights = weights.squeeze(0)
    return out, weights
