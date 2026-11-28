"""``nn.MultiheadAttention`` in mptorch's arithmetic: :class:`QMultiheadAttention`.

Its formats are a :class:`QAttentionFormats`: a ``QLinear``'s container for
the in-projections and one for the output projection, and an
:class:`~mptorch.quant.AttentionFormats` for the core. The layer is torch's
class with that ``formats`` attribute, so its parameters and ``state_dict``
are torch's, and :func:`~mptorch.quant.convert_` turns torch's modules into
it.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# A module reference rather than names: `attention` imports this package's
# Functions, and its names are read when a layer is built or runs.
from .. import attention as _attention
from .format import QAffineFormats, QMatmulFormats


class QAttentionFormats(nn.Module):
    """The formats of a :class:`QMultiheadAttention`: its projections' and its core's.

    It is a module, as :class:`~mptorch.quant.QAffineFormats` is, so that a
    quantizer that carries state is registered in the model's
    ``state_dict``. A part left ``None`` is PyTorch's.

    Args:
        in_proj (QAffineFormats, optional): the in-projections' container,
            for the packed weight or for each of the three separate ones.
            Default: ``None``, ``F.linear``
        out_proj (QAffineFormats, optional): the output projection's
            container. Default: ``None``, ``F.linear``
        attention (AttentionFormats, optional): the core's formats. The
            container builds its own containers for the core's two products
            from them, so no two layers share one. Default: ``None``,
            PyTorch's core
    """

    def __init__(
        self,
        in_proj: QAffineFormats | None = None,
        out_proj: QAffineFormats | None = None,
        attention: Any = None,
    ) -> None:
        super().__init__()
        if attention is not None and not isinstance(attention, _attention.AttentionFormats):
            raise TypeError(
                f"QAttentionFormats' attention is an AttentionFormats or None, "
                f"got {type(attention).__name__}"
            )
        self.in_proj = in_proj
        self.out_proj = out_proj
        self.attention = attention
        self.qk: QMatmulFormats | None = None
        self.pv: QMatmulFormats | None = None
        if attention is not None:
            self.qk = None if attention.qk is None else attention.qk.matmul()
            self.pv = None if attention.pv is None else attention.pv.matmul()

    @property
    def _plain(self) -> bool:
        """Whether every part is PyTorch's."""
        return (
            self.in_proj is None
            and self.out_proj is None
            and (self.attention is None or self.attention._plain)
        )


@dataclass(frozen=True)
class MultiheadAttentionFormats:
    """What a recipe gives an ``nn.MultiheadAttention``: a conversion's
    report row holds it, and :meth:`container` builds the layer's
    :class:`QAttentionFormats` from it.

    Args:
        in_proj (GemmFormats, optional): the ``linear`` kind's formats at the
            module's name, for its in-projections.
        out_proj (GemmFormats, optional): the ``linear`` kind's formats at its
            ``out_proj``'s name.
        attention (AttentionFormats, optional): the ``attention`` kind's
            formats at the module's name, for its core.
    """

    in_proj: Any
    out_proj: Any
    attention: Any

    def label(self) -> str:
        """A one-line summary, for reports."""
        parts = (
            f"{name} [{'torch' if value is None else value.label()}]"
            for name, value in (
                ("in_proj", self.in_proj),
                ("out_proj", self.out_proj),
                ("attention", self.attention),
            )
        )
        return "; ".join(parts)

    def container(self) -> QAttentionFormats:
        """A new container for one layer."""
        return QAttentionFormats(
            None if self.in_proj is None else self.in_proj.affine(),
            None if self.out_proj is None else self.out_proj.affine(),
            self.attention,
        )


class QMultiheadAttention(nn.MultiheadAttention):
    """:class:`torch.nn.MultiheadAttention` in the arithmetic of ``formats``.

    It takes torch's constructor arguments and has torch's parameters, so a
    pretrained module's ``state_dict`` loads, plus ``formats``. Its forward
    is torch's slow path, with each projection a ``QLinear``'s product over
    ``formats.in_proj`` or ``formats.out_proj`` and the core
    :func:`~mptorch.quant.qattention`'s steps over ``formats.attention``,
    with the scale ``1/sqrt(head_dim)``. A part left ``None`` is PyTorch's;
    with ``formats=None``, or every part ``None``, the forward is torch's,
    fast path included.

    With ``formats.attention`` set, two more things differ from torch's
    ``need_weights=True`` path than the arithmetic: the scores are the
    product times the scale, as in :func:`~mptorch.quant.qattention`, where
    torch's path scales the query first; and a query whose every key is
    masked attends to nothing (its output is the output projection's bias),
    where torch's path gives NaN (its ``need_weights=False`` path, SDPA,
    gives zeros there too).

    It takes ``nn.MultiheadAttention``'s arguments, and ``formats`` as a
    keyword.

    Args:
        formats (QAttentionFormats, optional): the formats.
            Default: ``None``, torch's arithmetic
    """

    #: ``mptorch.quant.simulate`` passes through every call made inside this
    #: layer: the layer already runs in mptorch's arithmetic.
    _mptorch_owned = True

    #: The kinds :meth:`_formats_from` reads, for ``convert_``'s ``kinds=``.
    _formats_kinds = ("linear", "attention")

    def __init__(self, *args: Any, formats: QAttentionFormats | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.formats = formats

    @classmethod
    def _formats_from(
        cls, recipe: Any, name: str, module: nn.Module, kinds: Any = None
    ) -> MultiheadAttentionFormats | None:
        """What ``recipe`` gives the ``nn.MultiheadAttention`` ``module``, named
        ``name``, or ``None`` for nothing: the ``linear`` kind at its name and
        at its ``out_proj``'s, the ``attention`` kind at its name, each only
        if ``kinds`` (``None`` for all) has it. ``convert_`` and
        ``simulate`` both resolve a module through this."""

        def get(kind: str, at: str, m: nn.Module) -> Any:
            return recipe.resolve(at, m, kind) if kinds is None or kind in kinds else None

        out = module.get_submodule("out_proj")
        parts = MultiheadAttentionFormats(
            get("linear", name, module),
            get("linear", f"{name}.out_proj" if name else "out_proj", out),
            get("attention", name, module),
        )
        if parts.in_proj is None and parts.out_proj is None and parts.attention is None:
            return None
        for part, gf in (("in_proj", parts.in_proj), ("out_proj", parts.out_proj)):
            if gf is not None and gf.palette:
                raise ValueError(
                    f"the recipe gives the {part} of the nn.MultiheadAttention at "
                    f"{name or '(the model)'!r} a palette fwd_mac: its in-projection packs the "
                    "query's, the key's and the value's into one product, whose width a map of "
                    "output neurons does not follow. Give its projections a single format"
                )
        return parts

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The attention, as ``nn.MultiheadAttention.forward`` computes it,
        in the arithmetic of ``formats``."""
        f = self.formats
        if f is None or f._plain:
            return super().forward(
                query,
                key,
                value,
                key_padding_mask,
                need_weights,
                attn_mask,
                average_attn_weights,
                is_causal,
            )
        with torch._C.DisableTorchFunction():  # the layer, not the calls it makes
            if query.is_nested or key.is_nested or value.is_nested:
                raise TypeError(
                    "QMultiheadAttention was given a NestedTensor, which only torch's fast path "
                    "takes: pad the sequences to one length and pass a key_padding_mask"
                )
            is_batched = query.dim() == 3
            key_padding_mask = F._canonical_mask(
                mask=key_padding_mask,
                mask_name="key_padding_mask",
                other_type=F._none_or_dtype(attn_mask),
                other_name="attn_mask",
                target_type=query.dtype,
            )
            attn_mask = F._canonical_mask(
                mask=attn_mask,
                mask_name="attn_mask",
                other_type=None,
                other_name="",
                target_type=query.dtype,
                check_other=False,
            )
            if self.batch_first and is_batched:
                # transposed so that `query is key is value` still holds where it did
                if key is value:
                    if query is key:
                        query = key = value = query.transpose(1, 0)
                    else:
                        query, key = (x.transpose(1, 0) for x in (query, key))
                        value = key
                else:
                    query, key, value = (x.transpose(1, 0) for x in (query, key, value))
            separate = not self._qkv_same_embed_dim
            out, weights = _attention._qmha_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=separate,
                q_proj_weight=self.q_proj_weight if separate else None,
                k_proj_weight=self.k_proj_weight if separate else None,
                v_proj_weight=self.v_proj_weight if separate else None,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                in_proj=f.in_proj,
                out_proj=f.out_proj,
                attention=f.attention,
                qk=f.qk,
                pv=f.pv,
            )
            if self.batch_first and is_batched:
                out = out.transpose(1, 0)
            return out, weights
