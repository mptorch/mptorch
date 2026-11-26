"""Counting copies of the opt-in kinds' formats, for ``simulate(..., events=True)``.

A recipe's activation, elementwise, embedding, softmax, norm, reduction and
attention formats are frozen values whose quantizer slots hold resolved
``Quant`` s and whose sums hold resolved GEMMs. A site's counting copy is the
same value with each slot's ``Quant`` wrapped in an `events._CountingQuant`
and each GEMM replaced by its counting spec (`recipe._counting`), every
counter bound to the site's tally. The functions that run the kinds read the
copy as they read the original, forward and backward, so the counts reach
the backward with the copy. A wrapped slot is not a ``Quant``, so the fused
pointwise and table ops decline it and the two passes run, where the value
before the rounding exists to recount (`ops._spec._round_result` rounds it
out of place); the words are the same either way.

A site's parts are named under their slots: a norm's sums under ``acc`` and
``bwd_acc``, an attention's products under ``qk`` and ``pv`` and its softmax
under ``softmax``. Its pass is ``fwd`` or, for a backward slot, ``bwd`` (an
attention's products take the GEMM kinds' ``fwd``, ``igrad`` and ``wgrad``).
"""

import copy
import dataclasses
from typing import Any

from .attention import AttentionFormats, FlashFormats
from .events import _Prefixed
from .norm import (
    _NORM_MACS,
    _NORM_QUANTS,
    _SOFTMAX_MACS,
    _SOFTMAX_QUANTS,
    NormFormats,
    SoftmaxFormats,
)
from .pointwise import ActivationFormats, PointwiseFormats
from .recipe import _counting, _counting_quant
from .reduce import ReductionFormats

# Each kind's quantizer slots, and its macs with their index in ``_specs``.
_SLOTS: dict[type, tuple[tuple[str, ...], tuple[str, ...]]] = {
    PointwiseFormats: (("quant", "grad_quant"), ()),
    ActivationFormats: (("inner", "out", "grad"), ()),
    SoftmaxFormats: (_SOFTMAX_QUANTS, _SOFTMAX_MACS),
    NormFormats: (_NORM_QUANTS, _NORM_MACS),
    ReductionFormats: (("div", "grad"), ()),
    FlashFormats: (("rescale", "combine", "lse"), ()),
}


def _pass_of(slot: str) -> str:
    return "bwd" if slot.startswith("bwd") or slot in ("grad", "grad_quant") else "fwd"


def _counting_formats(f: Any, tally: Any) -> Any:
    """A counting copy of an opt-in kind's formats ``f``, counting into
    ``tally``."""
    g = copy.copy(f)
    if isinstance(f, AttentionFormats):
        for name in ("qk", "pv"):
            part = getattr(f, name)
            if part is None:
                continue
            counted = _Prefixed(tally, name)
            object.__setattr__(g, f"_{name}", part.matmul(_tally=counted))
            if f.flash is not None:
                tile = dataclasses.replace(
                    part, **({"rotation": None} if name == "qk" else {"output_quant": None})
                )
                object.__setattr__(g, f"_{name}_tile", tile.matmul(_tally=counted))
        for name in ("scores", "softmax", "flash"):
            part = getattr(f, name)
            if part is not None:
                object.__setattr__(g, name, _counting_formats(part, _Prefixed(tally, name)))
        return g
    quants, macs = _SLOTS[type(f)]
    for name in quants:
        q = getattr(f, name)
        object.__setattr__(g, name, _counting_quant(q, tally, name, _pass_of(name)))
    if isinstance(f, ReductionFormats) and f._spec is not None:
        count = _counting(tally, "fwd")
        assert count is not None
        object.__setattr__(g, "_spec", count(f.mac, f._spec))
    if macs:
        specs = list(f._specs)
        for i, name in enumerate(macs):
            if specs[i] is not None:
                count = _counting(_Prefixed(tally, name), _pass_of(name))
                assert count is not None
                specs[i] = count(getattr(f, name), specs[i])
        object.__setattr__(g, "_specs", tuple(specs))
    return g
