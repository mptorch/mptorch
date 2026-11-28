"""Short labels of formats, macs and quantizers, for reports and reprs.

A recipe's rows, a profile's keys and the ``label`` of every formats value
(``NormFormats``, ``ReductionFormats``, ``AttentionFormats``, ...) name their
formats through these, so that one format reads the same everywhere.
"""

from typing import Any

from mptorch.number import BlockFormat
from mptorch.number import presets as number_presets
from mptorch.number.block import _block_label, _part_label

from .mac import BlockQuant, FusedMac, Quant, SplitMac, cast_quant
from .scaled import ScaledQuant


def _fmt_label(fmt: Any) -> str:
    """A format's short label: a preset's name (``MXFP8_E4M3``, ``E4M3``) if
    it is one, and ``carrier`` for ``None``, a sum left in the carrier."""
    if fmt is None:
        return "carrier"
    if isinstance(fmt, BlockFormat):
        for name in ("MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E2M3", "MXFP6_E3M2", "MXFP4_E2M1", "NVFP4"):
            if fmt == getattr(number_presets, name):
                return name
        return _block_label(fmt)
    return _part_label(fmt)


def _mac_label(mac: Any) -> str:
    """A mac's short label: each slot's format, or ``carrier`` for none, and
    a palette slot's entries in brackets."""
    if mac is None:
        return "torch"
    if isinstance(mac, SplitMac):
        return f"SplitMac({_slot_label(mac.mul)}, {_slot_label(mac.acc)})"
    if isinstance(mac, FusedMac):
        return f"FusedMac({_slot_label(mac.fma)})"
    a, b = cast_quant(mac.a).fmt, cast_quant(mac.b).fmt
    return f"BlockMac({_fmt_label(a)}, {_fmt_label(b)})"


def _slot_label(slot: Any) -> str:
    """A mac slot's label: its one format, or a palette's entries."""
    if slot is None:
        return _fmt_label(None)
    if len(slot) == 1:
        return _fmt_label(slot[0])
    return "[" + ", ".join(_fmt_label(f) for f in slot) + "]"


def _quant_label(q: Any) -> str:
    """A quantizer's short label: a format object's, or the callable's name."""
    if isinstance(q, tuple):  # a palette layer's output roundings, one per entry
        return "[" + ", ".join(_quant_label(e) for e in q) + "]"
    if isinstance(q, Quant | ScaledQuant | BlockQuant):
        return f"{type(q).__name__}({_fmt_label(q.fmt)}, {q.rounding.name})"
    return getattr(q, "__name__", type(q).__name__)
