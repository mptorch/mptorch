"""Counting a model's multiply-accumulates by format: :func:`profile`.

A mixed-precision study weighs accuracy against cost, and the cost of a
forward pass is mostly its products: how many multiply-accumulates (MACs)
each layer does, and in which arithmetic. :func:`profile` runs one forward
pass, records every GEMM-shaped layer's MACs with the formats it computes
in, and adds them up per format and against a cost per MAC.

It sees layers, through forward hooks: ``nn.Linear`` and ``nn.Conv1d/2d/3d``
(their mptorch subclasses included) and :class:`~mptorch.quant.QMatmul`. A
product computed by a function call rather than a layer (a ``torch.matmul``
in an attention block, GPT-2's ``Conv1D``, which calls ``torch.addmm``, or
``nn.MultiheadAttention``'s projections) is not seen.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
from torch import nn

from .gemm import cast_quant
from .mac import BlockMac, FusedMac, SplitMac
from .modules.matmul import QMatmul
from .recipe import GemmFormats, _mac_label

__all__ = ["profile", "Profile", "ProfileRecord"]

_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

# The key of a layer whose formats were not built by a GemmFormats, such as
# one configured with a *_gemm_formats factory: its arithmetic has no value
# to report it by.
_CUSTOM = "custom"


class ProfileRecord(NamedTuple):
    """One call of a layer during the profiled forward pass, as the GEMM the
    kernels run: ``batch`` products of an ``[M, K]`` and a ``[K, N]`` matrix,
    ``macs = batch * M * N * K``. ``formats`` is the layer's
    :class:`~mptorch.quant.GemmFormats` (``None`` for a PyTorch layer), and
    ``key`` what the costs and :meth:`Profile.by_format` file it under: the
    forward pass's mac, ``None`` for PyTorch's arithmetic, or ``"custom"``
    for an mptorch layer whose formats were built otherwise."""

    name: str
    kind: str
    batch: int
    M: int
    N: int
    K: int
    macs: int
    formats: GemmFormats | None
    key: Any


@dataclass(frozen=True)
class Profile:
    """The records of a profiled forward pass, and the cost table they are weighed by."""

    records: tuple[ProfileRecord, ...]
    cost: Mapping[Any, float] | None = None

    @property
    def macs(self) -> int:
        """The forward pass's MACs, over every layer recorded."""
        return sum(r.macs for r in self.records)

    def by_format(self) -> dict[Any, int]:
        """The MACs per key, in the order the keys were first met."""
        out: dict[Any, int] = {}
        for r in self.records:
            out[r.key] = out.get(r.key, 0) + r.macs
        return out

    def cost_of(self, key: Any) -> float:
        """The cost per MAC of ``key`` in the cost table: the key itself, and
        for a mac, then its product format (a ``SplitMac``'s ``mul``, a
        ``FusedMac``'s ``fma``, a ``BlockMac``'s ``a``), so that a table keyed
        by formats (``{BinaryK(8, 4): 1.0, None: 4.0}``) prices macs.

        Raises:
            ValueError: without a cost table.
            KeyError: for a key the table does not price.
        """
        if self.cost is None:
            raise ValueError("this profile has no cost table: pass cost= to profile")
        for k in (key, _product_format(key)):
            if k is not _NOTHING and k in self.cost:
                return float(self.cost[k])
        raise KeyError(f"the cost table has no entry for {key!r} or for its product format")

    @property
    def total_cost(self) -> float:
        """The sum over the records of ``macs * cost_of(key)``, which weighs
        each layer's MACs by the cost table."""
        return math.fsum(r.macs * self.cost_of(r.key) for r in self.records)

    def table(self) -> str:
        """The records as a table, one line per layer call, then the MACs per
        key with their share (and their cost, with a cost table)."""
        head = ("layer", "kind", "batch", "M", "N", "K", "MACs", "arithmetic")
        body = [
            (
                r.name or "(model)",
                r.kind,
                str(r.batch),
                str(r.M),
                str(r.N),
                str(r.K),
                f"{r.macs:,}",
                _key_label(r.key),
            )
            for r in self.records
        ]
        widths = [max(len(row[i]) for row in (head, *body)) for i in range(len(head))]
        lines = [
            "  ".join(
                c.rjust(w) if 2 <= i <= 6 else c.ljust(w)
                for i, (c, w) in enumerate(zip(row, widths, strict=True))
            ).rstrip()
            for row in (head, *body)
        ]
        total = self.macs or 1
        lines.append("")
        for key, macs in self.by_format().items():
            line = f"{_key_label(key)}: {macs:,} MACs ({macs / total:.1%})"
            if self.cost is not None:
                line += f", cost {macs * self.cost_of(key):,.6g}"
            lines.append(line)
        if self.cost is not None:
            lines.append(f"total cost {self.total_cost:,.6g}")
        lines.append(
            "(layers only: products computed by function calls, such as torch.matmul, "
            "are not counted)"
        )
        return "\n".join(lines)


_NOTHING = object()


def _product_format(key: Any) -> Any:
    """The format a mac rounds its products to, or ``_NOTHING``."""
    if isinstance(key, SplitMac):
        return key.mul[0]
    if isinstance(key, FusedMac):
        return _NOTHING if key.fma is None else key.fma[0]
    if isinstance(key, BlockMac):
        return cast_quant(key.a).fmt
    return _NOTHING


def _key_label(key: Any) -> str:
    if key is None:
        return "torch"
    if key == _CUSTOM:
        return "custom formats"
    return _mac_label(key)


def _key(module: nn.Module) -> tuple[GemmFormats | None, Any]:
    """A layer's formats and the key its MACs are filed under."""
    formats = getattr(module, "formats", None)
    if formats is None:
        return None, None
    source = getattr(formats, "_source", None)
    if source is not None:
        return source, source.fwd_mac
    return None, (None if formats.fwd_math is None else _CUSTOM)


def _linear_record(name, module, inputs, output) -> ProfileRecord:
    x = inputs[0]
    K, N = module.in_features, module.out_features
    M = x.numel() // K if K else 0
    formats, key = _key(module)
    return ProfileRecord(name, "linear", 1, M, N, K, M * N * K, formats, key)


def _conv_record(name, module, inputs, output) -> ProfileRecord:
    """A convolution's forward pass as its GEMMs: one per sample and group,
    of the group's output channels by the output positions, summing over the
    group's input channels times the kernel taps."""
    nd = len(module.kernel_size)
    batch = output.shape[0] if output.dim() == nd + 2 else 1
    groups = module.groups
    M = module.out_channels // groups
    N = math.prod(output.shape[-nd:])
    K = module.in_channels // groups * math.prod(module.kernel_size)
    formats, key = _key(module)
    return ProfileRecord(
        name, "conv", batch * groups, M, N, K, batch * groups * M * N * K, formats, key
    )


def _matmul_record(name, module, inputs, output) -> ProfileRecord:
    a, b = inputs[0], inputs[1]
    K = a.shape[-1]
    M = a.shape[-2] if a.dim() > 1 else 1
    N = b.shape[-1] if b.dim() > 1 else 1
    batch = output.numel() // (M * N) if M * N else 0
    formats, key = _key(module)
    return ProfileRecord(name, "matmul", batch, M, N, K, batch * M * N * K, formats, key)


def profile(
    model: nn.Module,
    *inputs: Any,
    cost: Mapping[Any, float] | None = None,
    **kwargs: Any,
) -> Profile:
    """Run one forward pass of ``model`` and count each layer's MACs by its formats.

    The pass is ``model(*inputs, **kwargs)``, under ``torch.no_grad()``. Each
    call of an ``nn.Linear``, an ``nn.Conv1d/2d/3d`` (an mptorch layer
    included) or a :class:`~mptorch.quant.QMatmul` is recorded as the GEMM it
    is (:class:`ProfileRecord`): a convolution as one GEMM per sample and
    group, as the kernels run it. Only the forward pass is counted, and only
    layers: a product computed by a function call is not seen.

    Args:
        model (nn.Module): the model.
        *inputs: its positional inputs.
        cost (Mapping, optional): a cost per MAC for each key (see
            :meth:`Profile.cost_of`), such as ``{BinaryK(8, 4): 1.0, None: 4.0}``
            for fp8 products against float32 ones. Default: ``None``
        **kwargs: its keyword inputs.

    Returns:
        Profile: the records, in the order the layers ran.

    Example::

        >>> import torch
        >>> from torch import nn
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import convert_, presets, profile
        >>> model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2))
        >>> _ = convert_(model, presets.uniform(BinaryK(8, 4)).override("2", linear=None))
        >>> p = profile(model, torch.randn(4, 16), cost={BinaryK(8, 4): 1.0, None: 4.0})
        >>> p.macs, p.total_cost
        (576, 640.0)
    """
    records: list[ProfileRecord] = []
    names = {m: n for n, m in model.named_modules()}

    def hook(module, args, output):
        name = names.get(module, "")
        if isinstance(module, nn.Linear):
            records.append(_linear_record(name, module, args, output))
        elif isinstance(module, _CONVS):
            records.append(_conv_record(name, module, args, output))
        else:
            records.append(_matmul_record(name, module, args, output))

    handles = [
        m.register_forward_hook(hook)
        for m in names
        if isinstance(m, nn.Linear | nn.Conv1d | nn.Conv2d | nn.Conv3d | QMatmul)
    ]
    try:
        with torch.no_grad():
            model(*inputs, **kwargs)
    finally:
        for h in handles:
            h.remove()
    return Profile(tuple(records), cost)
