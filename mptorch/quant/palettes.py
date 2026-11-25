"""Palettes in recipes: a format per output neuron, channel or column, chosen by a map.

A :class:`~mptorch.quant.GemmFormats` whose ``fwd_mac`` holds a
:class:`~mptorch.quant.Palette` computes each output neuron of a linear
layer (each output channel of a convolution, each output column of a
matmul) in the palette entry its *map* gives: one entry index per neuron,
channel or column. Its ``output_quant`` may give each entry a rounding of
its own, applied after the bias, as the products' palette is. The gradient
passes keep a single format each.

The map is the layer's state, not the recipe's: it comes from a
calibration, a threshold or a seeded draw once the model exists, and a
study may set several in turn. :func:`set_palette_maps_` writes maps into a
converted model's layers or into a :class:`~mptorch.quant.simulate`, and
:func:`palette_maps` reads them back. A map follows the layer through
``.to()``, and it is not part of the ``state_dict``, which stays the float
model's: save ``palette_maps(model)`` beside the checkpoint.

Until its map is set, a palette layer's map is ``None``, which allocates
nothing, and every output computes in entry 0, by entry 0's single-format
kernel: the words of a map of zeros. (A stochastic output rounding is the
exception: a map of zeros also rounds with the other entries, and so draws
more random bits.) A layer whose formats hold no palette carries none of
this.
"""

from collections.abc import Callable, Mapping
from dataclasses import replace
from functools import partial
from typing import Any

import torch
from torch import nn

from .conv import _conv_forward
from .gemm import _linear_forward, _output_rounding
from .mac import FusedMac, Quant, SplitMac, spec_for_mac
from .modules.format import QAffineFormats, QMatmulFormats
from .ops._conv import _ConvOp
from .ops._operands import _gemm_mixed_nd, _gemm_nd
from .ops._spec import _run_gemm, _run_gemm_mixed

__all__ = ["set_palette_maps_", "palette_maps"]


def _palette_length(mac: Any) -> int:
    """The number of entries of ``mac``'s palette, or 0 for a mac with none."""
    if isinstance(mac, SplitMac):
        slots = (mac.mul, mac.acc)
    elif isinstance(mac, FusedMac):
        slots = (mac.fma,)
    else:
        return 0
    return max((len(s) for s in slots if s is not None and len(s) > 1), default=0)


def _entry(mac: SplitMac | FusedMac, i: int) -> SplitMac | FusedMac:
    """The single-format mac of a palette mac's entry ``i``."""

    def pick(p):
        return None if p is None else p[i if len(p) > 1 else 0]

    if isinstance(mac, SplitMac):
        return replace(mac, mul=pick(mac.mul), acc=pick(mac.acc))
    return replace(mac, fma=pick(mac.fma))


def _mixed_2d(spec, prec_idx, a, b, trans_a, trans_b):
    return _run_gemm_mixed(spec, a, b, prec_idx, trans_a, trans_b)


class _Palette:
    """What a palette layer's forward runs, for one kind (``linear``,
    ``conv`` or ``matmul``): the palette op, entry 0's single-format op for
    an unset map, and each entry's output rounding. It holds nothing a layer
    changes, so every layer and every copy of one shares it
    (``copy.deepcopy`` returns it); the map is the layer's."""

    __slots__ = ("kind", "n", "spec", "spec0", "macs", "outputs", "first", "run0", "conv")

    def __init__(self, kind: str, mac: SplitMac | FusedMac, output_quant: Any):
        self.kind = kind
        self.n = _palette_length(mac)
        self.spec = spec_for_mac(mac)
        # entry 0 alone, with the palette op's checks: an unset layer refuses
        # what the palette op refuses (a storage dtype another entry cannot be
        # stored in), rather than run on until a map is set
        self.spec0 = spec_for_mac(_entry(mac, 0))._replace(
            stored=self.spec.stored, findings=self.spec.findings
        )
        self.macs = tuple(_entry(mac, i) for i in range(self.n))
        self.run0 = partial(_run_gemm, self.spec0)
        if kind == "conv":
            self.conv = (_ConvOp(self.spec0, None), _ConvOp(self.spec, None))
        else:
            self.conv = None
        # Each entry's output rounding: entry 0's in place where it can be (the
        # result is a tensor this forward allocated), the others into new
        # tensors, computed before entry 0 rounds the result in place.
        if output_quant is None:
            self.outputs, self.first = None, None
        else:
            quants = output_quant if isinstance(output_quant, tuple) else (output_quant,)
            self.outputs = tuple(
                replace(q, inplace=False) if type(q) is Quant and q.inplace else q for q in quants
            )
            self.first = _output_rounding(quants[0])

    def __deepcopy__(self, memo: dict) -> "_Palette":
        return self

    def __copy__(self) -> "_Palette":
        return self

    def forward(self, prec_idx: torch.Tensor | None, *args: Any, **kwargs: Any) -> torch.Tensor:
        """The layer's forward pass, each output in its map's entry."""
        if self.kind == "linear":
            q_input, q_weight, q_bias = args
            _check_width(prec_idx, q_weight.shape[0], "output neurons")
            matmul = self.run0 if prec_idx is None else partial(_mixed_2d, self.spec, prec_idx)
            out = _linear_forward(matmul, q_input, q_weight, q_bias)
            mask = None if prec_idx is None else prec_idx.view(-1)
        elif self.kind == "conv":
            assert self.conv is not None
            q_weight = args[1]
            _check_width(prec_idx, q_weight.shape[0], "output channels")
            run = self.conv[0] if prec_idx is None else partial(self.conv[1], prec_idx=prec_idx)
            out = _conv_forward(run, *args, **kwargs)
            nd = kwargs["nd"]
            mask = None if prec_idx is None else prec_idx.view(-1, *([1] * nd))
        else:
            q_a, q_b = args
            _check_width(prec_idx, q_b.shape[-1] if q_b.dim() > 1 else 1, "output columns")
            if prec_idx is None:
                out = _gemm_nd(self.spec0, q_a, q_b, False, False)
            else:
                out = _gemm_mixed_nd(self.spec, q_a, q_b, prec_idx, False, False)
            # a QMatmul has no weight to put its map on the operands' device
            mask = None if prec_idx is None else prec_idx.view(-1).to(out.device)
        return self._rounded(out, mask)

    def _rounded(self, out: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        outputs, first = self.outputs, self.first
        if outputs is None or first is None:
            return out
        if len(outputs) == 1 or mask is None:
            return first(out)
        # the higher entries first, then entry 0 in place, as
        # torch.where(high, q_high(v), q_low(v)) evaluates them
        others = [(i, outputs[i](out)) for i in range(len(outputs) - 1, 0, -1)]
        result = first(out)
        for i, rounded in others:
            result = torch.where(mask == i, rounded, result)
        return result


def _check_width(prec_idx: torch.Tensor | None, width: int, what: str) -> None:
    if prec_idx is not None and prec_idx.numel() != width:
        raise ValueError(
            f"this palette layer's map has {prec_idx.numel()} entries and the call has {width} "
            f"{what}: a map has one entry per output neuron, channel or column, so set a map "
            "of the call's width with set_palette_maps_"
        )


class _PaletteMap:
    """What a palette layer's container adds: the map, a buffer that is
    ``None`` until set and stays out of the ``state_dict``, and a forward
    that reads it at call time (a bound method, which ``copy.deepcopy`` binds
    to the copy, where a closure would keep reading the original's map)."""

    _palette: _Palette
    prec_idx: torch.Tensor | None
    fwd_math: Callable | None

    def _init_palette(self, palette: _Palette) -> None:
        self._palette = palette
        assert isinstance(self, nn.Module)
        self.register_buffer("prec_idx", None, persistent=False)
        # each entry's count of neurons, kept when the map is set, so that
        # profile can count a meta-device model's map, which has no values
        self._palette_counts: tuple[int, ...] | None = None
        self.fwd_math = self._palette_forward

    def _palette_forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self._palette.forward(self.prec_idx, *args, **kwargs)


class _PaletteAffineFormats(_PaletteMap, QAffineFormats):
    """A Linear layer's or a convolution's formats, its forward over a palette."""

    def __init__(self, palette: _Palette, **hooks: Any):
        QAffineFormats.__init__(self, **hooks)
        self._init_palette(palette)


class _PaletteMatmulFormats(_PaletteMap, QMatmulFormats):
    """A matmul's formats, its forward over a palette."""

    def __init__(self, palette: _Palette, **hooks: Any):
        QMatmulFormats.__init__(self, **hooks)
        self._init_palette(palette)


# --- setting and reading maps -------------------------------------------------------------

# how a map of one entry per neuron, channel or column is stored for each kind
_STORED_SHAPE = {"linear": (1, -1), "conv": (-1, 1), "matmul": (1, -1)}


def _layer_slots(model: nn.Module) -> dict[str, tuple[_PaletteMap, int | None, torch.device]]:
    """A converted model's palette layers: name -> (container, width, device)."""
    slots = {}
    for name, module in model.named_modules():
        formats = getattr(module, "formats", None)
        if not isinstance(formats, _PaletteMap):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            width, device = weight.shape[0], weight.device
        else:  # a QMatmul: its width is its calls', checked when they run
            first = next(model.parameters(), None)
            width, device = None, (torch.device("cpu") if first is None else first.device)
        slots[name] = (formats, width, device)
    return slots


def _simulate_slots(sim: Any) -> dict[str, tuple[_PaletteMap, int | None, torch.device]]:
    """A simulate's palette routes, named as set_palette_maps_ takes them."""
    state = sim._state
    modules = dict(sim.model.named_modules())
    slots = {}
    for (scope, kind, site), route in state.routes.items():
        if route is None or not isinstance(route.container, _PaletteMap):
            continue
        module = modules[scope]
        if kind == "matmul":
            name = f"{scope}#matmul" if site is None else f"{scope}#{site}"
        elif next(module.parameters(recurse=False), None) is None:
            continue  # a linear or conv route of a module without a weight of its own
        elif (kind == "conv") != isinstance(module, nn.modules.conv._ConvNd):
            continue  # every module has a route of each kind: a module's map is its own kind's
        else:
            name = scope
        first = next(module.parameters(), None)
        device = torch.device("cpu") if first is None else first.device
        slots[name] = (route.container, None, device)
    return slots


def _slots(target: Any) -> dict[str, tuple[_PaletteMap, int | None, torch.device]]:
    from .simulate import simulate  # imported here: simulate imports the recipe, which imports this

    if isinstance(target, simulate):
        return _simulate_slots(target)
    if isinstance(target, nn.Module):
        return _layer_slots(getattr(target, "_orig_mod", target))
    raise TypeError(f"target is a model or a simulate, got {type(target).__name__}")


def _alias(target: Any, name: str) -> str:
    """A simulate's name for a module reached under several names: the first."""
    aliases = getattr(getattr(target, "_state", None), "aliases", {})
    scope, sep, rest = name.partition("#")
    return aliases.get(scope, scope) + sep + rest


def set_palette_maps_(target: Any, maps: Mapping[str, torch.Tensor | None]) -> None:
    """Set the maps of a model's palette layers, or of a ``simulate``'s.

    Each map gives each output neuron of a linear layer (each output channel
    of a convolution, each output column of a matmul) the index of its
    palette entry: a 1D integer tensor, or a bool tensor that means entry 1
    where ``True`` and entry 0 elsewhere. ``None`` unsets a map, which then
    computes every output in entry 0. A map is checked once, here: its
    length against the layer's width where the layer knows it (a matmul's is
    checked when it runs) and its entries against the palette's length. It
    is stored as ``int32`` on the layer's device (a ``simulate``'s on its
    module's), so set maps after moving the model, and set them all before
    compiling it: a compiled model recompiles when an unset map is set, and
    keeps its graph when a map is replaced by one of the same shape.

    Args:
        target: a model whose palette layers were converted, or a
            ``simulate`` of a recipe with palettes.
        maps (Mapping[str, Tensor | None]): each layer's map, by the layer's
            name; for a ``simulate``, a matmul's maps are named
            ``"module#matmul#0"`` for the sites a rule names and
            ``"module#matmul"`` for the module's other sites, which share it.

    Raises:
        KeyError: for a name that is not a palette layer's.
        ValueError: for a map of the wrong length or with an entry out of the
            palette's range.
        TypeError: for a map that is not an integer or bool tensor.
    """
    slots = _slots(target)
    for name, value in maps.items():
        slot = slots.get(_alias(target, name))
        if slot is None:
            raise KeyError(
                f"{name!r} is not a palette layer of this target: its palette layers are "
                f"{', '.join(sorted(slots)) or 'none'}. For a simulate, a matmul site's map is "
                "named 'module#matmul#n' for a site a Rule names, and 'module#matmul' for the rest"
            )
        container, width, device = slot
        if value is None:
            container.prec_idx = None
            container._palette_counts = None
            continue
        idx = _checked_map(name, value, container._palette.n, width)
        stored = idx.reshape(_STORED_SHAPE[container._palette.kind]).to(
            device=device, dtype=torch.int32, copy=True
        )
        current = container.prec_idx
        if current is not None and current.shape == stored.shape and current.device == device:
            with torch.no_grad():
                current.copy_(stored)  # the same tensor: a compiled graph keeps it
        else:
            container.prec_idx = stored
        counts = torch.bincount(idx.long(), minlength=container._palette.n)
        container._palette_counts = tuple(int(c) for c in counts)


def _checked_map(name: str, value: Any, n: int, width: int | None) -> torch.Tensor:
    """``value`` as a 1D CPU index tensor, checked against the palette's length ``n``."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"the map of {name!r} is a tensor, got {type(value).__name__}")
    if value.dtype is torch.bool:
        value = value.to(torch.int64)
    elif value.is_floating_point() or value.is_complex():
        raise TypeError(
            f"the map of {name!r} is a {value.dtype} tensor: a map holds entry indices, so give "
            "an integer tensor, or a bool one for a two-entry palette"
        )
    idx = value.detach().reshape(-1).cpu()
    if width is not None and idx.numel() != width:
        raise ValueError(
            f"the map of {name!r} has {idx.numel()} entries and the layer has {width} outputs: "
            "a map has one entry per output neuron (channel, column)"
        )
    if idx.numel() and (int(idx.min()) < 0 or int(idx.max()) >= n):
        raise ValueError(
            f"the map of {name!r} holds entries from {int(idx.min())} to {int(idx.max())}, and "
            f"its palette has {n} entries: give indices from 0 to {n - 1}"
        )
    return idx


def palette_maps(target: Any) -> dict[str, torch.Tensor]:
    """The maps set on a model's palette layers, or on a ``simulate``'s, by
    name: each a 1D ``int32`` CPU tensor, ready to save beside a checkpoint
    and to give back to :func:`set_palette_maps_`. Unset maps are left out."""
    return {
        name: container.prec_idx.reshape(-1).cpu()
        for name, (container, _, _) in _slots(target).items()
        if container.prec_idx is not None
    }
