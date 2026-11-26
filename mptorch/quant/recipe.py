"""Recipes: the formats each part of a model computes in, as one value.

A model is converted (:func:`mptorch.quant.convert`) from a recipe rather
than layer by layer. The recipe names, for each *kind* of computation, the
formats it runs in, and rules override them for parts of the model chosen by
name, by type or by a predicate:

* :class:`GemmFormats` is the value of the three GEMM kinds: ``linear`` (a
  product with a weight, as in ``nn.Linear``), ``conv`` and ``matmul`` (a
  product of two activations). It holds the operand and gradient quantizers,
  the output rounding, and one mac for each of the three passes, and builds a
  fresh format container for a layer on each call
  (:meth:`GemmFormats.affine`, :meth:`GemmFormats.conv`,
  :meth:`GemmFormats.matmul`).
* :class:`~mptorch.quant.ActivationFormats` is the value of the
  ``activation`` kind (GELU, SiLU, sigmoid, tanh, hardswish, hardsigmoid,
  ReLU), and :class:`~mptorch.quant.PointwiseFormats` that of the
  ``elementwise`` kind (a model's additions, products, quotients and the
  other pointwise ops with an mptorch twin, which only
  :class:`~mptorch.quant.simulate` reaches) and of the ``embedding`` kind.
* :class:`~mptorch.quant.SoftmaxFormats` is the value of the ``softmax``
  kind, :class:`~mptorch.quant.NormFormats` that of the ``norm`` kind (layer,
  RMS, group and batch norms), and :class:`~mptorch.quant.ReductionFormats`
  that of the ``reduction`` kind (a model's ``torch.sum`` and ``torch.mean``
  calls, which only :class:`~mptorch.quant.simulate` reaches).
* :class:`~mptorch.quant.AttentionFormats` is the value of the ``attention`` kind
  (``F.scaled_dot_product_attention``, and ``nn.MultiheadAttention``'s core),
  made of the values above: a :class:`GemmFormats` for each of its two
  products, a softmax's and a pointwise op's formats.
* :class:`Rule` gives some kinds their values where its ``where`` matches.
* :class:`Recipe` is an ordered set of rules, of which the last that matches
  and names a kind wins: :meth:`Recipe.resolve` answers what one module
  computes in.

A kind a recipe does not mention, and a kind it sets to ``None``, stay plain
PyTorch.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import nn

from mptorch.number import BinaryK, BlockFormat, FixedPoint, Number, RoundMode, SuperFP
from mptorch.number import presets as number_presets
from mptorch.number.block import _block_label, _part_label

from .attention import AttentionFormats
from .conv import _conv_formats
from .events import _CountedCall, _CountingQuant, _SiteTally
from .gemm import (
    _block_linear_fwd,
    _block_linear_igrad,
    _block_linear_wgrad,
    _block_matmul_agrad,
    _block_matmul_bgrad,
    _block_matmul_fwd,
    _linear_fwd,
    _linear_igrad,
    _linear_wgrad,
    _matmul_agrad,
    _matmul_bgrad,
    _matmul_fwd,
    _plain_conv,
    _spec_matmul,
    _with_output,
    block_matmul_formats,
    cast_quant,
)
from .mac import (
    BlockMac,
    BlockQuant,
    FusedMac,
    Palette,
    Quant,
    SplitMac,
    _as_quant,
    spec_for_block_mac,
    spec_for_mac,
)
from .modules.format import QAffineFormats, QMatmulFormats
from .modules.linear import CustomArithLinear
from .modules.matmul import CustomArithMatmul
from .norm import NormFormats, SoftmaxFormats
from .ops._counted import _counted_parts, _counting_built
from .ops._operands import _gemm_nd
from .ops._spec import _GemmSpec, _run_gemm
from .palettes import _Palette, _palette_length, _PaletteAffineFormats, _PaletteMatmulFormats
from .pointwise import ActivationFormats, PointwiseFormats
from .reduce import ReductionFormats
from .rotation import Hadamard, _RotatedGrad, _RotatedQuant
from .scaled import ScaledQuant

if TYPE_CHECKING:
    from .convert import ConvertReport

__all__ = ["GemmFormats", "Rule", "Recipe"]

# The kinds a recipe can give formats to, and the three GEMM kinds among them,
# which `gemm` (a shorthand, not a kind) names at once.
_KINDS = (
    "linear",
    "conv",
    "matmul",
    "elementwise",
    "activation",
    "embedding",
    "softmax",
    "norm",
    "reduction",
    "attention",
)
_GEMM_KINDS = ("linear", "conv", "matmul")

# The default of `GemmFormats.uniform`'s gradient arguments: the forward's.
_SAME: Any = object()

_QUANT_SLOTS = (
    "input_quant",
    "weight_quant",
    "bias_quant",
    "output_quant",
    "igrad_quant",
    "wgrad_quant",
    "bgrad_quant",
)
_MAC_SLOTS = ("fwd_mac", "igrad_mac", "wgrad_mac")

# Each container's gradient hooks and the Function's default for an empty one,
# in the order of its operand slots, which GemmFormats._rotated wraps.
_AFFINE_GRADS = (
    ("bwd_igrad_math", CustomArithLinear._default_bwd_igrad),
    ("bwd_wgrad_math", CustomArithLinear._default_bwd_wgrad),
)
_MATMUL_GRADS = (
    ("bwd_agrad_math", CustomArithMatmul._default_agrad),
    ("bwd_bgrad_math", CustomArithMatmul._default_bgrad),
)


def _as_mac(slot: object, name: str) -> Any:
    """A pass's mac as :class:`GemmFormats` keeps it, resolved once so that an
    error raises when the recipe is built rather than when a layer runs."""
    if slot is None:
        return None
    if isinstance(slot, BlockFormat):
        return BlockMac(slot)
    if isinstance(slot, Number):
        slot = SplitMac(slot, slot)
    if isinstance(slot, SplitMac | FusedMac):
        if name != "fwd_mac" and spec_for_mac(slot).mixed:
            raise ValueError(
                f"{name} holds a palette: a GemmFormats takes a palette in fwd_mac only, whose "
                "map gives each output neuron its entry, and gives each gradient pass one mac. "
                "Give this pass a single format (with GemmFormats.uniform, pass grad_mac=), or "
                "build the layer's formats with matmul_formats or conv_formats, which take a map "
                "per pass"
            )
        return slot
    if isinstance(slot, BlockMac):
        return slot
    raise TypeError(
        f"{name} takes a SplitMac, a FusedMac, a BlockMac or a format, got {type(slot).__name__}"
    )


def _as_output_quant(slot: object, fwd_mac: Any) -> Any:
    """``output_quant`` as :class:`GemmFormats` keeps it: one quantizer, or,
    with a palette ``fwd_mac``, a tuple of one per entry (a ``Palette`` of
    formats meaning ``Quant(fmt)`` each)."""
    if isinstance(slot, Palette):
        slot = tuple(slot)
    if not isinstance(slot, tuple | list):
        return _as_quant(slot, "output_quant")
    n = _palette_length(fwd_mac)
    if len(slot) != n:
        raise ValueError(
            f"output_quant gives {len(slot)} roundings, one per palette entry, and fwd_mac "
            + (f"has a palette of {n}" if n else "has no palette")
            + ": give one rounding per entry of fwd_mac's palette, or a single rounding"
        )
    return tuple(_as_quant(q, f"output_quant[{i}]") for i, q in enumerate(slot))


# A pass's counting (simulate(events=True)): the resolved GEMM of a mac,
# with its op replaced by the counted twin; None for a pass not counted.
_Count = Callable[[Any, "_GemmSpec | None"], "_GemmSpec | None"] | None


def _linear_pass(
    mac: Any, hook: Callable, block_hook: Callable, where: str, count: _Count = None
) -> Callable | None:
    """One Linear pass's hook: none without a mac (PyTorch's own product),
    ``block_hook`` over a ``BlockMac``'s GEMM and its two operands' quantizers,
    and ``hook`` over any other mac's resolved GEMM, which refuses a wide
    fixed-point result as a ``where`` pass (``gemm._spec_matmul``). ``count``
    makes the GEMM's counting spec."""
    if mac is None:
        return None
    if isinstance(mac, BlockMac):
        if count is not None:
            count(mac, None)
        return block_hook(spec_for_block_mac(mac), cast_quant(mac.a), cast_quant(mac.b))
    spec = spec_for_mac(mac)
    if count is not None:
        spec = count(mac, spec)
    assert spec is not None
    return hook(_spec_matmul(spec, _run_gemm, where))


def _matmul_pass(
    mac: Any, hook: Callable, block_hook: Callable, where: str, count: _Count = None
) -> Callable | None:
    """One matmul pass's hook, as :func:`_linear_pass` builds a Linear one."""
    if mac is None:
        return None
    if isinstance(mac, BlockMac):
        if count is not None:
            count(mac, None)
        return block_hook(mac)
    spec = spec_for_mac(mac)
    if count is not None:
        spec = count(mac, spec)
    assert spec is not None
    return hook(_spec_matmul(spec, _gemm_nd, where))


def _wide_forward(
    mac: Any, hook: Callable, runner: Callable, count: _Count = None
) -> Callable | None:
    """The forward hook that leaves a wide fixed-point result in float64, for
    the container's ``_wide_fwd``, or ``None`` when ``mac``'s result is never
    wide."""
    if not isinstance(mac, SplitMac | FusedMac):
        return None
    spec = spec_for_mac(mac)
    if not spec.wide:
        return None
    if count is not None:
        spec = count(mac, spec)
    return hook(partial(runner, spec))


# --- counting (simulate(events=True)) ------------------------------------------

_PASSES = ("fwd", "igrad", "wgrad")
# Each quantizer slot's pass in a counting container's report.
_SLOT_PASS = {
    "input_quant": "fwd",
    "weight_quant": "fwd",
    "bias_quant": "fwd",
    "output_quant": "fwd",
    "igrad_quant": "igrad",
    "wgrad_quant": "wgrad",
    "bgrad_quant": "bgrad",
}


def _role_labels(mac: Any) -> tuple[str, str, str]:
    """The format of each of a single-format mac's roles (mul, acc, outer),
    or "" for a role it does not round."""
    if isinstance(mac, SplitMac):
        mul, acc = _slot_label(mac.mul), ("" if mac.acc is None else _slot_label(mac.acc))
    else:
        mul, acc = "", ("" if mac.fma is None else _slot_label(mac.fma))
    return mul, acc, ("" if mac.outer is None else _fmt_label(mac.outer))


def _uncounted_reason(mac: Any) -> str:
    """Why a mac's GEMM has no counted twin."""
    if isinstance(mac, BlockMac):
        return "a block GEMM"
    if _palette_length(mac):
        return "a palette mac"
    if mac.carrier_rounding is RoundMode.RO:
        return "a carrier step rounded to odd"
    return "a cross-family mac"


def _counting(tally: "_SiteTally | None", pass_: str) -> _Count:
    """``pass_``'s counting into ``tally``, or ``None`` without a tally."""
    if tally is None:
        return None

    def count(mac: Any, spec: "_GemmSpec | None") -> "_GemmSpec | None":
        parts = None if spec is None else _counted_parts(spec)
        if parts is None:
            tally.gap(pass_, "mac", _uncounted_reason(mac))
            return spec
        if not _counting_built():
            tally.gap(pass_, "mac", "built without MPTORCH_COUNTING=1")
            return spec
        assert spec is not None  # parts came from it
        (op, tail), conv = parts
        return spec._replace(op=_CountedCall(op, tail, conv, tally, pass_, _role_labels(mac)))

    return count


def _counting_quant(q: Any, tally: "_SiteTally | None", slot: str) -> Any:
    """A quantizer slot's value, counting its casts into ``tally`` when it is
    a Quant of a family the recount reads; the value itself otherwise, and
    without a tally."""
    if tally is None or q is None:
        return q
    pass_ = _SLOT_PASS.get(slot, slot.split("_")[0])
    if isinstance(q, Quant) and not q.inplace and isinstance(q.fmt, BinaryK | SuperFP | FixedPoint):
        return _CountingQuant(q, tally, pass_, slot, _fmt_label(q.fmt))
    what = "an in-place Quant" if isinstance(q, Quant) else f"a {type(q).__name__}"
    tally.gap(pass_, slot, f"{what} quantizer")
    return q


@dataclass(frozen=True, kw_only=True)
class GemmFormats:
    """The formats of one GEMM-shaped layer: its quantizers and one mac per pass.

    It is the value a :class:`Recipe` gives the ``linear``, ``conv`` and
    ``matmul`` kinds, and it builds the format container that a
    :class:`~mptorch.quant.QLinear`, a ``QConv1d/2d/3d`` or a
    :class:`~mptorch.quant.QMatmul` reads: :meth:`affine`, :meth:`conv` and
    :meth:`matmul` return a new one on each call, so that no two layers share
    a container.

    Each of the three passes (the forward product, the input gradient and the
    weight gradient) runs in its own mac, or in PyTorch's own arithmetic when
    its mac is ``None``. A mac is a :class:`~mptorch.quant.SplitMac`, a
    :class:`~mptorch.quant.FusedMac` or a :class:`~mptorch.quant.BlockMac`;
    a format means ``SplitMac(fmt, fmt)`` (a :class:`mptorch.BlockFormat`,
    ``BlockMac(fmt)``). A ``BlockMac``'s two operands are the pass's two, in
    order: (input, weight) for the forward, (gradient, weight) for the input
    gradient and (gradient, input) for the weight gradient, or (a, b), (grad,
    b) and (a, grad) for a matmul.

    ``fwd_mac`` may hold a :class:`~mptorch.quant.Palette`: the layer then
    computes each output neuron (a convolution's output channel, a matmul's
    output column) in the palette entry its map gives, and ``output_quant``
    may give each entry its own rounding, as a tuple of one quantizer per
    entry (or a ``Palette`` of formats, meaning ``Quant(fmt)`` each). The map
    is the layer's, set with :func:`~mptorch.quant.set_palette_maps_` once
    the model is converted or simulated; until then every output computes in
    entry 0. The gradient passes take one mac each.

    The quantizer slots are those of :class:`~mptorch.quant.QAffineFormats`,
    plus ``output_quant``, which rounds the forward result after the bias is
    added, with a straight-through gradient. Each takes a
    :class:`~mptorch.quant.Quant`, a :class:`~mptorch.quant.ScaledQuant`, a
    :class:`~mptorch.quant.BlockQuant`, a format (meaning ``Quant(fmt)``, or
    ``BlockQuant(fmt)`` for a block format) or any stateless callable. A
    module is refused, since every layer the recipe matches would share its
    state. For a matmul, ``input_quant`` and ``weight_quant`` are ``a``'s and
    ``b``'s, ``igrad_quant`` and ``wgrad_quant`` the two gradients', and the
    bias slots are not used. The recompute flags are passed on as they are
    (``recompute_a`` and ``recompute_b`` for a matmul).

    ``rotation`` (a :class:`~mptorch.quant.Hadamard`) rotates both operands
    of the forward product along K before their quantizers run, and rotates
    each gradient back, so the layer computes the same function in exact
    arithmetic and its gradients are those of that function. It helps the
    MX block formats and not NVFP4 (``kernels/block.rst``, "Rotating before
    block quantization"). The gradient passes multiply the rotated operands.
    A convolution takes no rotation.

    Args:
        input_quant, weight_quant, bias_quant (optional): the forward
            operands' quantizers. Default: ``None``
        output_quant (optional): rounds the forward result; with a palette
            ``fwd_mac``, one rounding per entry may be given, as a tuple.
            Default: ``None``
        fwd_mac, igrad_mac, wgrad_mac (optional): each pass's mac.
            Default: ``None``, PyTorch's arithmetic.
        igrad_quant, wgrad_quant, bgrad_quant (optional): the output
            gradient's quantizer for the input gradient, the weight gradient
            and the bias gradient. Default: ``None``
        recompute_input, recompute_weight (bool, optional): as for
            :class:`~mptorch.quant.QAffineFormats`. Default: ``None``
        rotation (Hadamard, optional): the rotation of the forward
            product's operands along K. Default: ``None``

    Raises:
        TypeError: for a slot of the wrong type, or a module in a quantizer
            slot.
        ValueError: for a palette in a gradient pass's mac, a tuple of output
            roundings whose length is not ``fwd_mac``'s palette's, or any
            error a mac raises when it is resolved.

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import GemmFormats, QLinear, Quant, SplitMac
        >>> p8 = BinaryK(8, 4)
        >>> gf = GemmFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
        >>> layer = QLinear(32, 8, formats=gf.affine())
    """

    input_quant: Any = None
    weight_quant: Any = None
    bias_quant: Any = None
    output_quant: Any = None
    fwd_mac: Any = None
    igrad_mac: Any = None
    wgrad_mac: Any = None
    igrad_quant: Any = None
    wgrad_quant: Any = None
    bgrad_quant: Any = None
    recompute_input: bool | None = None
    recompute_weight: bool | None = None
    rotation: Hadamard | None = None

    def __post_init__(self) -> None:
        for name in _MAC_SLOTS:
            object.__setattr__(self, name, _as_mac(getattr(self, name), name))
        for name in _QUANT_SLOTS:
            value = getattr(self, name)
            if name == "output_quant":
                value = _as_output_quant(value, self.fwd_mac)
            else:
                value = _as_quant(value, name)
            object.__setattr__(self, name, value)
        for name in ("recompute_input", "recompute_weight"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} is True, False or None, got {getattr(self, name)!r}")
        if self.rotation is not None and not isinstance(self.rotation, Hadamard):
            raise TypeError(f"rotation is a Hadamard or None, got {self.rotation!r}")

    @classmethod
    def uniform(
        cls, mac: Any, quant: Any = None, *, grad_mac: Any = _SAME, grad_quant: Any = _SAME
    ) -> "GemmFormats":
        """Every pass in ``mac``, and every operand and gradient rounded by ``quant``.

        ``grad_mac`` and ``grad_quant`` give the two gradient passes their own
        mac and the output gradient its own quantizer; they default to
        ``mac`` and ``quant``. The bias, its gradient and the output are left
        unrounded.
        """
        grad_mac = mac if grad_mac is _SAME else grad_mac
        grad_quant = quant if grad_quant is _SAME else grad_quant
        return cls(
            input_quant=quant,
            weight_quant=quant,
            fwd_mac=mac,
            igrad_mac=grad_mac,
            wgrad_mac=grad_mac,
            igrad_quant=grad_quant,
            wgrad_quant=grad_quant,
        )

    @property
    def palette(self) -> int:
        """The number of entries of ``fwd_mac``'s palette, or 0 without one."""
        return _palette_length(self.fwd_mac)

    @property
    def _has_block_mac(self) -> bool:
        """Whether a pass multiplies block formats, which no convolution does."""
        return any(isinstance(getattr(self, name), BlockMac) for name in _MAC_SLOTS)

    def _affine_slots(
        self, formats: QAffineFormats, tally: "_SiteTally | None" = None
    ) -> QAffineFormats:
        formats.input_quant = _counting_quant(self.input_quant, tally, "input_quant")
        formats.weight_quant = _counting_quant(self.weight_quant, tally, "weight_quant")
        formats.bias_quant = _counting_quant(self.bias_quant, tally, "bias_quant")
        formats.igrad_quant = _counting_quant(self.igrad_quant, tally, "igrad_quant")
        formats.wgrad_quant = _counting_quant(self.wgrad_quant, tally, "wgrad_quant")
        formats.bgrad_quant = _counting_quant(self.bgrad_quant, tally, "bgrad_quant")
        formats.recompute_input = self.recompute_input
        formats.recompute_weight = self.recompute_weight
        formats._source = self
        return self._rotated(formats, ("input_quant", "weight_quant"), _AFFINE_GRADS)

    def _rotated[FormatsT: (QAffineFormats, QMatmulFormats)](
        self, formats: FormatsT, slots: tuple[str, str], grads: tuple[tuple[str, Callable], ...]
    ) -> FormatsT:
        """``formats`` with its two operand slots rotating along K first, and
        each gradient hook (or the Function's default, for an empty one)
        rotating its result back; ``formats`` as it is without a rotation.
        The second operand is a matmul's ``b`` (dimension -2) when ``grads``
        names ``bwd_bgrad_math``, and a Linear's weight (dimension -1)
        otherwise."""
        if self.rotation is None:
            return formats
        rotation, b_dim = self.rotation, -2 if grads[1][0] == "bwd_bgrad_math" else -1
        for slot, dim in zip(slots, (-1, b_dim), strict=True):
            setattr(formats, slot, _RotatedQuant(rotation, getattr(formats, slot), dim))
        for (hook, default), dim in zip(grads, (-1, b_dim), strict=True):
            setattr(formats, hook, _RotatedGrad(rotation, getattr(formats, hook) or default, dim))
        formats._rotation = rotation
        return formats

    def affine(self, *, _tally: Any = None) -> QAffineFormats:
        """A new ``QAffineFormats`` for a Linear layer, with these formats.

        With a palette ``fwd_mac`` it holds the layer's map (``None`` until
        :func:`~mptorch.quant.set_palette_maps_` sets it), and its forward
        computes each output neuron in its entry, rounded by its entry's
        ``output_quant``. ``_tally`` (``simulate(events=True)``'s) makes a
        container that counts its casts there."""
        fwd_count, igrad_count, wgrad_count = (_counting(_tally, p) for p in _PASSES)
        igrad = _linear_pass(
            self.igrad_mac, _linear_igrad, _block_linear_igrad, "gradient", igrad_count
        )
        wgrad = _linear_pass(
            self.wgrad_mac, _linear_wgrad, _block_linear_wgrad, "gradient", wgrad_count
        )
        if self.palette:
            if _tally is not None:
                _tally.gap("fwd", "mac", "a palette mac")
            palette = _Palette("linear", self.fwd_mac, self.output_quant)
            hooks = {"bwd_igrad_math": igrad, "bwd_wgrad_math": wgrad}
            return self._affine_slots(_PaletteAffineFormats(palette, **hooks), _tally)
        formats = QAffineFormats(
            fwd_math=_linear_pass(
                self.fwd_mac, _linear_fwd, _block_linear_fwd, "forward", fwd_count
            ),
            bwd_igrad_math=igrad,
            bwd_wgrad_math=wgrad,
        )
        formats._wide_fwd = _wide_forward(self.fwd_mac, _linear_fwd, _run_gemm, fwd_count)
        if isinstance(self.fwd_mac, BlockMac):
            # the weight's quantizer, which QLinear.freeze_packed_ packs a frozen weight with
            formats._block_weight = cast_quant(self.fwd_mac.b)
        if self.output_quant is not None:
            output = _counting_quant(self.output_quant, _tally, "output_quant")
            _with_output(formats, output, F.linear)
        return self._affine_slots(formats, _tally)

    def conv(self, *, _tally: Any = None) -> QAffineFormats:
        """A new ``QAffineFormats`` for a ``QConv1d/2d/3d``, with these formats
        (``_tally``: as for :meth:`affine`).

        Raises:
            TypeError: if a pass's mac is a ``BlockMac``: there is no block
                convolution.
        """
        if self._has_block_mac:
            raise TypeError(
                "a convolution has no block-format arithmetic: give BlockMac formats to the "
                "linear and matmul kinds only"
            )
        if self.rotation is not None:
            raise TypeError(
                "a convolution takes no rotation: its reduction axis is its input channels times "
                "its kernel taps, which its kernels gather; give rotation to the linear and "
                "matmul kinds only"
            )
        specs = []
        for m, pass_ in zip((self.fwd_mac, self.igrad_mac, self.wgrad_mac), _PASSES, strict=True):
            count = _counting(_tally, pass_)
            spec = None if m is None else spec_for_mac(m)
            if spec is not None and count is not None and not (pass_ == "fwd" and self.palette):
                spec = count(m, spec)
            specs.append(spec)
        fwd, igrad, wgrad = specs
        if self.palette:
            if _tally is not None:
                _tally.gap("fwd", "mac", "a palette mac")
            grads = _conv_formats(None, igrad, wgrad)
            palette = _Palette("conv", self.fwd_mac, self.output_quant)
            formats = _PaletteAffineFormats(
                palette, bwd_igrad_math=grads.bwd_igrad_math, bwd_wgrad_math=grads.bwd_wgrad_math
            )
            return self._affine_slots(formats, _tally)
        formats = _conv_formats(fwd, igrad, wgrad)
        if self.output_quant is not None:
            output = _counting_quant(self.output_quant, _tally, "output_quant")
            _with_output(formats, output, _plain_conv)
        return self._affine_slots(formats, _tally)

    def matmul(self, *, _tally: Any = None) -> QMatmulFormats:
        """A new ``QMatmulFormats`` for a ``QMatmul``, with these formats
        (``_tally``: as for :meth:`affine`).

        When all three passes are ``BlockMac`` s it is
        :func:`~mptorch.quant.block_matmul_formats`'s container, which packs
        an operand in a square-tiled format once and reads it again in
        backward, with the quantizers this value sets in place of its own.
        """
        macs = (self.fwd_mac, self.igrad_mac, self.wgrad_mac)
        fwd_count, igrad_count, wgrad_count = (_counting(_tally, p) for p in _PASSES)
        if self.palette:
            if _tally is not None:
                _tally.gap("fwd", "mac", "a palette mac")
            formats = _PaletteMatmulFormats(
                _Palette("matmul", self.fwd_mac, self.output_quant),
                bwd_agrad_math=_matmul_pass(
                    macs[1], _matmul_agrad, _block_matmul_agrad, "gradient", igrad_count
                ),
                bwd_bgrad_math=_matmul_pass(
                    macs[2], _matmul_bgrad, _block_matmul_bgrad, "gradient", wgrad_count
                ),
            )
        elif all(isinstance(m, BlockMac) for m in macs):
            if _tally is not None:
                for pass_ in _PASSES:
                    _tally.gap(pass_, "mac", "a block GEMM")
            formats = block_matmul_formats(macs[0], agrad_mac=macs[1], bgrad_mac=macs[2])
        else:
            formats = QMatmulFormats(
                fwd_math=_matmul_pass(
                    macs[0], _matmul_fwd, _block_matmul_fwd, "forward", fwd_count
                ),
                bwd_agrad_math=_matmul_pass(
                    macs[1], _matmul_agrad, _block_matmul_agrad, "gradient", igrad_count
                ),
                bwd_bgrad_math=_matmul_pass(
                    macs[2], _matmul_bgrad, _block_matmul_bgrad, "gradient", wgrad_count
                ),
            )
            formats._wide_fwd = _wide_forward(macs[0], _matmul_fwd, _gemm_nd, fwd_count)
        for slot, name, value in (
            ("a_quant", "input_quant", self.input_quant),
            ("b_quant", "weight_quant", self.weight_quant),
            ("agrad_quant", "igrad_quant", self.igrad_quant),
            ("bgrad_quant", "wgrad_quant", self.wgrad_quant),
        ):
            if value is not None:
                setattr(formats, slot, _counting_quant(value, _tally, name))
        formats.recompute_a = self.recompute_input
        formats.recompute_b = self.recompute_weight
        if self.output_quant is not None and not self.palette:
            output = _counting_quant(self.output_quant, _tally, "output_quant")
            _with_output(formats, output, torch.matmul)
        formats._source = self
        return self._rotated(formats, ("a_quant", "b_quant"), _MATMUL_GRADS)

    def label(self) -> str:
        """A one-line summary of the formats, for reports: the mac (each
        pass's, if they differ), then the quantizers that are set, named once
        for the operands and once for the gradients when they agree."""
        macs = [_mac_label(getattr(self, name)) for name in _MAC_SLOTS]
        if macs[1:] == macs[:1] * 2:
            parts = [f"mac {macs[0]}"]
        else:
            parts = [f"{p} {m}" for p, m in zip(("fwd", "igrad", "wgrad"), macs, strict=True)]
        slots = {name.removesuffix("_quant"): getattr(self, name) for name in _QUANT_SLOTS}
        for group, names in (("operands", ("input", "weight")), ("grads", ("igrad", "wgrad"))):
            first, second = (slots[n] for n in names)
            if first is not None and first == second:
                slots[group] = slots.pop(names[0])
                del slots[names[1]]
        if "operands" in slots and slots["operands"] == slots.get("grads"):
            slots["quant"] = slots.pop("operands")
            del slots["grads"]
        parts += [f"{n} {_quant_label(q)}" for n, q in slots.items() if q is not None]
        if self.rotation is not None:
            parts.append(f"rotation {self.rotation.label()}")
        return ", ".join(parts)


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


# --- rules and recipes ---------------------------------------------------------

# The formats each kind past the GEMMs takes.
_KIND_VALUES: dict[str, type] = {
    "activation": ActivationFormats,
    "elementwise": PointwiseFormats,
    "embedding": PointwiseFormats,
    "softmax": SoftmaxFormats,
    "norm": NormFormats,
    "reduction": ReductionFormats,
    "attention": AttentionFormats,
}

Where = str | type | tuple[type, ...] | Callable[[str, nn.Module], bool]


def _kinds(kinds: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """A rule's kinds as ``(kind, value)`` pairs, ``gemm`` expanded to the
    three GEMM kinds (a kind named as well keeps its own value)."""
    out: dict[str, Any] = {}
    if "gemm" in kinds:
        gemm = _checked_kind("gemm", kinds["gemm"])
        out.update(dict.fromkeys(_GEMM_KINDS, gemm))
    for kind, value in kinds.items():
        if kind == "gemm":
            continue
        if kind not in _KINDS:
            raise TypeError(
                f"a recipe has no kind {kind!r}: the kinds are {', '.join(_KINDS)}, "
                "and gemm sets the three GEMM kinds at once"
            )
        out[kind] = _checked_kind(kind, value)
    return tuple(out.items())


def _checked_kind(kind: str, value: Any) -> Any:
    expected = _KIND_VALUES.get(kind, GemmFormats)
    if value is not None and not isinstance(value, expected):
        raise TypeError(
            f"the {kind} kind takes {'an' if expected.__name__[0] in 'AEIOU' else 'a'} "
            f"{expected.__name__} or None, got {type(value).__name__}"
        )
    if kind == "embedding" and value is not None and value.carrier_rounding is RoundMode.RO:
        raise ValueError(
            "the embedding kind's carrier_rounding is RoundMode.RO: an embedding gathers its "
            "rows, which leaves no carrier step to round to odd -- use RoundMode.RNE"
        )
    if kind in ("conv", "gemm") and value is not None and value._has_block_mac:
        raise ValueError(
            f"{kind}= would give a BlockMac to convolutions, which have no block-format "
            "arithmetic: give it as linear= (and matmul=) instead"
        )
    return value


@dataclass(frozen=True)
class Rule:
    """Formats for some kinds, wherever ``where`` matches.

    ``where`` chooses the modules by their qualified name, as
    ``model.named_modules()`` gives it (``""`` is the model itself), in one of
    three forms:

    * a glob, matched against the whole name with
      :func:`fnmatch.fnmatchcase`: ``*`` matches any run of characters, dots
      included, so ``"encoder.*"`` matches every module below ``encoder`` and
      ``"*"`` every module;
    * a type, or a tuple of types, matched with ``isinstance``;
    * a predicate, called as ``where(name, module)``.

    ``kinds`` maps a kind to its formats (a :class:`GemmFormats` for the
    three GEMM kinds, an :class:`~mptorch.quant.ActivationFormats` for
    ``activation``, a :class:`~mptorch.quant.PointwiseFormats` for
    ``elementwise`` and ``embedding``, a :class:`~mptorch.quant.SoftmaxFormats`
    for ``softmax``, a :class:`~mptorch.quant.NormFormats` for ``norm``, a
    :class:`~mptorch.quant.ReductionFormats` for ``reduction``, an
    :class:`~mptorch.quant.AttentionFormats` for ``attention``), or to
    ``None`` for plain PyTorch, and
    ``gemm`` names the three GEMM kinds at once. ``site``
    restricts the rule to one call site inside a module, such as the first of
    an attention block's two activation products; ``None`` matches every site.

    Raises:
        TypeError: for an unknown kind, a value that is not the kind's
            formats or ``None``, or a ``where`` of none of the three forms.
        ValueError: for a ``BlockMac`` given to the ``conv`` kind.
    """

    where: Where
    kinds: Any = field(default=())
    site: Any = None

    def __post_init__(self) -> None:
        where = self.where
        if not (
            isinstance(where, str | type)
            or (isinstance(where, tuple) and all(isinstance(t, type) for t in where))
            or callable(where)
        ):
            raise TypeError(
                "where is a name glob, a type, a tuple of types or a predicate "
                f"(name, module) -> bool, got {type(where).__name__}"
            )
        pairs = self.kinds.items() if isinstance(self.kinds, Mapping) else self.kinds
        object.__setattr__(self, "kinds", _kinds(dict(pairs)))

    def matches(self, name: str, module: nn.Module, site: Any = None) -> bool:
        """Whether this rule applies to ``module``, named ``name``, at ``site``."""
        if self.site is not None and self.site != site:
            return False
        where = self.where
        if isinstance(where, str):
            return fnmatchcase(name, where)
        if isinstance(where, type | tuple):
            return isinstance(module, where)
        return bool(where(name, module))

    def get(self, kind: str) -> tuple[bool, Any]:
        """``(True, value)`` if this rule names ``kind``, else ``(False, None)``."""
        for k, v in self.kinds:
            if k == kind:
                return True, v
        return False, None


class Recipe:
    """Which formats each part of a model computes in.

    ``Recipe(**kinds)`` gives every module the formats of each kind named
    (``linear``, ``conv``, ``matmul``, or ``gemm`` for all three;
    ``activation``, ``elementwise``, ``embedding``, ``softmax``, ``norm``,
    ``reduction``, ``attention``), and
    :meth:`override` adds a rule for some modules. Rules are applied in
    order, so a later rule wins for the kinds it names, and a kind it does not
    name keeps what the earlier rules gave it. A kind no rule names, and one
    set to ``None``, stays plain PyTorch. ``Recipe(rules, **kinds)`` starts
    from a sequence of :class:`Rule` values, after the keywords' rule.

    A recipe is immutable: :meth:`override` returns a new one. The same
    recipe can convert several models, and its formats build a new container
    for every layer.

    Example::

        >>> from mptorch import BinaryK
        >>> from mptorch.quant import GemmFormats, Quant, Recipe, SplitMac
        >>> p8 = BinaryK(8, 4)
        >>> gf = GemmFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
        >>> recipe = Recipe(gemm=gf).override("classifier.*", linear=None)
    """

    __slots__ = ("rules",)
    rules: tuple[Rule, ...]

    def __init__(self, rules: "tuple[Rule, ...] | list[Rule]" = (), /, **kinds: Any):
        rules = tuple(rules)
        for rule in rules:
            if not isinstance(rule, Rule):
                raise TypeError(f"a Recipe's rules are Rule values, got {type(rule).__name__}")
        if kinds:
            rules = (Rule("*", kinds), *rules)
        object.__setattr__(self, "rules", rules)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("a Recipe is immutable: override() returns a new one")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Recipe) and self.rules == other.rules

    def __hash__(self) -> int:
        return hash(self.rules)

    def __repr__(self) -> str:
        return f"Recipe({list(self.rules)!r})"

    def override(self, where: Where, *, site: Any = None, **kinds: Any) -> "Recipe":
        """A new recipe with one more rule, which wins over the earlier ones
        for the kinds it names (see :class:`Rule` for ``where`` and ``site``)."""
        return Recipe((*self.rules, Rule(where, kinds, site)))

    def attention_parts(self, where: Where, formats: AttentionFormats) -> "Recipe":
        """A new recipe that gives ``formats`` to the attention at ``where``,
        however the model spells it.

        Four rules: the ``attention`` kind, which
        ``F.scaled_dot_product_attention`` and ``nn.MultiheadAttention``
        take, and the parts of an eager attention, written out of two
        matmuls and a softmax as transformers' ``eager_attention_forward``
        is: ``formats.qk`` for the module's first product of two activations
        (site ``"matmul#0"``), ``formats.softmax`` for its softmax, and
        ``formats.pv`` for its second product (site ``"matmul#1"``). A model
        and its eager twin then compute the same words (see
        :func:`~mptorch.quant.qattention` for the cases where they cannot).

        Raises:
            TypeError: if ``formats`` is not an :class:`~mptorch.quant.AttentionFormats`.
            ValueError: if ``formats.scores`` is set: an eager attention
                multiplies its scores by the scale in a pointwise op, which
                no rule reaches apart from every other pointwise op of the
                module (the ``elementwise`` kind) -- round the product with
                ``qk``'s ``output_quant`` instead, or give the ``attention``
                kind alone. Or if ``formats.flash`` is set: an eager attention's
                two products and its softmax are three calls, which no rule
                can run as one flash loop.
        """
        if not isinstance(formats, AttentionFormats):
            raise TypeError(
                f"attention_parts takes an AttentionFormats, got {type(formats).__name__}"
            )
        if formats.scores is not None:
            raise ValueError(
                "attention_parts was given formats whose scores are rounded: an eager "
                "attention multiplies its scores by the scale in a pointwise op, which no "
                "rule reaches apart from the module's other pointwise ops -- round the "
                "product with qk's output_quant instead, or give the attention kind alone"
            )
        if formats.flash is not None:
            raise ValueError(
                "attention_parts was given formats with a flash part: an eager attention's two "
                "products and its softmax are three calls, which no rule can run as one flash "
                "loop -- give the attention kind alone, to a model that calls SDPA"
            )
        return Recipe(
            (
                *self.rules,
                Rule(where, {"attention": formats}),
                Rule(where, {"matmul": formats.qk}, "matmul#0"),
                Rule(where, {"softmax": formats.softmax}),
                Rule(where, {"matmul": formats.pv}, "matmul#1"),
            )
        )

    def resolve(self, name: str, module: nn.Module, kind: str, site: Any = None) -> Any:
        """The formats ``module``, named ``name``, gets for ``kind`` at ``site``:
        the value of the last rule that matches and names the kind, or
        ``None``. A predicate may look at the module, so nothing is cached."""
        if kind not in _KINDS:
            raise ValueError(f"a recipe has no kind {kind!r}: the kinds are {', '.join(_KINDS)}")
        for rule in reversed(self.rules):
            named, value = rule.get(kind)
            if named and rule.matches(name, module, site):
                return value
        return None

    def plan(self, model: nn.Module, **kwargs: Any) -> "ConvertReport":
        """What :func:`mptorch.quant.convert_` would do to ``model``, without
        doing it. It takes ``convert_``'s keyword arguments."""
        from .convert import _plan  # imported here: convert imports this module

        return _plan(model, self, **kwargs)
