"""The value types that describe a quantized computation.

:class:`Quant` describes an elementwise step, :class:`SplitMac` and
:class:`FusedMac` a reduction (a dot product), and :class:`Palette` a set of
simulated formats from which each output element selects one.
:class:`BlockQuant` and :class:`BlockMac` do the same for block formats.

Every step of a quantized layer is one of two kinds, and only the second
needs a policy for its internal arithmetic:

* An **elementwise** step (a weight, activation or gradient quantizer)
  rounds each value of a tensor to a simulated format, computing in the
  carrier (the binary float the cast rounds in). It needs only a format and
  a rounding mode, and :class:`Quant` makes it a callable.
* A **reduction** step is the GEMM, the only place where the arithmetic
  between roundings is simulated, so it is the only place where a
  multiply-accumulate (mac) policy matters. :class:`SplitMac` and
  :class:`FusedMac` are named after the two policies the kernels implement
  (``common/gemm_policy.h``).

The API is therefore a few value types whose number does not grow with the
number of ops, instead of one formats class per kind of layer, each with its
own slots and its own record of which slots are in use.

Describing a reduction as a value has two consequences:

* ``accumulate_quant`` and ``fma_quant`` are no longer booleans:
  ``SplitMac(mul=f, acc=None)`` *is* the case of a sum left in the carrier,
  so an accumulate format that is present but ignored cannot be expressed.
* A palette is no longer a separate family of functions. Every format slot
  takes a ``Number`` or a sequence of them, and a sequence of more than one
  format in any slot selects a ``_mixed`` op, which needs a ``prec_idx``.
  :func:`spec_for_mac` picks the op from the slots instead of the caller
  picking one of the flat wrappers. A :class:`SplitMac` whose two formats are
  of different families selects one of the cross-family ops, which take one
  format per slot.

Each mac is resolved to a ``_GemmSpec`` (the op and its arguments) once, and
the result is cached on the mac's value: the classes are frozen dataclasses,
so they hash by value, and :func:`spec_for_mac` caches on them. Working out a
format's defaults and carrier checks costs 2.3 to 6.4 microseconds, more than
the rest of the Python call path, so an object API that did it on every call
would be slower than the flat wrappers it is built on.
"""

import weakref
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import lru_cache, partial
from typing import Any, cast

import torch
from torch import nn
from torch._subclasses.fake_tensor import is_fake as _is_fake

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    FixedPoint,
    Number,
    Overflow,
    RoundMode,
    SuperFP,
)

from .ops import (
    BlockPacked,
    block_pack,
    block_quantize,
)
from .ops._accumulate import _checked_block_size
from .ops._checks import _checked_carrier
from .ops._spec import _GemmSpec, _QuantSpec, _run_quant
from .ops.binaryk import (
    _binaryK_accumulated_odd_spec,
    _binaryK_accumulated_spec,
    _binaryK_fma_accumulated_odd_spec,
    _binaryK_fma_accumulated_spec,
    _binaryK_fma_mixed_odd_spec,
    _binaryK_fma_mixed_spec,
    _binaryK_fma_odd_spec,
    _binaryK_fma_spec,
    _binaryK_mixed_odd_spec,
    _binaryK_mixed_spec,
    _binaryK_odd_spec,
    _binaryK_quant_spec,
    _binaryK_spec,
)
from .ops.block import _block_gemm_spec, _block_quant_spec, _BlockGemmSpec, _run_block_quant
from .ops.cross import (
    _binaryK_superfp_accumulated_odd_spec,
    _binaryK_superfp_accumulated_spec,
    _binaryK_superfp_odd_spec,
    _binaryK_superfp_spec,
    _superfp_binaryK_accumulated_odd_spec,
    _superfp_binaryK_accumulated_spec,
    _superfp_binaryK_odd_spec,
    _superfp_binaryK_spec,
)
from .ops.fixedpoint import _fixedpoint_fma_spec, _fixedpoint_quant_spec, _fixedpoint_spec
from .ops.superfp import (
    _superfp_accumulated_odd_spec,
    _superfp_accumulated_spec,
    _superfp_fma_accumulated_odd_spec,
    _superfp_fma_accumulated_spec,
    _superfp_fma_mixed_odd_spec,
    _superfp_fma_mixed_spec,
    _superfp_fma_odd_spec,
    _superfp_fma_spec,
    _superfp_mixed_odd_spec,
    _superfp_mixed_spec,
    _superfp_odd_spec,
    _superfp_quant_spec,
    _superfp_spec,
)

__all__ = ["Quant", "Palette", "SplitMac", "FusedMac", "BlockQuant", "BlockMac"]


# The simulated float formats the GEMM kernels of this module take.
# `_checked_accumulation` checks an outer format against this union. `Number`
# also covers `FixedPoint`, whose GEMM takes one format per slot and the NAIVE
# order only (`_checked_fixedpoint`), and `BlockFormat`, which has a GEMM of
# its own (`BlockMac`).
GemmFormat = BinaryK | SuperFP

# The fields that the `_mixed` schemas take once for a whole palette instead of
# per entry. The kernels tabulate only each entry's *widths* and keep one copy
# of these fields (`BinaryKCommon` and `SuperfpCommon` in common/gemm_args.h),
# so every entry of a palette must agree on them. A fixed-point slot is a
# Palette too, of one format (a mac refuses more), so that every reader of a
# slot reads it alike.
_SHARED_FIELDS: dict[type, tuple[str, ...]] = {
    BinaryK: ("is_signed", "prng_bits", "saturation", "subnormals"),
    SuperFP: ("is_signed", "prng_bits", "saturation"),
    FixedPoint: ("is_signed", "symmetric", "prng_bits"),
}

MAX_PALETTE = 8  # must equal MAX_GEMM_FORMATS in common/gemm_policy.h


@dataclass(frozen=True, init=False)
class Palette:
    """Up to eight simulated formats of one family, one selected per output element.

    A GEMM slot holds a palette when its format varies across the output. The
    ``_mixed`` ops take the palette's widths as a table, and a ``prec_idx``
    map gives the index of the entry each output element uses. A plain
    sequence of formats works anywhere a ``Palette`` does and is converted to
    one; the class exists to validate the entries. All entries must be of one
    family (all ``BinaryK`` or all ``SuperFP``), and they must agree on the
    fields the op takes once for the whole palette: the sign, the
    stochastic-rounding bits, the saturation mode and the subnormals mode. An
    entry that disagrees is rejected, with an error naming it, rather than
    silently given entry 0's value. A ``FixedPoint`` slot is a palette of one
    format: the fixed-point GEMM has no palette ops, so a mac refuses more.

    Args:
        formats (Number or Sequence[Number] or Palette): one format, a
            sequence of up to ``MAX_PALETTE`` formats, or another palette,
            whose formats are copied.

    Raises:
        ValueError: for an empty sequence, more than eight formats, or entries
            that disagree on a shared field.
        TypeError: for a format family no mac slot takes (a ``BlockFormat``
            is ``BlockMac``'s), or a mix of families.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import Palette, SplitMac, qmm
        >>> palette = Palette([BinaryK(8, 4), BinaryK(8, 5), BinaryK(16, 11)])
        >>> len(palette)
        3
        >>> prec_idx = torch.randint(3, (4, 3))
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), SplitMac(palette, palette),
        ...     prec_idx=prec_idx).shape
        torch.Size([4, 3])
    """

    formats: tuple[GemmFormat, ...]

    def __init__(self, formats: Number | Sequence[Number] | Palette):
        if isinstance(formats, Palette):
            formats = formats.formats
        elif isinstance(formats, Number):
            formats = (formats,)
        object.__setattr__(self, "formats", tuple(formats))
        self._validate()

    def _validate(self) -> None:
        if not self.formats:
            raise ValueError("a Palette needs at least one format")
        if len(self.formats) > MAX_PALETTE:
            raise ValueError(
                f"a Palette holds at most {MAX_PALETTE} formats, got {len(self.formats)}"
            )
        first = self.formats[0]
        if type(first) not in _SHARED_FIELDS:
            hint = ": a block format's GEMM is BlockMac's" if isinstance(first, BlockFormat) else ""
            raise TypeError(
                "a mac slot takes BinaryK, SuperFP or FixedPoint formats, got "
                f"{type(first).__name__}{hint}"
            )
        for i, fmt in enumerate(self.formats[1:], start=1):
            if type(fmt) is not type(first):
                raise TypeError(
                    f"every format in a Palette must be the same type: entry {i} is "
                    f"{type(fmt).__name__}, entry 0 is {type(first).__name__}"
                )
            for name in _SHARED_FIELDS[type(first)]:
                if getattr(fmt, name) != getattr(first, name):
                    raise ValueError(
                        f"a Palette shares {name} across its entries (the kernel tabulates "
                        f"only the format widths): entry {i} has {name}={getattr(fmt, name)!r}, "
                        f"entry 0 has {getattr(first, name)!r}"
                    )

    def __len__(self) -> int:
        return len(self.formats)

    def __iter__(self):
        return iter(self.formats)

    def __getitem__(self, i: int) -> GemmFormat:
        return self.formats[i]


def _as_palette(slot: Number | Sequence[Number] | Palette | None) -> Palette | None:
    """Convert a format slot to a ``Palette``. ``None``, which means the step
    is not rounded to a format, stays ``None``."""
    return None if slot is None else Palette(slot)


def _varies(slot: Palette | None) -> bool:
    """Whether this slot needs a ``_mixed`` op, that is, holds more than one
    format."""
    return slot is not None and len(slot) > 1


# --- an elementwise quantizer ------------------------------------------------


@dataclass(frozen=True)
class Quant:
    """A simulated format plus a rounding mode, as a callable ``Tensor -> Tensor``.

    Calling it rounds every element of a tensor to the format, computing in
    the carrier, and returns a tensor of the same dtype and shape. Every
    ``*_quant`` slot of a ``QAffineFormats`` or ``QMatmulFormats`` takes one,
    and a straight-through :class:`mptorch.quant.Quantizer` or a QAT observer
    can wrap one. The format is resolved once, when the ``Quant`` is built:
    the op and its arguments, and what each carrier and each storage dtype
    finds in the format. A call then makes only the checks that depend on the
    tensor's dtype, and gives the results, warnings and errors that
    :func:`mptorch.quant.binaryK_quantize` and its siblings give on every
    call. The class is frozen, so a ``Quant`` can be a dictionary key or a
    cache key, like the mac objects.

    Args:
        fmt (Number): the simulated format to round to: a
            :class:`mptorch.BinaryK`, a :class:`mptorch.SuperFP`, a
            :class:`mptorch.FixedPoint` or a :class:`mptorch.BlockFormat`. A
            block format's blocks run along the last axis; use
            :class:`BlockQuant` for another axis or for a fixed tensor scale.
        rounding (RoundMode): the rounding mode. Default: ``RoundMode.RNE``
        carrier (torch.dtype, optional): the carrier the cast rounds in, as
            for the argument of the same name of
            :func:`mptorch.quant.binaryK_quantize`. ``None`` uses the tensor's
            default carrier: binary32 for float16, bfloat16 and float32
            tensors, and binary64 for float64 ones. ``torch.float64`` rounds
            in binary64 whatever the tensor's dtype. ``torch.float32`` rounds
            in binary32 and refuses a float64 tensor. Default: ``None``
        inplace (bool): round the tensor in place and return it, instead of
            allocating a result. This calls
            :func:`mptorch.quant.binaryK_quantize_`,
            :func:`mptorch.quant.superfp_quantize_`,
            :func:`mptorch.quant.fixedpoint_quantize_` or
            :func:`mptorch.quant.block_quantize_`. Use it for a tensor the
            caller owns, such as a weight quantized once when it is loaded.
            Never use it in a ``*_quant`` slot of a layer's formats, because
            a layer's input belongs to the autograd graph of whatever produced
            it. The call refuses whatever that function refuses; for the
            elementwise quantizers, that is a tensor that is not contiguous, a
            CUDA view that does not start on a 16-byte boundary, and a
            ``carrier`` wider than the tensor. Default: ``False``

    Raises:
        TypeError: for a format that has no elementwise quantizer, or a
            ``carrier`` that is not a ``torch.dtype``.
        ValueError: for a ``carrier`` dtype that is not a carrier.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, RoundMode, SuperFP
        >>> from mptorch.quant import QAffineFormats, Quant
        >>> q = Quant(BinaryK(8, 4))
        >>> q(torch.tensor([1.0, 0.3, 0.7]))
        tensor([1.0000, 0.3125, 0.6875])
        >>> formats = QAffineFormats()
        >>> formats.input_quant = Quant(SuperFP(3, 4, 8, 7), RoundMode.SR)
    """

    fmt: Number
    rounding: RoundMode = RoundMode.RNE
    _: KW_ONLY
    carrier: torch.dtype | None = None
    inplace: bool = False
    _call: Callable[[torch.Tensor], torch.Tensor] = field(init=False, repr=False, compare=False)
    # the resolved call of an elementwise family (None for a block format),
    # which a pointwise op reads to round its own result (ops/_spec.py's
    # _round_result)
    _spec: _QuantSpec | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fmt, rm, carrier = self.fmt, self.rounding, _checked_carrier(self.carrier)
        inplace = self.inplace
        spec: _QuantSpec | None = None
        if isinstance(fmt, BinaryK):
            assert fmt.bias is not None  # resolved by BinaryK.__post_init__
            spec = _binaryK_quant_spec(
                fmt.K,
                fmt.P,
                fmt.bias,
                fmt.prng_bits,
                fmt.is_signed,
                rm,
                fmt.saturation,
                fmt.subnormals,
                carrier=carrier,
                in_place=inplace,
            )
            call = partial(_run_quant, spec)
        elif isinstance(fmt, SuperFP):
            spec = _superfp_quant_spec(
                fmt.man_bits,
                fmt.exp_bits,
                fmt.normal_binades,
                fmt.bias,
                fmt.prng_bits,
                fmt.is_signed,
                rm,
                fmt.saturation,
                carrier=carrier,
                in_place=inplace,
            )
            call = partial(_run_quant, spec)
        elif isinstance(fmt, FixedPoint):
            spec = _fixedpoint_quant_spec(
                fmt.wl,
                fmt.fl,
                fmt.prng_bits,
                fmt.is_signed,
                fmt.symmetric,
                rm,
                carrier=carrier,
                in_place=inplace,
            )
            call = partial(_run_quant, spec)
        elif isinstance(fmt, BlockFormat):
            call = partial(_run_block_quant, _block_quant_spec(fmt, rm, carrier, inplace))
        else:
            raise TypeError(f"no elementwise quantizer for {type(fmt).__name__}")
        object.__setattr__(self, "_call", call)
        object.__setattr__(self, "_spec", spec)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._call(x)


# --- a block format ------------------------------------------------------------


@dataclass(frozen=True)
class BlockQuant:
    """A block format plus a rounding mode: a fake quantizer and a packer.

    Calling it runs :func:`mptorch.quant.block_quantize` along ``axis``: it
    returns the decoded values, with the tensor's shape and dtype. That is
    what a ``*_quant`` slot of a layer's formats, or a
    :class:`mptorch.quant.Quantizer`, takes.

    :meth:`pack` runs :func:`mptorch.quant.block_pack` instead and returns the
    codes themselves, which the block GEMM of
    :func:`mptorch.quant.block_gemm_formats` and
    :func:`mptorch.quant.block_matmul_formats` multiplies directly, decoding
    them inside the kernel. Use ``BlockQuant(fmt, axis=1).pack`` as a
    ``QLinear``'s ``weight_quant`` to pack its weight once per weight update
    instead of once per hook. The packing is cached on the tensor, its
    version counter, the axis, its device and its dtype. It is therefore
    redone after each optimizer step (which writes the weight in place) and
    after the module is moved or converted (``.to()``, ``.cuda()``,
    ``.half()``, which keep the parameter object and its version), and at no
    other time. The cache misses a write that the version counter does not
    see, such as one made through ``.data``.

    Args:
        fmt (BlockFormat): the format.
        rounding (RoundMode): the elements' rounding mode. Default:
            ``RoundMode.RNE``
        axis (int): the axis the blocks run along. Default: ``-1``
        tensor_scale (float, optional): a fixed per-tensor scale, for a
            format that has one (NVFP4). ``None`` computes it from the tensor
            on each call. Default: ``None``
        carrier (torch.dtype, optional): as for
            :func:`mptorch.quant.block_quantize`. Default: ``None``

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import BlockQuant
        >>> q = BlockQuant(MXFP8_E4M3)
        >>> q(torch.tensor([[1.0, 0.1, -3.3, 100.0]]))
        tensor([[ 1.0000,  0.1016, -3.2500, 96.0000]])
        >>> w = torch.randn(16, 64)
        >>> q.pack(w, axis=1) is q.pack(w, axis=1)
        True
    """

    fmt: BlockFormat
    rounding: RoundMode = RoundMode.RNE
    _: KW_ONLY
    axis: int = -1
    tensor_scale: float | None = None
    carrier: torch.dtype | None = None
    # The last packing, as (weak reference to the tensor, its version, axis,
    # device, dtype, result). One entry is enough, since a BlockQuant packs one
    # weight. The device and the dtype are in the key because nn.Module's dtype
    # and device conversions (.to(), .half(), .cuda(), ...) keep the Parameter
    # object, so the weak reference still matches, and do not advance its
    # version: by default they assign `param.data` and keep the version, and
    # under torch.__future__.set_swap_module_params_on_conversion(True) they
    # swap the tensors and the version restarts at 0, so an old version can
    # come back (checked against torch 2.15's Module._apply).
    _memo: list = field(default_factory=lambda: [None], init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.fmt, BlockFormat):
            raise TypeError(f"BlockQuant takes a BlockFormat, got {type(self.fmt).__name__}")
        _checked_carrier(self.carrier)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return block_quantize(
            x,
            self.fmt,
            self.axis,
            rounding=self.rounding,
            tensor_scale=self.tensor_scale,
            carrier=self.carrier,
        )

    def pack(self, x: torch.Tensor, axis: int | None = None) -> BlockPacked:
        """:func:`mptorch.quant.block_pack` of ``x`` along ``axis``, cached on
        ``x``, its version counter, the axis, its device and its dtype.
        ``axis`` defaults to this quantizer's."""
        axis = self.axis if axis is None else axis
        memo = self._memo[0]
        if memo is not None:
            ref, version, memo_axis, device, dtype, packed = memo
            if (
                ref() is x
                and x._version == version
                and memo_axis == axis
                and x.device == device
                and x.dtype == dtype
            ):
                return packed
        packed = block_pack(
            x,
            self.fmt,
            axis,
            rounding=self.rounding,
            tensor_scale=self.tensor_scale,
            carrier=self.carrier,
        )
        # A real tensor packed under a fake tensor mode (profile(run=False)
        # packs one so) has fake codes, which a later real call on the same
        # tensor would return: they are not kept. Dynamo's trace, whose tensors
        # are all fake, keeps its packing as before.
        if not torch.compiler.is_compiling() and _is_fake(packed.data) and not _is_fake(x):
            return packed
        self._memo[0] = (weakref.ref(x), x._version, axis, x.device, x.dtype, packed)
        return packed

    def pack_fresh(self, x: torch.Tensor, axis: int) -> BlockPacked:
        """:func:`mptorch.quant.block_pack` of ``x`` along ``axis``, without the
        cache. Use it for an activation or a gradient, which is a new tensor
        on every call."""
        return block_pack(
            x,
            self.fmt,
            axis,
            rounding=self.rounding,
            tensor_scale=self.tensor_scale,
            carrier=self.carrier,
        )


def _as_quant(slot: object, name: str) -> Any:
    """A quantizer slot's value as a layer's formats take it: a format becomes
    a quantizer that rounds to it to nearest."""
    if slot is None:
        return None
    if isinstance(slot, BlockFormat):
        return BlockQuant(slot)
    if isinstance(slot, Number):
        return Quant(slot)
    if isinstance(slot, nn.Module):
        raise TypeError(
            f"{name} is a {type(slot).__name__}, a module: a recipe gives the same object to "
            "every layer it matches, so a module's state (a QAT observer's statistics) would be "
            "shared by all of them. Give the slot a Quant or a ScaledQuant (a Quantizer's "
            "forward quantizer, which is all a layer runs), or build those layers' formats "
            "yourself"
        )
    if not callable(slot):
        raise TypeError(
            f"{name} takes a format, a Quant, a ScaledQuant, a BlockQuant or a callable, got "
            f"{type(slot).__name__}"
        )
    return slot


def _as_block_quant(slot: BlockFormat | BlockQuant, name: str) -> BlockQuant:
    """Convert a block operand slot to a ``BlockQuant``; a bare format gets ``RoundMode.RNE``."""
    if isinstance(slot, BlockQuant):
        return slot
    if isinstance(slot, BlockFormat):
        return BlockQuant(slot)
    raise TypeError(f"{name} is a BlockFormat or a BlockQuant, got {type(slot).__name__}")


@dataclass(frozen=True)
class BlockMac:
    """The block GEMM: two block-format operands and the arithmetic of their product.

    ``a`` and ``b`` are the block formats that the left and right operands are
    packed in. Each is a :class:`mptorch.BlockFormat` (packed with
    ``RoundMode.RNE``) or a :class:`BlockQuant` (which also gives the rounding
    mode and a tensor scale). ``b`` defaults to ``a``.

    The GEMM decodes both operands in the carrier and sums their products as
    :func:`mptorch.quant.block_matmul` describes. In a split mac each product
    is rounded only by the carrier, and each partial sum to ``acc``. With
    ``fused=True`` each fused multiply-add is rounded to ``acc``.
    ``acc=None`` leaves the sum in the carrier. The accumulate algorithms are
    those of :class:`SplitMac`.

    :func:`mptorch.quant.qmatmul` and :class:`mptorch.quant.QMatmul` accept a
    ``BlockMac`` (:func:`mptorch.quant.block_matmul_formats` builds their
    formats). They also accept a bare ``BlockFormat``, which means
    ``BlockMac(fmt)``.

    Args:
        a (BlockFormat or BlockQuant): the left operand's format.
        b (BlockFormat or BlockQuant, optional): the right operand's format.
            Default: ``None``, meaning ``a``'s.
        acc (BinaryK, optional): the accumulate format. Default: ``None``
        fused (bool): use a fused multiply-add per step. Default: ``False``
        rounding (RoundMode): the rounding mode of the sum. Default:
            ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for :class:`SplitMac`
            (``TREE`` needs ``fused=False``). Default: ``AccumulateAlgorithm.NAIVE``
        carrier (torch.dtype, optional): ``torch.float64`` packs the operands
            and multiplies them in binary64, whatever their dtype (an operand
            given as a ``BlockQuant`` that names a carrier of its own keeps
            that carrier for packing). Default: ``None``, meaning the
            operands' carrier: binary64 for float64 tensors, binary32
            otherwise.
        block_size (int, optional): as for :class:`SplitMac`. Default: ``None``
        outer (BinaryK, optional): as for :class:`SplitMac`. Default: ``None``
        tensor_scale_epilogue (bool): apply the operands' per-tensor scales
            once, to the result, as NVIDIA's NVFP4 GEMMs do, instead of in
            each decoded element (:func:`mptorch.quant.block_matmul`).
            Default: ``False``

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, MXFP8_E4M3, MXFP4_E2M1
        >>> from mptorch.quant import BlockMac, qmm
        >>> mac = BlockMac(MXFP8_E4M3, MXFP4_E2M1, BinaryK(16, 11))
        >>> qmm(torch.randn(4, 64), torch.randn(64, 8), mac).shape
        torch.Size([4, 8])
    """

    a: BlockFormat | BlockQuant
    b: BlockFormat | BlockQuant | None = None
    acc: BinaryK | None = None
    _: KW_ONLY
    fused: bool = False
    rounding: RoundMode = RoundMode.RNE
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE
    carrier: torch.dtype | None = None
    block_size: int | None = None
    outer: BinaryK | None = None
    tensor_scale_epilogue: bool = False

    def __post_init__(self) -> None:
        def operand(slot: BlockFormat | BlockQuant, name: str) -> BlockQuant:
            q = _as_block_quant(slot, name)
            # the mac's carrier=torch.float64 applies to packing too, unless the
            # operand's BlockQuant names a carrier of its own
            if self.carrier is torch.float64 and q.carrier is None:
                q = replace(q, carrier=torch.float64)
            return q

        a = operand(self.a, "BlockMac's a")
        b = a if self.b is None else operand(self.b, "BlockMac's b")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)
        spec_for_block_mac(self)  # raise now any error the arithmetic would raise later


def cast_quant(slot: BlockFormat | BlockQuant | None) -> BlockQuant:
    """Return a ``BlockMac`` slot as the ``BlockQuant`` that ``BlockMac`` converted it to."""
    assert slot is not None  # BlockMac.__post_init__ sets an unset b to a
    return _as_block_quant(slot, "a BlockMac slot")


@lru_cache(maxsize=256)
def spec_for_block_mac(mac: BlockMac) -> _BlockGemmSpec:
    """Resolve a ``BlockMac`` to the block GEMM it names, cached on the mac's value."""
    return _block_gemm_spec(
        mac.acc,
        mac.fused,
        mac.rounding,
        mac.accumulate_algorithm,
        mac.block_size,
        mac.outer,
        mac.carrier,
        mac.tensor_scale_epilogue,
    )


# --- the two reduction policies ----------------------------------------------


def _checked_accumulation(
    mac: SplitMac | FusedMac, slots: tuple[Palette | None, ...], has_product: bool
) -> None:
    """Validate a mac's ``accumulate_algorithm``, ``block_size`` and ``outer``.

    ``slots`` is the mac's format slots after conversion to palettes:
    ``(mul, acc)`` for a split mac, ``(fma,)`` for a fused one. ``outer`` must
    be of the same family as the last slot that holds a format (the
    accumulate format if there is one, else the multiply format), because the
    outer format rounds totals of that format's values and no kernel converts
    them to another family. The block-size rule is the flat wrappers' own
    (:func:`mptorch.quant.ops._accumulate._checked_block_size`), so both API tiers raise
    the same errors.
    """
    outer = mac.outer
    if outer is not None and not isinstance(outer, GemmFormat):
        raise TypeError(
            f"outer must be one BinaryK or SuperFP format or None, got {type(outer).__name__}: "
            "a palette of outer formats is not implemented"
        )
    _checked_block_size(mac.accumulate_algorithm, mac.block_size, outer is not None, has_product)
    if mac.accumulate_algorithm is AccumulateAlgorithm.NAIVE:
        return
    if any(_varies(slot) for slot in slots):
        raise ValueError(
            f"AccumulateAlgorithm.{mac.accumulate_algorithm.name} is implemented for a single "
            "format per slot: the palette ops accumulate with NAIVE only, since no kernel runs "
            "a palette of accumulators or of outer formats; give each slot one format, or use "
            "NAIVE with the palette"
        )
    family = next((slot[0] for slot in reversed(slots) if slot is not None), None)
    if outer is not None and family is not None and type(outer) is not type(family):
        raise TypeError(
            f"outer must be of the accumulate format's family (the multiply format's when the "
            f"sum is left unrounded): got {type(outer).__name__} over "
            f"{type(family).__name__}; no kernel folds a total into another family"
        )


def _checked_carrier_rounding(mac: SplitMac | FusedMac, slots: tuple[Palette | None, ...]) -> None:
    """Validate a mac's ``carrier_rounding``: ``RNE`` (the default) or ``RO``.

    ``RO`` selects the ``custom_matmul_*_odd`` and ``custom_conv_*_odd`` ops,
    whose carrier operations round to odd, so that the simulated format's
    cast rounds the exact value. They exist for single-format, palette and
    cross-family macs, under every accumulate algorithm a mac of that kind
    takes (a palette takes only NAIVE, `_checked_accumulation`). A step
    rounded to odd is correct only if a format rounds it again afterwards, so
    ``RO`` is refused for ``FusedMac(None)``, whose fused step no format
    rounds. The bounds on the formats' precision depend on the carrier, so
    they are worked out when the mac is resolved (``ops._odd._odd_findings``) and
    reported on each call, for the call's carrier.
    """
    mode = mac.carrier_rounding
    if not isinstance(mode, RoundMode) or mode not in (RoundMode.RNE, RoundMode.RO):
        raise ValueError(
            f"carrier_rounding is RoundMode.RNE (the carrier's nearest step) or RoundMode.RO "
            f"(rounded to odd), got {mode!r}"
        )
    if mode is RoundMode.RNE:
        return
    if slots[0] is None:
        raise ValueError(
            "carrier_rounding=RoundMode.RO needs a format to round the fused step: a carrier "
            "step rounded to odd is only correct when a format rounds it again, and "
            "FusedMac(None) is the carrier's own nearest FMA"
        )


def _checked_fixedpoint(mac: SplitMac | FusedMac, slots: tuple[Palette | None, ...]) -> bool:
    """Validate a mac against the fixed-point GEMM's scope, and say whether it is one.

    A mac is fixed point when any slot holds a :class:`~mptorch.FixedPoint`.
    Its GEMM (:func:`mptorch.quant.fixedpoint_matmul`) takes one fixed-point
    format per slot, an accumulate format (a sum left in the carrier has no
    range to saturate or wrap at), the NAIVE order, and the carrier's
    nearest product: no palette, no other family beside it, no KAHAN,
    BLOCK or TREE, no ``carrier_rounding=RoundMode.RO``. A mac of the other
    families takes only ``Overflow.SATURATE``, which is what their formats'
    saturation modes do.
    """
    if not isinstance(mac.overflow, Overflow):
        raise TypeError(f"overflow must be an Overflow, got {mac.overflow!r}")
    present = [slot for slot in slots if slot is not None]
    if not any(isinstance(slot[0], FixedPoint) for slot in present):
        if mac.overflow is not Overflow.SATURATE:
            raise ValueError(
                "overflow=Overflow.WRAP needs a fixed-point accumulate format: a float "
                "format's overflow is its saturation mode"
            )
        return False
    families = sorted({type(slot[0]).__name__ for slot in present})
    if families != ["FixedPoint"]:
        raise TypeError(
            f"a fixed-point mac takes fixed-point formats only, got {' and '.join(families)}: no "
            "kernel pairs fixed point with another family; round the products with a Quant "
            "instead"
        )
    if isinstance(mac, SplitMac) and mac.acc is None:
        raise ValueError(
            "a fixed-point SplitMac needs an accumulate format: the sum saturates or wraps "
            "around at its range, and a sum left in the carrier has none"
        )
    if any(_varies(slot) for slot in present):
        raise ValueError(
            "a fixed-point mac takes one format per slot: no kernel runs a palette of "
            "fixed-point formats"
        )
    if mac.accumulate_algorithm is not AccumulateAlgorithm.NAIVE:
        raise ValueError(
            f"a fixed-point mac sums with AccumulateAlgorithm.NAIVE only, got "
            f"{mac.accumulate_algorithm.name}: every sum is exact already, so the other "
            "orders would only change where it saturates"
        )
    if mac.carrier_rounding is not RoundMode.RNE:
        raise ValueError(
            "a fixed-point mac takes carrier_rounding=RoundMode.RNE only: its sums are "
            "exact, and its product is the carrier's nearest; for an exact product, quantize "
            "the operands or use carrier=torch.float64"
        )
    acc = present[-1][0]
    assert isinstance(acc, FixedPoint)
    if mac.overflow is Overflow.WRAP and acc.is_signed and acc.symmetric:
        raise ValueError(
            f"{acc} is symmetric, so it has no modulus to wrap around: drop symmetric=True, or "
            "saturate"
        )
    return True


@dataclass(frozen=True)
class SplitMac:
    """Round the product and the running sum separately: two roundings per step.

    Each step of a dot product computes ``s = round_acc(s + round_mul(a * b))``
    in the carrier: the product is rounded to the ``mul`` format, and the new
    partial sum to the ``acc`` format. ``mul`` and ``acc`` are each a
    simulated format or a sequence of them. A sequence is a palette, and a
    palette of more than one format selects the ``_mixed`` op, which needs a
    ``prec_idx`` map at the call. ``acc`` (but not ``mul``) may also be
    ``None``, which leaves the running sum in the carrier's precision. The
    two formats may be of different families, a :class:`~mptorch.BinaryK`
    multiply with a :class:`~mptorch.SuperFP` accumulate or the reverse, as
    long as each slot holds one format, because the cross-family ops
    (:func:`mptorch.quant.binaryK_superfp_matmul`,
    :func:`mptorch.quant.superfp_binaryK_matmul`) have no palette variant.

    ``rounding`` is one mode for both roundings. The kernels take the rounding
    mode as a compile-time template parameter, because choosing it at run time
    inside the accumulation loop made the kernels 1.2 to 2.3 times slower, and
    a mode per format would square the number of kernels compiled. The
    saturation mode, the subnormals mode and the number of stochastic-rounding
    bits, on the other hand, *are* per format, and are set on the format
    objects.

    Both formats may be :class:`~mptorch.FixedPoint`, one per slot, with an
    accumulate format, under ``NAIVE``: every sum is then exact, the product
    the carrier's (:func:`mptorch.quant.fixedpoint_matmul`), and ``overflow``
    says whether the sum saturates or wraps around.

    Args:
        mul (Number or Sequence[Number] or Palette): the multiply format, or a
            palette of them.
        acc (Number or Sequence[Number] or Palette, optional): the accumulate
            format, or a palette of them; ``None`` leaves the running sum in
            the carrier. Default: ``None``
        rounding (RoundMode): the rounding mode of both roundings.
            Default: ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): how the products are
            summed. ``NAIVE`` is the step above, in order. ``KAHAN`` carries a
            correction term next to the sum, in the ``acc`` format. ``BLOCK``
            applies the step above within each block of ``block_size``
            products and adds each block's sum to a running total, rounding
            the total to the ``outer`` format. ``TREE`` adds each block's
            products in pairs, rounding every pair's sum to ``acc``, and adds
            the block's sum to the total as ``BLOCK`` does.
            :class:`mptorch.AccumulateAlgorithm` states each one precisely.
            ``KAHAN``, ``BLOCK`` and ``TREE`` run in both carriers, but only
            with one format per slot: a palette in any slot raises. Default:
            ``AccumulateAlgorithm.NAIVE``
        carrier (torch.dtype, optional): the carrier that both roundings and
            all the arithmetic between them are computed in, as for the
            argument of the same name of :func:`mptorch.quant.binaryK_matmul`.
            ``None`` uses the operands' carrier (binary64 for float64,
            binary32 for the other dtypes). ``torch.float64`` widens narrower
            operands to binary64 and narrows the result back to their dtype.
            ``torch.float32`` refuses float64 operands. Every call checks the
            formats against its carrier. Default: ``None``
        block_size (int, optional): the number of products per block.
            ``BLOCK`` requires one that divides 16 or is a multiple of 16.
            ``TREE`` takes a power of two in [2, 256] and defaults to 16.
            ``NAIVE`` and ``KAHAN`` take none. Default: ``None``
        outer (Number, optional): the format that ``BLOCK`` and ``TREE``
            round the total to after each block's sum is added. It must be of
            the accumulate format's family (the multiply format's when
            ``acc`` is ``None``). ``None`` leaves the total in the carrier, as
            ``acc=None`` does for the sum. When it is given, the result's
            values are this format's. Default: ``None``
        carrier_rounding (RoundMode): how the carrier rounds the product and
            the sum before the formats do. ``RoundMode.RNE`` is the carrier's
            ordinary multiply and add, rounded to nearest. ``RoundMode.RO``
            rounds both to odd instead, so that each format rounds the exact
            product and the exact sum (see :doc:`/kernels/arithmetic`, "What
            the carrier rounds"). That holds only for formats narrow enough,
            the ``outer`` format included, and a call with a wider format
            raises. In binary32, ``P`` must be at most 22 under ``RNE`` and
            ``RNA`` and at most 23 under the directed modes and ``RO``, and
            ``P + prng_bits`` at most 23 under ``SR`` and 22 under ``SR_B``
            and ``SR_C``. binary64's bounds are 29 higher.
            ``RO`` is implemented on CPU and CUDA tensors in both carriers,
            and on MPS tensors in binary32. It covers palettes under
            ``NAIVE``, and every other mac under every algorithm: under
            ``KAHAN``, ``BLOCK`` and ``TREE`` every carrier sum that a format
            rounds next is also rounded to odd, in a build with
            ``MPTORCH_ODD_ACCUMULATED=1`` (without it they raise). With
            ``acc=None`` only the product is rounded to odd, since no format
            rounds the sum afterwards. A fixed-point mac takes ``RNE`` only.
            Default: ``RoundMode.RNE``
        overflow (Overflow): what a fixed-point accumulate format's sum does
            past its range: ``SATURATE`` or ``WRAP`` (around modulo
            ``2**wl``). Every other mac takes ``SATURATE`` only. Default:
            ``Overflow.SATURATE``

    Raises:
        ValueError: if ``mul`` is ``None``, if both slots are palettes of
            different lengths, if ``mul`` and ``acc`` are of different
            families and either is a palette, if a palette is combined with
            an algorithm other than ``NAIVE``, if ``carrier`` is not a
            carrier, if ``block_size`` or ``outer`` does not fit the
            algorithm, or if ``carrier_rounding`` is neither ``RNE`` nor
            ``RO``; and for a fixed-point mac with no accumulate format, a
            palette, an algorithm other than ``NAIVE``, or ``RO``, and for
            ``Overflow.WRAP`` without a fixed-point accumulate format.
        TypeError: if a slot holds a format no GEMM takes or mixes families
            within a palette, if ``outer`` is not a single format or is of
            another family than the accumulate format, if a fixed-point
            format is beside another family, or if ``carrier`` or
            ``overflow`` is not of its type.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import SplitMac, qmm
        >>> binary8p4 = BinaryK(8, 4)
        >>> mac = SplitMac(binary8p4, BinaryK(16, 11))   # Binary8p4 products, Binary16p11 sums
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), mac).shape
        torch.Size([4, 3])
        >>> carrier_sum = SplitMac(binary8p4, None)   # products rounded, sums left in the carrier
        >>> from mptorch import AccumulateAlgorithm
        >>> blocked = SplitMac(binary8p4, binary8p4, block_size=8, outer=BinaryK(16, 11),
        ...                    accumulate_algorithm=AccumulateAlgorithm.BLOCK)
        >>> qmm(torch.randn(4, 64), torch.randn(64, 3), blocked).shape
        torch.Size([4, 3])
        >>> from mptorch import SuperFP
        >>> cross = SplitMac(binary8p4, SuperFP(3, 4, 8, 7))   # binaryK products, superfp sums
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), cross).shape
        torch.Size([4, 3])
    """

    mul: Number | Sequence[Number] | Palette
    acc: Number | Sequence[Number] | Palette | None = None
    _: KW_ONLY
    rounding: RoundMode = RoundMode.RNE
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE
    carrier: torch.dtype | None = None
    block_size: int | None = None
    outer: Number | None = None
    carrier_rounding: RoundMode = RoundMode.RNE
    overflow: Overflow = Overflow.SATURATE

    def __post_init__(self) -> None:
        _checked_carrier(self.carrier)
        mul, acc = _as_palette(self.mul), _as_palette(self.acc)
        if mul is None:
            raise ValueError("a SplitMac needs a multiply format")
        if _checked_fixedpoint(self, (mul, acc)):
            object.__setattr__(self, "mul", mul)
            object.__setattr__(self, "acc", acc)
            _checked_accumulation(self, (mul, acc), has_product=True)
            return
        if acc is not None:
            if type(acc[0]) is not type(mul[0]) and (_varies(mul) or _varies(acc)):
                raise ValueError(
                    f"a SplitMac whose multiply and accumulate formats are of different families "
                    f"({type(mul[0]).__name__} and {type(acc[0]).__name__}) takes one format per "
                    "slot: the cross-family ops have no palette variant"
                )
            if _varies(mul) and _varies(acc) and len(mul) != len(acc):
                raise ValueError(
                    "a SplitMac's multiply and accumulate palettes must have the same length, "
                    f"got {len(mul)} and {len(acc)}"
                )
        _checked_accumulation(self, (mul, acc), has_product=True)
        _checked_carrier_rounding(self, (mul, acc))
        object.__setattr__(self, "mul", mul)
        object.__setattr__(self, "acc", acc)


@dataclass(frozen=True)
class FusedMac:
    """One fused multiply-add per step, as in hardware, rounded to a format once.

    Each step computes ``s = round_fma(s + a * b)`` with the product exact, so
    each step is rounded to a format once, where :class:`SplitMac` rounds it
    twice. The carrier's FMA first rounds ``s + a * b`` to nearest, unless
    ``carrier_rounding=RoundMode.RO``. ``fma`` is a simulated format, a
    palette, or ``None``. ``None`` leaves each fused step rounded only by the
    carrier's own FMA, in order; with no format for ``prec_idx`` to select, it
    never selects a ``_mixed`` op.

    Args:
        fma (Number or Sequence[Number] or Palette, optional): the format each
            fused step is rounded to, a palette of them, or ``None``.
        rounding (RoundMode): the rounding mode of the fused step.
            Default: ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for :class:`SplitMac`,
            with the fused step in place of the split one. ``TREE`` raises,
            because a fused multiply-add has no product term to sum pairwise.
            Default: ``AccumulateAlgorithm.NAIVE``
        carrier (torch.dtype, optional): as for :class:`SplitMac`.
            Default: ``None``
        block_size (int, optional): as for :class:`SplitMac`. Default: ``None``
        outer (Number, optional): as for :class:`SplitMac`. With ``fma=None``
            it may be of either family. Default: ``None``
        carrier_rounding (RoundMode): as for :class:`SplitMac`, for the one
            carrier rounding of the fused step. ``RoundMode.RO`` needs a
            format to round the step afterwards, so ``fma=None`` raises.
            Default: ``RoundMode.RNE``
        overflow (Overflow): as for :class:`SplitMac`, for a fixed-point
            ``fma``, whose step ``s = round(a * b + s)`` is then exact before
            its one rounding (:func:`mptorch.quant.fixedpoint_matmul_fma`).
            Default: ``Overflow.SATURATE``

    Raises:
        ValueError: if ``carrier`` is not a carrier, a palette is invalid,
            ``block_size`` or ``outer`` does not fit the algorithm, or
            ``carrier_rounding`` does not fit the mac.
        TypeError: if ``carrier`` is not a ``torch.dtype``, or ``outer`` is of
            another family or not a single format.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import FusedMac, qmm
        >>> qmm(torch.randn(4, 8), torch.randn(8, 3), FusedMac(BinaryK(8, 4))).shape
        torch.Size([4, 3])
        >>> carrier_fma = FusedMac(None)   # each step rounded only by the carrier's FMA
    """

    fma: Number | Sequence[Number] | Palette | None
    _: KW_ONLY
    rounding: RoundMode = RoundMode.RNE
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE
    carrier: torch.dtype | None = None
    block_size: int | None = None
    outer: Number | None = None
    carrier_rounding: RoundMode = RoundMode.RNE
    overflow: Overflow = Overflow.SATURATE

    def __post_init__(self) -> None:
        _checked_carrier(self.carrier)
        fma = _as_palette(self.fma)
        _checked_fixedpoint(self, (fma,))
        _checked_accumulation(self, (fma,), has_product=False)
        _checked_carrier_rounding(self, (fma,))
        object.__setattr__(self, "fma", fma)


Mac = SplitMac | FusedMac


# --- resolution to the op ----------------------------------------------------
#
# `spec_for_mac` picks the op from the simulated formats' family, the kind of
# mac, whether a slot holds a palette, whether the two slots' families differ,
# the accumulate algorithm and the carrier rounding. A single-format slot
# opposite a palette is repeated to the palette's length (`_slot_fields`), so
# a SplitMac with a palette on one side and one format on the other needs no
# special case.


def _resolved(slot: object) -> Palette:
    """Return the ``Palette`` that a mac's ``__post_init__`` stored in a slot.

    The declared field types are what a *caller* may pass (a format, a
    sequence or a Palette), but what is stored is always a Palette. This
    function states that for the reader and checks it for the type checker.
    """
    assert isinstance(slot, Palette), f"mac slot was not normalized: {slot!r}"
    return slot


def _slot_fields(pal: Palette, prefix: str, n: int) -> dict[str, Any]:
    """Return one format slot as the op arguments that describe it.

    ``n`` is the call's palette size, the length of the longer slot, since a
    ``SplitMac`` may have a palette in one slot and a single format in the
    other. With ``n > 1`` a single-format slot is repeated to that length
    rather than passed as scalars: the ``_mixed`` schemas take every width as
    a list, read the palette size from the first one (``mul_K`` or
    ``mul_man_bits``), and require every other list to have that length.
    """
    first = pal[0]
    # Every entry has `first`'s type; Palette._validate guarantees it.
    entries = list(pal.formats) if len(pal) > 1 else [first] * n
    widths: dict[str, list]
    if isinstance(first, BinaryK):
        bk = cast(list[BinaryK], entries)
        widths = {
            f"{prefix}_K": [f.K for f in bk],
            f"{prefix}_P": [f.P for f in bk],
            f"{prefix}_bias": [f.bias for f in bk],
        }
    else:
        sfp = cast(list[SuperFP], entries)
        widths = {
            f"{prefix}_man_bits": [f.man_bits for f in sfp],
            f"{prefix}_exp_bits": [f.exp_bits for f in sfp],
            f"{prefix}_normal_binades": [f.normal_binades for f in sfp],
            f"{prefix}_bias": [f.bias for f in sfp],
        }
    fields: dict[str, Any] = widths if n > 1 else {k: v[0] for k, v in widths.items()}
    fields[f"{prefix}_is_signed"] = first.is_signed
    fields[f"{prefix}_prng_bits"] = first.prng_bits
    return fields


def _outer_fields(outer: Number | None) -> dict[str, Any]:
    """Return a mac's outer format as the op arguments that describe it (none for ``None``)."""
    if outer is None:
        return {}
    # `_checked_accumulation` has checked that it is a BinaryK or a SuperFP.
    assert isinstance(outer, GemmFormat), f"mac.outer was not validated: {outer!r}"
    fields: dict[str, Any] = {
        "outer_bias": outer.bias,
        "outer_is_signed": outer.is_signed,
        "outer_prng_bits": outer.prng_bits,
        "outer_saturation_mode": outer.saturation,
    }
    if isinstance(outer, BinaryK):
        return fields | {
            "outer_K": outer.K,
            "outer_P": outer.P,
            "outer_subnormals_mode": outer.subnormals,
        }
    return fields | {
        "outer_man_bits": outer.man_bits,
        "outer_exp_bits": outer.exp_bits,
        "outer_normal_binades": outer.normal_binades,
    }


@lru_cache(maxsize=256)
def spec_for_mac(mac: Mac) -> _GemmSpec:
    """Resolve a mac to the GEMM it names, as a ``_GemmSpec`` (the op and its arguments).

    The result is the same ``_GemmSpec`` that the flat wrapper in ``mptorch.quant.ops`` for
    that op builds from the same numbers. ``tests/test_number_formats.py``
    asserts that the two are equal, which keeps the two API tiers from
    drifting apart.

    The result is cached on the mac's value, which is why the mac classes are
    frozen (and so hashable by value). Without the cache, a
    ``qmatmul(a, b, SplitMac(...))`` written inside a loop would work out
    every default again on every call, at 2.3 to 6.4 microseconds a call. A
    cache miss costs exactly what a call of the flat wrapper costs.
    """
    # A fixed-point mac has its own builders, NAIVE only (`_checked_fixedpoint`).
    fixed = _fixedpoint_mac_spec(mac)
    if fixed is not None:
        return fixed
    fields: dict[str, Any] = {
        "accumulate_algorithm": mac.accumulate_algorithm,
        "rounding_mode": mac.rounding,
        "carrier": mac.carrier,
    }
    # A NAIVE mac goes through the NAIVE builders. Any other algorithm allows
    # only one format per slot (`_checked_accumulation`) and goes through the
    # builder of its family's `*_accumulated` op, which also takes the block
    # size and the outer format.
    naive = mac.accumulate_algorithm is AccumulateAlgorithm.NAIVE
    if not naive:
        fields |= {"block_size": mac.block_size} | _outer_fields(mac.outer)

    if isinstance(mac, FusedMac):
        if mac.fma is None:
            # No step is rounded to a format, so no cast reads the widths given
            # here (placeholders). The op's family is then the outer format's,
            # if there is one; without one the family does not matter, and the
            # binaryK op serves.
            if naive:
                return _binaryK_fma_spec(fma_K=2, fma_P=1, fma_quant=False, **fields)
            if isinstance(mac.outer, SuperFP):
                return _superfp_fma_accumulated_spec(
                    fma_man_bits=1,
                    fma_exp_bits=2,
                    fma_normal_binades=1,
                    fma_bias=1,
                    fma_quant=False,
                    **fields,
                )
            return _binaryK_fma_accumulated_spec(fma_K=2, fma_P=1, fma_quant=False, **fields)
        fma = _resolved(mac.fma)
        fma_0 = fma[0]
        binaryK = isinstance(fma_0, BinaryK)
        fields |= _slot_fields(fma, "fma", len(fma))
        fields["saturation_mode"] = fma_0.saturation
        if isinstance(fma_0, BinaryK):
            fields["subnormals_mode"] = fma_0.subnormals
        odd = mac.carrier_rounding is RoundMode.RO
        if _varies(fma):
            if odd:
                build = _binaryK_fma_mixed_odd_spec if binaryK else _superfp_fma_mixed_odd_spec
            else:
                build = _binaryK_fma_mixed_spec if binaryK else _superfp_fma_mixed_spec
            return build(**fields)
        if odd:
            if naive:
                build = _binaryK_fma_odd_spec if binaryK else _superfp_fma_odd_spec
            else:
                build = (
                    _binaryK_fma_accumulated_odd_spec
                    if binaryK
                    else _superfp_fma_accumulated_odd_spec
                )
            return build(**fields)
        if naive:
            build = _binaryK_fma_spec if binaryK else _superfp_fma_spec
        else:
            build = _binaryK_fma_accumulated_spec if binaryK else _superfp_fma_accumulated_spec
        return build(fma_quant=True, **fields)

    mul = _resolved(mac.mul)
    acc = None if mac.acc is None else _resolved(mac.acc)
    mul_0 = mul[0]
    # One palette size for the call: a SplitMac may hold a palette in its
    # multiply slot, its accumulate slot, or both, and the op takes one length.
    n = max(len(mul), len(acc) if acc is not None else 1)
    fields |= _slot_fields(mul, "mul", n)
    fields["saturation_mode"] = mul_0.saturation
    if isinstance(mul_0, BinaryK):
        fields["subnormals_mode"] = mul_0.subnormals
    fields["accumulate_quant"] = acc is not None
    if acc is not None:
        acc_0 = acc[0]
        fields |= _slot_fields(acc, "acc", n)
        fields["acc_saturation_mode"] = acc_0.saturation
        if isinstance(acc_0, BinaryK):
            fields["acc_subnormals_mode"] = acc_0.subnormals
    binaryK = isinstance(mul_0, BinaryK)
    odd = mac.carrier_rounding is RoundMode.RO
    if acc is not None and type(acc[0]) is not type(mul_0):
        # Cross-family: one format per slot (`SplitMac.__post_init__`), and
        # the sum is always rounded to the accumulate format, so these ops
        # take no accumulate_quant flag.
        del fields["accumulate_quant"]
        if odd and naive:
            build = _binaryK_superfp_odd_spec if binaryK else _superfp_binaryK_odd_spec
        elif odd:
            build = (
                _binaryK_superfp_accumulated_odd_spec
                if binaryK
                else _superfp_binaryK_accumulated_odd_spec
            )
        elif naive:
            build = _binaryK_superfp_spec if binaryK else _superfp_binaryK_spec
        else:
            build = (
                _binaryK_superfp_accumulated_spec if binaryK else _superfp_binaryK_accumulated_spec
            )
        return build(**fields)
    if n > 1:
        if odd:
            build = _binaryK_mixed_odd_spec if binaryK else _superfp_mixed_odd_spec
        else:
            build = _binaryK_mixed_spec if binaryK else _superfp_mixed_spec
        return build(**fields)
    if odd:
        if naive:
            build = _binaryK_odd_spec if binaryK else _superfp_odd_spec
        else:
            build = _binaryK_accumulated_odd_spec if binaryK else _superfp_accumulated_odd_spec
        return build(**fields)
    if naive:
        build = _binaryK_spec if binaryK else _superfp_spec
    else:
        build = _binaryK_accumulated_spec if binaryK else _superfp_accumulated_spec
    return build(**fields)


def _fixedpoint_fields(fmt: FixedPoint, prefix: str) -> dict[str, Any]:
    """A fixed-point slot as the fixed-point builders' arguments."""
    return {
        f"{prefix}_wl": fmt.wl,
        f"{prefix}_fl": fmt.fl,
        f"{prefix}_is_signed": fmt.is_signed,
        f"{prefix}_symmetric": fmt.symmetric,
        f"{prefix}_prng_bits": fmt.prng_bits,
    }


def _fixedpoint_mac_spec(mac: Mac) -> _GemmSpec | None:
    """A fixed-point mac's ``_GemmSpec``, from the flat wrappers' builders, or
    ``None`` for a mac of the other families."""
    slot = mac.fma if isinstance(mac, FusedMac) else mac.mul
    if slot is None:
        return None
    first = _resolved(slot)[0]
    if not isinstance(first, FixedPoint):
        return None
    common = {"rounding_mode": mac.rounding, "overflow": mac.overflow, "carrier": mac.carrier}
    if isinstance(mac, FusedMac):
        return _fixedpoint_fma_spec(**_fixedpoint_fields(first, "fma"), **common)
    acc = _resolved(mac.acc)[0]
    assert isinstance(acc, FixedPoint)  # `_checked_fixedpoint` refused a mix
    return _fixedpoint_spec(
        **_fixedpoint_fields(first, "mul"), **_fixedpoint_fields(acc, "acc"), **common
    )
