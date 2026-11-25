"""Block formats: packing a tensor into one, decoding it, and multiplying two packed tensors.

A block format (:class:`mptorch.BlockFormat`, such as OCP MX or NVFP4) stores
a tensor as narrow element codes in blocks of ``block_size`` consecutive
elements along one axis, the *packed* axis, and each block shares one scale.
This module holds the plain functions that call the six block ops, one
function per op and every op argument spelled out:

* :func:`block_pack` rounds a tensor to the format and stores the result as
  a :class:`BlockPacked` of ``uint8`` codes and scales;
* :func:`block_unpack` decodes a :class:`BlockPacked` back into a tensor;
* :func:`block_quantize` does both at once and returns the decoded values
  without building the packed tensor, and :func:`block_quantize_` writes
  them over its input;
* :func:`block_project` stores a tensor under scales the caller gives, as
  IEEE P3109's BlockProject does, instead of the ones the format's rule
  would choose;
* :func:`block_matmul` multiplies two :class:`BlockPacked` operands. The GEMM
  decodes the codes as it loads them and computes in the dot-product
  arithmetic it is given.

The same operations as reusable values are :class:`mptorch.quant.BlockQuant`
and :class:`mptorch.quant.BlockMac`, and the factories that configure a layer
with them are :func:`mptorch.quant.block_matmul_formats` and
:func:`mptorch.quant.block_gemm_formats`.

The kernels round in the *carrier*, the binary float the arithmetic runs in:
binary64 for a float64 tensor or under ``carrier=torch.float64`` (a narrower
tensor is then widened first, and the result narrowed back with a single
rounding), and binary32 otherwise. A :class:`BlockPacked` records the carrier
it was packed in, and unpacking or multiplying it runs in that carrier or a
wider one. On MPS, which has no float64, binary32 is the only carrier.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    FixedPoint,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
    SuperFP,
)
from mptorch.number.base import _report_per_call
from mptorch.number.binaryk import _binaryK_findings, check_binaryK_storage
from mptorch.number.block import (
    _block_label,
    _block_storage_findings,
    _minifloat,
    check_block_carrier,
    check_block_storage,
)

from ._accumulate import _checked_block_size
from ._checks import (
    _BELOW_BINARY64,
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _check_stored,
    _checked_carrier,
    _Findings,
    _format_findings,
    _narrowed,
    _report_findings,
    _Stored,
)

__all__ = [
    "BlockPacked",
    "block_pack",
    "block_unpack",
    "block_quantize",
    "block_quantize_",
    "block_project",
    "block_matmul",
]


# --- the format as the ops take it -------------------------------------------


@lru_cache(maxsize=256)
def _code_fields(
    f: "BinaryK | SuperFP | FixedPoint",
) -> tuple[int, int, int, int, int, int, int]:
    """Describe one code's family and layout as the descriptor lists them:
    ``(family, signed, exp_bits, man_bits, bias, subnormals, normal_binades)``.

    ``family`` is 0 for a binaryK and 1 for a superfp, and the field that
    only the other family uses is 0 (``normal_binades`` for a binaryK,
    ``subnormals`` for a superfp). A fixed-point element is family 0: it is
    described as the binaryK layout its integer code amounts to
    (`mptorch.number.block._minifloat`), under ``SUBNORMALS``."""
    mf = _minifloat(f)
    if isinstance(f, FixedPoint):
        sub = SubnormalsMode.SUBNORMALS.value
        return 0, int(f.is_signed), mf.exp_bits, mf.man_bits, mf.bias, sub, 0
    if isinstance(f, SuperFP):
        return 1, int(f.is_signed), mf.exp_bits, mf.man_bits, mf.bias, 0, f.normal_binades
    return 0, int(f.is_signed), mf.exp_bits, mf.man_bits, mf.bias, f.subnormals.value, 0


def _block_format_ints(fmt: BlockFormat) -> tuple[int, ...]:
    """Build the format's ``int[] fmt`` descriptor, the integer list the block
    ops take, in the order of ``BlockFmtField`` (``csrc/common/block_decode.h``):
    the element code's fields, the scale code's fields, then the block's size
    and rows. This is the only place that builds it, and ``block_decode.h``
    the only place that reads it."""
    e = fmt.elem
    efam, esg, ee, em, eb, esub, enb = _code_fields(e)
    s = fmt.scale
    if s is None:
        kind, sfields = 0, (0, 0, 0, 0, 0, 0, 0)
    else:
        kind = 1 if isinstance(s, BinaryK) and s.P == 1 else 2
        sfields = _code_fields(s)
    sfam, ssg, se, sm, sb, ssub, snb = sfields
    return (
        fmt.elem_bits,
        efam,
        esg,
        ee,
        em,
        eb,
        esub,
        enb,
        -1 if fmt.nan_code is None else fmt.nan_code,
        -1 if fmt.inf_code is None else fmt.inf_code,
        e.prng_bits,
        kind,
        sfam,
        ssg,
        se,
        sm,
        sb,
        ssub,
        snb,
        0 if fmt.scale_rule is None else fmt.scale_rule.value,
        fmt.block_size,
        fmt.block_rows,
    )


def _float32(v: float) -> float:
    """``v`` rounded to the nearest float32, as a Python float."""
    return torch.tensor(v, dtype=torch.float32).item()


def _amax(x: torch.Tensor) -> torch.Tensor:
    """Return ``max |x|`` as a 0-d tensor of ``x``'s dtype and device.

    It takes one reduction and allocates no ``|x|``-sized temporary, which
    would cost a weight's worth of memory at the peak of a layer's step.
    ``max(-min, max)`` is that value exactly, and a NaN or an infinity in
    ``x`` makes it non-finite, as ``vector_norm(x, inf)`` would. It uses
    ``torch.aminmax`` rather than ``vector_norm``: over 2^24 float32 elements
    on an Apple GPU, ``vector_norm`` took 25 ms and ``aminmax`` 1.5 ms, and
    over as many bfloat16 elements on the CPU, 4.5 ms and 0.3 ms."""
    lo, hi = torch.aminmax(x.detach())
    return torch.maximum(-lo, hi)


def _tensor_scale(
    x: torch.Tensor, fmt: BlockFormat, tensor_scale: float | None, wide: bool = False
) -> float:
    """Return the per-tensor scale a call packs with: 1 for a format without one.

    A format has a per-tensor scale when its block scale has a mantissa (a
    *cast* scale, as in NVFP4). The default is ``max(|x|) / (elem_max *
    top_scale)`` over the finite elements, computed in the carrier's
    arithmetic (float32, or float64 when ``wide``), so that the tensor's
    largest block gets the block scale ``top_scale``. That is the largest
    block scale, except under a Four Over Six rule, where it is the largest
    one that still leaves the block its second candidate (256 for E4M3). A
    tensor with no nonzero finite element gets 1. A scale the caller supplies
    (a static scale) is rounded to the carrier's precision (float32 or
    float64).
    """
    if not fmt.has_tensor_scale:
        if tensor_scale is not None:
            raise ValueError(
                f"{_block_label(fmt)} has no per-tensor scale: tensor_scale belongs to a format "
                "whose block scale has a mantissa (NVFP4)"
            )
        return 1.0
    if tensor_scale is not None:
        ts = float(tensor_scale) if wide else _float32(float(tensor_scale))
        if not ts > 0 or ts == float("inf"):
            raise ValueError(f"tensor_scale must be a positive finite float, got {tensor_scale}")
        return ts
    if x.numel() == 0:
        return 1.0
    assert fmt.elem_max is not None  # resolved by BlockFormat.__post_init__
    if wide:
        amax = _amax(x).cpu().double()
        if not torch.isfinite(amax):
            a = x.detach().abs().double()
            zero = torch.zeros((), dtype=a.dtype, device=a.device)
            amax = torch.where(torch.isfinite(a), a, zero).amax().cpu()
        ts = amax.item() / (fmt.elem_max * fmt.top_scale)
        return ts if ts > 0 else 1.0
    # Leave out the non-finite elements only when there are some, since doing
    # so allocates temporaries the size of x.
    amax = _amax(x).cpu().float()
    if not torch.isfinite(amax):
        a = x.detach().abs().float()
        amax = torch.where(torch.isfinite(a), a, torch.zeros((), device=a.device)).amax().cpu()
    ts = (amax / torch.tensor(fmt.elem_max * fmt.top_scale, dtype=torch.float32)).item()
    return ts if ts > 0 else 1.0


def _prologue(x: torch.Tensor, fmt: BlockFormat, carrier: torch.dtype | None, op: str) -> bool:
    """Run the checks every elementwise call makes first, and return whether
    the call rounds in binary64.

    It checks that ``fmt`` is a :class:`mptorch.BlockFormat` and that the
    carrier the call rounds in can hold it."""
    if not isinstance(fmt, BlockFormat):
        raise TypeError(f"{op} takes a BlockFormat, got {type(fmt).__name__}")
    wide, _ = _call_carrier(_checked_carrier(carrier), x.dtype)
    check_block_carrier(fmt, carrier=_CARRIER[wide])
    return wide


# --- a packed tensor --------------------------------------------------------


def _moved_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    """The logical shape with the packed axis moved last: the order the codes are stored in."""
    return (*shape[:axis], *shape[axis + 1 :], shape[axis])


def _storage_view(x: torch.Tensor, axis: int, op: str, in_place: bool = False) -> torch.Tensor:
    """Return ``x`` with its packed axis last, as the rank-2 or rank-3 tensor
    the ops read.

    A 1D tensor becomes a single row, and a tensor of rank 4 or more has its
    leading dimensions merged into one. The result is a view wherever one
    exists. When it does not, the out-of-place ops copy, and the in-place op
    (``in_place=True``) raises, since a copy is not what it was asked to
    write."""
    moved = x if axis == x.dim() - 1 else x.movedim(axis, -1)
    if moved.dim() == 1:
        return moved.unsqueeze(0)
    if moved.dim() <= 3:
        return moved
    batch = 1
    for d in moved.shape[:-2]:
        batch *= d
    if in_place:
        try:
            return moved.view(batch, *moved.shape[-2:])
        except RuntimeError:
            raise ValueError(
                f"{op}_ writes its argument in place, and folding this tensor's leading "
                "dimensions with its packed axis moved last needs a copy: use "
                f"{op}, which writes a new tensor"
            ) from None
    return moved.reshape(batch, *moved.shape[-2:])


class BlockPacked:
    """A tensor stored in a :class:`mptorch.BlockFormat`, as ``uint8`` codes and scales.

    This is what :func:`block_pack` returns. The *logical* tensor is the one
    that was packed. The fields are:

    - ``data``: the element codes of the logical tensor with its packed axis
      moved last, shaped ``[..., rows, row_bytes]``. The leading dimensions
      are merged into one, and a 1D tensor is stored as a single row.
    - ``scales``: the scale codes, shaped ``[..., row_tiles, n_blocks]``. A
      format without a scale has zero columns here.
    - ``shape``, ``axis`` and ``dtype``: the logical tensor's shape, the axis
      it was packed along, and the dtype it was packed from, which is the
      dtype :meth:`unpack` returns by default.
    - ``tensor_scale``: the per-tensor scale it was packed with (1 for a
      format without one).
    - ``carrier``: the carrier it was packed in, ``torch.float32`` (binary32)
      or ``torch.float64`` (binary64). :meth:`unpack` and
      :func:`block_matmul` decode it in that carrier or a wider one. If it
      is not given, it follows ``dtype``: binary64 for float64, binary32
      otherwise.

    :attr:`mT` is the logical transpose of the last two dimensions. It reads
    the same codes the other way round: nothing is copied, and only ``shape``
    and ``axis`` change. This lets :func:`block_matmul` use one packed weight
    in both a layer's forward pass and its input gradient.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_pack
        >>> w = torch.randn(64, 256)
        >>> p = block_pack(w, MXFP8_E4M3)
        >>> p.data.shape, p.scales.shape, p.nbytes
        (torch.Size([64, 256]), torch.Size([64, 8]), 16896)
        >>> p.mT.shape, p.mT.axis, p.mT.data.data_ptr() == p.data.data_ptr()
        ((256, 64), 0, True)
    """

    __slots__ = ("data", "scales", "fmt", "shape", "axis", "tensor_scale", "dtype", "carrier")

    def __init__(
        self,
        data: torch.Tensor,
        scales: torch.Tensor,
        fmt: BlockFormat,
        shape: tuple[int, ...],
        axis: int,
        tensor_scale: float,
        dtype: torch.dtype,
        carrier: torch.dtype | None = None,
    ):
        self.data = data
        self.scales = scales
        self.fmt = fmt
        self.shape = tuple(shape)
        self.axis = axis
        self.tensor_scale = tensor_scale
        self.dtype = dtype
        if carrier is None:
            carrier = torch.float64 if dtype is torch.float64 else torch.float32
        self.carrier = carrier

    def __repr__(self) -> str:
        return (
            f"BlockPacked(shape={self.shape}, axis={self.axis}, fmt={_block_label(self.fmt)}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    @property
    def device(self) -> torch.device:
        """The device the codes live on."""
        return self.data.device

    @property
    def nbytes(self) -> int:
        """The number of bytes of codes and scales together."""
        return self.data.numel() + self.scales.numel()

    @property
    def cols(self) -> int:
        """The length of the logical packed axis. Each row of ``data`` is
        padded up to a whole number of blocks."""
        return self.shape[self.axis]

    @property
    def mT(self) -> "BlockPacked":
        """The logical transpose of the last two dimensions, without a copy."""
        n = len(self.shape)
        if n < 2 or self.axis < n - 2:
            raise ValueError(
                "mT swaps a packed tensor's last two dimensions, which needs it packed along "
                f"one of them; this one is packed along axis {self.axis} of {n}"
            )
        shape = (*self.shape[:-2], self.shape[-1], self.shape[-2])
        axis = n - 1 if self.axis == n - 2 else n - 2
        return BlockPacked(
            self.data,
            self.scales,
            self.fmt,
            shape,
            axis,
            self.tensor_scale,
            self.dtype,
            self.carrier,
        )

    def to(self, device: torch.device | str) -> "BlockPacked":
        """The same codes on ``device``."""
        return BlockPacked(
            self.data.to(device),
            self.scales.to(device),
            self.fmt,
            self.shape,
            self.axis,
            self.tensor_scale,
            self.dtype,
            self.carrier,
        )

    def unpack(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """:func:`block_unpack` of this tensor."""
        return block_unpack(self, dtype=dtype)


def _format_args(p: BlockPacked, tensor_scale: float | None = None) -> tuple[Any, ...]:
    """Return an operand's format as the op schemas take it: the integer
    descriptor and its three floats (``elem_max``, ``scale_max``, the tensor
    scale), using ``tensor_scale`` in place of the operand's own when given."""
    fmt = p.fmt
    ts = p.tensor_scale if tensor_scale is None else tensor_scale
    return (_block_format_ints(fmt), fmt.elem_max, fmt.scale_max, ts)


# --- the elementwise ops -------------------------------------------------------


def block_pack(
    x: torch.Tensor,
    fmt: BlockFormat,
    axis: int = -1,
    *,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
    carrier: torch.dtype | None = None,
) -> BlockPacked:
    """Round ``x`` to a block format along ``axis`` and store it packed.

    Each run of ``fmt.block_size`` consecutive elements along ``axis`` shares
    one scale. With ``fmt.block_rows > 1`` the block also spans that many rows
    along the axis before ``axis``. The scale is chosen from the block's
    values as :class:`mptorch.BlockFormat` describes. Each element is then
    divided by the scale and rounded to the element format, in the rounding
    mode ``rounding``. The result stores the codes, ``elem_bits`` bits per
    element (a quarter of float32's size for an 8-bit element, an eighth for
    a 4-bit one), plus one byte per scale; :class:`BlockPacked` describes the
    layout. A transposed view, or a tensor packed along any ``axis``, is
    packed without a copy, because the kernel reads ``x`` through its strides
    (a tensor of rank 4 or more may need a copy to merge its leading
    dimensions).

    Under ``RoundMode.SR`` an element's random bits are selected by its index
    in the packed orientation (the tensor with ``axis`` moved last), so
    :func:`block_quantize` with the same generator state draws the same bits.

    Args:
        x (Tensor): float32, float64, float16 or bfloat16, of any rank.
        fmt (BlockFormat): the format.
        axis (int): the axis to pack along. Default: ``-1``
        rounding (RoundMode): the rounding mode for the elements. The scale is
            always chosen by the format's own rule. Default: ``RoundMode.RNE``
        tensor_scale (float, optional): the per-tensor scale of a format whose
            block scale has a mantissa (NVFP4). ``None`` computes
            ``max(|x|) / (elem_max * top_scale)``, which reads ``x`` once
            more and, on a GPU, waits for it to be ready. Default: ``None``
        carrier (torch.dtype, optional): the float type the rounding computes
            in. ``torch.float64`` packs any ``x`` in binary64, widening a
            narrower one first (the default tensor scale is then a float64
            quotient), and ``torch.float32`` names binary32, which is refused
            for a float64 ``x``. Default: ``None``, which follows ``x``:
            binary64 for float64, binary32 otherwise.

    Returns:
        BlockPacked: the codes, the scales and the information needed to
        decode them.

    Raises:
        ValueError: for a format the carrier cannot hold
            (:func:`mptorch.number.check_block_carrier`), ``carrier=torch.float32``
            on a float64 ``x``, a ``tensor_scale`` given to a format without
            one, or a 0-d ``x``.
        RuntimeError: for a tensor that requires grad, under grad mode.

    Example::

        >>> import torch
        >>> from mptorch import MXFP4_E2M1
        >>> from mptorch.quant import block_pack
        >>> p = block_pack(torch.tensor([[0.3, -1.0, 2.5, 6.0] * 8]), MXFP4_E2M1)
        >>> p.data.shape, p.scales.tolist()
        (torch.Size([1, 16]), [[127]])
        >>> p.unpack()[0, :4]
        tensor([ 0.5000, -1.0000,  2.0000,  6.0000])
    """
    wide = _prologue(x, fmt, carrier, "block_pack")
    if x.dim() == 0:
        raise ValueError("block_pack needs a tensor with at least one dimension")
    axis = axis % x.dim()
    src = x.double() if wide and x.dtype is not torch.float64 else x
    ts = _tensor_scale(src, fmt, tensor_scale, wide)
    data, scales = torch.ops.mptorch.block_pack.default(
        _storage_view(src, axis, "block_pack"),
        _block_format_ints(fmt),
        fmt.elem_max,
        fmt.scale_max,
        ts,
        rounding.value,
    )
    return BlockPacked(data, scales, fmt, tuple(x.shape), axis, ts, x.dtype, _CARRIER[wide])


def block_unpack(
    p: BlockPacked, *, dtype: torch.dtype | None = None, carrier: torch.dtype | None = None
) -> torch.Tensor:
    """Decode a :class:`BlockPacked` into a tensor of its logical shape.

    Each element is its code's value times its block's scale (times the
    tensor scale, if any), computed as one product in the carrier and stored
    in ``dtype``, the *storage dtype*. The carrier is the one ``p`` was packed
    in, or binary64 when ``dtype`` is float64 or under
    ``carrier=torch.float64``. When the storage dtype is narrower than the
    carrier, storing rounds the product a second time wherever the storage
    dtype cannot represent it, and :class:`mptorch.FormatRangeWarning` says
    so. That second rounding happens once, even from binary64, since a
    binary64 product is narrowed by ``narrow_float64`` rather than through
    float32. ``block_unpack(block_pack(x, fmt, axis))`` equals
    ``block_quantize(x, fmt, axis)`` element for element, under the same
    generator state.

    Args:
        p (BlockPacked): the packed tensor.
        dtype (torch.dtype, optional): the storage dtype: float32, float64,
            float16 or bfloat16. Default: ``None``, the dtype ``p`` was packed
            from.
        carrier (torch.dtype, optional): ``torch.float64`` decodes in
            binary64. ``torch.float32`` is refused for a ``p`` packed in
            binary64 and for a float64 ``dtype``. Default: ``None``, ``p``'s
            carrier, or binary64 for a float64 ``dtype``.

    Returns:
        Tensor: the decoded values, of shape ``p.shape`` (a view of the
        decoded storage with the packed axis moved back into place).

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_pack, block_unpack
        >>> x = torch.tensor([[1.0, 0.1, -3.3, 100.0]])
        >>> block_unpack(block_pack(x, MXFP8_E4M3, axis=0))  # one element per block
        tensor([[ 1.0000,  0.1016, -3.2500, 96.0000]])
    """
    dtype = p.dtype if dtype is None else dtype
    carrier = _checked_carrier(carrier)
    if carrier is torch.float32 and p.carrier is torch.float64:
        raise ValueError(
            "carrier=torch.float32 is narrower than binary64, which this tensor was packed in "
            "(its tensor scale and its format are binary64's); leave carrier unset"
        )
    wide, _ = _call_carrier(carrier, dtype)
    wide = wide or p.carrier is torch.float64
    check_block_carrier(p.fmt, carrier=_CARRIER[wide])
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_block_storage(p.fmt, storage=dtype)
    cols = p.cols
    out_dtype = torch.float64 if wide else dtype
    args = _format_args(p)
    out = torch.ops.mptorch.block_unpack.default(p.data, p.scales, cols, *args, out_dtype)
    if out_dtype is not dtype:
        out = _narrowed(out, dtype)
    moved = _moved_shape(p.shape, p.axis)
    if tuple(out.shape) != moved:
        out = out.reshape(moved)
    return out if p.axis == len(p.shape) - 1 else out.movedim(-1, p.axis)


def _quantize(
    x: torch.Tensor,
    fmt: BlockFormat,
    axis: int,
    rounding: RoundMode,
    tensor_scale: float | None,
    carrier: torch.dtype | None,
    in_place: bool,
) -> torch.Tensor:
    wide = _prologue(x, fmt, carrier, "block_quantize")
    widen = _quantizable(x, wide, in_place)
    if x.dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_block_storage(fmt, storage=x.dtype)
    ints = _block_format_ints(fmt)
    return _quantize_checked(x, fmt, ints, axis, rounding, tensor_scale, wide, widen, in_place)


def _quantizable(x: torch.Tensor, wide: bool, in_place: bool) -> bool:
    """Check that ``x`` can be quantized as asked, and return whether it must
    be widened to float64 first."""
    op = "block_quantize"
    if x.dim() == 0:
        raise ValueError(f"{op} needs a tensor with at least one dimension")
    widen = wide and x.dtype is not torch.float64
    if in_place and widen:
        raise ValueError(
            f"{op}_ with carrier=torch.float64 on a {x.dtype} tensor would round a float64 "
            f"copy and leave the tensor as it was: use {op}, which returns the result"
        )
    return widen


def _quantize_checked(
    x: torch.Tensor,
    fmt: BlockFormat,
    ints: tuple[int, ...],
    axis: int,
    rounding: RoundMode,
    tensor_scale: float | None,
    wide: bool,
    widen: bool,
    in_place: bool,
) -> torch.Tensor:
    """What :func:`block_quantize` and :func:`block_quantize_` do once the
    format and the tensor have passed their checks. ``ints`` is the format's
    descriptor, :func:`_block_format_ints`."""
    op = "block_quantize"
    axis = axis % x.dim()
    src = x.double() if widen else x
    ts = _tensor_scale(src, fmt, tensor_scale, wide)
    v = _storage_view(src, axis, op, in_place=in_place)
    args = (ints, fmt.elem_max, fmt.scale_max, ts, rounding.value)
    if in_place:
        torch.ops.mptorch.block_quant_.default(v, *args)
        return x
    out = torch.ops.mptorch.block_quant.default(v, *args)
    if widen:
        out = _narrowed(out, x.dtype)
    moved = _moved_shape(tuple(x.shape), axis)
    if tuple(out.shape) != moved:
        out = out.reshape(moved)
    return out if axis == x.dim() - 1 else out.movedim(-1, axis)


def block_quantize(
    x: torch.Tensor,
    fmt: BlockFormat,
    axis: int = -1,
    *,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round ``x`` to a block format along ``axis`` and return the decoded values.

    This does :func:`block_pack` and then :func:`block_unpack` in one op,
    without building the packed tensor. It is the *fake quantizer* (a
    function that returns quantized values in the input's dtype) that a
    ``*_quant`` slot of a layer's formats takes, and
    :class:`mptorch.quant.BlockQuant` is the same operation as a reusable
    value. The arguments are those of :func:`block_pack`, and the result has
    ``x``'s shape and dtype. The function is not
    differentiable: wrap it in :class:`mptorch.quant.Quantizer` to give it a
    straight-through gradient.

    Example::

        >>> import torch
        >>> from mptorch import NVFP4
        >>> from mptorch.quant import block_quantize
        >>> block_quantize(torch.tensor([[0.3, -1.0, 2.5, 6.0] * 4]), NVFP4)[0, :4]
        tensor([ 0.5000, -1.0000,  2.0000,  6.0000])
    """
    return _quantize(x, fmt, axis, rounding, tensor_scale, carrier, in_place=False)


def block_quantize_(
    x: torch.Tensor,
    fmt: BlockFormat,
    axis: int = -1,
    *,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """:func:`block_quantize`, written over ``x`` in place; returns ``x``.

    Use it for a tensor the caller owns, such as a weight quantized once when
    it is loaded. Never use it in a ``*_quant`` slot, because a layer's input
    belongs to the autograd graph of whatever produced it. ``x`` must be
    dense: its elements may not overlap or leave gaps in memory (a permuted
    contiguous tensor is fine, a strided slice such as ``x[:, ::2]`` is not).
    A tensor of rank 4 or more must also be able to merge its leading
    dimensions as a view once ``axis`` is moved last. The out-of-place op has
    neither restriction. The result equals that of :func:`block_quantize` bit
    for bit.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_quantize_
        >>> w = torch.tensor([[1.0, 0.1, -3.3, 100.0]])
        >>> block_quantize_(w, MXFP8_E4M3) is w
        True
    """
    return _quantize(x, fmt, axis, rounding, tensor_scale, carrier, in_place=True)


def block_project(
    x: torch.Tensor,
    fmt: BlockFormat,
    scales: torch.Tensor,
    axis: int = -1,
    *,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
    carrier: torch.dtype | None = None,
) -> BlockPacked:
    """Store ``x`` in a block format under the scale codes ``scales``: IEEE P3109's BlockProject.

    :func:`block_pack` chooses each block's scale by the format's rule;
    this function takes them as given, one code per block (or tile) in the
    layout of :attr:`BlockPacked.scales`. With ``S`` the value a block's
    code decodes to (times the tensor scale, for a cast scale), each element
    ``X`` of the block becomes ``Z``, the first case that applies:

    - NaN, if ``S`` or ``X`` is NaN;
    - 0, if ``S`` is 0 (an infinite ``X`` included);
    - ``X / S`` otherwise,

    and ``Z`` is rounded once, as :func:`block_pack` rounds an element: in
    the rounding mode ``rounding``, saturating to the element format's
    largest value. A power-of-two scale has no zero, and no scale format
    here has an infinity, which P3109's fourth case is for. Projecting onto
    the scales :func:`block_pack` chose returns its codes word for word,
    stochastic rounding included under the same generator state, except in a
    block that :func:`block_pack` marked with the scale format's NaN because
    it held a NaN its element format has no code for. There this function
    writes code 0 for every element, where :func:`block_pack` keeps their
    codes, and both decode to NaN.

    A NaN element whose block's scale is not NaN needs the element format's
    NaN code, which some formats lack (E2M1, E2M3 and E3M2, so MXFP4,
    MXFP6 and NVFP4). For those, the kernel reports such an element, and
    this function raises. Finding out waits for the kernel on a GPU, so a
    call in such a format synchronizes with the device. A format without a
    scale stores a NaN as its largest value, as :func:`block_pack` does.

    The result keeps a reference to ``scales`` as its own, and records the
    tensor scale.

    Args:
        x (Tensor): float32, float64, float16 or bfloat16, of any rank.
        fmt (BlockFormat): the format.
        scales (Tensor): the scale codes, ``uint8`` on ``x``'s device,
            shaped as ``block_pack(x, fmt, axis).scales`` is. A format
            without a scale takes a tensor with zero columns there.
        axis (int): the axis to pack along. Default: ``-1``
        rounding (RoundMode): the rounding mode for the elements. Default:
            ``RoundMode.RNE``
        tensor_scale (float, optional): the per-tensor scale of a format whose
            block scale has a mantissa (NVFP4), which the scale codes are
            decoded with. ``None`` computes :func:`block_pack`'s default from
            ``x``. Default: ``None``
        carrier (torch.dtype, optional): as for :func:`block_pack`.
            Default: ``None``

    Returns:
        BlockPacked: the codes, with ``scales`` as their scales.

    Raises:
        ValueError: as :func:`block_pack` does.
        RuntimeError: for scale codes of the wrong dtype, device or shape,
            for a NaN element the format cannot store, as described above, and
            for a tensor that requires grad, under grad mode.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_pack, block_project
        >>> x = torch.tensor([[1.0, -2.0, 3.0, 0.5] * 8])
        >>> p = block_project(x, MXFP8_E4M3, torch.tensor([[128]], dtype=torch.uint8))
        >>> p.unpack()[0, :4]  # code 128 is 2^(128 - 127): each element is stored as x / 2
        tensor([ 1.0000, -2.0000,  3.0000,  0.5000])
        >>> block_pack(x, MXFP8_E4M3).scales  # the format's rule would choose 2^-7
        tensor([[120]], dtype=torch.uint8)
    """
    wide = _prologue(x, fmt, carrier, "block_project")
    if x.dim() == 0:
        raise ValueError("block_project needs a tensor with at least one dimension")
    if not isinstance(scales, torch.Tensor):
        raise TypeError(
            f"block_project takes its scale codes as a uint8 tensor, got {type(scales).__name__}"
        )
    axis = axis % x.dim()
    src = x.double() if wide and x.dtype is not torch.float64 else x
    ts = _tensor_scale(src, fmt, tensor_scale, wide)
    data = torch.ops.mptorch.block_project.default(
        _storage_view(src, axis, "block_project"),
        scales,
        _block_format_ints(fmt),
        fmt.elem_max,
        fmt.scale_max,
        ts,
        rounding.value,
    )
    return BlockPacked(data, scales, fmt, tuple(x.shape), axis, ts, x.dtype, _CARRIER[wide])


class _BlockQuantSpec(NamedTuple):
    """One block quantizer call along the last axis, resolved once, for
    :class:`mptorch.quant.Quant`.

    It is the block counterpart of ``_QuantSpec`` (``ops/_spec.py``): the
    format's descriptor, binary32's refusal of the format (the message
    :func:`mptorch.number.check_block_carrier` raises, or ``None``), and the
    findings on storing a result in each dtype narrower than each carrier,
    keyed by that dtype and only where there is something to say.
    """

    fmt: BlockFormat
    ints: tuple[int, ...]
    rounding: RoundMode
    carrier: torch.dtype | None
    in_place: bool
    binary32_error: str | None
    stored: tuple[dict[torch.dtype, tuple[str | None, str | None]], ...]


def _block_quant_spec(
    fmt: BlockFormat, rounding: RoundMode, carrier: torch.dtype | None, in_place: bool
) -> _BlockQuantSpec:
    """Resolve a :func:`block_quantize` call along the last axis once."""
    try:
        check_block_carrier(fmt, carrier=torch.float32)
        binary32_error = None
    except ValueError as e:
        binary32_error = str(e)
    stored = tuple(
        {
            dtype: pair
            for dtype in narrower
            if (pair := _block_storage_findings(fmt, dtype, True)) != (None, None)
        }
        for narrower in (_NARROW_STORAGE, _BELOW_BINARY64)
    )
    ints = _block_format_ints(fmt)
    return _BlockQuantSpec(fmt, ints, rounding, carrier, in_place, binary32_error, stored)


def _run_block_quant(spec: _BlockQuantSpec, x: torch.Tensor) -> torch.Tensor:
    """:func:`block_quantize` (or :func:`block_quantize_`) of ``x`` along its
    last axis, with the format's checks resolved once; the checks run in the
    flat function's order and raise or warn as it does."""
    dtype = x.dtype
    if spec.carrier is None:
        wide = dtype is torch.float64
    else:
        wide, _ = _call_carrier(spec.carrier, dtype)
    if not wide and spec.binary32_error is not None:
        raise ValueError(spec.binary32_error)
    widen = _quantizable(x, wide, spec.in_place)
    stored = spec.stored[wide].get(dtype)
    if stored is not None:
        _report_per_call(*stored)
    return _quantize_checked(
        x, spec.fmt, spec.ints, -1, spec.rounding, None, wide, widen, spec.in_place
    )


# --- the GEMM -------------------------------------------------------------------


class _BlockGemmSpec(NamedTuple):
    """The block GEMM's arithmetic, converted into exactly the arguments the op takes.

    ``args`` holds every schema argument that follows the two operands'
    formats, in order: the multiply-accumulate step (fused or split, and the
    accumulate format), then the summation (algorithm, rounding mode, and the
    block size and outer format of ``BLOCK`` and ``TREE``). The carrier, the
    last argument, is not included, since it depends on the operands of each
    call. ``findings`` holds what binary32 and binary64 each report about the
    formats the GEMM rounds to, and ``stored`` the format whose values the
    result holds, for the check made when the result is stored in a dtype
    narrower than its carrier. The operands' own formats travel with the
    operands. ``epilogue`` applies the operands' tensor scales to the result
    instead of while decoding.
    """

    args: tuple[Any, ...]
    findings: tuple[_Findings, _Findings] = ((), ())
    stored: tuple[_Stored, ...] = ()
    carrier: torch.dtype | None = None
    epilogue: bool = False


def _binaryK_fields(f: BinaryK | None) -> tuple[Any, ...]:
    """A binaryK format as the op's fields, (quant, K, P, bias, is_signed, saturation,
    subnormals, prng_bits), with ``quant`` False and placeholders for ``None``."""
    if f is None:
        return (False, 0, 0, 0, True, 0, 0, 0)
    return (
        True,
        f.K,
        f.P,
        f.bias,
        f.is_signed,
        f.saturation.value,
        f.subnormals.value,
        f.prng_bits,
    )


@lru_cache(maxsize=256)
def _block_gemm_spec(
    acc: BinaryK | None,
    fused: bool,
    rounding: RoundMode,
    accumulate_algorithm: AccumulateAlgorithm,
    block_size: int | None,
    outer: BinaryK | None,
    carrier: torch.dtype | None,
    tensor_scale_epilogue: bool = False,
) -> _BlockGemmSpec:
    """Convert a block GEMM's arithmetic into a `_BlockGemmSpec`, cached on the arguments."""
    carrier = _checked_carrier(carrier)
    for name, f in (("acc", acc), ("outer", outer)):
        if f is not None and not isinstance(f, BinaryK):
            raise TypeError(
                f"the block GEMM's {name} format is a BinaryK or None, got {type(f).__name__}"
            )
    block = _checked_block_size(accumulate_algorithm, block_size, outer is not None, not fused)
    rounded = []
    for f in (acc, outer):
        if f is not None:
            fmt = (f.K, f.P, f.bias, f.is_signed, f.saturation, f.subnormals)
            rounded.append((_binaryK_findings, (*fmt, f.prng_bits)))
    findings = _format_findings(rounded)
    last = (
        outer
        if accumulate_algorithm in (AccumulateAlgorithm.BLOCK, AccumulateAlgorithm.TREE)
        else acc
    )
    stored: tuple[_Stored, ...] = ()
    if last is not None:
        stored = (
            (
                check_binaryK_storage,
                (last.K, last.P, last.bias, last.is_signed, last.saturation, last.subnormals),
            ),
        )
    a = _binaryK_fields(acc)
    o = _binaryK_fields(outer)
    args = (
        fused,
        a[0],
        a[1],
        a[2],
        a[3],
        a[4],
        accumulate_algorithm.value,
        rounding.value,
        a[5] if acc is not None else SaturationMode.SAT_NONE.value,
        a[6] if acc is not None else SubnormalsMode.SUBNORMALS.value,
        a[7],
        block,
        o[0],
        o[1],
        o[2],
        o[3],
        o[4],
        o[5] if outer is not None else SaturationMode.SAT_NONE.value,
        o[6] if outer is not None else SubnormalsMode.SUBNORMALS.value,
        o[7],
    )
    return _BlockGemmSpec(args, findings, stored, carrier, bool(tensor_scale_epilogue))


@dataclass(frozen=True)
class _Operand:
    """One packed operand as the op reads it."""

    data: torch.Tensor
    scales: torch.Tensor
    cols: int
    trans: bool
    lead: tuple[int, ...]


def _operand(p: BlockPacked, role: str) -> _Operand:
    """Return ``p`` as the GEMM's operand ``role``: ``"a"``, logically
    ``[..., M, K]``, or ``"b"``, logically ``[..., K, N]``.

    The kernel reads each operand along its K. An operand packed along K is
    read as stored, and one packed along its other trailing dimension is read
    transposed; any other packed axis raises."""
    if not isinstance(p, BlockPacked):
        raise TypeError(
            f"block_matmul's {role} is a BlockPacked (block_pack's result), got {type(p).__name__}"
        )
    n = len(p.shape)
    if n < 2:
        raise ValueError(f"block_matmul's operands have at least two dimensions; {role} has {n}")
    k_axis = n - 1 if role == "a" else n - 2
    if p.axis == k_axis:
        trans = False
    elif p.axis in (n - 1, n - 2):
        trans = True
    else:
        raise ValueError(
            f"block_matmul reads {role} packed along one of its last two dimensions, got axis "
            f"{p.axis} of {n}"
        )
    return _Operand(p.data, p.scales, p.cols, trans, p.shape[:-2])


def _gemm_carrier(spec: _BlockGemmSpec, a: BlockPacked, b: BlockPacked) -> bool:
    """Return whether a block GEMM runs in binary64.

    It does when the spec asks for binary64, or when either operand was
    packed in binary64, since a carrier is never narrower than the data it
    reads. ``carrier=torch.float32`` with a binary64 operand raises."""
    packed64 = a.carrier is torch.float64 or b.carrier is torch.float64
    if spec.carrier is torch.float32 and packed64:
        raise ValueError(
            "carrier=torch.float32 is narrower than binary64, which an operand was packed in; "
            "leave carrier unset"
        )
    return spec.carrier is torch.float64 or packed64


def _run_block_gemm(spec: _BlockGemmSpec, a: BlockPacked, b: BlockPacked) -> torch.Tensor:
    """Run the block GEMM on two packed operands: ``a @ b`` of their logical
    tensors, computed in the carrier (`_gemm_carrier`), as float32 or float64.

    Leading dimensions are broadcast when they are equal or when one operand
    has none. With ``spec.epilogue`` the kernel decodes each element with its
    block's scale alone (a tensor scale of 1), and the result is then
    multiplied once by ``alpha``, the product of the two tensor scales,
    rounded to the carrier. That makes two roundings in all. In binary32 the
    product of two float32 values is exact as a Python float, so rounding it
    to float32 is the first rounding; in binary64 Python's own multiply
    rounds it once. ``mul_`` by that carrier value is the second."""
    wide = _gemm_carrier(spec, a, b)
    found = spec.findings[wide]
    if found:
        _report_findings(found)
    if not wide:
        check_block_carrier(a.fmt, carrier=torch.float32)
        check_block_carrier(b.fmt, carrier=torch.float32)
    oa, ob = _operand(a, "a"), _operand(b, "b")
    if oa.lead and ob.lead and oa.lead != ob.lead:
        raise ValueError(
            f"block_matmul broadcasts leading dimensions only when they are equal or one "
            f"operand has none, got {oa.lead} and {ob.lead}"
        )
    unit = 1.0 if spec.epilogue else None
    out = torch.ops.mptorch.custom_matmul_block.default(
        oa.data,
        oa.scales,
        oa.cols,
        oa.trans,
        ob.data,
        ob.scales,
        ob.cols,
        ob.trans,
        *_format_args(a, unit),
        *_format_args(b, unit),
        *spec.args,
        wide,
    )
    if spec.epilogue:
        alpha = a.tensor_scale * b.tensor_scale
        if not wide:
            alpha = _float32(alpha)
        if alpha != 1.0:
            out.mul_(alpha)
    lead = oa.lead or ob.lead
    if len(lead) > 1:
        out = out.reshape(*lead, *out.shape[-2:])
    return out


def _stored_as(spec: _BlockGemmSpec, out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a GEMM result (float32 or float64, as its carrier) to ``dtype``,
    the storage dtype of the operands it was computed from.

    When the storage dtype is narrower than the carrier, the format the result
    holds is first checked against it (the storage check of
    :func:`mptorch.number.check_binaryK_storage`). A binary64 result is
    narrowed with a single rounding (``_checks._narrowed``)."""
    if dtype is out.dtype:
        return out
    wide = out.dtype is torch.float64
    if spec.stored and dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        _check_stored(spec.stored, dtype)
    return _narrowed(out, dtype) if wide else out.to(dtype)


def block_matmul(
    a: BlockPacked,
    b: BlockPacked,
    *,
    acc: BinaryK | None = None,
    fused: bool = False,
    rounding: RoundMode = RoundMode.RNE,
    accumulate_algorithm: AccumulateAlgorithm = AccumulateAlgorithm.NAIVE,
    block_size: int | None = None,
    outer: BinaryK | None = None,
    carrier: torch.dtype | None = None,
    tensor_scale_epilogue: bool = False,
) -> torch.Tensor:
    """Multiply two block-format tensors: ``a @ b`` of their logical values.

    ``a`` is logically ``[..., M, K]`` and ``b`` is ``[..., K, N]``. The
    kernel decodes each element as it loads it (its value times its block's
    scale, as :func:`block_unpack` would) and computes in the carrier:
    binary64 under ``carrier=torch.float64`` or when either operand was
    packed in binary64 (from a float64 tensor, or with that carrier), and
    binary32 otherwise. With a split multiply-accumulate (``fused=False``),
    each product is rounded to the carrier and the running sum to ``acc``.
    With a fused one, each fused multiply-add is rounded to ``acc``.
    ``acc=None`` leaves the sum in the carrier.

    The kernel reads each operand along K. An operand may be packed along its
    K (``a`` along its last axis, ``b`` along its second-to-last) or along its
    other trailing dimension, which the kernel then reads transposed. One
    packed weight therefore serves both ``x @ w.T`` and ``g @ w``, and
    :attr:`BlockPacked.mT` gives the transpose without a copy. The two
    operands may have different formats. Their leading dimensions must be
    equal, or absent from one of them. The function is not differentiable:
    for a gradient, use :func:`mptorch.quant.qmatmul` with a
    :class:`mptorch.quant.BlockMac`.

    A format with a per-tensor scale (NVFP4's E4M3 scale, or a superfp scale)
    decodes each element by default as its value times its block's scale times
    the tensor scale, inside the dot-product loop.
    ``tensor_scale_epilogue=True`` does what NVIDIA's NVFP4 GEMMs do instead:
    the loop sees only the elements times their block scales, and the result
    is multiplied once, at the end, by the product of the two tensor scales,
    rounded to the carrier. The two agree in exact arithmetic but round
    differently. With the epilogue, the accumulate format sees products that
    the tensor scales have not yet scaled, so its range must cover them.

    Args:
        a (BlockPacked): the left operand.
        b (BlockPacked): the right operand.
        acc (BinaryK, optional): the accumulate format. Default: ``None``
        fused (bool): round once per step (a fused multiply-add) instead of
            twice. Default: ``False``
        rounding (RoundMode): the rounding mode for ``acc`` and ``outer``.
            Default: ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :class:`mptorch.quant.SplitMac`. Default:
            ``AccumulateAlgorithm.NAIVE``
        block_size (int, optional): the block size of ``BLOCK`` and ``TREE``.
            Default: ``None``
        outer (BinaryK, optional): the outer format of ``BLOCK`` and
            ``TREE``, which sums the block sums. Default: ``None``
        carrier (torch.dtype, optional): ``torch.float64`` multiplies in
            binary64. ``torch.float32`` is refused when an operand was packed
            in binary64. Default: ``None``, the operands' carrier.
        tensor_scale_epilogue (bool): apply the operands' tensor scales to
            the result instead of while decoding. It has no effect on formats
            without a tensor scale (those with an E8M0 scale). Default:
            ``False``

    Returns:
        Tensor: ``[..., M, N]``, float32 in binary32 and float64 in binary64.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, MXFP8_E4M3
        >>> from mptorch.quant import block_matmul, block_pack
        >>> x, w = torch.randn(4, 64), torch.randn(16, 64)
        >>> y = block_matmul(block_pack(x, MXFP8_E4M3), block_pack(w, MXFP8_E4M3).mT,
        ...                  acc=BinaryK(16, 11))
        >>> y.shape
        torch.Size([4, 16])
    """
    spec = _block_gemm_spec(
        acc,
        fused,
        rounding,
        accumulate_algorithm,
        block_size,
        outer,
        carrier,
        tensor_scale_epilogue,
    )
    return _run_block_gemm(spec, a, b)
