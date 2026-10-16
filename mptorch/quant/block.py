"""Block formats: packing a tensor into one, reading it back, and multiplying two.

A block format (:class:`mptorch.BlockFormat`: OCP MX, NVFP4) stores a tensor
as narrow element codes that share a scale per block of ``block_size``
consecutive elements along one axis, the *packed* axis. This module is the
schema tier of the four elementwise ops and the GEMM over them:

* :func:`block_pack` rounds a tensor to the format and stores it, as a
  :class:`BlockPacked` of ``uint8`` codes and scales;
* :func:`block_unpack` decodes a :class:`BlockPacked`;
* :func:`block_quantize` and :func:`block_quantize_` are the two fused, the
  decoded values of the rounding with no packed intermediate (the second in
  place);
* :func:`block_matmul` multiplies two :class:`BlockPacked` operands, decoding
  them in the GEMM's tile loads, in the dot-product arithmetic it is given.

The value tier is :class:`mptorch.quant.BlockQuant` and
:class:`mptorch.quant.BlockMac`; the layer factories are
:func:`mptorch.quant.block_matmul_formats` and
:func:`mptorch.quant.block_gemm_formats`.

The kernels round in the carrier: binary64 for a float64 tensor or
``carrier=torch.float64`` (a narrower tensor is widened first and the result
narrowed back with one rounding), and binary32 otherwise. A
:class:`BlockPacked` remembers the carrier it was packed in, and unpacking it
or multiplying it runs in that carrier or a wider one. On MPS, which has no
float64, the binary32 carrier is the only one.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from mptorch.number import (
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
    SuperFP,
    _binaryK_findings,
    _block_label,
    _minifloat,
    check_binaryK_storage,
    check_block_carrier,
    check_block_storage,
)

from .ops import (
    _BELOW_BINARY64,
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _check_stored,
    _checked_block_size,
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
    "block_matmul",
]


# --- the format as the ops take it -------------------------------------------


@lru_cache(maxsize=256)
def _code_fields(f: "BinaryK | SuperFP") -> tuple[int, int, int, int, int, int, int]:
    """One code's family and layout, as the descriptor spells it: (family,
    signed, exp_bits, man_bits, bias, subnormals, normal_binades), family 0 a
    binaryK and 1 a superfp, each leaving the other's field 0."""
    mf = _minifloat(f)
    if isinstance(f, SuperFP):
        return 1, int(f.is_signed), mf.exp_bits, mf.man_bits, mf.bias, 0, f.normal_binades
    return 0, int(f.is_signed), mf.exp_bits, mf.man_bits, mf.bias, f.subnormals.value, 0


def _block_format_ints(fmt: BlockFormat) -> tuple[int, ...]:
    """The format's ``int[] fmt`` descriptor, in ``BlockFmtField``'s order
    (``csrc/common/block_decode.h``): the element code, the scale code, the
    block's size and rows. One builder here, one reader there."""
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
    """``max |x|`` as a 0-d tensor of ``x``'s dtype and device: one reduction,
    with no ``|x|``-sized temporary (a weight's worth of memory at the peak of
    a layer's step). ``max(-min, max)`` is that value exactly, and a NaN or an
    infinity in ``x`` makes it non-finite, as ``vector_norm(x, inf)`` does.
    ``torch.aminmax`` rather than ``vector_norm``, which took 25 ms against
    1.5 over 2^24 float32 elements on an Apple GPU, and 4.5 ms against 0.3 over
    bfloat16 ones on the CPU."""
    lo, hi = torch.aminmax(x.detach())
    return torch.maximum(-lo, hi)


def _tensor_scale(
    x: torch.Tensor, fmt: BlockFormat, tensor_scale: float | None, wide: bool = False
) -> float:
    """The per-tensor scale a call packs with: 1 for a format without one.

    NVFP4's default is ``max(|x|) / (elem_max * top_scale)`` over the finite
    elements, in the carrier's arithmetic (float32, or float64 when ``wide``),
    so that the tensor's largest block takes the scale ``top_scale``: the
    largest one, or under a Four Over Six rule the largest that leaves the
    block its second candidate (256 for E4M3). A tensor with no nonzero
    finite element takes 1. A caller-supplied (static) scale is rounded to
    the carrier's precision (float32 or float64).
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
    # The non-finite elements are left out only when there are some, which
    # costs the temporaries.
    amax = _amax(x).cpu().float()
    if not torch.isfinite(amax):
        a = x.detach().abs().float()
        amax = torch.where(torch.isfinite(a), a, torch.zeros((), device=a.device)).amax().cpu()
    ts = (amax / torch.tensor(fmt.elem_max * fmt.top_scale, dtype=torch.float32)).item()
    return ts if ts > 0 else 1.0


def _prologue(x: torch.Tensor, fmt: BlockFormat, carrier: torch.dtype | None, op: str) -> bool:
    """What every elementwise call checks first: the format, and the carrier the
    call rounds in, which the format must be one of; returns whether that is
    binary64."""
    if not isinstance(fmt, BlockFormat):
        raise TypeError(f"{op} takes a BlockFormat, got {type(fmt).__name__}")
    wide, _ = _call_carrier(_checked_carrier(carrier), x.dtype)
    check_block_carrier(fmt, carrier=_CARRIER[wide])
    return wide


# --- a packed tensor --------------------------------------------------------


def _moved_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    """The logical shape with the packed axis moved last: the storage order."""
    return (*shape[:axis], *shape[axis + 1 :], shape[axis])


def _storage_view(x: torch.Tensor, axis: int, op: str, in_place: bool = False) -> torch.Tensor:
    """``x`` with its packed axis last, as the rank-2 or rank-3 tensor the ops
    read: a view wherever one exists (a 1D tensor is one row, and the leading
    dimensions of a rank-4-or-more tensor are folded into one)."""
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
    """A tensor stored in a :class:`mptorch.BlockFormat`: ``uint8`` codes and scales.

    This is what :func:`block_pack` returns. Its fields are:

    - ``data``: the element codes of the logical tensor with its packed axis
      moved last, ``[..., rows, row_bytes]``. The leading dimensions are
      folded into one, and a 1D tensor is stored as one row.
    - ``scales``: the scale codes, ``[..., row_tiles, n_blocks]``. A format
      without a scale has no columns.
    - ``shape``, ``axis`` and ``dtype``: the logical tensor's shape, the axis
      it was packed along, and the dtype it was packed from, which
      :meth:`unpack` returns.
    - ``tensor_scale``: the per-tensor scale it was packed with (1 for a
      format without one).
    - ``carrier``: the carrier it was packed in, ``torch.float32`` (binary32)
      or ``torch.float64`` (binary64). :meth:`unpack` and
      :func:`block_matmul` decode it in that carrier or a wider one. It
      defaults to ``dtype``'s: binary64 for float64, binary32 otherwise.

    :attr:`mT` is the logical transpose of the last two dimensions. It reads
    the same codes the other way: nothing is copied, and only ``shape`` and
    ``axis`` change. This lets :func:`block_matmul` read one packed weight in
    both a layer's forward pass and its input gradient.

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
        """Bytes of codes and scales together."""
        return self.data.numel() + self.scales.numel()

    @property
    def cols(self) -> int:
        """The length of the packed axis. The rows of ``data`` round it up to whole
        blocks."""
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
    """An operand's format as the schemas take it: the descriptor and its floats,
    with ``tensor_scale`` in place of the operand's own when given."""
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
    one scale. For a 2D format the run also spans ``fmt.block_rows`` rows
    along the axis before it. The scale is chosen from the block's values as
    :class:`mptorch.BlockFormat` describes. Each element is then divided by
    the scale and rounded to the element format in ``rounding``. The result
    stores the codes, which take a quarter to an eighth of float32's bytes;
    :class:`BlockPacked` describes the layout. A transposed view is packed
    without a copy, and so is any ``axis``, because the kernel reads ``x``
    through its strides.

    Under ``RoundMode.SR`` an element's random bits are keyed on its index in
    the packed orientation (the tensor with ``axis`` moved last).
    :func:`block_quantize` with the same generator state therefore draws the
    same bits.

    Args:
        x (Tensor): float32, float64, float16 or bfloat16, of any rank.
        fmt (BlockFormat): the format.
        axis (int): the axis to pack along. Default: ``-1``
        rounding (RoundMode): the elements' rounding mode. The scale is
            always chosen by the format's rule. Default: ``RoundMode.RNE``
        tensor_scale (float, optional): the per-tensor scale of a format whose
            block scale has a mantissa (NVFP4). ``None`` takes
            ``max(|x|) / (elem_max * top_scale)``, which reads ``x`` once
            more and, on a GPU, synchronizes with it. Default: ``None``
        carrier (torch.dtype, optional): the arithmetic the rounding runs in:
            ``torch.float64`` packs any ``x`` in binary64, widening a narrower
            one first (the default tensor scale is then a float64 quotient),
            and ``torch.float32`` names binary32, which a float64 ``x``
            refuses. Default: ``None``, ``x``'s own: binary64 for float64,
            binary32 otherwise.

    Returns:
        BlockPacked: the codes, the scales and what describes them.

    Raises:
        ValueError: for a format the carrier does not hold
            (:func:`mptorch.number.check_block_carrier`), ``carrier=torch.float32``
            on a float64 ``x``, or a ``tensor_scale`` given to a format
            without one.
        RuntimeError: on a tensor that requires grad under grad mode.

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
    tensor scale), one product in the carrier, stored in ``dtype``. The
    carrier is the one ``p`` was packed in, or binary64 when ``dtype`` is
    float64 or ``carrier=torch.float64``. A result narrower than its carrier
    rounds that product a second time where the dtype does not hold it (once:
    a binary64 product reaches a narrower dtype through
    ``narrow_float64``), and :class:`mptorch.FormatRangeWarning` reports it.
    ``block_unpack(block_pack(x, fmt, axis))`` equals
    ``block_quantize(x, fmt, axis)`` element for element, under the same
    generator state.

    Args:
        p (BlockPacked): the packed tensor.
        dtype (torch.dtype, optional): float32, float64, float16 or bfloat16.
            Default: ``None``, the dtype ``p`` was packed from.
        carrier (torch.dtype, optional): ``torch.float64`` decodes in
            binary64; ``torch.float32`` refuses a ``p`` packed in binary64 and
            a float64 ``dtype``. Default: ``None``, ``p``'s carrier, or
            binary64 for a float64 ``dtype``.

    Returns:
        Tensor: the decoded values, of shape ``p.shape`` (a view with the
        packed axis moved back into place).

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_pack, block_unpack
        >>> x = torch.tensor([[1.0, 0.1, -3.3, 100.0]])
        >>> block_unpack(block_pack(x, MXFP8_E4M3, axis=0))  # a block per element
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
    op = "block_quantize"
    wide = _prologue(x, fmt, carrier, op)
    if x.dim() == 0:
        raise ValueError(f"{op} needs a tensor with at least one dimension")
    widen = wide and x.dtype is not torch.float64
    if in_place and widen:
        raise ValueError(
            f"{op}_ with carrier=torch.float64 on a {x.dtype} tensor would round a float64 "
            f"copy and leave the tensor as it was: use {op}, which returns the result"
        )
    if x.dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_block_storage(fmt, storage=x.dtype)
    axis = axis % x.dim()
    src = x.double() if widen else x
    ts = _tensor_scale(src, fmt, tensor_scale, wide)
    v = _storage_view(src, axis, op, in_place=in_place)
    args = (_block_format_ints(fmt), fmt.elem_max, fmt.scale_max, ts, rounding.value)
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

    This is :func:`block_pack` and :func:`block_unpack` fused, with no packed
    intermediate. It is the fake quantizer that a ``*_quant`` slot of a
    layer's formats takes, and :class:`mptorch.quant.BlockQuant` is its value
    spelling. The arguments are those of :func:`block_pack`, and the result
    has ``x``'s shape, dtype and layout. The function is not differentiable:
    wrap it in :class:`mptorch.quant.Quantizer` for a straight-through
    gradient.

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

    Use it for a tensor the caller owns, such as a weight quantized once at
    load. Never use it in a ``*_quant`` slot, because a layer's input belongs
    to the graph of whatever produced it. ``x`` must have no overlapping
    elements. A tensor of rank 4 or more must be able to fold its leading
    dimensions as a view with ``axis`` moved last. The out-of-place op has
    neither restriction. The result equals that of :func:`block_quantize`
    word for word.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_quantize_
        >>> w = torch.tensor([[1.0, 0.1, -3.3, 100.0]])
        >>> block_quantize_(w, MXFP8_E4M3) is w
        True
    """
    return _quantize(x, fmt, axis, rounding, tensor_scale, carrier, in_place=True)


# --- the GEMM -------------------------------------------------------------------


class _BlockGemmSpec(NamedTuple):
    """The block GEMM's arithmetic, resolved into exactly what the op takes.

    ``args`` is every schema argument after the two operands' formats, in
    order: the mac (fused or split, the accumulate format) and the
    accumulation (algorithm, rounding, the ``*_accumulated`` tail), but for
    the carrier, which is the call's. ``findings`` is what binary32 and
    binary64 each say about the formats it rounds with; ``stored`` the format
    a result narrower than its carrier holds. The operand formats travel with
    the operands. ``epilogue`` applies the operands' tensor scales to the
    result rather than in the decode.
    """

    args: tuple[Any, ...]
    findings: tuple[_Findings, _Findings] = ((), ())
    stored: tuple[_Stored, ...] = ()
    carrier: torch.dtype | None = None
    epilogue: bool = False


def _binaryK_fields(f: BinaryK | None) -> tuple[Any, ...]:
    """A binaryK slot as (quant, K, P, bias, is_signed, saturation, subnormals, prng)."""
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
    """Resolve a block GEMM's arithmetic once; memoized on the values."""
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
    """``p`` in the GEMM's ``role``: ``"a"``, logically ``[..., M, K]``, or
    ``"b"``, logically ``[..., K, N]``. Each is read along its K: stored with K
    along the packed axis unless it was packed along the other of its last two
    dimensions, which the kernel reads transposed."""
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
    """Whether a block GEMM runs in binary64: when the spec names it, or when
    either operand was packed in it (a carrier is never narrower than what it
    reads). ``carrier=torch.float32`` on a binary64 operand raises."""
    packed64 = a.carrier is torch.float64 or b.carrier is torch.float64
    if spec.carrier is torch.float32 and packed64:
        raise ValueError(
            "carrier=torch.float32 is narrower than binary64, which an operand was packed in; "
            "leave carrier unset"
        )
    return spec.carrier is torch.float64 or packed64


def _run_block_gemm(spec: _BlockGemmSpec, a: BlockPacked, b: BlockPacked) -> torch.Tensor:
    """Call the block GEMM on two packed operands: ``a @ b`` of their logical
    tensors, in the carrier (:func:`_gemm_carrier`), float32 or float64, the
    leading dimensions broadcast when one side has none or they are equal.

    With ``spec.epilogue`` the kernel decodes each element with its block's
    scale alone (a tensor scale of 1) and the result is multiplied, once, by
    ``alpha``, the product of the two tensor scales in the carrier: the
    product of two float32 values is exact in a Python float, so rounding it
    to float32 is the one rounding, a float64 product is rounded once by
    Python's own multiply, and ``mul_`` by a value of the carrier is the
    other."""
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
    """The result, float32 or float64 as its carrier, stored in ``dtype``, the
    operands', with the storage check a narrower dtype asks for; a binary64
    result is narrowed with one rounding (``ops._narrowed``)."""
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
    scale, as :func:`block_unpack` would) and multiplies in the carrier:
    binary64 when ``carrier=torch.float64`` or either operand was packed in
    binary64 (from a float64 tensor, or with that carrier), and binary32
    otherwise. A split mac (``fused=False``) rounds each product to the
    carrier and the running sum to ``acc``. A fused mac rounds each fused
    multiply-add to ``acc``. ``acc=None`` leaves the sum in the carrier.

    Each operand is read along K. It may be packed along its K (``a`` along
    its last axis, ``b`` along its second-to-last) or along its other
    dimension, which the kernel reads transposed. One packed weight therefore
    serves both ``x @ w.T`` and ``g @ w``; :attr:`BlockPacked.mT` is the
    transpose, without a copy. The two operands may have different formats.
    Leading dimensions must be equal, or absent on one side. The function is
    not differentiable: see :func:`mptorch.quant.qmatmul` with a
    :class:`mptorch.quant.BlockMac`.

    A format with a per-tensor scale (NVFP4's E4M3, or a superfp scale)
    decodes each element as its value times its block's scale times the
    tensor scale, inside the loop. ``tensor_scale_epilogue=True`` does what
    NVIDIA's NVFP4 GEMMs do instead. The loop sees the elements times their
    block scales only, and the result is multiplied once by the product of
    the two tensor scales, rounded to the carrier. The two agree in exact
    arithmetic and round differently. In the second, the accumulate format
    sees products that the tensor scales have not scaled, so it needs their
    range.

    Args:
        a (BlockPacked): the left operand.
        b (BlockPacked): the right operand.
        acc (BinaryK, optional): the accumulate format. Default: ``None``
        fused (bool): one rounding per step (a fused multiply-add) instead of
            two. Default: ``False``
        rounding (RoundMode): the rounding mode of the sum. Default:
            ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :class:`mptorch.quant.SplitMac`. Default:
            ``AccumulateAlgorithm.NAIVE``
        block_size (int, optional): BLOCK's and TREE's block size. Default:
            ``None``
        outer (BinaryK, optional): BLOCK's and TREE's outer format. Default:
            ``None``
        carrier (torch.dtype, optional): ``torch.float64`` multiplies in
            binary64; ``torch.float32`` refuses an operand packed in
            binary64. Default: ``None``, the operands' carrier.
        tensor_scale_epilogue (bool): apply the operands' tensor scales to
            the result instead of in the decode. It has no effect on formats
            without a tensor scale (E8M0). Default: ``False``

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
