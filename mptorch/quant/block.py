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

The kernels round in binary32 only so far (``dev/continuation_plan.md``, phase
G): a float64 tensor or ``carrier=torch.float64`` raises, and so does an MPS
tensor (phase H). Every entry point takes ``carrier`` all the same, as every
computing entry point of the library does.
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
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _check_stored,
    _checked_block_size,
    _checked_carrier,
    _Findings,
    _format_findings,
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


def _tensor_scale(x: torch.Tensor, fmt: BlockFormat, tensor_scale: float | None) -> float:
    """The per-tensor scale a call packs with: 1 for a format without one.

    NVFP4's default is ``amax(|x|) / (elem_max * scale_max)`` over the finite
    elements, in float32 arithmetic, so that the tensor's largest block takes
    the largest scale; a tensor with no nonzero finite element takes 1. A
    caller-supplied (static) scale is rounded to float32.
    """
    if not fmt.has_tensor_scale:
        if tensor_scale is not None:
            raise ValueError(
                f"{_block_label(fmt)} has no per-tensor scale: tensor_scale belongs to a format "
                "whose block scale has a mantissa (NVFP4)"
            )
        return 1.0
    if tensor_scale is not None:
        ts = _float32(float(tensor_scale))
        if not ts > 0 or ts == float("inf"):
            raise ValueError(f"tensor_scale must be a positive finite float, got {tensor_scale}")
        return ts
    if x.numel() == 0:
        return 1.0
    # max |x| as a reduction, with no |x|-sized temporary (a weight's worth of
    # memory at the peak of a layer's step); the non-finite elements are left
    # out only when there are some, which costs the temporaries.
    amax = torch.linalg.vector_norm(x.detach(), float("inf"), dtype=torch.float32).cpu()
    if not torch.isfinite(amax):
        a = x.detach().abs().float()
        amax = torch.where(torch.isfinite(a), a, torch.zeros((), device=a.device)).amax().cpu()
    assert fmt.elem_max is not None  # resolved by BlockFormat.__post_init__
    ts = (amax / torch.tensor(fmt.elem_max * fmt.scale_max, dtype=torch.float32)).item()
    return ts if ts > 0 else 1.0


def _prologue(x: torch.Tensor, fmt: BlockFormat, carrier: torch.dtype | None, op: str) -> None:
    """What every elementwise call checks first: the format, the carrier (binary32
    only), and a tensor dtype the kernels read."""
    if not isinstance(fmt, BlockFormat):
        raise TypeError(f"{op} takes a BlockFormat, got {type(fmt).__name__}")
    if x.dtype is torch.float64:
        check_block_carrier(fmt, carrier=torch.float64)
    wide, _ = _call_carrier(_checked_carrier(carrier), x.dtype)
    check_block_carrier(fmt, carrier=_CARRIER[wide])


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

    What :func:`block_pack` returns. ``data`` holds the element codes of the
    logical tensor with its packed axis moved last, ``[..., rows, row_bytes]``
    (the leading dimensions folded into one, a 1D tensor stored as one row),
    and ``scales`` the scale codes, ``[..., row_tiles, n_blocks]`` (no columns
    for a format without a scale). ``shape``, ``axis`` and ``dtype`` describe
    the logical tensor: its shape, the axis packed along, and the dtype it was
    packed from, which :meth:`unpack` returns. ``tensor_scale`` is the
    per-tensor scale it was packed with (1 for a format without one).

    :attr:`mT` is the logical transpose of the last two dimensions, the same
    codes read the other way: nothing is copied, only ``shape`` and ``axis``
    change, which is what lets :func:`block_matmul` read one packed weight in
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

    __slots__ = ("data", "scales", "fmt", "shape", "axis", "tensor_scale", "dtype")

    def __init__(
        self,
        data: torch.Tensor,
        scales: torch.Tensor,
        fmt: BlockFormat,
        shape: tuple[int, ...],
        axis: int,
        tensor_scale: float,
        dtype: torch.dtype,
    ):
        self.data = data
        self.scales = scales
        self.fmt = fmt
        self.shape = tuple(shape)
        self.axis = axis
        self.tensor_scale = tensor_scale
        self.dtype = dtype

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
        """The length of the packed axis, which ``data``'s rows round up to whole blocks."""
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
            self.data, self.scales, self.fmt, shape, axis, self.tensor_scale, self.dtype
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

    Each run of ``fmt.block_size`` consecutive elements along ``axis`` (times
    ``fmt.block_rows`` rows along the axis before it, for a 2D format) shares
    one scale, chosen from its largest magnitude as :class:`mptorch.BlockFormat`
    states, and each element is divided by it and rounded in ``rounding`` to
    the element format. The result stores the codes, a quarter to an eighth
    of float32's bytes: :class:`BlockPacked` says how. A transposed view packs
    without a copy, and so does any ``axis``: the kernel reads ``x`` through
    its strides.

    Under ``RoundMode.SR`` an element's random bits are keyed on its index in
    the packed orientation (the tensor with ``axis`` moved last), so
    :func:`block_quantize` with the same generator state draws the same bits.

    Args:
        x (Tensor): float32, float16 or bfloat16, of any rank.
        fmt (BlockFormat): the format.
        axis (int): the axis to pack along. Default: ``-1``
        rounding (RoundMode): the elements' rounding mode; the scale is
            always chosen by the format's rule. Default: ``RoundMode.RNE``
        tensor_scale (float, optional): the per-tensor scale of a format whose
            block scale has a mantissa (NVFP4); ``None`` takes ``amax(|x|) /
            (elem_max * scale_max)``, which reads ``x`` once more and, on a
            GPU, synchronizes with it. Default: ``None``
        carrier (torch.dtype, optional): the arithmetic the rounding runs in.
            Block formats have binary32 kernels only so far, so ``None`` and
            ``torch.float32`` work and ``torch.float64`` raises. Default:
            ``None``

    Returns:
        BlockPacked: the codes, the scales and what describes them.

    Raises:
        ValueError: for a float64 ``x`` or ``carrier=torch.float64`` (binary32
            only so far), or a ``tensor_scale`` given to a format without one.
        RuntimeError: on an MPS tensor (no Metal kernel yet), or on a tensor
            that requires grad under grad mode.

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
    _prologue(x, fmt, carrier, "block_pack")
    if x.dim() == 0:
        raise ValueError("block_pack needs a tensor with at least one dimension")
    axis = axis % x.dim()
    ts = _tensor_scale(x, fmt, tensor_scale)
    data, scales = torch.ops.mptorch.block_pack.default(
        _storage_view(x, axis, "block_pack"),
        _block_format_ints(fmt),
        fmt.elem_max,
        fmt.scale_max,
        ts,
        rounding.value,
    )
    return BlockPacked(data, scales, fmt, tuple(x.shape), axis, ts, x.dtype)


def block_unpack(
    p: BlockPacked, *, dtype: torch.dtype | None = None, carrier: torch.dtype | None = None
) -> torch.Tensor:
    """Decode a :class:`BlockPacked` into a tensor of its logical shape.

    Each element is its code's value times its block's scale (times the
    tensor scale), one binary32 product, stored in ``dtype``; a float16 or
    bfloat16 result rounds that product a second time where the dtype does not
    hold it, which :class:`mptorch.FormatRangeWarning` reports.
    ``block_unpack(block_pack(x, fmt, axis))`` equals ``block_quantize(x, fmt,
    axis)`` element for element, under the same generator state.

    Args:
        p (BlockPacked): the packed tensor.
        dtype (torch.dtype, optional): float32, float16 or bfloat16. Default:
            ``None``, the dtype ``p`` was packed from.
        carrier (torch.dtype, optional): as for :func:`block_pack`.
            Default: ``None``

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
    probe = torch.empty(0, dtype=dtype)
    _prologue(probe, p.fmt, carrier, "block_unpack")
    if dtype in _NARROW_STORAGE:
        check_block_storage(p.fmt, storage=dtype)
    cols = p.cols
    out = torch.ops.mptorch.block_unpack.default(p.data, p.scales, cols, *_format_args(p), dtype)
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
    _prologue(x, fmt, carrier, op)
    if x.dim() == 0:
        raise ValueError(f"{op} needs a tensor with at least one dimension")
    if x.dtype in _NARROW_STORAGE:
        check_block_storage(fmt, storage=x.dtype)
    axis = axis % x.dim()
    ts = _tensor_scale(x, fmt, tensor_scale)
    v = _storage_view(x, axis, op, in_place=in_place)
    args = (_block_format_ints(fmt), fmt.elem_max, fmt.scale_max, ts, rounding.value)
    if in_place:
        torch.ops.mptorch.block_quant_.default(v, *args)
        return x
    out = torch.ops.mptorch.block_quant.default(v, *args)
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

    :func:`block_pack` and :func:`block_unpack` fused, with no packed
    intermediate: the fake quantizer a ``*_quant`` slot of a layer's formats
    takes (:class:`mptorch.quant.BlockQuant` is its value spelling). The
    arguments are :func:`block_pack`'s, and the result has ``x``'s shape,
    dtype and layout. Not differentiable: wrap it in
    :class:`mptorch.quant.Quantizer` for a straight-through gradient.

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

    For a tensor the caller owns (a weight quantized once at load), never for
    a ``*_quant`` slot, whose input belongs to the graph of whatever produced
    it. ``x`` must have no overlapping elements, and a rank-4-or-more tensor
    must fold its leading dimensions as a view with ``axis`` moved last; the
    out-of-place op takes anything. The result equals :func:`block_quantize`'s
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
    accumulation (algorithm, rounding, the ``*_accumulated`` tail).
    ``findings`` is what binary32, the block kernels' carrier, says about the
    formats it rounds with; ``stored`` the format a result narrower than
    float32 holds. The operand formats travel with the operands. ``epilogue``
    applies the operands' tensor scales to the result rather than in the
    decode.
    """

    args: tuple[Any, ...]
    findings: _Findings = ()
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
    if carrier is torch.float64:
        raise ValueError(
            "block formats have binary32 kernels only so far, so the block GEMM takes no "
            "carrier=torch.float64 (dev/continuation_plan.md, phase G)"
        )
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
    found32, _ = _format_findings(rounded)
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
        a[5] if acc is not None else SaturationMode.OVF_INF.value,
        a[6] if acc is not None else SubnormalsMode.SUBNORMALS.value,
        a[7],
        block,
        o[0],
        o[1],
        o[2],
        o[3],
        o[4],
        o[5] if outer is not None else SaturationMode.OVF_INF.value,
        o[6] if outer is not None else SubnormalsMode.SUBNORMALS.value,
        o[7],
    )
    return _BlockGemmSpec(args, found32, stored, carrier, bool(tensor_scale_epilogue))


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


def _run_block_gemm(spec: _BlockGemmSpec, a: BlockPacked, b: BlockPacked) -> torch.Tensor:
    """Call the block GEMM on two packed operands: ``a @ b`` of their logical
    tensors, in float32, the leading dimensions broadcast when one side has
    none or they are equal.

    With ``spec.epilogue`` the kernel decodes each element with its block's
    scale alone (a tensor scale of 1) and the float32 result is multiplied,
    once, by ``alpha``, the float32 product of the two tensor scales: the
    product of two float32 values is exact in a Python float, so rounding it
    to float32 is the one rounding, and ``mul_`` by a float32 value is the
    other."""
    if spec.findings:
        _report_findings(spec.findings)
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
    )
    if spec.epilogue:
        alpha = _float32(a.tensor_scale * b.tensor_scale)
        if alpha != 1.0:
            out.mul_(alpha)
    lead = oa.lead or ob.lead
    if len(lead) > 1:
        out = out.reshape(*lead, *out.shape[-2:])
    return out


def _stored_as(spec: _BlockGemmSpec, out: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The float32 result stored in ``dtype``, the operands', with the storage
    check a narrower dtype asks for."""
    if dtype is torch.float32:
        return out
    if spec.stored and dtype in _NARROW_STORAGE:
        _check_stored(spec.stored, dtype)
    return out.to(dtype)


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

    ``a`` is logically ``[..., M, K]`` and ``b`` ``[..., K, N]``; the kernel
    decodes each element as it loads it (its value times its block's scale,
    as :func:`block_unpack` would) and multiplies in binary32. A split mac
    (``fused=False``) rounds each product to binary32 and the running sum to
    ``acc``; a fused one rounds each fused multiply-add to ``acc``; ``acc=None``
    leaves the sum in binary32. Each operand is read along K: packed along its
    K (``a`` along its last axis, ``b`` along its second-to-last) or along its
    other dimension, which the kernel reads transposed, so a packed weight
    serves ``x @ w.T`` and ``g @ w`` both (:attr:`BlockPacked.mT` is the
    transpose, without a copy). The two may be of different formats. Leading
    dimensions must be equal, or absent on one side. Not differentiable: see
    :func:`mptorch.quant.qmatmul` with a :class:`mptorch.quant.BlockMac`.

    A format with a per-tensor scale (NVFP4's E4M3, or a superfp scale)
    decodes each element as its value times its block's scale times the
    tensor scale, inside the loop. ``tensor_scale_epilogue=True`` does what
    NVIDIA's NVFP4 GEMMs do instead: the loop sees the elements times their
    block scales only, and the float32 result is multiplied once by the
    float32 product of the two tensor scales. The two agree in exact
    arithmetic and round differently; in the second the accumulate format
    sees the products unscaled by the tensor scales, so it needs their range.

    Args:
        a (BlockPacked): the left operand.
        b (BlockPacked): the right operand.
        acc (BinaryK, optional): the accumulate format. Default: ``None``
        fused (bool): one rounding per step (a fused multiply-add) rather
            than two. Default: ``False``
        rounding (RoundMode): the rounding mode of the sum. Default:
            ``RoundMode.RNE``
        accumulate_algorithm (AccumulateAlgorithm): as for
            :class:`mptorch.quant.SplitMac`. Default:
            ``AccumulateAlgorithm.NAIVE``
        block_size (int, optional): BLOCK's and TREE's block size. Default:
            ``None``
        outer (BinaryK, optional): BLOCK's and TREE's outer format. Default:
            ``None``
        carrier (torch.dtype, optional): ``None`` or ``torch.float32``;
            ``torch.float64`` raises (binary32 only so far). Default: ``None``
        tensor_scale_epilogue (bool): apply the operands' tensor scales to
            the result rather than in the decode. No effect on formats
            without one (E8M0). Default: ``False``

    Returns:
        Tensor: float32, ``[..., M, N]``.

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
