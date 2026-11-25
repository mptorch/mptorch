"""Operations on packed block tensors: IEEE P3109's block operations, and two of mptorch's own.

IEEE P3109 (arXiv:2606.04028, section VIII) defines its block operations on
a block given as a scale code and its elements' codes. Here a block is a
block of a :class:`~mptorch.quant.BlockPacked`, and each operation decodes
its operands into the carrier, computes there (rounding to nearest, as
every operation in mptorch computes between two roundings to a format) and
projects the result once:

* :func:`block_map` is P3109's BlockMap: a function applied to the decoded
  values of one or more packed tensors, its result projected onto the
  scale codes given, by :func:`~mptorch.quant.block_project`. With no scales
  given, the format's own rule chooses them (:func:`~mptorch.quant.block_pack`),
  which is P3109's variant that runs a scaling algorithm;
* :func:`block_exp` is BlockExp: :func:`block_map` of the exponential;
* :func:`block_scaled_add` is ScaledAdd: :func:`block_map` of a sum;
* :func:`convert_to_block_max_abs_finite` is ConvertToBlockMaxAbsFinite: a
  tensor stored with each block's scale the block's largest finite
  magnitude, so that its largest element becomes about 1.

P3109 names BlockReduceAdd and BlockDotProduct without defining them.
mptorch defines them as a reduction to one value per block, with the
arithmetic of a :func:`~mptorch.quant.qsum` or a
:func:`~mptorch.quant.qbmm`:

* :func:`block_reduce_add`: each block's decoded values summed, in order;
* :func:`block_dot`: each pair of blocks' decoded values multiplied and
  summed, as a dot product.

A block here is ``block_size`` consecutive elements along the packed axis,
which in a format with ``block_rows > 1`` is one row of a tile. None of
these functions is differentiable.
"""

from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from mptorch.number import BinaryK, BlockFormat, RoundMode, SaturationMode, SuperFP
from mptorch.number.block import _minifloat, _minifloat_value

from .mac import Quant
from .matmul import qbmm
from .ops._checks import _CARRIER, _call_carrier, _checked_carrier
from .ops.block import (
    BlockPacked,
    _float32,
    _storage_view,
    _tensor_scale,
    block_pack,
    block_project,
    block_unpack,
)
from .reduce import qsum

__all__ = [
    "block_map",
    "block_exp",
    "block_scaled_add",
    "convert_to_block_max_abs_finite",
    "block_reduce_add",
    "block_dot",
]


def _same_layout(packed: tuple[BlockPacked, ...], op: str) -> BlockPacked:
    """Check that ``packed`` is one or more :class:`BlockPacked` of one
    logical shape, packed along one axis, and return the first."""
    if not packed:
        raise TypeError(f"{op} takes at least one BlockPacked")
    for p in packed:
        if not isinstance(p, BlockPacked):
            raise TypeError(f"{op} takes BlockPacked operands, got {type(p).__name__}")
    first = packed[0]
    for p in packed[1:]:
        if p.shape != first.shape or p.axis != first.axis:
            raise ValueError(
                f"{op} pairs its operands' blocks, so they must have one shape and one packed "
                f"axis, got {first.shape} along axis {first.axis} and {p.shape} along axis "
                f"{p.axis}"
            )
    return first


def _carrier(packed: tuple[BlockPacked, ...]) -> torch.dtype:
    """The carrier the operands are decoded and computed in: binary64 if any
    of them was packed in it, binary32 otherwise."""
    return torch.float64 if any(p.carrier is torch.float64 for p in packed) else torch.float32


def _relabeled(q: BlockPacked, dtype: torch.dtype) -> BlockPacked:
    """``q`` recorded as packed from ``dtype``, the dtype it unpacks to by default."""
    return BlockPacked(q.data, q.scales, q.fmt, q.shape, q.axis, q.tensor_scale, dtype, q.carrier)


# --- P3109's operations ---------------------------------------------------------


def block_map(
    fn: Callable[..., torch.Tensor],
    *packed: BlockPacked,
    scales: torch.Tensor | None = None,
    fmt: BlockFormat | None = None,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
) -> BlockPacked:
    """Apply ``fn`` to the decoded values of packed tensors and store the result: P3109's BlockMap.

    Each operand is decoded (:func:`~mptorch.quant.block_unpack`) into the
    carrier, binary64 if any operand was packed in it and binary32
    otherwise, which holds every decoded value exactly. ``fn`` takes the
    decoded tensors, one per operand, and computes in that dtype. Its result
    is then stored in ``fmt``, rounded once in the rounding mode
    ``rounding``: under the scale codes ``scales``, by
    :func:`~mptorch.quant.block_project`, or, with ``scales=None``, under the
    scales the format's rule chooses for the result, by
    :func:`~mptorch.quant.block_pack`. The result is rounded twice in all:
    once by ``fn``'s arithmetic in the carrier, and once to the format.

    Args:
        fn (Callable): takes one tensor per operand and returns a tensor of
            their shape and the carrier's dtype.
        *packed (BlockPacked): the operands, of one logical shape and packed
            along one axis. Their formats may differ.
        scales (Tensor, optional): the result's scale codes, laid out as the
            operands' :attr:`BlockPacked.scales` are. Default: ``None``, the
            format's rule
        fmt (BlockFormat, optional): the result's format. Default: ``None``,
            the first operand's
        rounding (RoundMode): the rounding mode of the result's elements.
            Default: ``RoundMode.RNE``
        tensor_scale (float, optional): the result's per-tensor scale, for a
            format that has one. ``None`` takes the first operand's when
            ``scales`` is given and both formats have one, and otherwise
            :func:`~mptorch.quant.block_pack`'s default. Default: ``None``

    Returns:
        BlockPacked: the result, packed along the operands' axis, recorded as
        packed from the first operand's dtype.

    Raises:
        TypeError: for an operand that is not a :class:`BlockPacked`, or a
            result of ``fn`` that is not a tensor of the carrier's dtype.
        ValueError: for operands of different shapes or axes, or a result of
            another shape, and as :func:`~mptorch.quant.block_project` raises.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_map, block_pack
        >>> p = block_pack(torch.tensor([[0.5, -1.0, 2.0, 4.0] * 8]), MXFP8_E4M3)
        >>> block_map(torch.square, p).unpack()[0, :4]  # new scales, by the format's rule
        tensor([ 0.2500,  1.0000,  4.0000, 16.0000])
        >>> block_map(torch.square, p, scales=p.scales).unpack()[0, :4]  # p's: 16 saturates
        tensor([0.2500, 1.0000, 4.0000, 7.0000])
    """
    first = _same_layout(packed, "block_map")
    carrier = _carrier(packed)
    values = [block_unpack(p, dtype=carrier) for p in packed]
    y = fn(*values)
    if not isinstance(y, torch.Tensor) or y.dtype is not carrier:
        raise TypeError(
            f"block_map's fn must return a {carrier} tensor, the carrier's dtype, got "
            f"{y.dtype if isinstance(y, torch.Tensor) else type(y).__name__}"
        )
    if tuple(y.shape) != first.shape:
        raise ValueError(
            f"block_map's fn must return a tensor of its operands' shape {first.shape}, got "
            f"{tuple(y.shape)}"
        )
    out_fmt = first.fmt if fmt is None else fmt
    if scales is None:
        q = block_pack(
            y, out_fmt, first.axis, rounding=rounding, tensor_scale=tensor_scale, carrier=carrier
        )
    else:
        if tensor_scale is None and isinstance(out_fmt, BlockFormat):
            if out_fmt.has_tensor_scale and first.fmt.has_tensor_scale:
                tensor_scale = first.tensor_scale
        q = block_project(
            y,
            out_fmt,
            scales,
            first.axis,
            rounding=rounding,
            tensor_scale=tensor_scale,
            carrier=carrier,
        )
    return _relabeled(q, first.dtype)


def block_exp(
    p: BlockPacked,
    scales: torch.Tensor | None = None,
    *,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
) -> BlockPacked:
    """The exponential of a packed tensor, stored in its format: P3109's BlockExp.

    It is ``block_map(torch.exp, p, scales=scales, ...)``: each block is
    decoded, ``torch.exp`` runs in the carrier, and the result is stored
    under the scale codes ``scales``, or under the ones the format's rule
    chooses with ``scales=None``. P3109 requires the second spelling to give
    what the first gives on the scales it chose, which holds here, since
    :func:`~mptorch.quant.block_project` onto :func:`~mptorch.quant.block_pack`'s
    scales returns its codes.

    Args:
        p (BlockPacked): the operand.
        scales (Tensor, optional): the result's scale codes. Default: ``None``
        rounding (RoundMode): the elements' rounding mode. Default:
            ``RoundMode.RNE``
        tensor_scale (float, optional): as for :func:`block_map`. Default:
            ``None``

    Returns:
        BlockPacked: ``exp`` of ``p``, in ``p``'s format.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_exp, block_pack
        >>> p = block_pack(torch.tensor([[0.0, 1.0, -1.0, 2.0] * 8]), MXFP8_E4M3)
        >>> block_exp(p).unpack()[0, :4]
        tensor([1.0000, 2.7500, 0.3750, 7.0000])
    """
    return block_map(torch.exp, p, scales=scales, rounding=rounding, tensor_scale=tensor_scale)


def block_scaled_add(
    a: BlockPacked,
    b: BlockPacked,
    scales: torch.Tensor | None = None,
    *,
    fmt: BlockFormat | None = None,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
) -> BlockPacked:
    """The sum of two packed tensors, stored in a block format: P3109's ScaledAdd.

    P3109 defines ScaledAdd on single-element blocks: each operand decoded
    with its scale, the two added, and the sum projected into the result's
    format. Here the operands are packed tensors of one shape, in any block
    formats, and this is ``block_map(torch.add, a, b, ...)``: the sum is
    computed in the carrier and stored under ``scales``, or under the scales
    the format's rule chooses with ``scales=None``.

    Args:
        a, b (BlockPacked): the operands, of one shape and packed along one
            axis.
        scales (Tensor, optional): the result's scale codes. Default: ``None``
        fmt (BlockFormat, optional): the result's format. Default: ``None``,
            ``a``'s
        rounding (RoundMode): the elements' rounding mode. Default:
            ``RoundMode.RNE``
        tensor_scale (float, optional): as for :func:`block_map`. Default:
            ``None``

    Returns:
        BlockPacked: ``a + b``.

    Example::

        >>> import torch
        >>> from mptorch import MXFP4_E2M1
        >>> from mptorch.quant import block_pack, block_scaled_add
        >>> a = block_pack(torch.tensor([[1.0, 2.0, 3.0, 4.0] * 8]), MXFP4_E2M1)
        >>> b = block_pack(torch.tensor([[0.5, 0.5, 0.5, 1.5] * 8]), MXFP4_E2M1)
        >>> block_scaled_add(a, b).unpack()[0, :4]  # 2.5, 3.5 and 5.5 round to E2M1
        tensor([1.5000, 2.0000, 4.0000, 6.0000])
    """
    return block_map(
        torch.add, a, b, scales=scales, fmt=fmt, rounding=rounding, tensor_scale=tensor_scale
    )


def _tile_amax(a: torch.Tensor, fmt: BlockFormat) -> torch.Tensor:
    """The largest of ``a``'s values in each tile: ``a`` is ``[(batch,) rows,
    cols]``, non-negative, and the result ``[(batch,) row_tiles, n_blocks]``,
    as a :class:`BlockPacked`'s scales are laid out."""
    bs, br = fmt.block_size, fmt.block_rows
    rows, cols = a.shape[-2:]
    n_blocks, row_tiles = -(-cols // bs), -(-rows // br)
    batched = a.dim() == 3
    a = a if batched else a.unsqueeze(0)
    a = F.pad(a, (0, n_blocks * bs - cols, 0, row_tiles * br - rows))
    m = a.reshape(a.shape[0], row_tiles, br, n_blocks, bs).amax(dim=(2, 4))
    return m if batched else m.squeeze(0)


@lru_cache(maxsize=64)
def _scale_values(scale: BinaryK | SuperFP) -> tuple[float, ...]:
    """The value of each magnitude code of a cast scale below its NaN, in
    code order, which is increasing order."""
    mf = _minifloat(scale)
    nan_mag = (1 << (mf.bits - int(mf.is_signed))) - 1
    return tuple(float(_minifloat_value(mf, mag)) for mag in range(nan_mag))


def _scale_codes(m: torch.Tensor, fmt: BlockFormat, ts: float) -> torch.Tensor:
    """The scale code nearest each block's largest finite magnitude ``m``
    (divided by the tensor scale ``ts`` for a cast scale), as ``uint8``.

    A power-of-two scale rounds ``m`` to the nearest power of two, a tie to
    the even exponent as every cast in the library breaks one, and clamps it
    to the scale's codes; a block of zeros gets code 0, the smallest scale.
    A cast scale is the scale format's own rounding of ``m / ts`` to nearest
    even, saturating to ``scale_max``, so a block whose ``m / ts`` is below
    half the smallest scale gets the zero scale."""
    scale = fmt.scale
    assert scale is not None
    if isinstance(scale, BinaryK) and scale.P == 1:
        assert scale.bias is not None
        mant, e = torch.frexp(m)  # m = mant * 2^e, mant in [0.5, 1)
        k = e - 1  # floor(log2(m))
        up = (mant > 0.75) | ((mant == 0.75) & (k % 2 != 0))
        code = (k + up.to(k.dtype) + scale.bias).clamp(0, (1 << scale.K) - 2)
        return torch.where(m > 0, code, torch.zeros_like(code)).to(torch.uint8)
    cast = Quant(replace(scale, saturation=SaturationMode.SAT_FINITE, domain=None), RoundMode.RNE)
    q = cast(m / torch.tensor(ts, dtype=m.dtype, device=m.device))
    table = torch.tensor(_scale_values(scale), dtype=m.dtype, device=m.device)
    q = torch.minimum(q, table[-1])
    return torch.searchsorted(table, q.contiguous()).to(torch.uint8)


def convert_to_block_max_abs_finite(
    x: torch.Tensor,
    fmt: BlockFormat,
    axis: int = -1,
    *,
    rounding: RoundMode = RoundMode.RNE,
    tensor_scale: float | None = None,
    carrier: torch.dtype | None = None,
) -> BlockPacked:
    """Store ``x`` scaled by each block's largest finite magnitude: ConvertToBlockMaxAbsFinite.

    Each block's scale is the block's largest finite magnitude ``m``,
    rounded into the scale format, so that the block's largest element is
    stored as about ±1, where :func:`~mptorch.quant.block_pack`'s rules
    store it near the element format's largest value. The elements are then
    stored under those scales by :func:`~mptorch.quant.block_project`, in the
    rounding mode ``rounding``, so an infinite element saturates to the
    largest element value and a NaN follows :func:`~mptorch.quant.block_project`.

    The scale is rounded to nearest. A power-of-two scale (E8M0) is the power
    of two nearest ``m``, a tie going to the even exponent, which puts the
    largest element in [2/3, 4/3]; a block of zeros gets the smallest scale.
    A cast scale (NVFP4's E4M3) is ``m / tensor_scale`` rounded by the scale
    format's cast to nearest even, at most its largest value, and the
    default tensor scale is ``max(|x|) / scale_max`` over the finite
    elements, so the tensor's largest block gets the largest scale. A block
    whose ``m / tensor_scale`` rounds to zero gets the zero scale, and P3109
    then stores every element of it as 0.

    Args:
        x (Tensor): float32, float64, float16 or bfloat16, of any rank.
        fmt (BlockFormat): a format with a scale.
        axis (int): the axis to pack along. Default: ``-1``
        rounding (RoundMode): the elements' rounding mode. Default:
            ``RoundMode.RNE``
        tensor_scale (float, optional): the per-tensor scale of a format that
            has one. Default: ``None``, ``max(|x|) / scale_max``
        carrier (torch.dtype, optional): as for :func:`~mptorch.quant.block_pack`.
            Default: ``None``

    Returns:
        BlockPacked: the codes and the scales chosen.

    Raises:
        TypeError: for a ``fmt`` that is not a :class:`mptorch.BlockFormat`.
        ValueError: for a format without a scale, and as
            :func:`~mptorch.quant.block_project` raises.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import convert_to_block_max_abs_finite
        >>> x = torch.tensor([[0.25, -1.0, 2.0, 4.0] * 8])
        >>> convert_to_block_max_abs_finite(x, MXFP8_E4M3).data[0, :4]  # 1/16, -1/4, 1/2, 1
        tensor([ 24, 168,  48,  56], dtype=torch.uint8)
    """
    op = "convert_to_block_max_abs_finite"
    if not isinstance(fmt, BlockFormat):
        raise TypeError(f"{op} takes a BlockFormat, got {type(fmt).__name__}")
    if fmt.scale is None:
        raise ValueError(f"{op} chooses each block's scale, so it needs a format with a scale")
    if x.dim() == 0:
        raise ValueError(f"{op} needs a tensor with at least one dimension")
    wide, _ = _call_carrier(_checked_carrier(carrier), x.dtype)
    dtype = _CARRIER[wide]
    axis = axis % x.dim()
    a = _storage_view(x.detach(), axis, op).abs().to(dtype)
    a = torch.where(torch.isfinite(a), a, torch.zeros((), dtype=dtype, device=a.device))
    ts = 1.0
    if fmt.has_tensor_scale:
        if tensor_scale is not None:
            ts = _tensor_scale(x, fmt, tensor_scale, wide)
        else:
            amax = a.amax().item() if a.numel() else 0.0
            ts = amax / fmt.scale_max if amax > 0 else 1.0
            ts = ts if wide else _float32(ts)
    elif tensor_scale is not None:
        _tensor_scale(x, fmt, tensor_scale, wide)  # raises: the format has no tensor scale
    codes = _scale_codes(_tile_amax(a, fmt), fmt, ts)
    return block_project(
        x,
        fmt,
        codes,
        axis,
        rounding=rounding,
        tensor_scale=ts if fmt.has_tensor_scale else None,
        carrier=carrier,
    )


# --- mptorch's reductions -------------------------------------------------------


def _per_block(
    v: torch.Tensor, block_size: int, reduce: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """``reduce`` over each block of ``v``'s last dimension: ``reduce`` takes
    ``[..., n, k]`` and returns ``[..., n]``. A last block shorter than the
    others is reduced on its own, rather than padded with zeros, which a
    compensated or stochastic sum would not treat as nothing."""
    cols = v.shape[-1]
    full = cols // block_size
    parts = []
    if full:
        parts.append(reduce(v[..., : full * block_size].unflatten(-1, (full, block_size))))
    if cols > full * block_size:
        parts.append(reduce(v[..., full * block_size :].unsqueeze(-2)))
    return torch.cat(parts, -1)


def block_reduce_add(p: BlockPacked, mac: Any = None) -> torch.Tensor:
    """Each block's sum, as mptorch defines P3109's BlockReduceAdd.

    Each block's elements are decoded into the carrier and summed in order
    by :func:`~mptorch.quant.qsum` under ``mac``, which gives the sum's
    arithmetic: each step rounded to a format, any accumulate algorithm, any
    rounding mode. With ``mac=None`` the sum is ``torch.sum``'s in the
    carrier. The last block of a row whose length is not a multiple of
    ``block_size`` is summed over the elements it has.

    Args:
        p (BlockPacked): the packed tensor.
        mac: as :func:`~mptorch.quant.qsum`'s. Default: ``None``

    Returns:
        Tensor: ``p``'s shape with the packed axis's length replaced by its
        number of blocks, in the carrier's dtype (float32, or float64 for a
        tensor packed in binary64).

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, MXFP8_E4M3
        >>> from mptorch.quant import block_pack, block_reduce_add
        >>> p = block_pack(torch.tensor([[1.0, 0.0625, 0.0625, 0.0] * 8]), MXFP8_E4M3)
        >>> block_reduce_add(p), block_reduce_add(p, mac=BinaryK(8, 4))
        (tensor([[9.]]), tensor([[8.]]))
    """
    if not isinstance(p, BlockPacked):
        raise TypeError(f"block_reduce_add takes a BlockPacked, got {type(p).__name__}")
    v = block_unpack(p, dtype=p.carrier).movedim(p.axis, -1)
    out = _per_block(v, p.fmt.block_size, lambda t: qsum(t, -1, mac))
    return out.movedim(-1, p.axis)


def block_dot(a: BlockPacked, b: BlockPacked, mac: Any = None) -> torch.Tensor:
    """Each pair of blocks' dot product, as mptorch defines P3109's BlockDotProduct.

    ``a`` and ``b`` have one shape, axis and block size, so their blocks
    pair up. Each pair is decoded into the carrier and multiplied as a dot
    product, a ``[1, k] @ [k, 1]`` GEMM, by :func:`~mptorch.quant.qbmm` under
    ``mac``: any mac a GEMM takes, its accumulate algorithm and rounding mode
    included. With ``mac=None`` it is ``torch.bmm``'s in the carrier. A last
    block shorter than the others is multiplied over the elements it has.

    Args:
        a, b (BlockPacked): the operands.
        mac: the dot products' arithmetic, as :func:`~mptorch.quant.qmatmul`
            takes it. Default: ``None``

    Returns:
        Tensor: the operands' shape with the packed axis's length replaced by
        its number of blocks, in the carrier's dtype.

    Example::

        >>> import torch
        >>> from mptorch import MXFP8_E4M3
        >>> from mptorch.quant import block_dot, block_pack
        >>> a = block_pack(torch.ones(1, 64), MXFP8_E4M3)
        >>> b = block_pack(torch.full((1, 64), 0.5), MXFP8_E4M3)
        >>> block_dot(a, b)
        tensor([[16., 16.]])
    """
    first = _same_layout((a, b), "block_dot")
    if a.fmt.block_size != b.fmt.block_size:
        raise ValueError(
            "block_dot pairs its operands' blocks, so they must have one block size, got "
            f"{a.fmt.block_size} and {b.fmt.block_size}"
        )
    carrier = _carrier((a, b))
    va = block_unpack(a, dtype=carrier).movedim(first.axis, -1)
    vb = block_unpack(b, dtype=carrier).movedim(first.axis, -1)
    pairs = torch.stack((va, vb), 0)

    def dots(t: torch.Tensor) -> torch.Tensor:
        lead, k = t.shape[1:-1], t.shape[-1]
        x = t[0].reshape(-1, 1, k)
        y = t[1].reshape(-1, k, 1)
        return qbmm(x, y, mac).reshape(lead)

    out = _per_block(pairs, a.fmt.block_size, dots)
    return out.movedim(-1, first.axis)
