"""Scaled quantization: multiply a tensor into a format's range, round, and divide back out.

An 8-bit format covers a few dozen binades, and a gradient of size ``1e-3``
falls mostly below E4M3's smallest value, so most of it rounds to zero. The
remedy FP8 training recipes use is a scale: multiply the tensor so that its
largest magnitude lands on the format's largest value, round, and divide the
scale back out. :class:`ScaledQuant` is that step as a quantizer, usable in
any ``*_quant`` slot.
"""

import math
from dataclasses import KW_ONLY, dataclass, field

import torch

from mptorch.number import BinaryK, BlockFormat, FixedPoint, Number, RoundMode, SuperFP

from .mac import Quant
from .ops._checks import _call_carrier, _checked_carrier, _narrowed
from .ops.block import _amax

__all__ = ["ScaledQuant"]

_GRANULARITIES = ("tensor", "row", "block")


def _maxabs(t: torch.Tensor) -> torch.Tensor:
    """``max |t|`` along the last dimension, kept as a dimension of size 1,
    without an ``|t|``-sized temporary (as :func:`_amax` does for a whole tensor)."""
    lo, hi = torch.aminmax(t, dim=-1, keepdim=True)
    return torch.maximum(-lo, hi)


@dataclass(frozen=True)
class ScaledQuant:
    """Round a tensor to a format after scaling its largest magnitude onto ``target``.

    A call computes ``s = target / max|x|`` and returns ``q(x * s) / s``,
    where ``q`` rounds to ``fmt`` with ``rounding``. The tensor's largest
    magnitude therefore lands on ``target``, by default the format's largest
    finite value, and the whole of the format's range is used. The result
    has ``x``'s shape and dtype. Like :class:`Quant` it is stateless (the
    scale is computed from each tensor, as a *current* scaling recipe does),
    so a layer may run it again in backward instead of keeping its result.

    Every step runs in the *carrier*, the binary float the cast rounds in: a
    float32, float16 or bfloat16 tensor is copied to float32 (a float64 one,
    or any tensor under ``carrier=torch.float64``, to float64), then the
    scale is computed, the copy is multiplied by it, rounded to ``fmt`` and
    divided by it, each operation rounded to the carrier's nearest value, and
    the result is converted back to ``x``'s dtype with a single rounding. A
    float16 tensor is therefore not limited by float16's range while it is
    scaled (a scale of ``5.7e7``, E5M2's 57344 over a gradient of size
    ``1e-3``, overflows float16), and its values are not rounded to float16
    between the multiplication and the format. ``x`` itself is never written.
    A call reads and writes the tensor four times, where a :class:`Quant`
    does once: for the largest magnitudes, for the product, which goes
    straight into the carrier, for the rounding, and for the quotient, which
    goes straight into the result. A narrower tensor on the CPU or MPS is
    converted in passes of its own, six in all.

    A tensor (or a row or block, see ``granularity``) whose largest magnitude
    is zero, infinite or NaN is scaled by 1, and so is one whose scale would
    overflow or underflow the carrier. The scale is chosen on the tensor's
    device, so a call never waits for it, and it compiles. A NaN or an
    infinity then goes through the format as its saturation mode says: under
    ``SaturationMode.SAT_FINITE`` an infinity becomes the largest finite
    value, which hides an overflow from a loss scaler.

    Scaling suits a format with a saturating mode: with
    ``SaturationMode.SAT_FINITE`` the largest magnitude rounds onto ``target``
    under every rounding mode, where under ``SAT_NONE`` a directed or
    stochastic rounding may carry it to infinity. The format is used as it is
    given.

    Args:
        fmt (Number): the format: a :class:`mptorch.BinaryK`, a
            :class:`mptorch.SuperFP` or a :class:`mptorch.FixedPoint`. A
            :class:`mptorch.BlockFormat` scales each block itself, so it is
            refused here: use :class:`Quant` or :class:`BlockQuant`.
        rounding (RoundMode): the rounding mode of the format.
            Default: ``RoundMode.RNE``
        target (float, optional): where the largest magnitude is scaled to.
            Default: ``None``, the format's largest finite value
            (``max_finite``, or ``max_value`` for a fixed-point format). OCP's
            E4M3 and E5M2 stop at 448 and 57344, below the largest values of
            the binaryK layouts, which count code points as P3109 does; pass
            those to simulate OCP's formats.
        margin (int): scale to ``target / 2**margin`` instead, leaving
            ``margin`` binades of headroom. Default: ``0``
        granularity (str): what one scale covers: ``"tensor"``, the whole
            tensor; ``"row"``, each slice along ``axis`` (one scale per
            output channel of a weight, or per token of an activation); or
            ``"block"``, each block of ``block_size`` consecutive elements
            along ``axis``, the last of which may be shorter.
            Default: ``"tensor"``
        axis (int): the axis that ``"row"`` reduces along and that
            ``"block"`` cuts into blocks. Default: ``-1``
        block_size (int, optional): the block length, which ``"block"``
            needs and the other granularities refuse. Default: ``None``
        pow2 (bool): round each scale down to a power of two, so that
            multiplying and dividing by it are exact (barring overflow and
            underflow): the only rounding left is the format's.
            Default: ``False``
        carrier (torch.dtype, optional): ``torch.float64`` runs every step in
            binary64 whatever ``x``'s dtype; ``torch.float32`` names binary32,
            which a float64 tensor refuses. Default: ``None``, ``x``'s own
            carrier.
        eps (float): the smallest magnitude the scale is computed from, which
            bounds the scale of a nearly zero tensor. Default: ``1e-30``

    Raises:
        TypeError: for a format that is not a ``BinaryK``, ``SuperFP`` or
            ``FixedPoint``, or a ``carrier`` that is not a ``torch.dtype``.
        ValueError: for a ``target`` or ``eps`` that is not positive and
            finite, an unknown ``granularity``, a ``block_size`` that does not
            fit the granularity, or a ``carrier`` dtype that is not a carrier.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK, SaturationMode
        >>> from mptorch.quant import Quant, ScaledQuant
        >>> e5m2 = BinaryK(8, 3, bias=15, saturation=SaturationMode.SAT_FINITE)
        >>> g = torch.tensor([1e-6, -3e-7, 2e-8])
        >>> Quant(e5m2)(g)  # all below E5M2's smallest value, 2**-16
        tensor([0., 0., 0.])
        >>> ScaledQuant(e5m2, target=57344.0)(g)  # OCP E5M2's largest value
        tensor([ 1.0000e-06, -2.8571e-07,  1.7857e-08])
    """

    fmt: Number
    rounding: RoundMode = RoundMode.RNE
    _: KW_ONLY
    target: float | None = None
    margin: int = 0
    granularity: str = "tensor"
    axis: int = -1
    block_size: int | None = None
    pow2: bool = False
    carrier: torch.dtype | None = None
    eps: float = 1e-30
    _quant: Quant = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        fmt = self.fmt
        if isinstance(fmt, BlockFormat):
            raise TypeError(
                "ScaledQuant takes a BinaryK, SuperFP or FixedPoint format, and a block format "
                "scales each of its blocks itself: use Quant(fmt) or BlockQuant(fmt)"
            )
        if not isinstance(fmt, BinaryK | SuperFP | FixedPoint):
            raise TypeError(
                "ScaledQuant takes a BinaryK, SuperFP or FixedPoint format, "
                f"got {type(fmt).__name__}"
            )
        target = self.target
        if target is None:
            target = fmt.max_value if isinstance(fmt, FixedPoint) else fmt.max_finite
        target = float(target)
        if not (math.isfinite(target) and target > 0):
            raise ValueError(f"target must be a positive finite value, got {self.target}")
        object.__setattr__(self, "target", target)
        if not (math.isfinite(self.eps) and self.eps > 0):
            raise ValueError(f"eps must be a positive finite value, got {self.eps}")
        if self.granularity not in _GRANULARITIES:
            raise ValueError(
                f"granularity is one of {', '.join(map(repr, _GRANULARITIES))}, "
                f"got {self.granularity!r}"
            )
        if self.granularity == "block":
            bs = self.block_size
            if not isinstance(bs, int) or isinstance(bs, bool) or bs < 1:
                raise ValueError(
                    f"granularity='block' needs a block_size of at least 1, got {bs!r}"
                )
        elif self.block_size is not None:
            raise ValueError(
                f"block_size belongs to granularity='block', and this one is {self.granularity!r}"
            )
        _checked_carrier(self.carrier)
        # The tensor the format rounds is already in the carrier, so the inner
        # quantizer names none, and it rounds that tensor in place.
        object.__setattr__(self, "_quant", Quant(fmt, self.rounding, inplace=True))

    def _scale(self, amax: torch.Tensor) -> torch.Tensor:
        """The scale of each slice whose largest magnitude is ``amax``."""
        assert self.target is not None  # resolved by __post_init__
        s = self.target / amax.clamp_min(self.eps)
        if self.margin:
            s = s / 2.0**self.margin
        if self.pow2:
            # the largest power of two at most s; log2 may round onto the
            # power of two above or below, which the two selects correct
            p = torch.exp2(torch.floor(torch.log2(s)))
            p = torch.where(p > s, p * 0.5, p)
            s = torch.where(p * 2 <= s, p * 2, p)
        ok = torch.isfinite(amax) & (amax > 0) & torch.isfinite(s) & (s > 0)
        return torch.where(ok, s, 1.0)

    def _views(self, t: torch.Tensor) -> list[torch.Tensor]:
        """Views of ``t`` that together cover it once, one per scale or
        tensor of scales: ``t`` itself, ``t`` with the scaled axis last, or
        its full blocks and its shorter last block. The input, the carrier
        tensor and the result take the same views, so they line up."""
        if self.granularity == "tensor":
            return [t]
        moved = t.movedim(self.axis, -1)
        if self.granularity == "row":
            return [moved]
        bs = self.block_size
        assert bs is not None  # checked by __post_init__
        length = moved.shape[-1]
        full = length // bs * bs
        views = []
        if full:
            views.append(moved.narrow(-1, 0, full).unflatten(-1, (full // bs, bs)))
        if full < length:
            views.append(moved.narrow(-1, full, length - full))
        return views

    def _largest(self, view: torch.Tensor) -> torch.Tensor:
        """``max |view|`` per scale, with as many dimensions as ``view``: a
        tensor of scales broadcasts against its view, and torch's type
        promotion then computes in the scales' dtype, the carrier, even for a
        narrower view (a 0-d scale would leave the product in the view's
        dtype)."""
        if self.granularity == "tensor":
            return _amax(view).reshape((1,) * view.dim())
        return _maxabs(view)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        wide, _ = _call_carrier(self.carrier, dtype)
        carrier = torch.float64 if wide else torch.float32
        if not x.numel():
            return torch.empty_like(x, memory_format=torch.contiguous_format)
        # The largest magnitudes are read from x itself, whose values the
        # carrier holds exactly. The product goes straight from x into the
        # carrier tensor y, and the quotient straight from y into the result,
        # where torch's op computes in one dtype and stores in another at full
        # speed: on CUDA, and wherever the two dtypes are the same. That makes four
        # passes over the tensor (the largest magnitudes, the product, the
        # rounding in place, the quotient). On the CPU an op whose operands'
        # dtypes differ runs a scalar casting loop, slower than converting in
        # a pass of its own (2^24 bfloat16 elements: 308 ms against 165 on one
        # thread), so a narrower tensor is converted there, as it is on MPS,
        # in two passes more.
        in_op = dtype is carrier or x.device.type == "cuda"
        sources = self._views(x)
        scales = [self._scale(self._largest(v).to(carrier)) for v in sources]
        if in_op:
            y = torch.empty_like(x, dtype=carrier, memory_format=torch.contiguous_format)
            targets = self._views(y)
            for src, dst, s in zip(sources, targets, scales, strict=True):
                _mul_into(src, s, dst)
        else:
            y = x.to(carrier, memory_format=torch.contiguous_format, copy=True)
            targets = self._views(y)
            for dst, s in zip(targets, scales, strict=True):
                dst.mul_(s)
        self._quant(y)
        # binary64 into float16 or bfloat16 takes _narrowed's pass, since
        # torch's own conversion would round twice
        if not in_op or dtype is carrier or (wide and dtype is not torch.float32):
            for dst, s in zip(targets, scales, strict=True):
                dst.div_(s)
            if dtype is carrier:
                return y
            return _narrowed(y, dtype) if wide else y.to(dtype)
        out = torch.empty_like(y, dtype=dtype)
        for dst, res, s in zip(targets, self._views(out), scales, strict=True):
            _div_into(dst, s, res)
        return out


# Dynamo does not trace an `out=` tensor that is not contiguous, as a row along
# another axis or a block is, so under torch.compile those views take a copy
# and an in-place op instead, which compute the same values (and which
# Inductor fuses into one loop anyway).


def _mul_into(src: torch.Tensor, s: torch.Tensor, dst: torch.Tensor) -> None:
    """``dst = src * s``, computed in ``dst``'s dtype, the carrier."""
    if torch.compiler.is_compiling() and not dst.is_contiguous():
        dst.copy_(src).mul_(s)
    else:
        torch.mul(src, s, out=dst)


def _div_into(src: torch.Tensor, s: torch.Tensor, dst: torch.Tensor) -> None:
    """``dst = src / s``, computed in the carrier and rounded once into
    ``dst``'s narrower dtype."""
    if torch.compiler.is_compiling() and not dst.is_contiguous():
        dst.copy_(torch.div(src, s))
    else:
        torch.div(src, s, out=dst)
