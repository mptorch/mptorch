"""Per-tensor scaling: an elementwise format with a power-of-two scale per tensor.

:class:`TensorQuant` multiplies a tensor by ``2**s``, rounds it to a
:class:`mptorch.BinaryK`, :class:`mptorch.SuperFP` or :class:`mptorch.FixedPoint`
format, and multiplies the result by ``2**-s``. ``s`` puts the tensor's largest
magnitude in the format's top binade (Perez et al., arXiv:2309.17224, Eq. 3);
for a fixed-point format that is a per-tensor choice of its fractional length,
as in dynamic fixed point. A :class:`mptorch.BlockFormat` carries its own scales
(:mod:`mptorch.quant.block`).

``s`` is read from ``max |x|`` on the host, as a block format's tensor scale is,
so a call on a GPU synchronizes with it. Both multiplications are exact in the
carrier: binary64 for a float64 tensor or ``carrier=torch.float64``, and
binary32 otherwise.
"""

import math
from dataclasses import KW_ONLY, dataclass, field
from fractions import Fraction
from functools import lru_cache

import torch

from mptorch.number import (
    _BINARY64,
    BinaryK,
    BlockFormat,
    FixedPoint,
    RoundMode,
    SuperFP,
    _binaryK_extent,
    _binaryK_label,
    _minifloat,
    _superfp_extent,
    _superfp_label,
)

from .block import _amax
from .mac import Quant
from .ops import _NARROW_STORAGE, _call_carrier, _checked_carrier, _narrowed

__all__ = ["TensorQuant", "max_finite"]


@lru_cache(maxsize=256)
def max_finite(fmt: BinaryK | SuperFP | FixedPoint) -> float:
    """The largest finite value of a :class:`mptorch.BinaryK`, :class:`mptorch.SuperFP`
    or :class:`mptorch.FixedPoint`.

    For a binaryK or a superfp it depends on the saturation mode. Under
    ``SAT_FINITE`` the top code is a finite value. Under the other modes the top
    code is the infinity, so the largest finite value is one code below it. An
    unsigned binaryK also gives up a code for its NaN. A fixed-point format has
    no infinity, so it is just ``fmt.max_value``.

    Example::

        >>> from mptorch import E4M3, E5M2, BinaryK, FixedPoint, SaturationMode
        >>> from mptorch.quant import max_finite
        >>> max_finite(E4M3), max_finite(E5M2)
        (448.0, 98304.0)
        >>> max_finite(BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE))
        480.0
        >>> max_finite(FixedPoint(8, 4))
        7.9375
    """
    if isinstance(fmt, FixedPoint):
        return fmt.max_value
    if isinstance(fmt, SuperFP):
        label = _superfp_label(fmt.man_bits, fmt.exp_bits, fmt.normal_binades, fmt.bias)
        ext = _superfp_extent(
            label,
            fmt.man_bits,
            fmt.exp_bits,
            fmt.normal_binades,
            fmt.bias,
            fmt.saturation,
            _BINARY64,
        )
    elif isinstance(fmt, BinaryK):
        mf = _minifloat(fmt)
        label = _binaryK_label(fmt.K, fmt.P, mf.bias, fmt.is_signed, fmt.subnormals)
        ext = _binaryK_extent(
            label,
            mf.man_bits,
            mf.exp_bits,
            mf.bias,
            fmt.is_signed,
            fmt.saturation,
            fmt.subnormals,
            _BINARY64,
        )
    else:
        raise TypeError(
            f"max_finite takes a BinaryK, a SuperFP or a FixedPoint, got {type(fmt).__name__}"
        )
    top = Fraction((1 << ext.man_bits) + ext.top_code) * Fraction(2) ** (ext.top_exp - ext.man_bits)
    return float(top)


def _finite_amax(x: torch.Tensor) -> float:
    """The largest finite ``|x|``, as a Python float (0 if there are none).

    This is the same reduction ``block.py`` uses to set a block format's
    per-tensor scale. Reading the value back to the host synchronizes with the
    GPU.
    """
    amax = _amax(x).item()
    if math.isfinite(amax):
        return amax
    a = x.detach().abs()
    zero = torch.zeros((), dtype=a.dtype, device=a.device)
    return torch.where(torch.isfinite(a), a, zero).amax().item()


def _scale_exponent(amax: float, top: float) -> int:
    """The largest ``s`` with ``amax * 2**s <= top``, for a positive ``amax``.

    With ``amax = amax_mantissa * 2**amax_exponent`` and
    ``top = top_mantissa * 2**top_exponent``, both mantissas in [0.5, 1), it is
    ``top_exponent - amax_exponent``, less one when ``amax_mantissa > top_mantissa``.
    """
    amax_mantissa, amax_exponent = math.frexp(amax)
    top_mantissa, top_exponent = math.frexp(top)
    return top_exponent - amax_exponent - (amax_mantissa > top_mantissa)


def _mul_pow2(x: torch.Tensor, s: int) -> torch.Tensor:
    """Return ``x * 2**s``, rounded once, even when ``2**s`` doesn't fit in
    ``x``'s dtype.

    Outside the dtype's normal exponents ([-126, 127] for float32, [-1022, 1023]
    for float64) ``2**s`` would be inf or 0, so it is applied in steps: the
    leftover part of ``s`` first, then the largest power of two the dtype holds,
    as many times as needed. Only the last step can round, so the result
    matches a single multiplication.
    """
    lo, hi = (-1022, 1023) if x.dtype is torch.float64 else (-126, 127)
    step = hi if s > 0 else lo
    whole = 0
    while not lo <= s <= hi:
        s -= step
        whole += 1
    x = x * 2.0**s
    for _ in range(whole):
        x = x * 2.0**step
    return x


@dataclass(frozen=True)
class TensorQuant:
    """Quantize a tensor to a format, using a power-of-two scale chosen per tensor.

    A call returns ``Quant(fmt, rounding)(x * 2**s) * 2**-s``, with the same
    dtype and shape as ``x``. ``s`` is the largest integer that keeps
    ``max |x| * 2**s`` at or below the format's largest finite value
    (:func:`max_finite`). This lets a tensor of small values use the format's
    whole range instead of rounding to zero (see the example).

    - Only finite elements count toward ``max |x|``. A tensor that is empty, all
      zeros, or entirely non-finite is not scaled.
    - Finding ``s`` reads ``max |x|`` back to the host, so each call on a GPU
      synchronizes.
    - A :class:`mptorch.FixedPoint` format has no infinity, so an infinite
      element comes back as ``max_finite(fmt) * 2**-s`` rather than ``inf``.

    It can be used in any ``*_quant`` slot of a ``QAffineFormats`` or
    ``QMatmulFormats``, just like a :class:`Quant`.

    Args:
        fmt (BinaryK, SuperFP or FixedPoint): the format to round to.
        rounding (RoundMode): the rounding mode. Default: ``RoundMode.RNE``
        carrier (torch.dtype, optional): as for :class:`Quant`. Default: ``None``

    Raises:
        TypeError: for a format other than a ``BinaryK``, a ``SuperFP`` or a
            ``FixedPoint``, or a ``carrier`` that is not a ``torch.dtype``.
        ValueError: for a ``carrier`` dtype that names no carrier.

    Example::

        >>> import torch
        >>> from mptorch import E4M3
        >>> from mptorch.quant import Quant, TensorQuant
        >>> x = torch.tensor([2.0**-12, 2.0**-14])
        >>> Quant(E4M3)(x)
        tensor([0., 0.])
        >>> TensorQuant(E4M3)(x)
        tensor([2.4414e-04, 6.1035e-05])
    """

    fmt: BinaryK | SuperFP | FixedPoint
    rounding: RoundMode = RoundMode.RNE
    _: KW_ONLY
    carrier: torch.dtype | None = None
    _quant: Quant = field(init=False, repr=False, compare=False)
    _top: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.fmt, BlockFormat):
            raise TypeError(
                "TensorQuant takes a BinaryK, a SuperFP or a FixedPoint: a BlockFormat "
                "carries its own scales (BlockQuant)"
            )
        if not isinstance(self.fmt, BinaryK | SuperFP | FixedPoint):
            raise TypeError(
                "TensorQuant takes a BinaryK, a SuperFP or a FixedPoint, "
                f"got {type(self.fmt).__name__}"
            )
        _checked_carrier(self.carrier)
        object.__setattr__(self, "_quant", Quant(self.fmt, self.rounding))
        object.__setattr__(self, "_top", max_finite(self.fmt))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        _, widen = _call_carrier(self.carrier, x.dtype)
        if widen:
            src = x.double()
        elif x.dtype in _NARROW_STORAGE:
            src = x.float()
        else:
            src = x
        amax = _finite_amax(src) if src.numel() else 0.0
        top = min(self._top, torch.finfo(src.dtype).max)
        s = _scale_exponent(amax, top) if amax > 0 else 0
        out = _mul_pow2(self._quant(_mul_pow2(src, s)), -s)
        if out.dtype is x.dtype:
            return out
        return _narrowed(out, x.dtype) if widen else out.to(x.dtype)
