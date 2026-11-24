"""Fixed point's flat functions: the elementwise quantizers. There is no fixed-point GEMM."""

import torch

from mptorch import RoundMode
from mptorch.number.fixedpoint import (
    _fixedpoint_findings,
    _fixedpoint_storage_findings,
    check_fixedpoint_carrier,
    check_fixedpoint_storage,
)

from ._checks import (
    _BELOW_BINARY64,
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _narrowed,
    _quantizer_operand,
    _refuse_widening_in_place,
)
from ._spec import _quant_spec, _QuantSpec


def fixedpoint_quantize(
    x: torch.Tensor,
    wl: int,
    fl: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    symmetric: bool = False,
    rounding_mode: RoundMode = RoundMode.RNE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a fixed-point format.

    The format's values are the integers that a ``wl``-bit word holds, times
    the step ``2**-fl``: ``[-2**(wl - 1 - fl), 2**(wl - 1 - fl) - 2**-fl]`` in
    two's complement, one step narrower at the bottom with ``symmetric``, and
    ``[0, 2**(wl - fl) - 2**-fl]`` unsigned (see :class:`mptorch.FixedPoint`).
    A value beyond the range, including an infinite one, saturates to the end
    of the range on its side, since a fixed-point word has no code for
    infinity; hence there is no ``saturation_mode``. NaN inputs pass through.

    Dtypes, the carrier, the storage check, ``prng_bits`` and
    differentiability are as for :func:`binaryK_quantize`.

    Args:
        x (Tensor): the tensor to round; float32, float64, float16 or
            bfloat16.
        wl (int): word length in bits, including the sign bit.
        fl (int): fractional length; the step between values is ``2**-fl``.
        prng_bits (int): the number of random bits the stochastic rounding
            modes draw below the step; ignored by the other modes.
            Default: ``0``
        is_signed (bool): whether the format has a sign bit. Default: ``True``
        symmetric (bool): drop the most negative code, so that the range is
            symmetric about zero; signed formats only. Default: ``False``
        rounding_mode (RoundMode): how a value that falls between two of the
            format's values is rounded. Default: ``RoundMode.RNE``
        carrier (torch.dtype, optional): as for :func:`binaryK_quantize`.
            Default: ``None``, ``x``'s own carrier.

    Returns:
        Tensor: a new tensor of ``x``'s shape and dtype holding the rounded
        values.

    Raises:
        ValueError: if the carrier cannot hold the format (more magnitude
            bits than its significand has, ``prng_bits`` included, a largest
            value below its smallest normal, or an end of the range beyond
            its largest finite value), for an invalid layout (``wl`` too
            short, a symmetric unsigned format, a negative ``prng_bits``), if
            the storage dtype can store none of the format's nonzero values,
            or if ``carrier`` is a dtype that names no carrier or is narrower
            than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode.

    Warns:
        FormatRangeWarning: if the format's range exceeds the carrier's, or a
            result at an edge of the format's range is not a value of a
            narrower storage dtype.

    Example::

        >>> x = torch.tensor([1.1, 3.3, -9.0, 0.03])
        >>> fixedpoint_quantize(x, wl=8, fl=4)
        tensor([ 1.1250,  3.3125, -8.0000,  0.0000])

    With ``fl=4`` the step is 1/16: 1.1 and 3.3 round to the nearest
    sixteenth, -9.0 saturates to the bottom of the range, -8, and 0.03 is
    nearer to zero than to the step.
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_fixedpoint_carrier(wl, fl, is_signed, symmetric, prng_bits, carrier=_CARRIER[wide])
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_fixedpoint_storage(
            wl,
            fl,
            is_signed,
            symmetric,
            storage=dtype,
            elementwise=True,
        )

    out = torch.ops.mptorch.fixedpoint_quant.default(
        _quantizer_operand(x, widen),
        wl,
        fl,
        prng_bits,
        is_signed,
        symmetric,
        rounding_mode.value,
    )
    return _narrowed(out, dtype) if widen else out


def fixedpoint_quantize_(
    x: torch.Tensor,
    wl: int,
    fl: int,
    prng_bits: int = 0,
    is_signed: bool = True,
    symmetric: bool = False,
    rounding_mode: RoundMode = RoundMode.RNE,
    *,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Round every element of ``x`` to a fixed-point format, in place.

    The in-place version of :func:`fixedpoint_quantize`: the same arguments,
    checks and result, written over ``x``, with no output allocated. Its
    purpose and the inputs it refuses (a tensor that is not contiguous, a CUDA
    view that does not start on a 16-byte boundary, a ``carrier`` wider than
    ``x``) are as for :func:`binaryK_quantize_`.

    Args:
        x (Tensor): the tensor to round and overwrite; float32, float64,
            float16 or bfloat16, contiguous. The format and mode arguments
            between it and ``carrier`` are :func:`fixedpoint_quantize`'s.
        carrier (torch.dtype, optional): ``None`` or the carrier ``x`` already
            has. Default: ``None``

    Returns:
        Tensor: ``x`` itself, holding the rounded values.

    Raises:
        ValueError: as for :func:`fixedpoint_quantize`, and if ``carrier`` is
            wider than ``x``.
        TypeError: if ``carrier`` is neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if ``x`` requires grad under grad mode, is not
            contiguous, or is a CUDA view off a 16-byte boundary.

    Warns:
        FormatRangeWarning: as for :func:`fixedpoint_quantize`.

    Example::

        >>> x = torch.tensor([1.1, 3.3, -9.0, 0.03])
        >>> fixedpoint_quantize_(x, wl=8, fl=4)
        tensor([ 1.1250,  3.3125, -8.0000,  0.0000])
        >>> x
        tensor([ 1.1250,  3.3125, -8.0000,  0.0000])
    """
    dtype = x.dtype
    wide, widen = _call_carrier(carrier, dtype)
    check_fixedpoint_carrier(wl, fl, is_signed, symmetric, prng_bits, carrier=_CARRIER[wide])
    if dtype in (_BELOW_BINARY64 if wide else _NARROW_STORAGE):
        check_fixedpoint_storage(
            wl,
            fl,
            is_signed,
            symmetric,
            storage=dtype,
            elementwise=True,
        )
    if widen:
        _refuse_widening_in_place("fixedpoint_quantize", dtype)

    torch.ops.mptorch.fixedpoint_quant_.default(
        x,
        wl,
        fl,
        prng_bits,
        is_signed,
        symmetric,
        rounding_mode.value,
    )
    return x


def _fixedpoint_quant_spec(
    wl: int,
    fl: int,
    prng_bits: int,
    is_signed: bool,
    symmetric: bool,
    rounding_mode: RoundMode,
    *,
    carrier: torch.dtype | None,
    in_place: bool,
) -> _QuantSpec:
    """Resolve a :func:`fixedpoint_quantize` call (or a
    :func:`fixedpoint_quantize_` one) once, for :class:`mptorch.quant.Quant`."""
    ops = (torch.ops.mptorch.fixedpoint_quant.default, torch.ops.mptorch.fixedpoint_quant_.default)
    fields = (wl, fl, is_signed, symmetric)
    return _quant_spec(
        ops,
        (wl, fl, prng_bits, is_signed, symmetric, rounding_mode.value),
        "fixedpoint_quantize",
        in_place,
        carrier,
        lambda c: _fixedpoint_findings(*fields, prng_bits, c),
        lambda dtype, elementwise: _fixedpoint_storage_findings(*fields, dtype, elementwise),
    )
