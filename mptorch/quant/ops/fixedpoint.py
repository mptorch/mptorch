"""Fixed point's flat functions: the elementwise quantizers and the GEMMs.

The GEMMs, :func:`fixedpoint_matmul` (a split mac) and
:func:`fixedpoint_matmul_fma` (a fused one), each come after the builder of
their ``_GemmSpec``. Their arithmetic is the C++ header
``csrc/common/fixedpoint_gemm.h``'s, and the rules that choose how a call runs
and what it refuses are ``mptorch.number.fixedpoint``'s ``_fixedpoint_gemm``,
which the C++ host applies too.
"""

import torch

from mptorch import FixedPoint, Overflow, RoundMode
from mptorch.number.fixedpoint import (
    _fixedpoint_findings,
    _fixedpoint_gemm,
    _fixedpoint_gemm_wide,
    _fixedpoint_label,
    _fixedpoint_storage_findings,
    check_fixedpoint_carrier,
    check_fixedpoint_storage,
)

from ._checks import (
    _BELOW_BINARY64,
    _CARRIER,
    _NARROW_STORAGE,
    _call_carrier,
    _checked_carrier,
    _narrowed,
    _quantizer_operand,
    _refuse_widening_in_place,
)
from ._operands import _gemm_nd
from ._spec import _GemmSpec, _quant_spec, _QuantSpec


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


# --- the GEMMs --------------------------------------------------------------------

# The operand dtypes a binary32 call stores its result in, besides float32.
_NARROWER = (torch.float16, torch.bfloat16)


def _checked_overflow(overflow: Overflow) -> bool:
    """``overflow`` as the ops' ``wrap`` flag."""
    if not isinstance(overflow, Overflow):
        raise TypeError(f"overflow must be an Overflow, got {overflow!r}")
    return overflow is Overflow.WRAP


def _check_gemm_stored(
    wl: int, fl: int, is_signed: bool, symmetric: bool, *, storage: torch.dtype
) -> None:
    """Check storing a fixed-point GEMM's result in ``storage``: under the rule
    for a computed result, unless the result is wide there, in which case the
    op computes it into float64 instead and nothing is stored in ``storage``."""
    if not _fixedpoint_gemm_wide(FixedPoint(wl, fl, is_signed=is_signed), storage):
        check_fixedpoint_storage(wl, fl, is_signed, symmetric, storage=storage, elementwise=False)


def _fixedpoint_gemm_spec(
    op, args: tuple, mul: FixedPoint | None, acc: FixedPoint, wrap: bool, carrier
) -> _GemmSpec:
    """The ``_GemmSpec`` of a fixed-point GEMM call, its rules found once.

    Each carrier's finding is the GEMM's rule (`_fixedpoint_gemm`), which
    refuses what neither of the call's paths can compute; binary64's refusal
    is one no carrier escapes, so it raises here. A binary32 call also gets
    the finding on storing a float32 result, when the result is not wide in
    float32; ``stored`` checks the narrower dtypes on each call, and ``wide``
    lists the dtypes whose result is float64.
    """
    fields = (acc.wl, acc.fl, acc.is_signed, acc.symmetric)
    rule64 = _fixedpoint_gemm(mul, acc, wrap, torch.float64).error
    if rule64 is not None:
        raise ValueError(rule64)
    rule32 = _fixedpoint_gemm(mul, acc, wrap, torch.float32).error
    b32: tuple = ((rule32, None),) if rule32 is not None else ()
    if rule32 is None and not _fixedpoint_gemm_wide(acc, torch.float32):
        found = _fixedpoint_storage_findings(*fields, torch.float32, False)
        if found != (None, None):
            b32 = (found,)
    wide = frozenset(d for d in (torch.float32, *_NARROWER) if _fixedpoint_gemm_wide(acc, d))
    label = _fixedpoint_label(*fields)
    return _GemmSpec(
        op,
        args,
        ((_check_gemm_stored, fields),),
        carrier=_checked_carrier(carrier),
        findings=(b32, ()),
        wide=wide,
        wide_label=f"{label} has {acc.mag_bits} magnitude bits",
    )


def _fixedpoint_spec(
    *,
    mul_wl: int,
    mul_fl: int,
    mul_is_signed: bool = True,
    mul_symmetric: bool = False,
    mul_prng_bits: int = 0,
    acc_wl: int,
    acc_fl: int,
    acc_is_signed: bool = True,
    acc_symmetric: bool = False,
    acc_prng_bits: int = 0,
    rounding_mode: RoundMode = RoundMode.RNE,
    overflow: Overflow = Overflow.SATURATE,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`fixedpoint_matmul`'s formats into a ``_GemmSpec``.

    That function documents the arguments."""
    wrap = _checked_overflow(overflow)
    mul = FixedPoint(
        mul_wl, mul_fl, is_signed=mul_is_signed, symmetric=mul_symmetric, prng_bits=mul_prng_bits
    )
    acc = FixedPoint(
        acc_wl, acc_fl, is_signed=acc_is_signed, symmetric=acc_symmetric, prng_bits=acc_prng_bits
    )
    args = (
        mul_wl,
        mul_fl,
        mul_is_signed,
        mul_symmetric,
        acc_wl,
        acc_fl,
        acc_is_signed,
        acc_symmetric,
        rounding_mode.value,
        mul_prng_bits,
        acc_prng_bits,
        wrap,
    )
    op = torch.ops.mptorch.custom_matmul_fixedpoint.default
    return _fixedpoint_gemm_spec(op, args, mul, acc, wrap, carrier)


def fixedpoint_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    mul_wl: int,
    mul_fl: int,
    mul_is_signed: bool = True,
    mul_symmetric: bool = False,
    mul_prng_bits: int = 0,
    acc_wl: int,
    acc_fl: int,
    acc_is_signed: bool = True,
    acc_symmetric: bool = False,
    acc_prng_bits: int = 0,
    rounding_mode: RoundMode = RoundMode.RNE,
    overflow: Overflow = Overflow.SATURATE,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM in fixed point, with the product and the sum rounded separately.

    Each step of a dot product is ``p = Q_M(a * b)`` and ``s = Q_A(s + p)``,
    from ``s = 0``, in order: the carrier's product rounded to the multiply
    format, and the exact sum rounded to the accumulate format. Every sum is
    exact, so the result depends only on the products and the two
    roundings, never on a carrier's rounding of the sum. The product is the
    carrier's, as in every mptorch mac: exact when the operands are values of
    fixed-point formats whose significands fit the carrier together (8- to
    12-bit operands in binary32), and for any operands with
    ``carrier=torch.float64`` up to 53 bits.

    The sum's rounding saturates to the accumulate format's range, or with
    ``overflow=Overflow.WRAP`` wraps around modulo ``2**acc_wl``, as integer
    hardware does. The product's rounding always saturates.

    Two paths compute the same bits: an int64 accumulator, which is the
    definition and holds accumulate formats of up to 53 magnitude bits, and,
    where the carrier computes every step exactly, the shipped fixed-point
    casts in the carrier, which is faster. The call picks one from the
    formats alone (see :doc:`/kernels/arithmetic`, "Fixed point").

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        mul_wl (int): the multiply format's word length, its sign bit included.
        mul_fl (int): its fractional length; its step is ``2**-mul_fl``.
        mul_is_signed (bool): whether it has a sign bit. Default: ``True``
        mul_symmetric (bool): whether it drops its most negative code.
            Default: ``False``
        mul_prng_bits (int): random bits stochastic rounding draws for the
            product, as :func:`fixedpoint_quantize`'s ``prng_bits``. Default: 0
        acc_wl (int): the accumulate format's word length.
        acc_fl (int): its fractional length.
        acc_is_signed (bool): whether it has a sign bit. Default: ``True``
        acc_symmetric (bool): whether it drops its most negative code.
            Default: ``False``
        acc_prng_bits (int): random bits drawn for the sum, at most the
            carrier's mantissa (23 in binary32, 52 in binary64). Default: 0
        rounding_mode (RoundMode): the mode of both roundings, any
            :class:`~mptorch.RoundMode`. Default: ``RoundMode.RNE``
        overflow (Overflow): what the sum does past the accumulate format's
            range. Default: ``Overflow.SATURATE``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, with the shape ``torch.matmul`` would give: in
        the operands' dtype, or in float64 when the accumulate format has more
        significant bits than that dtype holds (24 for float32, 11 for
        float16, 8 for bfloat16), so that no value of the format is rounded
        again when it is stored.

    Raises:
        ValueError: for a format ``FixedPoint`` refuses, a multiply format the
            carrier's cast does not round correctly, an accumulate format past
            53 magnitude bits or outside float64's range, more random bits than
            the carrier's draw holds, wrap-around of a symmetric format, or a
            sum past the integer accumulator's 62 bits.
        TypeError: if ``overflow`` is not an ``Overflow``, or ``carrier`` is
            neither a ``torch.dtype`` nor ``None``.
        RuntimeError: if the inner dimensions do not match, the operands
            differ in dtype or device, or an operand requires grad under grad
            mode.

    Warns:
        FormatRangeWarning: if a result stored in the operands' dtype can hold
            a value that dtype does not represent.

    Example::

        Products and sums on a grid of quarters, where ``a @ a`` is
        ``[[2.75, -0.3125], [-0.5, 3.0625]]``: ``0.5 * 1.25 = 0.625`` rounds to
        ``0.5`` (a tie, to even) and ``-0.9375`` to ``-1``::

        >>> a = torch.tensor([[0.5, 1.25], [2.0, -0.75]])
        >>> fixedpoint_matmul(a, a, mul_wl=8, mul_fl=2, acc_wl=12, acc_fl=2)
        tensor([[ 2.7500, -0.5000],
                [-0.5000,  3.0000]])
        >>> fixedpoint_matmul(a, a, mul_wl=16, mul_fl=8, acc_wl=32, acc_fl=8).dtype
        torch.float64
    """
    spec = _fixedpoint_spec(
        mul_wl=mul_wl,
        mul_fl=mul_fl,
        mul_is_signed=mul_is_signed,
        mul_symmetric=mul_symmetric,
        mul_prng_bits=mul_prng_bits,
        acc_wl=acc_wl,
        acc_fl=acc_fl,
        acc_is_signed=acc_is_signed,
        acc_symmetric=acc_symmetric,
        acc_prng_bits=acc_prng_bits,
        rounding_mode=rounding_mode,
        overflow=overflow,
        carrier=carrier,
    )
    return _gemm_nd(spec, a, b, trans_a, trans_b)


def _fixedpoint_fma_spec(
    *,
    fma_wl: int,
    fma_fl: int,
    fma_is_signed: bool = True,
    fma_symmetric: bool = False,
    fma_prng_bits: int = 0,
    rounding_mode: RoundMode = RoundMode.RNE,
    overflow: Overflow = Overflow.SATURATE,
    carrier: torch.dtype | None = None,
) -> _GemmSpec:
    """Resolve :func:`fixedpoint_matmul_fma`'s format into a ``_GemmSpec``.

    That function documents the arguments."""
    wrap = _checked_overflow(overflow)
    fma = FixedPoint(
        fma_wl, fma_fl, is_signed=fma_is_signed, symmetric=fma_symmetric, prng_bits=fma_prng_bits
    )
    args = (
        fma_wl,
        fma_fl,
        fma_is_signed,
        fma_symmetric,
        rounding_mode.value,
        fma_prng_bits,
        wrap,
    )
    op = torch.ops.mptorch.custom_matmul_fixedpoint_fma.default
    return _fixedpoint_gemm_spec(op, args, None, fma, wrap, carrier)


def fixedpoint_matmul_fma(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    trans_a: bool = False,
    trans_b: bool = False,
    fma_wl: int,
    fma_fl: int,
    fma_is_signed: bool = True,
    fma_symmetric: bool = False,
    fma_prng_bits: int = 0,
    rounding_mode: RoundMode = RoundMode.RNE,
    overflow: Overflow = Overflow.SATURATE,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """Quantized GEMM in fixed point with a fused multiply-add: one rounding per step.

    The fused counterpart of :func:`fixedpoint_matmul`: each step is
    ``s = Q_F(a * b + s)``, the exact sum of the carrier's product and the
    running sum rounded once to the format. It always runs the int64
    accumulator, which aligns the carrier's product exactly to the format's
    grid.

    Args:
        a (Tensor): the left operand; float32, float64, float16 or bfloat16.
        b (Tensor): the right operand, of ``a``'s dtype and device.
        trans_a (bool): read ``a`` transposed in its last two dimensions.
            Default: ``False``
        trans_b (bool): read ``b`` transposed in its last two dimensions.
            Default: ``False``
        fma_wl (int): the format's word length, its sign bit included.
        fma_fl (int): its fractional length.
        fma_is_signed (bool): whether it has a sign bit. Default: ``True``
        fma_symmetric (bool): whether it drops its most negative code.
            Default: ``False``
        fma_prng_bits (int): random bits drawn for each step, at most the
            carrier's mantissa (23 in binary32, 52 in binary64). Default: 0
        rounding_mode (RoundMode): the step's rounding mode. Default:
            ``RoundMode.RNE``
        overflow (Overflow): what the sum does past the format's range.
            Default: ``Overflow.SATURATE``
        carrier (torch.dtype, optional): as for :func:`binaryK_matmul`.
            Default: ``None``, the operands' own carrier.

    Returns:
        Tensor: the product, in the operands' dtype or, for a format wider than
        it, in float64, as for :func:`fixedpoint_matmul`.

    Raises:
        ValueError: as for :func:`fixedpoint_matmul`, for the one format.
        TypeError: as for :func:`fixedpoint_matmul`.
        RuntimeError: as for :func:`fixedpoint_matmul`.

    Warns:
        FormatRangeWarning: as for :func:`fixedpoint_matmul`.

    Example::

        >>> a = torch.tensor([[0.5, 1.25], [2.0, -0.75]])
        >>> fixedpoint_matmul_fma(a, a, fma_wl=12, fma_fl=2)
        tensor([[ 2.7500, -0.5000],
                [-0.5000,  3.0000]])
    """
    spec = _fixedpoint_fma_spec(
        fma_wl=fma_wl,
        fma_fl=fma_fl,
        fma_is_signed=fma_is_signed,
        fma_symmetric=fma_symmetric,
        fma_prng_bits=fma_prng_bits,
        rounding_mode=rounding_mode,
        overflow=overflow,
        carrier=carrier,
    )
    return _gemm_nd(spec, a, b, trans_a, trans_b)
