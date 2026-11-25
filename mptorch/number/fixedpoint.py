"""Fixed-point formats, and checks that a carrier and a storage dtype can hold one.

:class:`FixedPoint` describes the format. It has no exponent field, so its
checks use neither `_carrier`'s `_Extent` nor `_storage`'s GEMM rule: they
compare its precision and the two ends of its range with the carrier's
directly. Its storage rules are the elementwise one and the computed
result's (`_fixedpoint_value_set_findings`). The end of the module holds the
rules of the fixed-point GEMM (phase T): whether the carrier or the integer
accumulator computes a call, what each refuses, and when a result is wide.
"""

import warnings
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache
from typing import NamedTuple

import torch

from ._carrier import _CARRIERS, _Carrier, _pointing_wider
from ._storage import _STORAGE, _in_storage, _Storage
from .base import FormatRangeWarning, Number, _report_per_call

#: The most random bits a stochastic rounding reads: the mantissa of a binary64 draw.
_DRAW_BITS = 52


@dataclass(frozen=True)
class FixedPoint(Number):
    """A fixed-point format: ``wl`` bits, of which ``fl`` are fractional.

    Defined as in Gupta et al., *Deep Learning with Limited Numerical
    Precision* (arXiv:1502.02551). Its values are the integers that a
    ``wl``-bit word holds, times the step ``2**-fl``. With ``n = wl - 1``
    magnitude bits if signed and ``n = wl`` if unsigned, the range is:

    * ``[-2**(n - fl), 2**(n - fl) - 2**-fl]`` if signed (two's complement);
    * ``[-(2**(n - fl) - 2**-fl), 2**(n - fl) - 2**-fl]`` if signed with
      ``symmetric=True``;
    * ``[0, 2**(n - fl) - 2**-fl]`` if unsigned.

    ``fl`` may be any integer: negative for a step larger than 1, and larger
    than ``wl`` for a format whose values are all small fractions.

    Every :class:`RoundMode` applies, as for :class:`BinaryK`. A magnitude
    below the step rounds to zero or to ``2**-fl``, chosen as
    :class:`SubnormalsMode` describes. A result beyond the range, including an
    infinite input, saturates to the end of the range on its side, since a
    fixed-point word has no code for infinity; hence there is no
    ``saturation`` parameter. A NaN input stays a NaN, the only zero is
    unsigned (no cast returns ``-0.0``), and an unsigned format rounds every
    negative value to zero. The object is frozen for the same reason as a
    :class:`BinaryK`.

    Args:
        wl (int): word length in bits, including the sign bit: at least 2 if
            signed, 1 if unsigned.
        fl (int): fractional length; the step between values is ``2**-fl``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        symmetric (bool): drop the most negative code, so that the range is
            symmetric about zero; signed formats only. Default: ``False``.
        prng_bits (int): number of random bits drawn below the step by the
            stochastic rounding modes; ignored by the other modes. Default: 0.

    Raises:
        ValueError: for a ``wl`` that is too short, a symmetric unsigned
            format, a ``prng_bits`` below 0 or above 52 (the most a binary64
            draw holds), or a format that binary64 cannot simulate: more than
            53 magnitude bits, a largest value below binary64's smallest
            normal, or an end of the range beyond binary64's largest finite
            value. Each call checks the format against its own carrier (see
            :class:`FormatRangeWarning`): an elementwise quantizer rounds in
            the carrier's significand, so it also needs ``mag_bits +
            prng_bits`` within the carrier's precision, which a fixed-point
            GEMM's integer accumulator does not.

    Example::

        >>> from mptorch import FixedPoint
        >>> f = FixedPoint(8, 4)  # steps of 1/16
        >>> f.min_value, f.max_value, f.step
        (-8.0, 7.9375, 0.0625)
        >>> FixedPoint(8, 4, symmetric=True).min_value
        -7.9375
    """

    wl: int
    fl: int
    _: KW_ONLY
    is_signed: bool = True
    symmetric: bool = False
    prng_bits: int = 0

    def __post_init__(self) -> None:
        min_wl = 2 if self.is_signed else 1
        if self.wl < min_wl:
            raise ValueError(
                f"FixedPoint needs wl >= {min_wl} "
                f"({'signed' if self.is_signed else 'unsigned'}), got wl={self.wl}"
            )
        if self.symmetric and not self.is_signed:
            raise ValueError("FixedPoint needs is_signed=True to be symmetric")
        if self.prng_bits < 0:
            raise ValueError(f"FixedPoint prng_bits must be non-negative, got {self.prng_bits}")
        if self.prng_bits > _DRAW_BITS:
            raise ValueError(
                f"FixedPoint draws at most {_DRAW_BITS} random bits, the mantissa of a binary64 "
                f"draw, got prng_bits={self.prng_bits}"
            )

        # Raise only for what no use of the format can do. Its random bits are
        # left out: an elementwise call rounds in the carrier's significand and
        # checks the format with them against its own carrier
        # (`check_fixedpoint_carrier`), and a fixed-point GEMM's integer
        # accumulator reads them from a draw of their own (`_fixedpoint_gemm`).
        check_fixedpoint(
            self.wl,
            self.fl,
            self.is_signed,
            self.symmetric,
            0,
            carrier=torch.float64,
            warn=False,
        )

    @property
    def mag_bits(self) -> int:
        """The number of magnitude bits, ``wl`` less the sign bit if there is one."""
        return self.wl - (1 if self.is_signed else 0)

    @property
    def step(self) -> float:
        """The step between neighbouring values, ``2**-fl``."""
        return 2.0**-self.fl

    @property
    def max_value(self) -> float:
        """The largest value, ``(2**mag_bits - 1) * 2**-fl``."""
        return ((1 << self.mag_bits) - 1) * self.step

    @property
    def min_value(self) -> float:
        """The smallest value. It is 0 if unsigned, ``-max_value`` if
        symmetric, and one step below ``-max_value`` otherwise."""
        if not self.is_signed:
            return 0.0
        if self.symmetric:
            return -self.max_value
        return -(1 << self.mag_bits) * self.step


def check_fixedpoint(
    wl: int,
    fl: int,
    is_signed: bool = True,
    symmetric: bool = False,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype = torch.float32,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Check whether a carrier can hold a fixed-point format.

    Behaves as :func:`check_binaryK`.

    Args:
        wl (int): word length, the sign bit included.
        fl (int): fractional length; the step between values is ``2**-fl``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        symmetric (bool): whether the most negative code is dropped.
            Default: ``False``.
        prng_bits (int): random bits drawn for stochastic rounding. Default: 0.
        carrier (torch.dtype): ``torch.float32`` or ``torch.float64``.
            Default: ``torch.float32``.
        warn (bool): whether to issue the warning. Default: ``True``.
        stacklevel (int): passed to ``warnings.warn``, as for
            :func:`check_binaryK`. Default: 3.

    Raises:
        ValueError: for an invalid layout (see :class:`~mptorch.FixedPoint`), more
            magnitude bits than the carrier's significand has (``prng_bits``
            included), a largest value below the carrier's smallest normal, or
            an end of the range beyond the carrier's largest finite value.
    """
    error, warning = _fixedpoint_findings(wl, fl, is_signed, symmetric, prng_bits, carrier)
    if error is not None:
        raise ValueError(error)
    if warn and warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


def check_fixedpoint_carrier(
    wl: int,
    fl: int,
    is_signed: bool = True,
    symmetric: bool = False,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype,
) -> None:
    """The fixed-point counterpart of :func:`check_binaryK_carrier`.

    The arguments are those of :func:`check_fixedpoint` without ``warn`` and
    ``stacklevel``, and ``carrier`` is required.

    Raises:
        ValueError: as :func:`check_fixedpoint`.
    """
    _report_per_call(*_fixedpoint_findings(wl, fl, is_signed, symmetric, prng_bits, carrier))


def check_fixedpoint_storage(
    wl: int,
    fl: int,
    is_signed: bool = True,
    symmetric: bool = False,
    *,
    storage: torch.dtype,
    elementwise: bool = False,
) -> None:
    """The fixed-point counterpart of :func:`check_binaryK_storage`.

    It takes the format's parameters as :func:`check_fixedpoint` does, and the
    same ``storage`` and ``elementwise`` keywords as
    :func:`check_binaryK_storage`. ``elementwise=False`` is the rule for a
    computed result, which can be any of the format's values: a pointwise
    op's, or a fixed-point GEMM's accumulate or fused format (a wide one's
    result is float64 and is not checked: the forward pass's output rounding
    stores it).

    Raises:
        ValueError: if the fixed-point format's largest value is below the
            storage dtype's smallest positive value, so that none of its
            nonzero values can be stored, or, under ``elementwise=False``,
            if it has more significant bits than the storage dtype.
    """
    _report_per_call(
        *_fixedpoint_storage_findings(wl, fl, is_signed, symmetric, storage, elementwise)
    )


# --- how the checks reach their findings -------------------------------------


def _fixedpoint_label(wl: int, fl: int, is_signed: bool, symmetric: bool) -> str:
    """The format written as a constructor call, for messages, leaving out
    ``is_signed`` and ``symmetric`` when they have their default values."""
    return (
        f"FixedPoint(wl={wl}, fl={fl}"
        + ("" if is_signed else ", is_signed=False")
        + (", symmetric=True" if symmetric else "")
        + ")"
    )


def _fixedpoint_layout_error(
    label: str, wl: int, is_signed: bool, symmetric: bool, prng_bits: int
) -> str | None:
    """Return an error for an invalid fixed-point layout, or ``None``.

    :class:`FixedPoint` rejects these in `__post_init__`, but the wrappers in
    `mptorch.quant.ops` that take plain integers come here unchecked.
    """
    if prng_bits < 0:
        return f"{label}: prng_bits must be >= 0, got {prng_bits}"

    # room for the sign bit, if any, and at least one magnitude bit
    min_wl = 2 if is_signed else 1
    if wl < min_wl:
        return f"{label}: wl must be >= {min_wl}, got {wl}"

    # symmetric drops the most negative code, which an unsigned format lacks
    if symmetric and not is_signed:
        return f"{label}: an unsigned format cannot be symmetric"

    return None


@lru_cache(maxsize=512)
def _fixedpoint_findings(
    wl: int,
    fl: int,
    is_signed: bool,
    symmetric: bool,
    prng_bits: int,
    carrier: torch.dtype,
) -> tuple[str | None, str | None]:
    """Return the findings of ``carrier`` on a fixed-point format, as ``(error, warning)``.

    The fixed-point counterpart of `_binaryK_findings`. A fixed-point format
    has no exponent field, so `_width_error` and `_range_findings` do not
    apply; its precision, its range and its step are compared with the
    carrier here instead:

    * Every value is a ``mag_bits``-bit integer times the step, so, as for a
      binaryK's precision, ``mag_bits`` must fit in the carrier's
      significand.
    * Stochastic rounding draws its random bits below the step, in the same
      significand, so ``mag_bits + prng_bits`` must fit in it too.
    * The largest value must not be below the carrier's smallest normal.
    * Both ends of the range must be carrier values. In two's complement the
      end farther from zero is the bottom, ``-2**(mag_bits - fl)``, one
      binade beyond the top.
    * A step below the lowest value the carrier's cast handles correctly
      warns.

    These limits were measured with
    ``dev/benchmarks/formats/format_limits.py fixedpoint ... --audit`` and the
    ``fixed-beyond`` modes of the cast sweeps (2026-11-10). An invalid layout
    is reported first, since no carrier can hold it.
    """
    label = _fixedpoint_label(wl, fl, is_signed, symmetric)

    # error: an invalid layout (wl too short, symmetric unsigned, or negative prng_bits)
    layout_error = _fixedpoint_layout_error(label, wl, is_signed, symmetric, prng_bits)
    if layout_error is not None:
        return layout_error, None

    mag_bits = wl - (1 if is_signed else 0)  # every value is a mag_bits-bit integer times 2^-fl
    top_exp = mag_bits - 1 - fl  # the exponent of the largest value's leading bit
    exp_step = -fl

    # the exponent of the leading bit of the end farther from zero: the top,
    # or in two's complement the bottom, -2^(mag_bits - fl), one binade higher
    end_exp = top_exp + 1 if is_signed and not symmetric else top_exp

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        # error: the carrier's significand must hold a mag_bits-bit integer
        if mag_bits > c.precision:
            return (
                f"{label} needs {mag_bits} bits of precision (one per magnitude bit), but it "
                f"is rounded in {c.name}, which has {c.precision}: {c.name} cannot represent "
                "every value of the format"
            ), None

        # error: stochastic rounding's random bits sit below the step, in the same significand
        if mag_bits + prng_bits > c.precision:
            return (
                f"{label} has too many stochastic-rounding bits: its {mag_bits} magnitude bits "
                f"and {prng_bits} random bits must fit together in the {c.precision} bits of "
                f"{c.name}'s significand"
            ), None

        # error: the whole range is below the carrier's smallest normal
        if top_exp < c.min_normal_exp:
            return (
                f"{label}'s largest value is below {c.name}'s smallest normal, "
                f"2^{c.min_normal_exp}: the cast cannot place any of the format's values"
            ), None

        # error: an end of the range is beyond the carrier's largest finite value
        if end_exp > c.top_exp:
            if end_exp > top_exp:
                end = f"most negative value, -2^{end_exp},"
            else:
                end = f"largest value, in the 2^{top_exp} binade,"
            return (
                f"{label}'s {end} is beyond {c.name}'s range, which ends below "
                f"2^{c.top_exp + 1}: a fixed-point format saturates to the ends of its range, "
                f"so {c.name} must be able to represent both"
            ), None

        # warn: the step is below the lowest value the carrier's cast handles
        # correctly (`_Carrier.min_simulable_exp`)
        if exp_step < c.min_simulable_exp:
            return None, (
                f"{label}'s step, 2^{exp_step}, is below 2^{c.min_simulable_exp}, the lowest "
                f"the cast handles correctly in {c.name}: values near zero may round incorrectly"
            )
        return None, None

    return _pointing_wider(find, carrier)


@lru_cache(maxsize=256)
def _fixedpoint_storage_findings(
    wl: int,
    fl: int,
    is_signed: bool,
    symmetric: bool,
    storage: torch.dtype,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    """Return the findings on storing a fixed-point format's results in
    ``storage``, as ``(error, warning)``.

    The fixed-point counterpart of `_binaryK_storage_findings`. Under the
    elementwise rule the inputs are already values of the storage dtype, so
    only the ends of the fixed-point range can produce a result the storage
    dtype does not hold. Under the rule for a computed result (a pointwise
    op's, or a fixed-point GEMM's), every value of the format must be a value
    of the storage dtype (`_fixedpoint_value_set_findings`). An invalid
    layout returns ``(None, None)``, as for binaryK.
    """
    label = _fixedpoint_label(wl, fl, is_signed, symmetric)
    if _fixedpoint_layout_error(label, wl, is_signed, symmetric, 0) is not None:
        return None, None

    st = _STORAGE[storage]
    mag_bits = wl - (1 if is_signed else 0)
    top_exp = mag_bits - 1 - fl

    # the fixed-point format's largest value exceeds the storage dtype's: it is in
    # a higher binade, or in the same binade with more significant bits (both
    # are all ones)
    max_above = top_exp > st.top_exp or (top_exp == st.top_exp and mag_bits > st.man_bits + 1)
    # the fixed-point format's smallest value, -2^(mag_bits - fl), is beyond the
    # storage dtype's range
    min_above = is_signed and not symmetric and mag_bits - fl > st.top_exp

    # the storage dtype's largest value is on the fixed-point grid (a multiple of 2^-fl)
    dtype_max_on_grid = st.top_exp - st.man_bits >= -fl

    # the storage dtype can hold no nonzero value of the fixed-point format
    if top_exp < st.min_exp:
        return (
            f"{label}'s largest value is below {st.name}'s smallest positive value, "
            f"2^{st.min_exp}: a {st.name} result stores none of the format's nonzero values"
        ), None
    if not elementwise:
        return _fixedpoint_value_set_findings(label, mag_bits, fl, max_above or min_above, st)
    # an input near the storage dtype's largest value can round up past it, to infinity
    if (max_above or min_above) and not dtype_max_on_grid:
        return None, (
            f"{label} reaches beyond {st.name}'s largest finite value, which is not a value "
            f"of the format: a {st.name} input near it can round up out of {st.name}'s range, "
            "and is then stored as infinity"
        )
    # an infinite input saturates to an end of the fixed-point range, which the
    # storage dtype cannot hold
    if max_above or min_above:
        return None, (
            f"{label} reaches beyond {st.name}'s largest finite value: an infinite input "
            f"saturates to an end of the format's range, which {st.name} stores as infinity"
        )
    # an input above the fixed-point format's largest value saturates to it,
    # which the storage dtype does not hold
    if not max_above and not _in_storage((1 << mag_bits) - 1, -fl, st):
        return None, (
            f"{label}'s largest value is not a {st.name} value: an input above it saturates "
            f"to it, and storing that result in {st.name} rounds it a second time"
        )
    return None, None


def _fixedpoint_value_set_findings(
    label: str, mag_bits: int, fl: int, above: bool, st: _Storage
) -> tuple[str | None, str | None]:
    """The rule for a computed result: whether every value of a fixed-point
    format is a value of the storage dtype, as ``(error, warning)``.

    Its values are the multiples of ``2**-fl`` of up to ``mag_bits``
    significant bits (and, in two's complement, the power of two at the
    bottom). With at most the storage dtype's precision, each is a normal of
    the storage dtype, or a multiple of ``2**-fl`` below its smallest normal,
    which is one of its subnormals when ``-fl`` is not below the exponent of
    the smallest. ``above`` says whether an end of the range is past the
    storage dtype's largest value.
    """
    name = st.name
    if mag_bits > st.man_bits + 1:
        return (
            f"{label} has {mag_bits} significant bits, and a {name} result holds "
            f"{st.man_bits + 1}: a computed result can be any of the format's values, and "
            f"storing one that {name} cannot represent rounds it a second time"
        ), None
    if above:
        return None, (
            f"{label} reaches beyond {name}'s largest finite value: storing a result in "
            f"{name} turns the format's values past it into infinity"
        )
    if -fl < st.min_exp:
        return None, (
            f"{label}'s values are 2^{-fl} apart, finer than {name}'s smallest subnormal, "
            f"2^{st.min_exp}: storing a result in {name} rounds the smallest of them a "
            f"second time, onto {name}'s coarser spacing"
        )
    return None, None


# --- the fixed-point GEMM (phase T) ------------------------------------------
#
# A mac whose formats are fixed point computes, for operands a and b in the
# carrier c, and Q_F the rounding to format F with saturation (or, for the
# accumulate format, wrap-around):
#
#     split:  p = Q_M(RN_c(a * b)),  s = Q_A(s + p)
#     fused:                         s = Q_F(RN_c(a * b) + s)
#
# with every sum exact. The product is the carrier's, as every mptorch mac's
# is. The definition is the integer accumulator's: an int64 sum, each step
# rounded to the accumulate format's grid once. Where the carrier holds every
# step exactly, the carrier computes it instead, the faster way, with the
# same bits. These are the rules that choose, and that refuse what neither
# can do; the C++ host applies the same ones.

#: The accumulate format's widest magnitude: a result is held in float64,
#: whose 53 significant bits hold every value of such a format.
_INTEGER_ACC_BITS = 53

#: The widest exact sum the integer accumulator forms, in an int64.
_INTEGER_SPAN_BITS = 62


def _fixedpoint_span(*formats: FixedPoint) -> int:
    """The significant bits an exact sum of one value of each format can need.

    Each value is an integer times its format's step, of magnitude at most
    ``2**(mag_bits - fl)`` (the two's complement bottom), so a sum of them is
    a multiple of the finest step, ``2**-max(fl)``, below
    ``2**(max(mag_bits - fl) + 1)``."""
    top = max(f.mag_bits - f.fl for f in formats) + 1
    return top + max(f.fl for f in formats)


class _FixedPointGemm(NamedTuple):
    """How a fixed-point GEMM call is computed.

    ``in_carrier`` says that the carrier computes every step exactly, so the
    sum in the carrier gives the integer accumulator's bits. ``error`` is
    why the call is refused, or ``None``."""

    in_carrier: bool
    error: str | None


@lru_cache(maxsize=256)
def _fixedpoint_gemm(
    mul: FixedPoint | None, acc: FixedPoint, wrap: bool, carrier: torch.dtype
) -> _FixedPointGemm:
    """The fixed-point GEMM's rules for a call in ``carrier`` (``torch.float32``
    for binary32, ``torch.float64`` for binary64).

    ``mul`` is a split mac's multiply format, or ``None`` for a fused mac;
    ``acc`` is the accumulate format, or the fused mac's; ``wrap`` asks for
    wrap-around on the accumulate format's overflow, where saturation is the
    default. The carrier computes the call when the mac is split, the sum
    saturates, the carrier's cast rounds both formats correctly (no finding,
    not even a warning: the cast misrounds near zero below its lowest step,
    where the definition is exact), and an exact sum of a product and a
    running sum fits the carrier's significand (`_fixedpoint_span`). Since
    every step saturates to its format, that depends on the formats alone,
    never on the length of the sum. Every other call runs the integer
    accumulator, which refuses:

    - a multiply format the carrier's cast does not round correctly, an
      error or a warning of its elementwise rule (on both paths the product
      is rounded by that cast);
    - an accumulate format past `_INTEGER_ACC_BITS`, or one that float64
      cannot hold;
    - stochastic rounding with more random bits than the carrier's draw
      holds below its sign and exponent;
    - wrap-around of a symmetric format, which has no modulus;
    - a sum past `_INTEGER_SPAN_BITS`: a split mac's exact sum, or a fused
      mac's, with the guard bits and the random bits its rounding reads.
    """
    c = _CARRIERS[carrier]
    acc_label = _fixedpoint_label(acc.wl, acc.fl, acc.is_signed, acc.symmetric)
    if mul is not None:
        mul_error, mul_warning = _fixedpoint_findings(
            mul.wl, mul.fl, mul.is_signed, mul.symmetric, mul.prng_bits, carrier
        )
        if mul_error is not None or mul_warning is not None:
            return _FixedPointGemm(
                False, f"the multiply format is rounded in the carrier: {mul_error or mul_warning}"
            )
    acc_fields = (acc.wl, acc.fl, acc.is_signed, acc.symmetric)
    in_carrier = (
        mul is not None
        and not wrap
        and _fixedpoint_findings(*acc_fields, acc.prng_bits, carrier) == (None, None)
        and _fixedpoint_span(mul, acc) <= c.precision
    )
    if in_carrier:
        return _FixedPointGemm(True, None)

    if acc.mag_bits > _INTEGER_ACC_BITS:
        return _FixedPointGemm(
            False,
            f"{acc_label} has {acc.mag_bits} magnitude bits, and a fixed-point GEMM accumulates "
            f"at most {_INTEGER_ACC_BITS}, which float64 holds exactly: use a narrower "
            "accumulate format",
        )
    acc_error = _fixedpoint_findings(*acc_fields, 0, torch.float64)[0]
    if acc_error is not None:
        return _FixedPointGemm(False, f"the result is held in float64: {acc_error}")
    if acc.prng_bits > c.man_bits:
        return _FixedPointGemm(
            False,
            f"{acc_label} draws {acc.prng_bits} random bits, and a {c.name} draw holds "
            f"{c.man_bits} below its sign and exponent: use at most {c.man_bits}",
        )
    if wrap and acc.symmetric:
        return _FixedPointGemm(
            False,
            f"{acc_label} is symmetric, so it has no modulus to wrap around: drop "
            "symmetric=True, or saturate",
        )
    if mul is not None:
        span = _fixedpoint_span(mul, acc)
        mul_label = _fixedpoint_label(mul.wl, mul.fl, mul.is_signed, mul.symmetric)
        what = f"a sum of {mul_label} and {acc_label}"
    else:
        # the carrier's product, aligned to the format's grid with a guard
        # bit and the random bits below it, beside the running sum
        span = acc.mag_bits + acc.prng_bits + 4
        what = f"a fused step into {acc_label}, with its guard and random bits,"
    if span > _INTEGER_SPAN_BITS:
        return _FixedPointGemm(
            False,
            f"{what} needs {span} bits, and the integer accumulator holds "
            f"{_INTEGER_SPAN_BITS}: narrow the formats, or bring their steps closer",
        )
    return _FixedPointGemm(False, None)


def _fixedpoint_gemm_wide(acc: FixedPoint, dtype: torch.dtype) -> bool:
    """Whether a fixed-point GEMM's result is computed into float64 for
    operands of ``dtype``: when the accumulate format has more significant
    bits than ``dtype`` holds. The result is then rounded once, by the
    output rounding, before it is stored in ``dtype``."""
    st = _STORAGE.get(dtype)
    return st is not None and acc.mag_bits > st.man_bits + 1
