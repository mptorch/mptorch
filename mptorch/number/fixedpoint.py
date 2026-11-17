"""Fixed-point formats, and the checks that hold one to a carrier and a dtype.

:class:`FixedPoint` is the format. It has no exponent field, so its findings
use neither `_carrier`'s `_Extent` nor `_storage`'s value-set rule. They hold
its precision and the two ends of its range against the carrier's directly,
and its storage rule is the elementwise one alone, since there is no
fixed-point GEMM.
"""

import warnings
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache

import torch

from ._carrier import _Carrier, _pointing_wider
from ._storage import _STORAGE, _in_storage
from .base import FormatRangeWarning, Number, _report_per_call


@dataclass(frozen=True)
class FixedPoint(Number):
    """A fixed-point format: ``wl`` bits, of which ``fl`` are fractional.

    Defined as in Gupta et al., *Deep Learning with Limited Numerical
    Precision* (arXiv:1502.02551). Its values are the multiples of the step
    ``2**-fl`` that a ``wl``-bit integer holds: with ``n = wl - 1`` magnitude
    bits in a signed format and ``n = wl`` in an unsigned one, the range is
    ``[-2**(n - fl), 2**(n - fl) - 2**-fl]`` in two's complement,
    ``[-(2**(n - fl) - 2**-fl), 2**(n - fl) - 2**-fl]`` with ``symmetric``,
    and ``[0, 2**(n - fl) - 2**-fl]`` unsigned. ``fl`` may be any integer:
    negative for a step coarser than 1, above ``wl`` for a format of pure
    fractions.

    Every :class:`RoundMode` applies, as for :class:`BinaryK`. Below the step
    the two candidates are zero and ``2**-fl``, picked as :class:`SubnormalsMode`
    describes. A result beyond the range saturates to its end, an infinite
    input included, because a fixed-point word has no code for an infinity, so
    there is no ``saturation`` field. A NaN passes through, the one zero is
    unsigned (no cast returns ``-0.0``), and an unsigned format rounds every
    negative value to zero. The instance is frozen for the same reason
    :class:`BinaryK` is.

    Args:
        wl (int): word length, the sign bit included; at least 2 signed and 1
            unsigned.
        fl (int): fractional length; the step between values is ``2**-fl``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        symmetric (bool): drop the most negative code, so the range is
            symmetric about zero; signed formats only. Default: ``False``.
        prng_bits (int): width of the random bits drawn by
            :attr:`RoundMode.SR` below the step, ignored under every other
            rounding mode. Default: 0.

    Raises:
        ValueError: for a ``wl`` too short, a symmetric unsigned format, a
            negative ``prng_bits``, or what binary64 cannot simulate: more
            than 52 magnitude bits, with or without ``prng_bits``, a
            largest value below its normals, or a range reaching its top
            binade. Each call checks the format
            against its own carrier (see :class:`FormatRangeWarning`).

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

        # Raise for what no carrier can do, and nothing more (`check_fixedpoint`)
        check_fixedpoint(
            self.wl,
            self.fl,
            self.is_signed,
            self.symmetric,
            self.prng_bits,
            carrier=torch.float64,
            warn=False,
        )

    @property
    def mag_bits(self) -> int:
        """Magnitude bits: ``wl`` less the sign bit, if there is one."""
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
        """The smallest value: 0 unsigned, ``-max_value`` symmetric, else one step below it."""
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
    """Hold a fixed-point format against what a carrier can hold.

    The same contract as :func:`check_binaryK`.

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
        stacklevel (int): how far above this function the caller's own code
            is. Default: 3.

    Raises:
        ValueError: for a layout with no values (see :class:`FixedPoint`),
            more magnitude bits than the carrier's mantissa has, with or
            without ``prng_bits``, a largest value below the carrier's
            normals, or a range reaching the carrier's top binade.
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
    """:func:`check_binaryK_carrier` for a fixed-point format.

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
    """:func:`check_binaryK_storage` for a fixed-point format.

    The same contract, with the format spelled as :func:`check_fixedpoint`
    spells it and the same ``storage`` and ``elementwise`` keywords. Only the
    elementwise rule is implemented, since there is no fixed-point GEMM yet.

    Raises:
        ValueError: where the format's largest value is below the dtype's
            smallest, so nothing of it is stored.
        NotImplementedError: for ``elementwise=False``, the GEMM rule.
    """
    _report_per_call(
        *_fixedpoint_storage_findings(wl, fl, is_signed, symmetric, storage, elementwise)
    )


# --- what the checks find ----------------------------------------------------


def _fixedpoint_label(wl: int, fl: int, is_signed: bool, symmetric: bool) -> str:
    """The format as its constructor call, for messages; defaults are omitted."""
    return (
        f"FixedPoint(wl={wl}, fl={fl}"
        + ("" if is_signed else ", is_signed=False")
        + (", symmetric=True" if symmetric else "")
        + ")"
    )


def _fixedpoint_layout_error(
    label: str, wl: int, is_signed: bool, symmetric: bool, prng_bits: int
) -> str | None:
    """The fixed-point layouts no range can be read from, or ``None``.

    :class:`FixedPoint` refuses these in `__post_init__`; the plain-integer
    wrappers in `mptorch.quant.ops` reach here directly.
    """
    if prng_bits < 0:
        return f"{label}: prng_bits must be >= 0, got {prng_bits}"

    # a sign bit and at least one magnitude bit
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
    """What ``carrier`` says about a fixed-point format, as ``(error, warning)``.

    The fixed-point twin of `_binaryK_findings`. A fixed-point format has no
    exponent field, so `_width_error` and `_range_findings` do not apply: its
    range is its precision, the leading bit of its largest value, and its
    step, each held against the carrier's here. Every value is a mag_bits-bit
    integer times the step, so the precision rule is binaryK's on precision,
    mag_bits against the carrier's significand; stochastic rounding draws its
    bits below the step in the same significand, so they share it; and both
    ends of the range must be carrier values, which in two's complement is the
    bottom, -2^(mag_bits - fl), one binade past the top. Measured with
    ``dev/benchmarks/formats/format_limits.py fixedpoint ... --audit`` and the
    ``fixed-beyond`` modes of the cast sweeps (2026-11-10), after SR became
    integer arithmetic on the word; until then it added 2^(mag_bits - fl) in
    the carrier, which cost one bit and one binade. The layout is checked
    first, and returned as is, since no carrier holds it.
    """
    label = _fixedpoint_label(wl, fl, is_signed, symmetric)

    # error: a layout with no values (wl too short, symmetric unsigned, negative prng_bits)
    layout_error = _fixedpoint_layout_error(label, wl, is_signed, symmetric, prng_bits)
    if layout_error is not None:
        return layout_error, None

    mag_bits = wl - (1 if is_signed else 0)  # every value is a mag_bits-bit integer times 2^-fl
    top_exp = mag_bits - 1 - fl  # the largest value's leading bit
    exp_step = -fl

    # the leading bit of the end farthest from zero: the top, or in two's
    # complement the bottom, -2^(mag_bits - fl), a binade past it
    end_exp = top_exp + 1 if is_signed and not symmetric else top_exp

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        # error: every value is a mag_bits-bit integer, which the carrier's
        # significand holds only up to its precision
        if mag_bits > c.precision:
            return (
                f"{label} asks for {mag_bits} bits of precision, its magnitude bits, and it "
                f"is rounded in {c.name}, which has {c.precision}"
            ), None

        # error: stochastic rounding draws its bits below the step, in the same significand
        if mag_bits + prng_bits > c.precision:
            return (
                f"{label} asks for {prng_bits} stochastic-rounding bits below {mag_bits} "
                f"magnitude bits; they are drawn from the {c.name} significand the rounding "
                f"happens in, which has {c.precision} bits to share"
            ), None

        # error: the whole range is below the carrier's normals
        if top_exp < c.min_normal_exp:
            return (
                f"{label}'s largest value is below {c.name}'s smallest normal, 2^{c.min_normal_exp}"
            ), None

        # error: an end of the range is past the carrier's largest value
        if end_exp > c.top_exp:
            if end_exp > top_exp:
                end = f"bottom, -2^{end_exp},"
            else:
                end = f"largest value, in 2^{top_exp},"
            return (
                f"{label}'s {end} is past {c.name}'s range, which stops below 2^{c.top_exp + 1}"
            ), None

        # warn: the step is below the floor the casts place values by (`_Carrier`)
        if exp_step < c.min_simulable_exp:
            return None, (
                f"{label}'s step, 2^{exp_step}, is below 2^{c.min_simulable_exp}; "
                f"values near zero may round incorrectly in {c.name}"
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
    """What storing a fixed-point result in ``storage`` says, as ``(error, warning)``.

    The fixed-point twin of `_binaryK_storage_findings`. Only the elementwise
    rule is implemented: its inputs are already the dtype's values, so only
    the ends of the range can put a result off the dtype's grid. The GEMM rule
    raises until there is a fixed-point GEMM. A layout the carrier check has
    rejected yields ``(None, None)``, as for binaryK.
    """
    if not elementwise:
        raise NotImplementedError("there is no fixed-point GEMM yet, so no GEMM storage rule")
    label = _fixedpoint_label(wl, fl, is_signed, symmetric)
    if _fixedpoint_layout_error(label, wl, is_signed, symmetric, 0) is not None:
        return None, None

    st = _STORAGE[storage]
    mag_bits = wl - (1 if is_signed else 0)
    top_exp = mag_bits - 1 - fl

    # the format's max starts a binade higher, or has more bits in the same one
    max_above = top_exp > st.top_exp or (top_exp == st.top_exp and mag_bits > st.man_bits + 1)
    # the format's min, -2^(mag_bits - fl), is past the dtype's range
    min_above = is_signed and not symmetric and mag_bits - fl > st.top_exp

    # the dtype's max is a multiple of the step 2^-fl
    dtype_max_on_grid = st.top_exp - st.man_bits >= -fl

    # nothing but zero is stored
    if top_exp < st.min_exp:
        return (f"{label}'s largest value is below {st.name}'s smallest, 2^{st.min_exp}"), None
    # an input near the dtype's largest value can round up past it
    if (max_above or min_above) and not dtype_max_on_grid:
        return None, (
            f"{label} exceeds {st.name}'s largest value; inputs near it may round to infinity"
        )
    # an infinite input saturates onto an end of the range the dtype cannot hold
    if max_above or min_above:
        return None, (
            f"{label} exceeds {st.name}'s largest value; an infinite input saturates "
            f"onto the end of its range, which {st.name} stores as infinity"
        )
    # an input past the format's largest value saturates onto it
    if not max_above and not _in_storage((1 << mag_bits) - 1, -fl, st):
        return None, (
            f"{label}'s largest value is not a {st.name} value; saturated results are rounded again"
        )
    return None, None
