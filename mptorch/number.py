"""Number formats, the modes that round into them, and what a float makes of them.

The enums here (:class:`RoundMode`, :class:`SaturationMode`,
:class:`SubnormalsMode`, :class:`AccumulateAlgorithm`) are mirrored one to one,
by name and by integer value, by the C++ enums in ``csrc/common/modes.h``; the
kernels receive ``.value`` across the ``torch.ops`` boundary, so the two must
stay in sync. :class:`BinaryK` and :class:`SuperFP` are the frozen value types
that name a format once. The ``check_*`` functions say what a *carrier* (the
binary float a cast rounds in) and a *storage* dtype (the tensor a result is
written to) make of a format, as a :exc:`ValueError` where the format cannot
work at all and a :class:`FormatRangeWarning` where it works over part of its
range.
"""

__all__ = [
    "SaturationMode",
    "SubnormalsMode",
    "RoundMode",
    "AccumulateAlgorithm",
    "Number",
    "FloatFormat",
    "BinaryK",
    "SuperFP",
    "BlockFormat",
    "FormatRangeWarning",
    "E8M0",
    "E4M3",
    "E5M2",
    "E2M1",
    "E2M3",
    "E3M2",
    "MXFP8_E4M3",
    "MXFP8_E5M2",
    "MXFP6_E2M3",
    "MXFP6_E3M2",
    "MXFP4_E2M1",
    "NVFP4",
]

import os
import sys
import warnings
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass, field
from enum import Enum
from fractions import Fraction
from functools import lru_cache
from typing import NamedTuple

import torch


class SaturationMode(Enum):
    """What a format does with a value beyond its largest finite one.

    Mirrors ``SaturationMode`` in ``csrc/common/modes.h``, by name and value.
    The three members are IEEE P3109's saturation modes, and they also select
    P3109's *domain*: a finite-domain format admits only ``SatFinite``, so
    :attr:`SAT_FINITE` is the finite domain, in which the code points the
    extended domain spends on the infinities hold finite values instead, and
    the other two are the extended domain. The mode is a property of each
    format slot, so a GEMM whose multiply saturates and whose accumulate
    overflows to infinity is expressible. A NaN input passes through every
    mode unchanged.

    Example::

        >>> from mptorch import BinaryK, SaturationMode
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> x = torch.tensor([1.1, 3.3, 500.0])
        >>> Quant(BinaryK(8, 4, bias=7))(x).tolist()
        [1.125, 3.25, inf]
        >>> sat = SaturationMode.SAT_FINITE
        >>> Quant(BinaryK(8, 4, bias=7, saturation=sat))(x).tolist()
        [1.125, 3.25, 480.0]
    """

    #: P3109 ``SatFinite``, finite domain: every result beyond the largest
    #: finite value, an infinite input included, clamps to that value.
    SAT_FINITE = 0
    #: P3109 ``SatPropagate``, extended domain: a finite result beyond the
    #: largest finite value clamps to it, and an infinite one stays infinite.
    SAT_PROPAGATE = 1
    #: P3109 ``SatNone``, extended domain, the default: a result beyond the
    #: largest finite value becomes :math:`\pm\infty` under every rounding mode,
    #: because the overflow test reads the rounded value. This differs from IEEE
    #: 754, whose ``roundTowardZero`` never rounds a finite value to infinity.
    OVF_INF = 2


class SubnormalsMode(Enum):
    """How a format handles the range below its smallest normal number.

    Mirrors ``SubnormalsMode`` in ``csrc/common/modes.h``, by name and value.
    Only :attr:`SUBNORMALS` is IEEE P3109's (every P3109 format with more than
    one bit of precision has subnormals), so the other two give formats
    outside the standard. The three differ only in where the bottom of the
    range is. Below it they all behave alike, because the region has the same
    shape in each: the two candidates for the magnitude of an input ``x`` are
    zero and the smallest value, ``x_min``, and the rounding mode picks
    between them. Nearest takes the nearer, and on a tie
    :attr:`RoundMode.RNE` takes the zero and :attr:`RoundMode.RNA` takes
    ``x_min``; the directed modes take their own direction;
    :attr:`RoundMode.RO` takes ``x_min``, the nonzero one; and
    :attr:`RoundMode.SR` takes ``x_min`` with probability ``|x| / x_min``.

    All three keep the same two special codes, which the casts' NaN detection
    and sign handling rely on: the all-zero code is the one zero, unsigned as
    in P3109 (no cast ever returns ``-0.0``), and in a signed format the code
    with only the sign bit set, the one that would encode ``-0.0``, is the NaN.

    Example::

        >>> from mptorch import BinaryK, SubnormalsMode
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> x = torch.tensor([2**-7, 2**-6])  # below and at the smallest normal
        >>> Quant(BinaryK(8, 4, bias=7))(x).tolist()
        [0.0078125, 0.015625]
        >>> flush = SubnormalsMode.NORMALS
        >>> Quant(BinaryK(8, 4, bias=7, subnormals=flush))(x).tolist()
        [0.0, 0.015625]
    """

    SUBNORMALS = 0  #: Gradual underflow, the default: exponent code 0 holds subnormals.
    NORMALS = 1  #: No subnormals: below the smallest normal, zero or that normal.
    #: No subnormals, and exponent code 0 is one more binade of normal numbers,
    #: except its mantissa-zero code, which is still the zero (and with the
    #: sign bit the NaN), so that binade's values start one step above the
    #: power of two at its foot.
    EXTENDED_NORMALS = 2


class RoundMode(Enum):
    """How a value is rounded onto a format's grid.

    Mirrors ``RoundMode`` in ``csrc/common/modes.h``, by name and value. Each
    member is one of IEEE P3109's rounding modes, whose name is given in
    parentheses, and defines the result :math:`y` of rounding an input
    :math:`x`. The rounding mode is a property of the operation, not of the
    format: the kernels take it as a template parameter, because a runtime
    ``switch`` inside a GEMM's accumulate step cost 1.2 to 2.3 times the
    kernel time, so :class:`mptorch.quant.Quant` and
    :class:`mptorch.quant.SplitMac` carry it next to the format. Stochastic
    rounding draws each output element's random bits from a stream keyed on
    that element's index in the output, so the result does not depend on how
    the work is split across threads or batches.

    Example::

        >>> from mptorch import BinaryK, RoundMode
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> x = torch.tensor([1.1, 3.3])
        >>> Quant(BinaryK(8, 4, bias=7), RoundMode.RNE)(x).tolist()
        [1.125, 3.25]
        >>> Quant(BinaryK(8, 4, bias=7), RoundMode.RZ)(x).tolist()
        [1.0, 3.25]
        >>> Quant(BinaryK(8, 4, bias=7), RoundMode.RU)(x).tolist()
        [1.125, 3.5]
    """

    RNE = 0  #: Round to nearest, ties to even (``NearestTiesToEven``)
    RNA = 1  #: Round to nearest, ties to away from zero (``NearestTiesToAway``)
    RU = 2  #: The smallest :math:`y` such that :math:`y \ge x` (``TowardPositive``)
    RD = 3  #: The largest :math:`y` such that :math:`y \le x` (``TowardNegative``)
    RZ = 4  #: The largest :math:`y` such that :math:`|y| \le |x|` (``TowardZero``)
    RO = 5  #: Round to odd (``ToOdd``)
    SR = 6  #: Stochastic rounding with ``prng_bits`` random bits (``StochasticA``)


class AccumulateAlgorithm(Enum):
    """How the partial products of a dot product are folded into its sum.

    Mirrors ``AccumulateAlgorithm`` in ``csrc/common/modes.h``, by name and
    value, and selects the reduction inside a custom-arithmetic GEMM
    (:class:`mptorch.quant.SplitMac`, :class:`mptorch.quant.FusedMac`, the
    flat ``*_matmul`` wrappers). Below, ``a`` and ``b`` are the two operands
    of one step, ``s`` is the running sum, and ``round_mul`` and ``round_acc``
    are the mac's two roundings (``round_acc`` is the identity with no
    accumulate format, and in a fused mac it is the fused format's rounding,
    the step being ``round_acc(fma(a, b, s))`` in place of
    ``round_acc(s + round_mul(a * b))``). ``round_out`` is the rounding of the
    mac's ``outer`` format (the identity without one), and the products are
    taken in order, ``k = 1 .. K``:

    * :attr:`NAIVE`: ``s = round_acc(s + round_mul(a * b))``, which is what a
      hardware accumulator of the accumulate format does. The result is ``s``.
    * :attr:`KAHAN`: a compensation ``c`` is carried alongside the sum, in
      the accumulate format like it: ``y = round_acc(round_mul(a * b) - c)``,
      ``t = round_acc(s + y)``, ``c = round_acc(round_acc(t - s) - y)``,
      ``s = t``. The result is ``s``. With no accumulate format this is
      Kahan's summation in the carrier.
    * :attr:`BLOCK`: the ``NAIVE`` step runs on a block's sum ``s``, and after
      every ``block_size`` products, and after the last, the block is folded
      into a total, ``tot = round_out(tot + s)``, and ``s`` restarted from
      zero. Two levels of precision: a narrow accumulate format inside a
      block, a wider outer format across them. The result is ``tot``.
    * :attr:`TREE`: the rounded products ``p1, p2, ...`` of a block of
      ``block_size = 2**L`` are summed pairwise,
      ``round_acc(round_acc(p1 + p2) + round_acc(p3 + p4))`` and so on up,
      in the order they arrive, and the block's root is folded as in
      ``BLOCK``. A last, partial block merges the subtrees it has, smallest
      first. It needs a product term, so a fused mac refuses it.

    Under :attr:`RoundMode.SR` every rounding that is not an identity draws
    once, in the order written. Nothing is padded, so a result does not
    depend on how a backend tiles ``K``; the CPU and CUDA kernels agree to the
    bit in every deterministic mode. The three past ``NAIVE`` are implemented
    for the single-format ops, in binary32: float64 operands,
    ``carrier=torch.float64``, a palette and an MPS tensor raise.

    Example::

        >>> from mptorch import AccumulateAlgorithm
        >>> AccumulateAlgorithm.NAIVE.value
        0
    """

    NAIVE = 0  #: Quantize the running sum after every accumulation step.
    KAHAN = 1  #: Kahan-compensated summation, the compensation in the accumulate format.
    BLOCK = 2  #: Two-level (FABSum-style) block summation with an outer format.
    TREE = 3  #: Pairwise summation within a block, folded with an outer format.


# --- what a carrier can hold -------------------------------------------------
#
# The casts do not build encodings. They round a value of their *carrier*, the
# binary float the arithmetic runs in, onto the format's grid and hand back a
# value of the carrier (`docs/source/concepts.rst`, "What the carrier can
# hold"). The carrier is the operand's by default, binary64 for a float64
# tensor and binary32 for float32, float16 and bfloat16, and binary64 for
# those three too where a call asks for it with `carrier=torch.float64`. It is
# never narrower than the tensor. A format with more range or precision than
# its carrier is quantized *partially*: the result is still a tensor of
# plausible numbers, and some of the format's values are simply never among
# them. The checks below are what stops that being invisible.
# `dev/benchmarks/format_limits.py` measures the bounds stated here against
# the kernels, over ~17 M inputs per format in every rounding mode, and
# `tests/test_format_limits.py` pins the boundaries it found.
#
# Two bounds, not one, and they differ by a binade:
#
#   representable  every value of the format is a value of the carrier.
#                  Arithmetic alone: its significand bits (24, 53), nothing
#                  above its largest finite value, nothing spaced finer than
#                  its smallest subnormal (2**-149, 2**-1074).
#   faithful       the cast reaches those values, for every input. Stronger,
#                  because a cast places its input by the *exponent field*
#                  of the word and all of a carrier's subnormals share field
#                  0, so nothing below its smallest normal (2**-126, 2**-1022)
#                  can be told apart: a binade above that (2**-125, 2**-1021)
#                  is the lowest value a format may have, for a cast that
#                  reads the field down there. `_binaryK_extent` derives the
#                  floor per `SubnormalsMode`, since two of them compare
#                  magnitudes instead and reach a binade lower.
#
# Together the top and bottom leave 253 binades to spend in binary32 and 2045
# in binary64, which allows at most seven and ten exponent bits.
#
# What separates a warning from an error here is not which of those two bounds
# is missed but whether the format still *works*. A format whose range outruns
# its carrier quantizes correctly over the part that fits (a wide binaryK used
# as a precision-only target is a real thing, and `tests/` uses several on
# purpose); it is simply partial, and being told so is the whole point. That
# is a warning. An error is for a format that cannot function at all: a grid
# finer than the carrier's (nothing survives), an exponent field wider than
# the kernels shift by, stochastic bits with no significand left to draw
# from, a superfp whose regions come out misordered, or one whose largest
# finite code is below the carrier's normals, where the largest finite value
# is zero and every nonzero result overflows.
#
# Which carrier a format meets belongs to the call, not to the format, so a
# carrier's findings are reported per call, from `mptorch.quant.ops`, as the
# storage checks further down are. All `BinaryK`/`SuperFP` check when they are
# built is what no carrier can do (binary64's errors, binary64 being the
# wider), and they warn about nothing.


class _Carrier(NamedTuple):
    """The binary float a cast rounds in, as the range checks read it.

    Two instances exist, ``_BINARY32`` and ``_BINARY64``. Every exponent is an
    unbiased power of two: ``man_bits`` is ``precision - 1`` (the stored
    mantissa bits), ``min_normal_exp`` is the exponent of the smallest normal,
    which is also the exponent that field 1 encodes, and ``min_simulable_exp``
    is one binade above it, the lowest a format's smallest value may sit when
    the cast places inputs by their exponent field.
    """

    name: str
    precision: int  # significand bits, the implicit one included
    man_bits: int
    top_exp: int  # exponent of the largest finite value
    min_normal_exp: int
    min_subnormal_exp: int  # exponent of the smallest positive value
    min_simulable_exp: int  # the lowest a format's own smallest value may be


_BINARY32 = _Carrier("binary32", 24, 23, 127, -126, -149, -125)
_BINARY64 = _Carrier("binary64", 53, 52, 1023, -1022, -1074, -1021)
# Keyed on the dtype a `carrier=` argument names, which is the float the
# carrier is; float16 and bfloat16 never carry, they are rounded in binary32.
_CARRIERS: dict[torch.dtype, _Carrier] = {torch.float32: _BINARY32, torch.float64: _BINARY64}
_MAX_EXP_BITS = 30  # `1 << exp_bits` is a C++ `int` in the kernels


class FormatRangeWarning(UserWarning):
    """Warns that a format's range outruns the float it is rounded or stored in.

    Quantizing to such a format is partial: the cast is correct over the part
    of the format the carrier holds, and some of the format's values are never
    returned. The carrier is the operand's, binary64 for a float64 tensor and
    binary32 for float32, float16 and bfloat16, unless the call names binary64
    with ``carrier=torch.float64``. The quantizers and GEMMs of
    :mod:`mptorch.quant` warn per call when a format reaches above the
    carrier's largest finite value, spaces its values finer than the carrier's
    smallest subnormal (:math:`2^{-149}` in binary32, :math:`2^{-1074}` in
    binary64), or has a smallest positive value below the floor the carrier
    gives it. That floor depends on how the cast finds the bottom of the
    format:

    * :math:`2^{-125}` (:math:`2^{-1021}`) for a :class:`BinaryK` with
      ``SubnormalsMode.SUBNORMALS`` and for a :class:`SuperFP`. Their casts
      place an input on the subnormal or supernormal grid by its exponent
      field, which all of the carrier's subnormals share, so a format reaching
      lower would have its bottom binade rounded onto the wrong grid.
    * :math:`2^{-126}` (:math:`2^{-1022}`), the carrier's smallest normal, for
      a :class:`BinaryK` with ``SubnormalsMode.NORMALS`` or
      ``SubnormalsMode.EXTENDED_NORMALS``, whose casts find the bottom by
      comparing magnitudes and read no field. Two cases stop a binade short,
      at the first floor: ``P = 1`` under ``NORMALS``, and
      ``EXTENDED_NORMALS`` at the carrier's full precision (``P = 24``,
      ``P = 53``).

    :doc:`/concepts` ("What the carrier can hold") derives both.
    :class:`BinaryK` and :class:`SuperFP` do not warn when they are built,
    since a format does not know which carrier it will meet; they raise only
    for what neither carrier can do.

    The same warning is issued when the result is stored in a dtype narrower
    than the carrier (float16 or bfloat16 under binary32, and float32 as well
    under binary64) and the format's range outruns *that* dtype: above its
    largest finite value, or spaced finer than its smallest subnormal. The
    cast rounds in the carrier, and storing the result rounds a second time,
    onto the dtype's grid, with a value above the dtype's range stored as an
    infinity whatever the format's saturation mode.

    It is a warning rather than an error because such a format is still
    useful: a wide binaryK is a reasonable precision-only target, where the
    unreachable top of the range is exactly the point. Each warning names the
    line that made the call, so the default filter reports it once per line
    rather than once per training step.

    Example::

        >>> import warnings
        >>> import torch
        >>> from mptorch import BinaryK, FormatRangeWarning
        >>> from mptorch.quant import Quant
        >>> q = Quant(BinaryK(16, 8, bias=255))  # values down to 2**-261
        >>> warnings.simplefilter("ignore", FormatRangeWarning)
        >>> y = q(torch.ones(2))  # rounded in binary32, without the warning
    """


def _top_of_range(
    exp_bits: int, man_bits: int, bias: int, reserved: int, carrier: _Carrier
) -> tuple[int, int, int]:
    """The largest finite value of a format, in the kernel's terms.

    Returns ``(exponent field, exponent, mantissa code)`` of the largest
    finite value, transcribed from ``make_normal_range_params`` in
    ``bit_helper.h``. P3109 counts that value in code points: of the top
    binade's codes the last ``reserved`` hold no finite value (an infinity
    outside the finite domain, and the NaN of an unsigned binaryK), and with
    one or two mantissa bits those can outnumber the binade's codes, which
    moves the largest finite value a binade or two down, so the loop borrows
    from lower binades until the code is non-negative. Counted in
    ``min(man_bits, carrier.man_bits)`` codes, as the kernel counts them; a
    format asking for more than that has already been rejected for precision
    before this is read.
    """
    codes = 1 << min(man_bits, carrier.man_bits)
    top_code = codes - 1 - reserved
    top_field = (1 << exp_bits) - 1
    while top_code < 0:
        top_code += codes
        top_field -= 1
    return top_field, top_field - bias, top_code


def _width_error(
    label: str, exp_bits: int, man_bits: int, prng_bits: int, carrier: _Carrier
) -> str | None:
    """The bounds that must hold before anything shifts by ``exp_bits``.

    Returns the message for the first one violated, or ``None``: a negative
    ``prng_bits``, an exponent field wider than ``_MAX_EXP_BITS``, a mantissa
    wider than the carrier's, or stochastic-rounding bits that do not fit next
    to the mantissa in the carrier's significand. Each is a format that cannot
    function, never a partial one.
    """
    if prng_bits < 0:
        # `BinaryK`/`SuperFP` refuse a negative count in `__post_init__`; the
        # plain-integer wrappers in `mptorch.quant.ops` reach here directly
        return f"{label} asks for {prng_bits} stochastic-rounding bits, which cannot be negative"
    if exp_bits > _MAX_EXP_BITS:
        binades = carrier.top_exp - carrier.min_simulable_exp + 1
        return (
            f"{label} has {exp_bits} exponent bits; the kernels derive the format's "
            f"range with `1 << exp_bits` in a 32-bit int, and {carrier.name} spans "
            f"{binades} binades in any case, so at most {binades.bit_length() - 1} are usable"
        )
    if man_bits > carrier.man_bits:
        return (
            f"{label} asks for {man_bits + 1} bits of precision, and it is rounded in "
            f"{carrier.name}, which has {carrier.precision}: the format's grid is finer "
            f"than anything that can be returned"
        )
    if man_bits + prng_bits > carrier.man_bits:
        return (
            f"{label} asks for {prng_bits} stochastic-rounding bits below a "
            f"{man_bits}-bit mantissa; they are drawn from the {carrier.name} significand "
            f"the rounding happens in, which has {carrier.man_bits} bits to share"
        )
    return None


class _Extent(NamedTuple):
    """A format's range, in the only terms the range checks read.

    Derived once from the parameters by `_binaryK_extent` or `_superfp_extent`
    and held against two things: the carrier a cast rounds in, and a result
    narrower than that carrier, which the rounded value is stored in. Every
    exponent is an unbiased power of two.
    """

    label: str
    exp_bits: int
    man_bits: int
    top_field: int  # the largest finite value's exponent field
    top_exp: int  # ... its exponent
    top_code: int  # ... and its mantissa code, of `man_bits`
    bottom_exp: int  # the exponent of the smallest positive value
    step_exp: int  # the exponent of the finest spacing anywhere in the format
    min_bottom: int  # how low `bottom_exp` may go before the carrier's cast loses it
    bottom_why: str  # ... and what goes wrong below that, for the warning
    # What lies below the binades that carry a full mantissa, which start at
    # `normal_exp`: the name of one of `SubnormalsMode`'s three, or
    # "SUPERNORMALS" for superfp's powers of two.
    normal_exp: int
    below: str


def _range_findings(ext: _Extent, carrier: _Carrier) -> tuple[str | None, str | None]:
    """The top and bottom of the range against the carrier's, after the widths.

    Returns ``(error, warning)``, at most one of them set. ``bottom_exp`` is
    the exponent of the format's smallest positive value and ``step_exp`` that
    of its finest spacing; for a format whose lowest binade carries a mantissa
    the two differ by ``man_bits``. ``min_bottom`` is how low that smallest
    value may go, which `_binaryK_extent` derives per subnormals mode.

    Only the first finding is returned, so a format that outruns its carrier
    at both ends says so about the top first and about the bottom once that is
    fixed. The error is the one case that stops the format working at all: no
    finite value the carrier holds, so every nonzero result overflows.
    """
    label, top_exp, bottom_exp, step_exp = ext.label, ext.top_exp, ext.bottom_exp, ext.step_exp
    name = carrier.name
    if ext.top_field < 1 or top_exp < carrier.min_normal_exp:
        return (
            f"{label} has no finite normal value {name} can hold: the largest "
            f"finite code sits at 2^{top_exp}, so every nonzero result overflows"
        ), None
    if top_exp > carrier.top_exp:
        return None, (
            f"{label} reaches 2^{top_exp}, above {name}'s 2^{carrier.top_exp}: the cast "
            f"saturates at {name}'s largest value on the format's grid instead, and "
            f"the codes above it are unreachable (needs bias >= {ext.top_field - carrier.top_exp})"
        )
    if step_exp < carrier.min_subnormal_exp:
        return None, (
            f"{label} spaces its values 2^{step_exp} apart at the bottom, below "
            f"{name}'s 2^{carrier.min_subnormal_exp}: values that far down are not "
            f"{name} values at all, so the bottom of the range is unreachable"
        )
    if bottom_exp < ext.min_bottom:
        return None, (
            f"{label}'s smallest value is 2^{bottom_exp}, below 2^{ext.min_bottom}: "
            f'{ext.bottom_why}. See concepts.rst, "What the carrier can hold".'
        )
    return None, None


# The three things that can go wrong below `_Extent.min_bottom`, each as the
# sentence `_Extent.bottom_why` carries into the warning.


def _bottom_by_field(c: _Carrier) -> str:
    """Why a cast that reads the exponent field loses values below the floor."""
    return (
        f"the casts place an input by its {c.name} exponent field, which every {c.name} "
        f"subnormal shares, so this format's values below 2^{c.min_normal_exp} are "
        "rounded onto the wrong grid and some are never returned"
    )


def _bottom_tie_by_field(c: _Carrier) -> str:
    """Why a one-code-per-binade format's floor cannot sit at the smallest normal."""
    return (
        "with one code per binade the cast rounds by exponent field alone, and half that "
        f"value, 2^{c.min_normal_exp - 1}, is a {c.name} subnormal, in the field every "
        "one of them shares -- so round-to-nearest-even reads it as a tie between two "
        "binades and returns the smallest value, where the format's answer is zero"
    )


def _bottom_half_unheld(c: _Carrier) -> str:
    """Why a full-precision extended-normals floor needs a step the carrier lacks."""
    return (
        f"half that value, 2^{c.min_normal_exp - 1} x (1 + 2^-{c.man_bits}), is the "
        f"round-to-nearest boundary below it and needs a step {c.name} does not have, so "
        f"the cast holds it as 2^{c.min_normal_exp - 1} -- and round-to-nearest-away "
        "returns the smallest value for that input, where the format's answer is zero"
    )


def _binaryK_label(K: int, P: int, bias: int, is_signed: bool, subnormals: SubnormalsMode) -> str:
    """The format as its constructor call, for messages; defaults are omitted."""
    return (
        f"BinaryK(K={K}, P={P}, bias={bias}"
        + ("" if is_signed else ", is_signed=False")
        + ("" if subnormals is SubnormalsMode.SUBNORMALS else f", {subnormals.name}")
        + ")"
    )


def _binaryK_extent(
    label: str,
    man_bits: int,
    exp_bits: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    carrier: _Carrier,
) -> _Extent:
    """A binaryK format's range, as an `_Extent`.

    Call it only once `_width_error` has passed, since this shifts by
    ``exp_bits``. The top comes from `_top_of_range`; the bottom and the finest
    step depend on the subnormals mode, and so does the floor the carrier's
    cast is exact down to.
    """
    # P3109's reserved codes: +infinity outside the finite domain, and NaN
    # above it in an unsigned format (`make_binaryK_params`)
    reserved = (saturation is not SaturationMode.SAT_FINITE) + (0 if is_signed else 1)
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    min_exp = 1 - bias  # the smallest normal's exponent
    if subnormals is SubnormalsMode.SUBNORMALS:
        # the subnormals are one grid of fixed spacing, and its step is both
        # the smallest value and the finest spacing in the format
        bottom_exp = step_exp = min_exp - man_bits
    elif subnormals is SubnormalsMode.NORMALS:
        bottom_exp, step_exp = min_exp, min_exp - man_bits
    else:
        # EXTENDED_NORMALS: one more binade of normals below the rest, minus
        # its mantissa-zero code, which is still the zero, so the smallest
        # value is one step into that binade; with man_bits == 0 the step is
        # the whole binade and the floor is the one NORMALS has
        bottom_exp = min_exp - 1 if man_bits else min_exp
        step_exp = bottom_exp - man_bits
    # How low the floor may go is not the same for the three modes, and the
    # difference is which of them reads an exponent field down there. In
    # binary32's numbers, binary64's following in parentheses:
    # `SUBNORMALS` derives its grid from the input's exponent field, which
    # every subnormal of the carrier shares, so it is blind below 2**-126
    # (2**-1022) and needs its floor a binade above that. `NORMALS` and
    # `EXTENDED_NORMALS` decide by comparing magnitudes as words, which reads
    # no field at all, so they are exact to 2**-126, and no further, because
    # below it the floor leaves the carrier's normals and nothing flushes. Two
    # of their formats stop a binade short of that, both at half the floor,
    # the one point the compare needs besides the floor itself:
    #
    #   man_bits == 0   the round before the compare reads a field after all.
    #                   With no mantissa it rounds between binades, and a floor
    #                   at 2**-126 puts half of it, 2**-127, in field 0, where
    #                   the round sees a tie between binades and carries it up
    #                   to the floor before the compare can send it to zero.
    #   EXTENDED, full  at the carrier's full precision (P = 24, 53) half the
    #   precision       floor, 2**-127 * (1 + 2**-23), needs a 2**-150 step
    #                   (2**-1075); the carrier holds it as 2**-127, and
    #                   round-to-nearest-away takes that input up to the floor.
    #
    # `dev/benchmarks/cast_all_modes_sweep.cu` found both in binary32 by
    # checking every rounding mode over all 2**32 inputs; binary64 is exact
    # at both points, and the same bounds apply one carrier wider.
    if subnormals is SubnormalsMode.SUBNORMALS:
        min_bottom, bottom_why = carrier.min_simulable_exp, _bottom_by_field(carrier)
    elif man_bits == 0:
        min_bottom, bottom_why = carrier.min_simulable_exp, _bottom_tie_by_field(carrier)
    elif subnormals is SubnormalsMode.EXTENDED_NORMALS:
        # half the floor's last bit, 2**(bottom_exp - 1 - man_bits), must exist
        min_bottom = max(carrier.min_normal_exp, man_bits + 1 + carrier.min_subnormal_exp)
        bottom_why = _bottom_half_unheld(carrier)
    else:
        min_bottom, bottom_why = carrier.min_normal_exp, _bottom_by_field(carrier)
    if bottom_exp < carrier.min_normal_exp:
        # the floor itself is out of reach, which says more
        bottom_why = _bottom_by_field(carrier)
    return _Extent(
        label,
        exp_bits,
        man_bits,
        top_field,
        top_exp,
        top_code,
        bottom_exp,
        step_exp,
        min_bottom,
        bottom_why,
        min_exp,
        subnormals.name,
    )


def _superfp_label(man_bits: int, exp_bits: int, normal_binades: int, bias: int) -> str:
    """The format as its constructor call, for messages."""
    return (
        f"SuperFP(man_bits={man_bits}, exp_bits={exp_bits}, "
        f"normal_binades={normal_binades}, bias={bias})"
    )


def _superfp_regions_error(label: str, exp_bits: int, normal_binades: int) -> str | None:
    """The one superfp layout no range can be read from, or ``None``."""
    binades = 1 << exp_bits
    if normal_binades >= binades:
        # no supernormal codes left, which puts the supernormal floor one
        # *above* the normal one; the cast tests supernormal-then-underflow,
        # so the format's lowest normal binade would flush to zero
        return (
            f"{label} leaves no supernormal codes: normal_binades must be at most "
            f"2**exp_bits - 1 = {binades - 1}, or the cast flushes the format's "
            f"lowest normal binade to zero"
        )
    return None


def _superfp_extent(
    label: str,
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    carrier: _Carrier,
) -> _Extent:
    """A superfp format's range, as an `_Extent`.

    Call it only once `_width_error` and `_superfp_regions_error` have passed.
    The normal region holds the top ``normal_binades`` binades with a full
    mantissa; below it the ``(binades - normal_binades) * 2**man_bits`` codes
    the other binades would have spent on mantissas encode the zero and, all
    but that one, further powers of two, the supernormals, and everything
    below those flushes.
    """
    binades = 1 << exp_bits
    # superfp spends no code on a NaN, signed or not (`make_superfp_params`)
    reserved = saturation is not SaturationMode.SAT_FINITE
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, carrier)
    normal_cutoff = binades - bias - normal_binades
    # the smallest supernormal, `superfp_region_cutoffs`
    bottom_exp = normal_cutoff - (binades - normal_binades) * (1 << man_bits) + 1
    step_exp = min(bottom_exp, normal_cutoff - man_bits)
    return _Extent(
        label,
        exp_bits,
        man_bits,
        top_field,
        top_exp,
        top_code,
        bottom_exp,
        step_exp,
        carrier.min_simulable_exp,  # the supernormal arm reads the exponent field
        _bottom_by_field(carrier),
        normal_cutoff,
        "SUPERNORMALS",
    )


# The derivations are memoized and the reporting is not, deliberately. The
# elementwise quantizers look their format up on every call, and a flat GEMM
# wrapper resolves its formats on every call too, so this sits on a per-call
# path: a cache hit is 0.25 us against 1.04 to redo the derivation. Caching
# the *warning* instead would make it fire once per process and then go
# quiet, which is a worse contract than `warnings`' own once-per-location
# default and would make the tests order-dependent.
#
# A binary32 finding that binary64 does not share says so, since the cure is
# then the binary64 carrier rather than a different format.


def _pointing_wider(
    find: Callable[[_Carrier], tuple[str | None, str | None]], carrier: torch.dtype
) -> tuple[str | None, str | None]:
    """Run ``find`` for ``carrier`` and, if binary32 objects where binary64
    would not, append the way out (``carrier=torch.float64``) to its message.
    """
    error, warning = found = find(_CARRIERS[carrier])
    if carrier is torch.float32 and found != (None, None) and find(_BINARY64) == (None, None):

        def wider(message: str) -> str:
            end = "" if message.endswith(".") else "."
            return (
                f"{message}{end} binary64 holds all of it: a float64 tensor's carrier, "
                "which carrier=torch.float64 gives a narrower tensor too."
            )

        return error and wider(error), warning and wider(warning)
    return found


@lru_cache(maxsize=512)
def _binaryK_findings(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    prng_bits: int,
    carrier: torch.dtype,
) -> tuple[str | None, str | None]:
    """What ``carrier`` says about a binaryK format, as ``(error, warning)``.

    Memoized on the whole parameter tuple, which is why it takes plain values
    rather than a :class:`BinaryK`. The widths are checked first, since the
    range derivation shifts by the exponent width.
    """
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = _width_error(label, exp_bits, man_bits, prng_bits, c)
        if error is not None:
            return error, None
        ext = _binaryK_extent(label, man_bits, exp_bits, bias, is_signed, saturation, subnormals, c)
        return _range_findings(ext, c)

    return _pointing_wider(find, carrier)


@lru_cache(maxsize=512)
def _superfp_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    prng_bits: int,
    carrier: torch.dtype,
) -> tuple[str | None, str | None]:
    """What ``carrier`` says about a superfp format, as ``(error, warning)``.

    The superfp twin of `_binaryK_findings`, with the region-layout check
    added to the width checks.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)

    def find(c: _Carrier) -> tuple[str | None, str | None]:
        error = _width_error(label, exp_bits, man_bits, prng_bits, c) or _superfp_regions_error(
            label, exp_bits, normal_binades
        )
        if error is not None:
            return error, None
        ext = _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation, c)
        return _range_findings(ext, c)

    return _pointing_wider(find, carrier)


def check_binaryK(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype = torch.float32,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Hold a binaryK format against what a carrier can hold.

    A format that cannot function in the carrier is an error; one whose range
    merely outruns it quantizes partially rather than wrongly, and is a
    warning. :class:`BinaryK` calls this with ``carrier=torch.float64`` and
    ``warn=False``, which raises for what no carrier can do and says nothing
    else, because which carrier a format meets is the call's to know; the
    call holds it to that one with :func:`check_binaryK_carrier`.

    Args:
        K (int): total bits of the format.
        P (int): bits of precision, the implicit leading bit included.
        bias (int): the resolved exponent bias, never ``None``.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.OVF_INF``.
        subnormals (SubnormalsMode): what lies below the smallest normal.
            Default: ``SubnormalsMode.SUBNORMALS``.
        prng_bits (int): random bits drawn for stochastic rounding. Default: 0.
        carrier (torch.dtype): ``torch.float32`` for binary32 or
            ``torch.float64`` for binary64. Default: ``torch.float32``.
        warn (bool): whether to issue the warning. Default: ``True``.
        stacklevel (int): how far above this function the caller's own code
            is, so the warning points there. Default: 3.

    Raises:
        ValueError: for a format that cannot function in the carrier: more
            precision than the carrier has, stochastic bits with no
            significand left to draw from, an exponent field past 30 bits, a
            negative ``prng_bits``, or no finite value the carrier holds.
    """
    error, warning = _binaryK_findings(
        K, P, bias, is_signed, saturation, subnormals, prng_bits, carrier
    )
    if error is not None:
        raise ValueError(error)
    if warn and warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


def check_superfp(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype = torch.float32,
    warn: bool = True,
    stacklevel: int = 3,
) -> None:
    """Hold a superfp format against what a carrier can hold.

    The same contract as :func:`check_binaryK`. There is no ``is_signed``:
    superfp spends no code point on a NaN either way, so signedness does not
    move the top of the range.

    Args:
        man_bits (int): stored mantissa bits.
        exp_bits (int): exponent field width.
        normal_binades (int): binades at the top of the range that keep their
            mantissa; the rest encode supernormal powers of two.
        bias (int): the exponent bias.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.OVF_INF``.
        prng_bits (int): random bits drawn for stochastic rounding. Default: 0.
        carrier (torch.dtype): ``torch.float32`` or ``torch.float64``.
            Default: ``torch.float32``.
        warn (bool): whether to issue the warning. Default: ``True``.
        stacklevel (int): how far above this function the caller's own code
            is. Default: 3.

    Raises:
        ValueError: as :func:`check_binaryK`, and also for a
            ``normal_binades`` that leaves no supernormal codes.
    """
    error, warning = _superfp_findings(
        man_bits, exp_bits, normal_binades, bias, saturation, prng_bits, carrier
    )
    if error is not None:
        raise ValueError(error)
    if warn and warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


def check_binaryK_carrier(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype,
) -> None:
    """:func:`check_binaryK`, reported the way a call reports it.

    The same findings, with the warning naming the first frame outside mptorch
    and torch, the line that made the call, however deep inside them the call
    was reached from (a layer's backward, for instance). The arguments are
    those of :func:`check_binaryK` without ``warn`` and ``stacklevel``, and
    ``carrier`` is required.

    Raises:
        ValueError: as :func:`check_binaryK`.
    """
    _report_per_call(
        *_binaryK_findings(K, P, bias, is_signed, saturation, subnormals, prng_bits, carrier)
    )


def check_superfp_carrier(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    prng_bits: int = 0,
    *,
    carrier: torch.dtype,
) -> None:
    """:func:`check_binaryK_carrier` for a superfp format.

    The arguments are those of :func:`check_superfp` without ``warn`` and
    ``stacklevel``, and ``carrier`` is required.

    Raises:
        ValueError: as :func:`check_superfp`.
    """
    _report_per_call(
        *_superfp_findings(man_bits, exp_bits, normal_binades, bias, saturation, prng_bits, carrier)
    )


# --- what a tensor narrower than its carrier can store ------------------------
#
# A float16 or bfloat16 tensor is rounded in binary32, its default carrier,
# and the result is converted back when it is written: `SIMDTraits`' store in
# the elementwise quantizers, `store_elem` in the GEMM, a `static_cast` on the
# host, round to nearest on both backends. Under `carrier=torch.float64`
# float32 joins them: the operands are widened, rounded in binary64, and the
# result narrowed by `mptorch.quant.ops._narrowed`, once, to nearest. Either
# conversion is a second rounding. A format whose results are not all values
# of the dtype is quantized partially, as one that outruns its carrier is,
# with one difference worth saying out loud: a result above the dtype's range
# is stored as an infinity *whatever the format's saturation mode says*, so a
# `SAT_FINITE` format overflows too.
#
# Which results a call can produce depends on the op, and the two answers
# differ enough to need two rules:
#
#   a GEMM       rounds sums and products, which land anywhere, so every value
#                of its last format is some result: the format's value set
#                must be the dtype's. The carrier's "representable" question
#                asked of a narrower float:
#
#                                   float16     bfloat16    float32
#                  precision        P <= 11     P <= 8      P <= 24      raises
#                  largest value    < 2**16     < 2**128    < 2**128     warns
#                  finest step      >= 2**-24   >= 2**-133  >= 2**-149   warns
#
#   elementwise  rounds the tensor's own values, which are already the
#                dtype's. A format at least as fine as the dtype somewhere
#                hands those back unchanged there, so precision and a fine
#                bottom cost nothing, and a result off the dtype's grid comes
#                from three places only, at the format's edges: its largest
#                value inside the dtype's range but not a value of it (an
#                input above saturates onto it, in a saturating mode); the
#                dtype's largest value not on the format's grid while the
#                format reaches past it (an input there rounds up, out of the
#                dtype's range); and `EXTENDED_NORMALS`' hole, an input the
#                dtype has and the format does not, whose way up is not the
#                dtype's. Each warns.
#
# Both raise for the storage twin of the carrier's "no finite value" error, a
# format whose largest value is below the dtype's smallest, of which nothing
# is stored. And neither has the carrier's "faithful" floor: that exists
# because the casts read an exponent field, where storing a value reads
# nothing and rounds it correctly. Under binary32, for bfloat16 only the
# GEMM's precision bound is new, since its range is binary32's to within its
# precision, so a format binary32 holds with at most eight bits of precision
# is inside it at both ends. Under binary64 the carrier passes ranges no
# narrower dtype holds, so every row can speak. The rule does not otherwise
# depend on the carrier, which is wider than the dtype either way and rounds
# the dtype's own values exactly.
#
# The dtype belongs to the call, not the format (a `BinaryK` does not know
# what it will be stored in), so these run per call, from `mptorch.quant.ops`,
# on the format whose values the result holds: the elementwise quantizer's,
# and a GEMM's accumulate or fused format. A GEMM's multiply format never
# reaches storage (its products are intermediates in the carrier), and with
# the running sum left unquantized no format's values are stored at all.


class _Storage(NamedTuple):
    """A dtype a rounded result is stored in, as `_storage_findings` reads it."""

    name: str
    man_bits: int  # stored significand bits
    top_exp: int  # the exponent of the largest finite value
    min_exp: int  # the exponent of the smallest positive value, a subnormal


_STORAGE: dict[torch.dtype, _Storage] = {
    torch.float16: _Storage("float16", 10, 15, -24),
    torch.bfloat16: _Storage("bfloat16", 7, 127, -133),
    torch.float32: _Storage("float32", 23, 127, -149),
}


def _normalized(M: int, k: int) -> tuple[int, int, int]:
    """``M * 2**k``, for ``M > 0``, as (odd significand, its exponent, the
    exponent of its leading bit).

    Strips the trailing zeros of ``M`` into the exponent so that two spellings
    of one value compare equal.
    """
    tz = (M & -M).bit_length() - 1
    M >>= tz
    return M, k + tz, k + tz + M.bit_length() - 1


def _in_storage(M: int, k: int, st: _Storage) -> bool:
    """Whether ``M * 2**k`` is a finite value of the dtype.

    True when the value is on the dtype's grid at that magnitude (the normals'
    ``2**(E - man_bits)``, or the subnormals' fixed step below them) and not
    above its largest value, which with at most ``man_bits + 1`` significant
    bits is a test on the leading bit's binade.
    """
    _, k, lead = _normalized(M, k)
    min_normal_exp = st.min_exp + st.man_bits
    return lead <= st.top_exp and k >= max(lead, min_normal_exp) - st.man_bits


def _on_format_grid(M: int, k: int, ext: _Extent) -> bool:
    """Whether ``M * 2**k`` sits on the format's grid, ignoring its top.

    The top is ignored on purpose: the question asked here is which
    neighbours an input at that magnitude has, which the grid's continuation
    above the largest value answers. Below the full-mantissa binades the
    answer depends on what the format keeps down there.
    """
    M, k, lead = _normalized(M, k)
    m = ext.man_bits
    if lead >= ext.normal_exp:
        return k >= lead - m
    if ext.below == "SUBNORMALS":
        return k >= ext.normal_exp - m
    if ext.below == "EXTENDED_NORMALS" and m and lead == ext.normal_exp - 1:
        return k >= lead - m and M != 1  # that binade's power of two is the hole
    if ext.below == "SUPERNORMALS":
        return M == 1 and lead >= ext.bottom_exp
    return False


def _storage_value_set_findings(ext: _Extent, st: _Storage) -> tuple[str | None, str | None]:
    """A GEMM's rule: the format's value set against the dtype's.

    Returns ``(error, warning)``. Precision is checked first, because the two
    range bounds are only exact once it holds. With at most the dtype's
    significand bits, a format value in a binade the dtype has normals for is
    one of them, and one below that is a multiple of ``2**step_exp`` under the
    dtype's smallest normal, which is a dtype subnormal as long as
    ``step_exp`` is not below the dtype's own step; and every format value
    under ``2**(top_exp + 1)`` is at most the dtype's largest finite value as
    long as ``top_exp`` is not above its own.
    """
    label, name = ext.label, st.name
    if ext.man_bits > st.man_bits:
        return (
            f"{label} has {ext.man_bits + 1} bits of precision, and a {name} result holds "
            f"{st.man_bits + 1}: a GEMM's result can be any of the format's values, and "
            f"storing it rounds the ones {name} lacks a second time, onto {name}'s grid"
        ), None
    if ext.top_exp > st.top_exp:
        return None, (
            f"{label} reaches 2^{ext.top_exp}, above {name}'s largest finite value "
            f"2^{st.top_exp + 1} - 2^{st.top_exp - st.man_bits}: a {name} result stores "
            f"the format's values above that as infinity, whatever the saturation mode "
            f"(needs bias >= {ext.top_field - st.top_exp})"
        )
    if ext.step_exp < st.min_exp:
        return None, (
            f"{label} spaces its values 2^{ext.step_exp} apart at the bottom, below "
            f"{name}'s 2^{st.min_exp}: a {name} result rounds the values down there "
            f"a second time, onto {name}'s coarser grid, when it is stored"
        )
    return None, None


def _storage_edge_findings(
    ext: _Extent, st: _Storage, saturation: SaturationMode
) -> tuple[str | None, str | None]:
    """The elementwise rule: which of the dtype's own values round off its grid.

    Returns ``(error, warning)``; only warnings arise here. An input already
    on the format's grid comes back unchanged, and one that is not sits where
    the format's grid is coarser than the dtype's, whose neighbours there are
    therefore values of the dtype too. What is left is the three edges the
    module note above names, and nothing else.

    The second edge is reached only by saturating. A largest value that is not
    the dtype's has more significant bits than the dtype, so the format is
    finer there, and every input of the dtype above it is already on the
    format's continued grid: rounding leaves it where it is and `OVF_INF`
    makes it an infinity, which is a value of the dtype.
    """
    label, name, m = ext.label, st.name, ext.man_bits
    top = ((1 << m) + ext.top_code, ext.top_exp - m)  # the largest finite value
    dtype_top = ((1 << (st.man_bits + 1)) - 1, st.top_exp - st.man_bits)
    if ext.top_exp != st.top_exp:
        # different binades decide it, and an exponent field of up to 30 bits
        # would make lining the two up as integers a very large shift
        above, below = ext.top_exp > st.top_exp, ext.top_exp < st.top_exp
    else:
        finer = max(m, st.man_bits)
        f, d = top[0] << (finer - m), dtype_top[0] << (finer - st.man_bits)
        above, below = f > d, f < d
    if above and not _on_format_grid(*dtype_top, ext):
        return None, (
            f"{label} reaches past {name}'s largest finite value, which is not on the "
            f"format's grid: a {name} input near it rounds up out of {name}'s range and "
            f"is stored as infinity, whatever the saturation mode"
        )
    saturates = saturation is not SaturationMode.OVF_INF
    if below and saturates and not _in_storage(*top, st):
        return None, (
            f"{label}'s largest finite value, ({top[0]}) * 2^{top[1]}, is not a {name} "
            f"value: a {name} input above it saturates onto it, and storing the result "
            f"rounds it a second time"
        )
    hole = ext.normal_exp - 1
    if (
        ext.below == "EXTENDED_NORMALS"
        and m
        and _in_storage(1, hole, st)
        and not _in_storage((1 << m) + 1, hole - m, st)
    ):
        return None, (
            f"{label} has no value at 2^{hole}, and {name} does: the first value above "
            f"that hole, (2^{m} + 1) * 2^{hole - m}, is where a {name} input there rounds, "
            f"and it is not a {name} value"
        )
    return None, None


def _storage_findings(
    ext: _Extent, storage: torch.dtype, elementwise: bool, saturation: SaturationMode
) -> tuple[str | None, str | None]:
    """Hold an `_Extent` against a storage dtype under the rule ``elementwise``
    selects, after the one error both rules share: a format with no nonzero
    finite value the dtype can store.
    """
    st = _STORAGE[storage]
    if ext.top_exp < st.min_exp:
        return (
            f"{ext.label}'s largest finite value is below 2^{ext.top_exp + 1}, under "
            f"{st.name}'s smallest positive value 2^{st.min_exp}: a {st.name} result "
            f"stores none of the format's nonzero finite values"
        ), None
    if elementwise:
        return _storage_edge_findings(ext, st, saturation)
    return _storage_value_set_findings(ext, st)


@lru_cache(maxsize=256)
def _binaryK_storage_findings(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    storage: torch.dtype,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    """What storing a binaryK result in ``storage`` says, as ``(error, warning)``.

    Memoized on the parameter tuple, like `_binaryK_findings`. A format the
    carrier check has already rejected yields ``(None, None)``, since no range
    can be read from it and the carrier's error is the one to report.
    """
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)
    # Read against binary64, the wider carrier. The extent a storage rule reads
    # (the top, the step, the floor) is the format's own and the same in
    # either carrier for a format binary32 holds, and a format only binary64
    # holds can reach storage too, under `carrier=torch.float64`.
    if _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None:
        # the carrier's to report, and the carrier check runs before any call
        # reaches this: no range can be read from such a format
        return None, None
    ext = _binaryK_extent(
        label, man_bits, exp_bits, bias, is_signed, saturation, subnormals, _BINARY64
    )
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None  # likewise: no finite value at all, the carrier's error
    return _storage_findings(ext, storage, elementwise, saturation)


@lru_cache(maxsize=256)
def _superfp_storage_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    storage: torch.dtype,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    """What storing a superfp result in ``storage`` says, as ``(error, warning)``.

    The superfp twin of `_binaryK_storage_findings`.
    """
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)
    if (
        _width_error(label, exp_bits, man_bits, 0, _BINARY64) is not None
        or _superfp_regions_error(label, exp_bits, normal_binades) is not None
    ):
        return None, None  # as for binaryK: the carrier check has said so already
    ext = _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation, _BINARY64)
    if ext.top_field < 1 or ext.top_exp < _BINARY64.min_normal_exp:
        return None, None
    return _storage_findings(ext, storage, elementwise, saturation)


@lru_cache(maxsize=1)
def _library_frames() -> tuple[str, ...]:
    """The source trees a per-call warning looks past to name its caller.

    Returns the mptorch and torch package directories, each with a trailing
    separator, for ``warnings.warn``'s ``skip_file_prefixes``. A storage
    check runs where the dtype is known, inside the op, and the op is reached
    from a wrapper, a layer's math hook or an autograd Function's backward, at
    different depths and some of them under torch's own frames
    (``Module.__call__``, ``Function.apply``). So rather than count frames
    with ``stacklevel`` the warning names the first frame outside both trees,
    which is the line that made the call: the location worth reading, and the
    one ``warnings``' default once-per-location rule then keys on, so a
    training loop hears it once.
    """
    roots = [os.path.dirname(os.path.abspath(__file__))]
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.__file__:
        roots.append(os.path.dirname(os.path.abspath(torch_module.__file__)))
    return tuple(os.path.join(root, "") for root in roots)


def _report_per_call(error: str | None, warning: str | None) -> None:
    """Raise the error if there is one, else warn, naming the caller's line."""
    if error is not None:
        raise ValueError(error)
    if warning is not None:
        warnings.warn(warning, FormatRangeWarning, skip_file_prefixes=_library_frames())


def check_binaryK_storage(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    *,
    storage: torch.dtype,
    elementwise: bool = False,
) -> None:
    """Hold a binaryK format against the dtype a result is written in.

    ``storage`` is a dtype narrower than the carrier the cast rounds in:
    ``torch.float16`` or ``torch.bfloat16`` under binary32, and those or
    ``torch.float32`` under binary64. Writing the result rounds it a second
    time, onto the dtype's grid. :func:`check_binaryK_carrier` holds the
    format against the carrier, and this does not repeat it. Which results
    can land off the dtype's grid depends on what was rounded, so there are
    two rules (the module note above gives both): a quantizer's inputs are
    already the dtype's values, and only three edges of the format's range
    can move one off the grid, whereas a GEMM's sums reach every value the
    format has, so its whole value set must be the dtype's. The dtype belongs
    to the call, not the format, so :mod:`mptorch.quant.ops` runs this per
    call on the format whose values the result holds, and the warning names
    the first frame outside mptorch and torch.

    Args:
        K (int): total bits of the format.
        P (int): bits of precision, the implicit leading bit included.
        bias (int): the resolved exponent bias.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.OVF_INF``.
        subnormals (SubnormalsMode): what lies below the smallest normal.
            Default: ``SubnormalsMode.SUBNORMALS``.
        storage (torch.dtype): the dtype the result is stored in.
        elementwise (bool): ``True`` for a quantizer, ``False`` for a GEMM.
            Default: ``False``.

    Raises:
        ValueError: where nothing of the format survives being stored (its
            largest value is below the dtype's smallest), or, for a GEMM,
            where the format has more precision than the dtype.
    """
    _report_per_call(
        *_binaryK_storage_findings(
            K, P, bias, is_signed, saturation, subnormals, storage, elementwise
        )
    )


def check_superfp_storage(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    *,
    storage: torch.dtype,
    elementwise: bool = False,
) -> None:
    """:func:`check_binaryK_storage` for a superfp format.

    The same contract, with the format spelled as :func:`check_superfp`
    spells it and the same ``storage`` and ``elementwise`` keywords.

    Raises:
        ValueError: as :func:`check_binaryK_storage`.
    """
    _report_per_call(
        *_superfp_storage_findings(
            man_bits, exp_bits, normal_binades, bias, saturation, storage, elementwise
        )
    )


# --- format objects ----------------------------------------------------------
#
# One value type per simulated format, carrying exactly what that format's
# cast reads and nothing else. They exist so a caller can name a format once,
# `BinaryK(8, 4)`, instead of spelling its five or six schema arguments at
# each of the places a GEMM, a quantizer or a layer takes one, and so the two
# halves of a mixed-precision configuration can differ where the kernel
# already lets them differ.
#
# They are frozen and they resolve their derived constants in `__post_init__`,
# because the thing that reads them memoizes on the value: re-deriving a
# format per call costs 2.3-6.4 us, which is the whole reason
# `mptorch.quant.gemm`'s factories resolve once per layer. Frozen also makes
# them hashable, which is what the memo keys on, and means a format's hash
# cannot drift from the constants derived at construction.


@dataclass(frozen=True)
class Number:
    """Base class for every simulated number format.

    Subclassed by :class:`FloatFormat` (the element-by-element float
    families) and :class:`BlockFormat` (block minifloats: OCP MX and NVFP4,
    whose elements share a scale per block). Fixed-point, block floating
    point with integer mantissas, logarithmic and tapered (posit) formats
    belong here too; each needs a kernel first, so none of them is declared
    as an empty class in the meantime.

    Example::

        >>> from mptorch import BinaryK, Number
        >>> isinstance(BinaryK(8, 4), Number)
        True
    """


@dataclass(frozen=True)
class FloatFormat(Number):
    """Base class for the two floating-point families the kernels implement.

    :class:`BinaryK` and :class:`SuperFP` are its subclasses; a quantizer or
    GEMM factory that takes a ``FloatFormat`` accepts either.

    Example::

        >>> from mptorch import FloatFormat, SuperFP
        >>> isinstance(SuperFP(3, 4, 2, 7), FloatFormat)
        True
    """


@dataclass(frozen=True)
class BinaryK(FloatFormat):
    """An IEEE P3109 binaryK float: ``K`` total bits of which ``P`` are precision.

    P3109, the draft Standard for Arithmetic Formats for Machine Learning
    (Fitzgibbon, Wintersteiger and Sarnoff give an overview in
    arXiv:2606.04028), parameterizes its binary formats by bitwidth,
    precision (the implicit leading bit included), signedness and domain, and
    names them ``Binary<K>p<P><s|u><e|f>``. ``K``, ``P`` and ``is_signed`` are
    the first three, and ``saturation`` carries the domain (see
    :class:`SaturationMode`), so ``BinaryK(8, 4)`` is ``Binary8p4se`` and
    ``BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE)`` is ``Binary8p4sf``.
    Three of P3109's conventions are worth knowing: the format has one zero,
    unsigned, at the all-zero code, so no cast returns ``-0.0``; in a signed
    format the code with only the sign bit set (which would hold ``-0.0``) is
    the NaN, and in an unsigned one the top code is; and the largest finite
    value is counted in code points, so with one or two mantissa bits the
    codes reserved for infinity and NaN can push it a binade or two down.
    :doc:`/concepts` maps the rest of the standard onto this class and says
    where MPTorch goes beyond it.

    The rounding mode is not a property of the format but of the operation
    (see :class:`mptorch.quant.Quant` and :class:`mptorch.quant.SplitMac`),
    because the kernels take it as a template parameter shared by a multiply
    and its accumulate. The instance is frozen, and so hashable, because the
    code that resolves a format memoizes on its value; the derived ``bias``
    is filled in at construction rather than at each call site.

    Args:
        K (int): total bits, sign included.
        P (int): bits of precision, the implicit leading bit included, so
            ``P - 1`` mantissa bits are stored; ``1 <= P <= K``.
        bias (int, optional): exponent bias. Default: P3109's, ``2**(K-P-1)``
            signed and ``2**(K-P)`` unsigned, which encodes 1.0 at the middle
            code point. Any other bias gives a format outside the standard;
            that is how the layout of OCP's E4M3, ``BinaryK(8, 4, bias=7)``, is
            spelled.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        prng_bits (int): width of the random mantissa drawn by
            :attr:`RoundMode.SR`, ignored under every other rounding mode.
            Default: 0.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.OVF_INF``.
        subnormals (SubnormalsMode): what lies below the smallest normal; any
            value but ``SUBNORMALS`` gives a format outside the standard.
            Default: ``SubnormalsMode.SUBNORMALS``.

    Raises:
        ValueError: only for a format neither carrier can simulate: more than
            53 bits of precision, ``P - 1 + prng_bits`` above 52, an exponent
            field past 30 bits, a negative ``prng_bits``, or no finite value
            binary64 holds. Nothing warns here; whether float32 or float64
            arithmetic carries the rest is the call's to decide, so each call
            checks the format against its own carrier (see
            :class:`FormatRangeWarning`).

    Example::

        >>> from mptorch import BinaryK
        >>> BinaryK(8, 4).bias  # P3109 Binary8p4se
        8
        >>> BinaryK(8, 4, is_signed=False).bias  # Binary8p4ue
        16
        >>> BinaryK(8, 4, bias=7).man_bits  # OCP E4M3's layout
        3
    """

    K: int
    P: int
    _: KW_ONLY
    bias: int | None = None
    is_signed: bool = True
    prng_bits: int = 0
    saturation: SaturationMode = SaturationMode.OVF_INF
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS

    def __post_init__(self) -> None:
        if self.P < 1 or self.K < self.P:
            raise ValueError(f"BinaryK needs 1 <= P <= K, got K={self.K}, P={self.P}")
        if self.prng_bits < 0:
            raise ValueError(f"BinaryK prng_bits must be non-negative, got {self.prng_bits}")
        if self.bias is None:
            middle = 2 ** (self.K - self.P - 1) if self.is_signed else 2 ** (self.K - self.P)
            object.__setattr__(self, "bias", middle)
        assert self.bias is not None
        # Raise for what no carrier can do, and nothing more: the carrier
        # belongs to the call, which holds the format to it (`check_binaryK`)
        check_binaryK(
            self.K,
            self.P,
            self.bias,
            self.is_signed,
            self.saturation,
            self.subnormals,
            self.prng_bits,
            carrier=torch.float64,
            warn=False,
        )

    @property
    def man_bits(self) -> int:
        """Mantissa bits, as the cast counts them (the implicit bit is not one)."""
        return self.P - 1


@dataclass(frozen=True)
class SuperFP(FloatFormat):
    """A supernormal float, as ``common/cast_superfp.h`` implements it.

    The top ``normal_binades`` binades of the exponent range keep a full
    ``man_bits``-bit mantissa; the codes the remaining binades would have
    spent on mantissas, ``n = (2**exp_bits - normal_binades) * 2**man_bits``
    of them, encode the zero and ``n - 1`` further powers of two below them
    (the supernormals), and everything below those flushes to zero. The
    format therefore trades precision at the bottom of the range for dynamic
    range at the same width.

    Deliberately not symmetric with :class:`BinaryK`. ``bias`` is required and
    has no default, because the cast has no derivation rule for one; there is
    a single ``normal_binades`` rather than one per end of the range; and
    there is no ``subnormals`` mode, because the format has no subnormals. It
    spends no code point on a NaN, so signedness does not move the top of the
    range, and its one zero is unsigned like binaryK's. The instance is
    frozen for the same reason :class:`BinaryK` is.

    Args:
        man_bits (int): stored mantissa bits in the normal region.
        exp_bits (int): exponent field width, at least 1.
        normal_binades (int): binades at the top of the range that keep their
            mantissa; at least 1 and at most ``2**exp_bits - 1``, so that at
            least one supernormal code remains.
        bias (int): exponent bias.
        is_signed (bool): whether the format has a sign bit. Default: ``True``.
        prng_bits (int): width of the random mantissa drawn by
            :attr:`RoundMode.SR`. Default: 0.
        saturation (SaturationMode): overflow behaviour and domain.
            Default: ``SaturationMode.OVF_INF``.

    Raises:
        ValueError: only for what neither carrier can simulate, as for
            :class:`BinaryK`, and here also for a ``normal_binades`` that
            leaves no supernormal codes. Each call checks the format against
            its own carrier (see :class:`FormatRangeWarning`).

    Example::

        >>> from mptorch import SuperFP
        >>> from mptorch.quant import Quant
        >>> import torch
        >>> f = SuperFP(man_bits=3, exp_bits=4, normal_binades=2, bias=7)
        >>> x = torch.tensor([300.0, 3.0, 0.001])  # normal, supernormal, supernormal
        >>> Quant(f)(x).tolist()
        [288.0, 4.0, 0.0009765625]
    """

    man_bits: int
    exp_bits: int
    normal_binades: int
    bias: int
    _: KW_ONLY
    is_signed: bool = True
    prng_bits: int = 0
    saturation: SaturationMode = SaturationMode.OVF_INF

    def __post_init__(self) -> None:
        if self.man_bits < 0 or self.exp_bits < 1:
            raise ValueError(
                f"SuperFP needs man_bits >= 0 and exp_bits >= 1, "
                f"got man_bits={self.man_bits}, exp_bits={self.exp_bits}"
            )
        if self.normal_binades < 1:
            raise ValueError(f"SuperFP needs normal_binades >= 1, got {self.normal_binades}")
        if self.prng_bits < 0:
            raise ValueError(f"SuperFP prng_bits must be non-negative, got {self.prng_bits}")

        # Raise for what no carrier can do, and nothing more (`check_superfp`)
        check_superfp(
            self.man_bits,
            self.exp_bits,
            self.normal_binades,
            self.bias,
            self.saturation,
            self.prng_bits,
            carrier=torch.float64,
            warn=False,
        )


# --- block formats -------------------------------------------------------------
#
# A block format stores a tensor as narrow codes that share one scale per
# block: OCP's MX formats (Microscaling Formats v1.0: an E8M0 power-of-two
# scale per 32 elements) and NVIDIA's NVFP4 (an E4M3 scale per 16 elements,
# times a float32 scale per tensor). The element and the scale are each a
# *minifloat code*: a sign-magnitude layout of `exp_bits` exponent and
# `man_bits` mantissa bits over a bias, with exponent field 0 holding
# subnormals (or, under `SubnormalsMode.EXTENDED_NORMALS`, one more binade of
# normals). A `BinaryK` names that layout, so one value type covers every
# element and scale the formats use, and `csrc/common/block_decode.h` has one
# decode and one encode for all of them.
#
# The element is rounded by the `BinaryK` cast of its own format under
# `SAT_FINITE` and then clamped to `elem_max`, which is where the OCP formats
# part from P3109's: E4M3's `SAT_FINITE` top is 480 and OCP's 448, since OCP
# spends the code above on NaN; E5M2's are 114688 and 57344, below its
# infinity. The clamp is exact in every rounding mode: every value between
# `elem_max` and the format's own top rounds to a code the clamp folds onto
# `elem_max`. A scale with no mantissa (`P = 1`) is read as E8M0 is, a power
# of two `2**(code - bias)` for every code but the all-ones one, its NaN, and
# is set from the block's largest magnitude by integer arithmetic, not by a
# cast; a scale with a mantissa is cast (to nearest even, under `SAT_FINITE`)
# and spends its all-ones magnitude code on NaN in the same way.
#
# The block kernels round in binary32 only so far (`dev/continuation_plan.md`,
# phase G), so a block format's element, and a scale that is cast, must be
# formats binary32 holds entirely: those are what `BlockFormat` checks when it
# is built, as `BinaryK` checks what no carrier can do. A call on float64
# tensors, or with `carrier=torch.float64`, raises per call.


class _Minifloat(NamedTuple):
    """A `BinaryK`'s or a `SuperFP`'s code layout, as the block kernels encode
    and decode it."""

    is_signed: bool
    exp_bits: int
    man_bits: int
    bias: int
    extended: bool  # EXTENDED_NORMALS: exponent field 0 is a binade of normals
    # superfp: magnitude codes below `super_codes` are the supernormals, code c
    # the power of two 2**(super_cutoff + c - 1); 0 for a binaryK layout
    super_codes: int = 0
    super_cutoff: int = 0

    @property
    def bits(self) -> int:
        """Bits per code, the sign's included."""
        return int(self.is_signed) + self.exp_bits + self.man_bits


def _bias(f: BinaryK) -> int:
    """A built ``BinaryK``'s bias, which ``__post_init__`` has resolved."""
    assert f.bias is not None
    return f.bias


def _minifloat(f: "BinaryK | SuperFP") -> _Minifloat:
    """The code layout of ``f``: P3109's field widths in sign-magnitude for a
    ``BinaryK``, and for a ``SuperFP`` its fields and its supernormal codes
    (``cast_superfp.h``'s ``superfp_region_cutoffs``)."""
    if isinstance(f, SuperFP):
        normal_cutoff = (1 << f.exp_bits) - f.normal_binades - f.bias
        super_codes = ((1 << f.exp_bits) - f.normal_binades) << f.man_bits
        return _Minifloat(
            f.is_signed,
            f.exp_bits,
            f.man_bits,
            f.bias,
            False,
            super_codes,
            normal_cutoff - super_codes + 1,
        )
    exp_bits = f.K - f.P if f.is_signed else f.K - f.P + 1
    return _Minifloat(
        f.is_signed, exp_bits, f.P - 1, _bias(f), f.subnormals is SubnormalsMode.EXTENDED_NORMALS
    )


def _minifloat_value(mf: _Minifloat, mag: int) -> Fraction:
    """The magnitude a code (sign bit removed) encodes, specials aside."""
    if mag < mf.super_codes:
        return Fraction(0) if mag == 0 else Fraction(2) ** (mf.super_cutoff + mag - 1)
    e, m = mag >> mf.man_bits, mag & ((1 << mf.man_bits) - 1)
    if e == 0 and not mf.extended:
        return Fraction(m) * Fraction(2) ** (1 - mf.bias - mf.man_bits)
    if e == 0 and m == 0:
        return Fraction(0)  # EXTENDED_NORMALS keeps its binade's foot for the zero
    return Fraction((1 << mf.man_bits) + m) * Fraction(2) ** (e - mf.bias - mf.man_bits)


def _floor_log2(value: Fraction) -> int:
    """``floor(log2(value))`` of a positive rational, exactly."""
    num, den = value.numerator, value.denominator
    e = num.bit_length() - den.bit_length()
    if (num << -e if e < 0 else num) < (den if e < 0 else den << e):
        e -= 1
    return e


def _minifloat_code(mf: _Minifloat, value: Fraction) -> int | None:
    """The magnitude code of a positive ``value``, or ``None`` if the layout has
    no code for it."""
    e = _floor_log2(value)
    if mf.super_codes and e - mf.super_cutoff + 1 < mf.super_codes:
        # below a superfp's normal binades only its powers of two are values
        code = e - mf.super_cutoff + 1
        return code if code >= 1 and value == Fraction(2) ** e else None
    lowest = -mf.bias if mf.extended else 1 - mf.bias
    if e >= lowest:
        field_ = e + mf.bias
        m = (value / Fraction(2) ** e - 1) * (1 << mf.man_bits)
        if m.denominator != 1 or field_ >= (1 << mf.exp_bits) or (field_ == 0 and m == 0):
            return None
        return (field_ << mf.man_bits) | int(m)
    if mf.extended:
        return None
    m = value / Fraction(2) ** (1 - mf.bias - mf.man_bits)
    return int(m) if m.denominator == 1 else None


def _significant_bits(value: Fraction) -> int:
    """How many bits the significand of a dyadic ``value`` needs."""
    num = value.numerator
    return (num >> ((num & -num).bit_length() - 1)).bit_length()


def _block_part_error(role: str, f: "BinaryK | SuperFP") -> str | None:
    """Why a block format cannot use ``f`` as its element or cast scale, or ``None``.

    The block kernels round in binary32 only, and assemble decoded values on
    the binary32 word, so the format rounded with (``f`` under ``SAT_FINITE``)
    has to be one binary32 holds entirely: no error and no range warning.
    """
    if isinstance(f, SuperFP):
        error, warning = _superfp_findings(
            f.man_bits,
            f.exp_bits,
            f.normal_binades,
            f.bias,
            SaturationMode.SAT_FINITE,
            f.prng_bits,
            torch.float32,
        )
    else:
        error, warning = _binaryK_findings(
            f.K,
            f.P,
            f.bias,
            f.is_signed,
            SaturationMode.SAT_FINITE,
            f.subnormals,
            f.prng_bits,
            torch.float32,
        )
    found = error or warning
    if found is None:
        return None
    return (
        f"a BlockFormat's {role} must be a format binary32 holds entirely, since the block "
        f"kernels round in binary32 only so far (dev/continuation_plan.md, phase G): {found}"
    )


class ScaleRounding(Enum):
    """How a :class:`BlockFormat` chooses a block's scale from the block's
    largest magnitude ``amax``.

    Mirrors ``ScaleRounding`` in ``csrc/common/block_decode.h``, by name and
    value. Below, ``r = amax / elem_max`` (divided by the tensor scale too, for
    a cast scale) is the scale that would take ``amax`` exactly onto
    ``elem_max``, and ``T`` is ``elem_max`` plus half the element format's step
    above it (7 for E2M1, whose values end 4, 6), the largest scaled element
    that rounding to nearest would still take to ``elem_max``. Every rule is
    then clamped to the scale's codes (and a zero cast scale raised to the
    smallest positive one), and none depends on the elements' rounding mode.
    With E2M1 elements a block's largest element lands, in E2M1's units, in
    ``[4, 8)`` under :attr:`OCP`, ``[4.5, 9]`` under :attr:`NEAREST` (from a
    power-of-two scale), ``(3, 6]`` under :attr:`UP` and ``(3.5, 7]`` under
    :attr:`SELECTIVE`; anything above 6 saturates to 6.

    Example::

        >>> import dataclasses
        >>> from mptorch import MXFP4_E2M1, ScaleRounding
        >>> MXFP4_E2M1.scale_rule
        <ScaleRounding.OCP: 0>
        >>> dataclasses.replace(MXFP4_E2M1, scale_rounding=ScaleRounding.UP).scale_rule
        <ScaleRounding.UP: 2>
    """

    #: OCP MX's rule, for a power-of-two scale only: ``2**(floor(log2(amax)) -
    #: emax)``. A block's largest element can pass ``elem_max`` by up to
    #: ``2**(emax + 1) / elem_max`` (4/3 for E2M1) and saturate. The default
    #: for a power-of-two scale (E8M0).
    OCP = 0
    #: ``r`` rounded to nearest even in the scale format. The scale can fall
    #: half a step of it below ``r``: a factor of 1.0625 in E4M3's normals,
    #: which E2M1's top step absorbs, and of 1.5 where the scale's values are
    #: powers of two, which saturates the largest element by up to a third.
    #: NVFP4's rule, and the default for a binaryK scale with a mantissa
    #: (E4M3).
    NEAREST = 1
    #: The smallest scale at least ``r``: no element passes ``elem_max``, at
    #: the price of a grid up to twice as coarse for the rest of the block.
    #: The rule NVIDIA's MXFP8 pretraining recipe uses for E8M0 (arXiv
    #: 2506.08027).
    UP = 2
    #: :attr:`NEAREST`, or the next scale up where the block's largest
    #: element would land above ``T``: only where the clamp at ``elem_max``
    #: would move it further than rounding to nearest does. From a power of
    #: two, the smallest scale with ``amax / scale <= T``. The default for a
    #: :class:`SuperFP` scale, whose values below its normal binades are
    #: powers of two.
    SELECTIVE = 3


@dataclass(frozen=True)
class BlockFormat(Number):
    """A block format: narrow element codes that share one scale per block.

    Covers OCP's Microscaling (MX) formats and NVIDIA's NVFP4, and the same
    construction over :class:`SuperFP` elements and scales. A tensor is split
    into blocks of ``block_size`` consecutive elements along one axis (the
    *packed* axis, which the functions that take a block format name with
    ``axis``), and each block stores one ``scale`` code and ``block_size``
    element codes. An element decodes to its ``elem`` value times the block's
    scale, times a per-tensor float32 scale when ``scale`` is cast rather than
    a power of two (NVFP4's E4M3, or a superfp). :func:`mptorch.quant.block_quantize` rounds a
    tensor to the format, :func:`mptorch.quant.block_pack` stores it in
    ``uint8`` codes, and :func:`mptorch.quant.block_matmul` multiplies two
    packed operands.

    With ``block_rows > 1`` a scale is shared by a *tile* of ``block_rows``
    consecutive rows of ``block_size`` elements: rows along the axis before
    the packed one, never across a batch dimension. A square tile
    (``block_rows == block_size``) quantizes a matrix to the same values
    whichever of its two axes is packed, in every deterministic rounding
    mode, so one packed weight serves a layer's forward pass and its input
    gradient (read transposed). ``dataclasses.replace(NVFP4, block_rows=16)``
    is NVFP4 with 16 x 16 weight tiles.

    The block scale is chosen from the block's largest magnitude ``amax`` by
    ``scale_rounding`` (:class:`ScaleRounding`). By default a power-of-two
    scale (``scale.P == 1``, E8M0) is ``2**(floor(log2(amax)) - emax)``,
    clamped to its codes, where ``emax = floor(log2(elem_max))``, as OCP MX
    specifies; a ``BinaryK`` scale with a mantissa (E4M3) is ``amax /
    (elem_max * tensor_scale)`` rounded to nearest even by the scale format's
    cast, as NVFP4 specifies; and a ``SuperFP`` scale rounds that ratio the
    same way unless the block's largest element would then saturate past
    what rounding to nearest allows, where it takes the next scale up. A
    cast scale is clamped to its largest value and raised to its smallest
    positive one; the tensor scale defaults to ``max(|x|) / (elem_max *
    scale_max)`` over the whole tensor ``x``. Each element is divided by its
    scale and rounded, in the call's rounding mode, by ``elem``'s cast (a
    binaryK or a superfp one) under ``SaturationMode.SAT_FINITE``, then
    clamped to ``elem_max`` (``elem``'s own ``saturation`` is not read; its
    ``subnormals`` and ``prng_bits`` are). An infinite input saturates to
    ``elem_max``. A NaN input takes ``nan_code`` when the element format has
    one; otherwise the block's scale becomes the scale format's NaN, so the
    whole block (every element of a 2D tile) decodes to NaN; with neither
    (``scale=None`` and no ``nan_code``) it becomes ``elem_max``. No value
    decodes to ``-0.0``.

    A superfp code is laid out as its format is: the normal binades' codes as
    a binaryK's, and below them, from code 1, the supernormal powers of two
    (see :class:`SuperFP`). It spends no code on NaN or infinity unless
    ``nan_code``/``inf_code`` say so, as for a binaryK element, and a superfp
    scale spends its all-ones code on NaN, as every cast scale does.

    Args:
        elem (BinaryK or SuperFP): the element format, at most 16 bits, one
            binary32 holds entirely.
        scale (BinaryK or SuperFP, optional): the scale format, one byte: E8M0
            (``BinaryK(8, 1, bias=127, is_signed=False,
            subnormals=SubnormalsMode.EXTENDED_NORMALS)``), a binaryK with a
            mantissa such as E4M3, or a superfp; the last two also bring a
            per-tensor scale. ``None`` stores the element codes unscaled.
        block_size (int): elements per block along the packed axis, a power
            of two in [2, 128] with ``elem_bits * block_size`` a multiple of
            8.
        block_rows (int): rows per scale tile, a power of two in [1, 128]
            with ``block_rows * block_size <= 16384``; 1 without a scale.
            Default: ``1``
        elem_max (float, optional): the largest magnitude an element takes, a
            value of ``elem``. Default: ``None``, ``elem``'s largest finite
            value under ``SAT_FINITE``.
        nan_code (int, optional): the magnitude code (sign bit clear) the
            element format spends on NaN, above ``elem_max``'s. Default:
            ``None``
        inf_code (int, optional): the magnitude code it spends on infinity,
            decoded but never produced. Default: ``None``
        scale_rounding (ScaleRounding, optional): how a block's scale is
            chosen; ``OCP`` needs a power-of-two scale. Default: ``None``,
            the scale's own rule: ``OCP`` for a power-of-two scale,
            ``NEAREST`` for a binaryK with a mantissa, ``SELECTIVE`` for a
            superfp. It is resolved against the scale each time the format
            is built, so ``dataclasses.replace(NVFP4, scale=SuperFP(...))``
            takes the superfp's rule.

    Derived when the format is built, and read-only: ``scale_rule``, the
    :class:`ScaleRounding` in effect (``None`` without a scale), which is
    what formats compare by, so leaving ``scale_rounding`` to its default and
    naming the default make equal formats; ``elem_bits``, the
    bits per element code, the sign's included (``elem.K`` for a binaryK);
    ``emax``, ``floor(log2(elem_max))``; ``scale_max``, the largest scale, the
    value of the code below the scale's NaN (448 for E4M3, ``2**127`` for
    E8M0, 0 without a scale); ``has_tensor_scale``, whether the scale is cast
    (anything but E8M0-style), which brings the per-tensor float32 scale;
    ``bytes_per_block``,
    the bytes of element codes in one row of a block; and ``is_square``,
    ``block_rows == block_size``.

    Raises:
        ValueError: for sizes outside the ranges above, an element over 16
            bits or a scale over 8, an element or cast scale binary32 does not
            hold entirely, an ``elem_max`` that is not a value of ``elem``, a
            ``nan_code`` or ``inf_code`` at or below ``elem_max``'s code, a
            power-of-two scale that is signed or reaches past ``2**-127`` or
            ``2**127`` (the pack multiplies by its reciprocal), or a
            ``scale_rounding`` of ``OCP`` on a cast scale or of anything
            without a scale.
        TypeError: if ``elem`` or ``scale`` is not a ``BinaryK`` or a
            ``SuperFP``.

    Example::

        >>> from mptorch import MXFP8_E4M3, NVFP4
        >>> MXFP8_E4M3.elem_max, MXFP8_E4M3.emax, MXFP8_E4M3.bytes_per_block
        (448.0, 8, 32)
        >>> NVFP4.scale_max, NVFP4.has_tensor_scale
        (448.0, True)
        >>> import dataclasses
        >>> dataclasses.replace(NVFP4, block_rows=16).is_square
        True
        >>> from mptorch import E2M1, E8M0, BlockFormat, SuperFP
        >>> sfp4 = BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32)  # 0 .. 6, down to 1/8
        >>> sfp4.elem_bits, sfp4.elem_max, sfp4.emax
        (4, 6.0, 2)
        >>> BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16).scale_rule
        <ScaleRounding.SELECTIVE: 3>
    """

    elem: "BinaryK | SuperFP"
    scale: "BinaryK | SuperFP | None"
    block_size: int
    _: KW_ONLY
    block_rows: int = 1
    elem_max: float | None = None
    nan_code: int | None = None
    inf_code: int | None = None
    scale_rounding: ScaleRounding | None = field(default=None, compare=False)
    scale_rule: ScaleRounding | None = field(init=False, repr=False)
    elem_bits: int = field(init=False, repr=False, compare=False)
    emax: int = field(init=False, repr=False, compare=False)
    scale_max: float = field(init=False, repr=False, compare=False)
    has_tensor_scale: bool = field(init=False, repr=False, compare=False)
    bytes_per_block: int = field(init=False, repr=False, compare=False)
    is_square: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        elem, scale = self.elem, self.scale
        if not isinstance(elem, BinaryK | SuperFP):
            raise TypeError(
                f"a BlockFormat's element is a BinaryK or a SuperFP, got {type(elem).__name__} "
                "(fixed-point elements are dev/continuation_plan.md's phase K)"
            )
        mf = _minifloat(elem)
        bits = mf.bits
        if bits > 16:
            raise ValueError(f"a BlockFormat's element codes are at most 16 bits, got {bits}")
        bs, br = self.block_size, self.block_rows
        if not 2 <= bs <= 128 or bs & (bs - 1):
            raise ValueError(f"block_size must be a power of two in [2, 128], got {bs}")
        if (bits * bs) % 8:
            raise ValueError(
                f"a block of {bs} {bits}-bit codes is {bits * bs} bits; a block must fill "
                "whole bytes, so elem_bits * block_size must be a multiple of 8"
            )
        if not 1 <= br <= 128 or br & (br - 1) or br * bs > 16384:
            raise ValueError(
                f"block_rows must be a power of two in [1, 128] with block_rows * block_size "
                f"<= 16384, got block_rows={br} and block_size={bs}"
            )
        error = _block_part_error("element", elem)
        if error is not None:
            raise ValueError(error)

        # P3109's NaN at the top of an unsigned binaryK; superfp spends no code on one
        reserved = 0 if elem.is_signed or isinstance(elem, SuperFP) else 1
        _, top_exp, top_code = _top_of_range(mf.exp_bits, mf.man_bits, mf.bias, reserved, _BINARY64)
        top = Fraction((1 << mf.man_bits) + top_code) * Fraction(2) ** (top_exp - mf.man_bits)
        if self.elem_max is None:
            elem_max = top
        else:
            elem_max = Fraction(self.elem_max)
            if not 0 < elem_max <= top or _minifloat_code(mf, elem_max) is None:
                raise ValueError(
                    f"elem_max must be a positive value of the element format {elem} no larger "
                    f"than its largest finite value {float(top)}, got {self.elem_max}"
                )
        max_code = _minifloat_code(mf, elem_max)
        assert max_code is not None
        magnitudes = 1 << (bits - int(elem.is_signed))
        for name, code in (("nan_code", self.nan_code), ("inf_code", self.inf_code)):
            if code is not None and not max_code < code < magnitudes:
                raise ValueError(
                    f"{name} is a magnitude code above elem_max's ({max_code:#x}) and below "
                    f"{magnitudes:#x}, which the encoder never produces for a value; got {code}"
                )
        if self.nan_code is not None and self.nan_code == self.inf_code:
            raise ValueError("nan_code and inf_code must differ")

        rounding = self.scale_rounding
        if rounding is not None and not isinstance(rounding, ScaleRounding):
            raise TypeError(
                f"scale_rounding is a ScaleRounding or None, got {type(rounding).__name__}"
            )
        if scale is None:
            if br != 1:
                raise ValueError("block_rows is the rows a scale is shared by, and needs a scale")
            if rounding is not None:
                raise ValueError("scale_rounding chooses a block's scale, and needs a scale")
            scale_max, tensor_scaled = 0.0, False
        else:
            if not isinstance(scale, BinaryK | SuperFP):
                raise TypeError(
                    "a BlockFormat's scale is a BinaryK, a SuperFP or None, got "
                    f"{type(scale).__name__}"
                )
            smf = _minifloat(scale)
            if smf.bits > 8:
                raise ValueError(f"a BlockFormat's scale is one byte, got {smf.bits} bits")
            if isinstance(scale, BinaryK) and scale.P == 1:
                if scale.is_signed:
                    raise ValueError(
                        "a power-of-two scale (P=1, E8M0) is unsigned: its codes are exponents"
                    )
                lo, hi = -_bias(scale), (1 << scale.K) - 2 - _bias(scale)
                if lo < -127 or hi > 127:
                    raise ValueError(
                        f"a power-of-two scale's codes are 2^{lo} .. 2^{hi}, and must lie in "
                        "2^-127 .. 2^127, where binary32 holds the reciprocal the pack "
                        "multiplies by as well"
                    )
                scale_max, tensor_scaled = 2.0**hi, False
            else:
                error = _block_part_error("scale", scale)
                if error is not None:
                    raise ValueError(error)
                nan_mag = (1 << (smf.bits - int(scale.is_signed))) - 1
                scale_max, tensor_scaled = float(_minifloat_value(smf, nan_mag - 1)), True
                if scale_max <= 0:
                    raise ValueError(
                        f"the scale format {scale} has no positive value below its NaN"
                    )
                if rounding is ScaleRounding.OCP:
                    raise ValueError(
                        "ScaleRounding.OCP is OCP MX's rule for a power-of-two scale, and "
                        f"{scale} is a cast scale: use NEAREST, UP or SELECTIVE"
                    )
            if rounding is None:
                if not tensor_scaled:
                    rounding = ScaleRounding.OCP
                elif isinstance(scale, SuperFP):
                    rounding = ScaleRounding.SELECTIVE
                else:
                    rounding = ScaleRounding.NEAREST

        derived = {
            "scale_rule": rounding,
            "elem_max": float(elem_max),
            "elem_bits": bits,
            "emax": _floor_log2(elem_max),
            "scale_max": scale_max,
            "has_tensor_scale": tensor_scaled,
            "bytes_per_block": bits * bs // 8,
            "is_square": br == bs,
        }
        for name, value in derived.items():
            object.__setattr__(self, name, value)


# The per-call findings of a block format. The carrier: block kernels exist in
# binary32 only, which `BlockFormat` has already checked the format against.
_BLOCK_BINARY32_ONLY = (
    "block formats have binary32 kernels only so far, so they take float32, float16 or "
    "bfloat16 tensors and no carrier=torch.float64 (dev/continuation_plan.md, phase G)"
)


def check_block_carrier(fmt: BlockFormat, *, carrier: torch.dtype) -> None:
    """Hold a block format against the carrier a call rounds in.

    The element and scale formats were held against binary32 when ``fmt`` was
    built; what is left per call is the carrier itself, and binary64 has no
    block kernels yet.

    Raises:
        ValueError: for ``carrier=torch.float64``.
    """
    del fmt  # every BlockFormat binary32 holds was checked when it was built
    if carrier is torch.float64:
        raise ValueError(_BLOCK_BINARY32_ONLY)


def _block_label(fmt: BlockFormat) -> str:
    """The format as a short constructor-like label, for messages."""
    rows = f", block_rows={fmt.block_rows}" if fmt.block_rows != 1 else ""
    return f"BlockFormat({fmt.elem}, {fmt.scale}, {fmt.block_size}{rows})"


@lru_cache(maxsize=256)
def _block_storage_findings(
    fmt: BlockFormat, storage: torch.dtype, elementwise: bool
) -> tuple[str | None, str | None]:
    """What storing a block format's decoded values in ``storage`` says.

    Returns ``(error, warning)``; only warnings arise. A decoded value is an
    element value times its block's scale (times the tensor scale). The
    elementwise rule, for a quantizer or an unpack of a tensor of that dtype:
    with a power-of-two scale the scale follows the block's largest input, so
    a decoded value is the input itself wherever the scaled element grid is
    at least as fine as the dtype's and a point of a coarser binary grid
    elsewhere, a value of the dtype either way, except where a block
    saturates onto ``elem_max`` times its scale, which needs ``elem_max``'s
    significand to fit the dtype's. With a mantissa-bearing scale a decoded
    value is a binary32 product the dtype does not hold in general. The
    value-set rule (``elementwise=False``) asks the same of every value the
    scale's range can produce.
    """
    st = _STORAGE[storage]
    label = _block_label(fmt)
    if fmt.has_tensor_scale:
        return None, (
            f"{label}'s values are an element times its block's scale times the tensor scale, "
            f"products rounded in binary32 that {st.name} does not hold in general: a {st.name} "
            "result rounds them a second time when it is stored"
        )
    assert fmt.elem_max is not None  # resolved by __post_init__
    top = Fraction(fmt.elem_max)
    if _significant_bits(top) > st.man_bits + 1:
        return None, (
            f"{label}'s largest element value {fmt.elem_max} has {_significant_bits(top)} "
            f"significant bits and {st.name} holds {st.man_bits + 1}: a block that saturates "
            f"stores it rounded a second time"
        )
    if fmt.scale is None:
        lo, hi = 0, 0
    elif elementwise:
        return None, None
    else:
        assert isinstance(fmt.scale, BinaryK)  # a cast scale has a tensor scale, above
        lo, hi = -_bias(fmt.scale), (1 << fmt.scale.K) - 2 - _bias(fmt.scale)
    mf = _minifloat(fmt.elem)
    top_exp = fmt.emax + hi
    if top_exp > st.top_exp:
        return None, (
            f"{label} reaches 2^{top_exp}, above {st.name}'s largest finite value: a "
            f"{st.name} result stores those values as infinity"
        )
    if mf.super_codes:
        step_exp = mf.super_cutoff + lo  # the smallest supernormal, the finest spacing
    else:
        step_exp = (-mf.bias if mf.extended else 1 - mf.bias) - mf.man_bits + lo
    if not elementwise and step_exp < st.min_exp:
        return None, (
            f"{label} spaces its values 2^{step_exp} apart at the bottom, below {st.name}'s "
            f"2^{st.min_exp}: a {st.name} result rounds them a second time when it is stored"
        )
    return None, None


def check_block_storage(
    fmt: BlockFormat, *, storage: torch.dtype, elementwise: bool = True
) -> None:
    """Hold a block format's decoded values against the dtype a result is written in.

    ``storage`` is ``torch.float16`` or ``torch.bfloat16``, whose result
    rounds a decoded value a second time where the dtype does not hold it.
    ``elementwise=True`` is the rule for :func:`mptorch.quant.block_quantize`
    and :func:`mptorch.quant.block_unpack` on a tensor of that dtype (its
    inputs are already the dtype's values); ``False`` holds every value the
    format can decode to. See :func:`check_binaryK_storage` for the two rules.

    Warns:
        FormatRangeWarning: where a decoded value can land off the dtype's grid.
    """
    _report_per_call(*_block_storage_findings(fmt, storage, elementwise))


# The element and scale formats of OCP MX v1.0 and NVFP4, spelled as the
# binaryK layouts they are (OCP's biases are IEEE 754's, one below P3109's).
E8M0 = BinaryK(8, 1, bias=127, is_signed=False, subnormals=SubnormalsMode.EXTENDED_NORMALS)
E4M3 = BinaryK(8, 4, bias=7)
E5M2 = BinaryK(8, 3, bias=15)
E2M1 = BinaryK(4, 2, bias=1)
E2M3 = BinaryK(6, 4, bias=1)
E3M2 = BinaryK(6, 3, bias=3)

# OCP MX v1.0's concrete formats (block 32, E8M0 scale) and NVFP4 (block 16,
# E4M3 scale and a float32 tensor scale). E4M3 and E5M2 spend codes on NaN and
# infinity, so their largest values are OCP's, below their SAT_FINITE tops.
MXFP8_E4M3 = BlockFormat(E4M3, E8M0, 32, elem_max=448.0, nan_code=0x7F)
MXFP8_E5M2 = BlockFormat(E5M2, E8M0, 32, elem_max=57344.0, nan_code=0x7F, inf_code=0x7C)
MXFP6_E2M3 = BlockFormat(E2M3, E8M0, 32)
MXFP6_E3M2 = BlockFormat(E3M2, E8M0, 32)
MXFP4_E2M1 = BlockFormat(E2M1, E8M0, 32)
NVFP4 = BlockFormat(E2M1, E4M3, 16)
