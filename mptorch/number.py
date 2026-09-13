__all__ = [
    "SaturationMode",
    "SubnormalsMode",
    "RoundMode",
    "AccumulateAlgorithm",
    "Number",
    "FloatFormat",
    "BinaryK",
    "SuperFP",
    "FormatRangeWarning",
]

import os
import sys
import warnings
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from functools import lru_cache
from typing import NamedTuple


class SaturationMode(Enum):
    """
    Enum for what a format does with a value beyond its largest finite one.

    Mirrors ``SaturationMode`` in ``csrc/common/modes.h``. The three are IEEE
    P3109's saturation modes, and they also carry P3109's *domain*, since a
    finite-domain format admits only ``SatFinite``: ``SAT_FINITE`` is the
    finite domain, in which the code points the extended domain spends on the
    infinities hold finite values, and the other two are the extended domain.
    NaN inputs pass through every mode unchanged.
    """

    #: P3109 ``SatFinite``, finite domain: everything, infinities included,
    #: clamps to the largest finite value.
    SAT_FINITE = 0
    #: P3109 ``SatPropagate``, extended domain: a finite result beyond the largest
    #: finite value clamps to it, and an infinite one stays infinite.
    SAT_PROPAGATE = 1
    #: P3109 ``SatNone``, extended domain (the default): a result beyond the
    #: largest finite value becomes :math:`\pm\infty` under every rounding mode,
    #: because the rounding comes first -- unlike IEEE 754's ``roundTowardZero``.
    OVF_INF = 2


class SubnormalsMode(Enum):
    """
    Enum for how a format handles the range below its smallest normal number.

    Mirrors ``SubnormalsMode`` in ``csrc/common/modes.h``. Only ``SUBNORMALS``
    is IEEE P3109's -- every P3109 format with more than one bit of precision
    has subnormals -- so the other two give formats outside the standard.

    They differ only in where the bottom of the range is. Below it all three
    behave alike, because the region has the same shape in each: the two
    candidates are zero and the smallest value, and the rounding mode picks
    between them -- nearest takes the nearer and a tie the zero, the directed
    modes take their own direction, :attr:`RoundMode.RO` takes the nonzero one,
    and :attr:`RoundMode.SR` takes it with probability :math:`|x|` over it.
    """

    SUBNORMALS = 0  #: Gradual underflow: exponent code 0 encodes subnormal numbers (the default).
    NORMALS = 1  #: No subnormals; values below the smallest normal flush to zero.
    #: No subnormals; exponent code 0 is one more binade of normal numbers --
    #: except its mantissa-zero code, which is still the zero (and with the sign
    #: bit the NaN), so that binade's values start one step above the power of
    #: two at its foot.
    EXTENDED_NORMALS = 2


class RoundMode(Enum):
    """
    Enum for floating-point rounding modes.

    Result :math:`y` is obtained from an input :math:`x` depending on the rounding mode.
    Each is one of IEEE P3109's rounding modes, whose name is given in parentheses.
    """

    RNE = 0  #: Round to nearest, ties to even (``NearestTiesToEven``)
    RNA = 1  #: Round to nearest, ties to away from zero (``NearestTiesToAway``)
    RU = 2  #: Return the smallest :math:`y` such that :math:`y \ge x` (``TowardPositive``)
    RD = 3  #: Return the largest :math:`y` such that :math:`y \le x` (``TowardNegative``)
    RZ = 4  #: Return the largest :math:`y` such that :math:`|y| \le |x|` (``TowardZero``)
    RO = 5  #: Round to odd (``ToOdd``)
    SR = 6  #: Stochastic rounding with ``prng_bits`` random bits (``StochasticA``)


class AccumulateAlgorithm(Enum):
    """
    Enum selecting how the partial products of a dot product are folded
    into its running sum inside a custom-arithmetic GEMM core (see
    :mod:`mptorch.quant.gemm`).

    Only :attr:`NAIVE` is implemented today; :attr:`KAHAN`, :attr:`BLOCK`,
    and :attr:`TREE` are reserved for future compensated-summation /
    block-summation / tree-summation accumulators (see
    ``dev/gemm_roadmap.md``).
    """

    NAIVE = 0  #: Quantize the running sum after every accumulation step.
    KAHAN = 1  #: Kahan-compensated summation (not yet implemented).
    BLOCK = 2  #: Two-level (FABSum-style) block summation (not yet implemented).
    TREE = 3  #: Pairwise tree-reduction summation (not yet implemented).


# --- what binary32 can carry -------------------------------------------------
#
# The casts do not build encodings. They round a binary32 value onto the
# format's grid and hand back a binary32 value (see `docs/source/concepts.rst`,
# "What float32 can carry"), so a format with more range or precision than
# binary32 is quantized *partially*: the result is still a tensor of plausible
# numbers, and some of the format's values are simply never among them. The
# checks below are what stops that being invisible. `dev/gemm_roadmap.md` (T4)
# derives the bounds and `dev/benchmarks/format_limits.py` measures them
# against the kernels, over ~17 M inputs per format in every rounding mode.
#
# Two bounds, not one, and they differ by a binade:
#
#   representable  every value of the format is a binary32 value. Arithmetic
#                  alone -- 24 significand bits, nothing above binary32's
#                  largest finite value, nothing spaced finer than 2**-149.
#   faithful       the cast reaches those values, for every binary32 input.
#                  Stronger, because every cast places its input by the
#                  *exponent field* of the word and all binary32 subnormals
#                  share field 0, so nothing below 2**-126 can be told apart:
#                  2**-125 is the lowest value a format may have.
#
# Together the top and bottom leave 253 binades to spend, which allows at most
# seven exponent bits in either family.
#
# What separates a warning from an error here is not which of those two bounds
# is missed but whether the format still *works*. A format whose range outruns
# binary32 quantizes perfectly well over the part that fits -- a wide binaryK
# used as a precision-only target is a real thing, and `tests/` uses several
# on purpose -- it is simply partial, and being told so is the whole point.
# That is a warning. An error is for a format that cannot function at all: a
# grid finer than binary32's (nothing survives), an exponent field wider than
# the kernels shift by, stochastic bits with no significand left to draw from,
# a superfp whose regions come out misordered, or one whose largest finite
# code is below binary32's normals, where `max_num` is zero and every nonzero
# result overflows.

_F32_PRECISION = 24  # binary32 significand bits, the implicit one included
_F32_MAN_BITS = 23
_F32_TOP_EXP = 127  # exponent of the largest finite binary32
_F32_MIN_NORMAL_EXP = -126
_F32_MIN_SUBNORMAL_EXP = -149  # exponent of the smallest positive binary32
_F32_MIN_SIMULABLE_EXP = -125  # the lowest a format's own smallest value may be
_MAX_EXP_BITS = 30  # `1 << exp_bits` is a C++ `int` in the kernels


class FormatRangeWarning(UserWarning):
    """A format whose range outruns binary32, or the float16 or bfloat16
    tensor its results are stored in, so quantizing to it is partial.

    Warned by :class:`BinaryK`, :class:`SuperFP` and the plain-integer
    wrappers of :mod:`mptorch.quant.ops` when a format reaches above
    binary32's largest finite value, spaces its values finer than
    :math:`2^{-149}`, or has a smallest value below :math:`2^{-125}` -- the
    last because every cast places an input by its binary32 exponent field,
    which all binary32 subnormals share, so the bottom binade is rounded onto
    the wrong grid. In each case the format quantizes correctly over the part
    of it binary32 holds and some of its values are simply never returned.

    Warned again, per call, when the result is a float16 or bfloat16 tensor
    and the format's range outruns *that* dtype: above its largest finite
    value, or spaced finer than its smallest subnormal. The cast still rounds
    in binary32, and storing the result rounds a second time, onto the
    dtype's grid -- with a value above the dtype's range stored as an
    infinity whatever the format's saturation mode.

    It is a warning rather than an error because such a format is still
    useful: a wide binaryK is a reasonable precision-only target, where the
    unreachable top of the range is exactly the point. Filter it with
    :func:`warnings.simplefilter` when that is what is meant.
    """


def _top_of_range(exp_bits: int, man_bits: int, bias: int, reserved: int) -> tuple[int, int, int]:
    """``make_normal_range_params``' top of the range, as (exponent field,
    exponent, mantissa code) of the largest finite value.

    Transcribed from ``bit_helper.h``: of the top binade's codes the last
    ``reserved`` hold no finite value, and with one or two mantissa bits those
    can outnumber the binade's, which moves the largest finite value a binade
    or two down. Counted in ``min(man_bits, 23)`` codes, as the kernel counts
    them -- a format asking for more than that has already been rejected for
    precision by the time this is read.
    """
    codes = 1 << min(man_bits, _F32_MAN_BITS)
    top_code = codes - 1 - reserved
    top_field = (1 << exp_bits) - 1
    while top_code < 0:
        top_code += codes
        top_field -= 1
    return top_field, top_field - bias, top_code


def _width_error(label: str, exp_bits: int, man_bits: int, prng_bits: int) -> str | None:
    """The bounds that must hold before anything shifts by ``exp_bits``."""
    if prng_bits < 0:
        # `BinaryK`/`SuperFP` refuse this before they get here; the
        # plain-integer wrappers reach it with nothing in between
        return f"{label} asks for {prng_bits} stochastic-rounding bits, which cannot be negative"
    if exp_bits > _MAX_EXP_BITS:
        return (
            f"{label} has {exp_bits} exponent bits; the kernels derive the format's "
            f"range with `1 << exp_bits` in a 32-bit int, and binary32 spans 253 "
            f"binades in any case, so at most 7 are usable"
        )
    if man_bits > _F32_MAN_BITS:
        return (
            f"{label} asks for {man_bits + 1} bits of precision; the casts round in "
            f"binary32, which has {_F32_PRECISION}, so the format's grid is finer than "
            f"anything that can be returned"
        )
    if man_bits + prng_bits > _F32_MAN_BITS:
        return (
            f"{label} asks for {prng_bits} stochastic-rounding bits below a "
            f"{man_bits}-bit mantissa; they are drawn from the binary32 significand "
            f"the rounding happens in, which has {_F32_MAN_BITS} bits to share"
        )
    return None


class _Extent(NamedTuple):
    """A format's range, in the only terms the range checks read.

    Derived once from the parameters and held against two carriers: binary32,
    which every cast rounds in, and a float16 or bfloat16 result, which the
    rounded value is stored in.
    """

    label: str
    exp_bits: int
    man_bits: int
    top_field: int  # the largest finite value's exponent field
    top_exp: int  # ... its exponent
    top_code: int  # ... and its mantissa code, of `man_bits`
    bottom_exp: int  # the exponent of the smallest positive value
    step_exp: int  # the exponent of the finest spacing anywhere in the format
    min_bottom: int  # how low `bottom_exp` may go before a binary32 cast loses it
    # What lies below the binades that carry a full mantissa, which start at
    # `normal_exp`: one of `SubnormalsMode`'s three, or superfp's supernormals.
    normal_exp: int
    below: str


def _range_findings(ext: _Extent) -> tuple[str | None, str | None]:
    """The top and bottom of the range against binary32's, after the widths.

    ``bottom_exp`` is the exponent of the format's smallest positive value and
    ``step_exp`` that of its finest spacing; for a format whose lowest binade
    carries a mantissa the two differ by ``man_bits``. ``min_bottom`` is how
    low that smallest value may go, which is not the same for every format --
    see `_binaryK_extent`.

    Only the first finding is returned, so a format that outruns binary32 at
    both ends says so about the top first and about the bottom once that is
    fixed. The error is the one case that stops the format working at all.
    """
    label, top_exp, bottom_exp, step_exp = ext.label, ext.top_exp, ext.bottom_exp, ext.step_exp
    if ext.top_field < 1 or top_exp < _F32_MIN_NORMAL_EXP:
        return (
            f"{label} has no finite normal value binary32 can hold: the largest "
            f"finite code sits at 2^{top_exp}, so every nonzero result overflows"
        ), None
    if top_exp > _F32_TOP_EXP:
        return None, (
            f"{label} reaches 2^{top_exp}, above binary32's 2^{_F32_TOP_EXP}: the cast "
            f"saturates at binary32's largest value on the format's grid instead, and "
            f"the codes above it are unreachable (needs bias >= {(1 << ext.exp_bits) - 128})"
        )
    if step_exp < _F32_MIN_SUBNORMAL_EXP:
        return None, (
            f"{label} spaces its values 2^{step_exp} apart at the bottom, below "
            f"binary32's 2^{_F32_MIN_SUBNORMAL_EXP}: values that far down are not "
            f"binary32 values at all, so the bottom of the range is unreachable"
        )
    if bottom_exp < ext.min_bottom:
        return None, (
            f"{label}'s smallest value is 2^{bottom_exp}, below 2^{ext.min_bottom}: the "
            f"casts place an input by its binary32 exponent field, which every binary32 "
            f"subnormal shares, so this format's values below 2^{_F32_MIN_NORMAL_EXP} "
            f"are rounded onto the wrong grid and some are never returned. See "
            f'concepts.rst, "What float32 can carry".'
        )
    return None, None


def _binaryK_label(K: int, P: int, bias: int, is_signed: bool, subnormals: SubnormalsMode) -> str:
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
) -> _Extent:
    """A binaryK format's range -- once `_width_error` has passed, since this
    shifts by ``exp_bits``."""
    # P3109's reserved codes: +infinity outside the finite domain, and NaN
    # above it in an unsigned format (`make_binaryK_params`)
    reserved = (saturation is not SaturationMode.SAT_FINITE) + (0 if is_signed else 1)
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved)
    min_exp = 1 - bias  # the smallest normal's exponent
    if subnormals is SubnormalsMode.SUBNORMALS:
        # the subnormals are one grid of fixed spacing, and its step is both
        # the smallest value and the finest spacing in the format
        bottom_exp = step_exp = min_exp - man_bits
    elif subnormals is SubnormalsMode.NORMALS:
        bottom_exp, step_exp = min_exp, min_exp - man_bits
    else:
        # EXTENDED_NORMALS: one more binade of normals below the rest, minus
        # its mantissa-zero code, which is still the zero -- so the smallest
        # value is one step into that binade, and with man_bits == 0 the step
        # is the whole binade and the floor is the one NORMALS has
        bottom_exp = min_exp - 1 if man_bits else min_exp
        step_exp = bottom_exp - man_bits
    # How low the floor may go is not the same for the three modes, and the
    # difference is which of them reads an exponent field down there.
    # `SUBNORMALS` derives its grid from the input's, which every binary32
    # subnormal shares, so it is blind below 2**-126 and needs its floor a
    # binade above that. `NORMALS` and `EXTENDED_NORMALS` decide by comparing
    # magnitudes as words, which reads no field at all, so they are exact to
    # 2**-126 -- and no further, because below it `min_exponent_store` reaches
    # zero, the floor leaves binary32's normals and nothing flushes (T5).
    min_bottom = (
        _F32_MIN_SIMULABLE_EXP if subnormals is SubnormalsMode.SUBNORMALS else _F32_MIN_NORMAL_EXP
    )
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
        min_exp,
        subnormals.name,
    )


def _superfp_label(man_bits: int, exp_bits: int, normal_binades: int, bias: int) -> str:
    return (
        f"SuperFP(man_bits={man_bits}, exp_bits={exp_bits}, "
        f"normal_binades={normal_binades}, bias={bias})"
    )


def _superfp_regions_error(label: str, exp_bits: int, normal_binades: int) -> str | None:
    """The one superfp layout no range can be read from."""
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
) -> _Extent:
    """A superfp format's range -- once `_width_error` and
    `_superfp_regions_error` have passed."""
    binades = 1 << exp_bits
    # superfp spends no code on a NaN, signed or not (`make_superfp_params`)
    reserved = saturation is not SaturationMode.SAT_FINITE
    top_field, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved)
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
        _F32_MIN_SIMULABLE_EXP,  # the supernormal arm rounds the input's exponent field (T4)
        normal_cutoff,
        "SUPERNORMALS",
    )


# The derivations are memoized and the reporting is not, deliberately.
# `binaryK_matmul` resolves its formats on every call (finding P4), so this
# sits on a per-call path: a cache hit is 0.25 us against 1.04 to redo the
# derivation, and `check_binaryK` around it is 0.40. Caching the *warning*
# instead would make it fire once per process and then go quiet, which is a
# worse contract than `warnings`' own once-per-location default and would make
# the tests order-dependent.


@lru_cache(maxsize=256)
def _binaryK_findings(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: SaturationMode,
    subnormals: SubnormalsMode,
    prng_bits: int,
) -> tuple[str | None, str | None]:
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)
    error = _width_error(label, exp_bits, man_bits, prng_bits)
    if error is not None:
        return error, None
    return _range_findings(
        _binaryK_extent(label, man_bits, exp_bits, bias, is_signed, saturation, subnormals)
    )


@lru_cache(maxsize=256)
def _superfp_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    prng_bits: int,
) -> tuple[str | None, str | None]:
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)
    error = _width_error(label, exp_bits, man_bits, prng_bits) or _superfp_regions_error(
        label, exp_bits, normal_binades
    )
    if error is not None:
        return error, None
    return _range_findings(
        _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation)
    )


def check_binaryK(
    K: int,
    P: int,
    bias: int,
    is_signed: bool = True,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    subnormals: SubnormalsMode = SubnormalsMode.SUBNORMALS,
    prng_bits: int = 0,
    *,
    stacklevel: int = 3,
) -> None:
    """Hold a binaryK format against what binary32 can carry.

    Raises :exc:`ValueError` for a format that cannot function -- see the note
    above for the five ways -- and warns with :class:`FormatRangeWarning` for
    one whose range outruns binary32, which quantizes partially rather than
    wrongly. ``bias`` is the resolved one, not ``None``.

    :class:`BinaryK` calls this, and so do the plain-integer wrappers in
    :mod:`mptorch.quant.ops`, which never see a format object -- one rule, so
    ``binaryK_matmul(a, b, mul_K=16, mul_P=8)`` says what ``BinaryK(16, 8)``
    says. ``stacklevel`` is how far above this the caller's own code is, so
    the warning points there: 3 from a wrapper that calls this directly, 4
    from one more frame up.
    """
    error, warning = _binaryK_findings(K, P, bias, is_signed, saturation, subnormals, prng_bits)
    if error is not None:
        raise ValueError(error)
    if warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


def check_superfp(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode = SaturationMode.OVF_INF,
    prng_bits: int = 0,
    *,
    stacklevel: int = 3,
) -> None:
    """:func:`check_binaryK` for a superfp format -- see it for the contract.

    There is no ``is_signed``: superfp spends no code point on a NaN either
    way, so signedness does not move the top of the range.
    """
    error, warning = _superfp_findings(
        man_bits, exp_bits, normal_binades, bias, saturation, prng_bits
    )
    if error is not None:
        raise ValueError(error)
    if warning is not None:
        warnings.warn(warning, FormatRangeWarning, stacklevel=stacklevel)


# --- what float16 and bfloat16 can store --------------------------------------
#
# The casts round in binary32 whatever the operand dtype, and a float16 or
# bfloat16 result is converted back when it is written -- `SIMDTraits`' store
# in the elementwise quantizers, `store_elem` in the GEMM, a `static_cast` on
# the host, round to nearest on both backends. That conversion is a second
# rounding. A format whose results are not all values of the dtype is
# quantized partially in the way T4's are, with one difference worth saying
# out loud: a result above the dtype's range is stored as an infinity
# *whatever the format's saturation mode says*, so a `SAT_FINITE` format
# overflows too.
#
# Which results a call can produce depends on the op, and the two answers
# differ enough to need two rules (`dev/gemm_roadmap.md`, T6):
#
#   a GEMM       rounds sums and products, which land anywhere, so every value
#                of its last format is some result: the format's value set
#                must be the dtype's. T4's "representable" question with a
#                narrower carrier --
#
#                                   float16     bfloat16
#                  precision        P <= 11     P <= 8      raises
#                  largest value    < 2**16     < 2**128    warns
#                  finest step      >= 2**-24   >= 2**-133  warns
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
# Both raise for the storage twin of T4's `max_num == 0`, a format whose
# largest value is below the dtype's smallest, of which nothing is stored. And
# neither has T4's "faithful" floor: that exists because the casts read a
# binary32 exponent field, where storing a value reads nothing and rounds it
# correctly. For bfloat16 only the GEMM's precision bound is new -- its range is
# binary32's to within its precision, so a format T4 passes with at most eight
# bits of precision is inside it at both ends.
#
# The dtype belongs to the call, not the format -- a `BinaryK` does not know
# what it will be stored in -- so these run per call, from `mptorch.quant.ops`,
# on the format whose values the result holds: the elementwise quantizer's,
# and a GEMM's accumulate or fused format. A GEMM's multiply format never
# reaches storage (its products are binary32 intermediates), and with the
# running sum left unquantized no format's values are stored at all.


class _Storage(NamedTuple):
    """A dtype a rounded result is stored in, as `_storage_findings` reads it."""

    name: str
    man_bits: int  # stored significand bits
    top_exp: int  # the exponent of the largest finite value
    min_exp: int  # the exponent of the smallest positive value, a subnormal


_STORAGE: dict[str, _Storage] = {
    "float16": _Storage("float16", 10, 15, -24),
    "bfloat16": _Storage("bfloat16", 7, 127, -133),
}


def _normalized(M: int, k: int) -> tuple[int, int, int]:
    """``M * 2**k``, for ``M > 0``, as (odd significand, its exponent, the
    exponent of its leading bit)."""
    tz = (M & -M).bit_length() - 1
    M >>= tz
    return M, k + tz, k + tz + M.bit_length() - 1


def _in_storage(M: int, k: int, st: _Storage) -> bool:
    """Whether ``M * 2**k`` is a finite value of the dtype.

    On its grid at that magnitude -- the normals' ``2**(E - man_bits)``, or
    the subnormals' fixed step below them -- and not above its largest value,
    which with at most ``man_bits + 1`` significant bits is the binade test.
    """
    _, k, lead = _normalized(M, k)
    min_normal_exp = st.min_exp + st.man_bits
    return lead <= st.top_exp and k >= max(lead, min_normal_exp) - st.man_bits


def _on_format_grid(M: int, k: int, ext: _Extent) -> bool:
    """Whether ``M * 2**k`` sits on the format's grid, below its largest value
    or not: the question is which neighbours an input there has."""
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

    Precision first, because the two range bounds are only exact once it
    holds. With at most the dtype's significand bits, a format value in a
    binade the dtype has normals for is one of them, and one below that is a
    multiple of ``2**step_exp`` under the dtype's smallest normal, which is a
    dtype subnormal as long as ``step_exp`` is not below the dtype's own
    step; and every format value under ``2**(top_exp + 1)`` is at most the
    dtype's largest finite value as long as ``top_exp`` is not above its own.
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

    An input already on the format's grid comes back unchanged, and one that
    is not sits where the format's grid is coarser than the dtype's, whose
    neighbours there are therefore values of the dtype too. What is left is
    the three edges the note above names, and nothing else.

    The second is reached only by saturating. A largest value that is not the
    dtype's has more significant bits than the dtype, so the format is finer
    there, and every input of the dtype above it is already on the format's
    continued grid: rounding leaves it where it is and `OVF_INF` makes it an
    infinity, which is a value of the dtype.
    """
    label, name, m = ext.label, st.name, ext.man_bits
    top = ((1 << m) + ext.top_code, ext.top_exp - m)  # the format's largest finite value
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
    ext: _Extent, storage: str, elementwise: bool, saturation: SaturationMode
) -> tuple[str | None, str | None]:
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
    storage: str,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    label = _binaryK_label(K, P, bias, is_signed, subnormals)
    if _width_error(label, exp_bits, man_bits, 0) is not None:
        # binary32's to report, and `check_binaryK` runs before any call
        # reaches this: no range can be read from such a format
        return None, None
    ext = _binaryK_extent(label, man_bits, exp_bits, bias, is_signed, saturation, subnormals)
    if ext.top_field < 1 or ext.top_exp < _F32_MIN_NORMAL_EXP:
        return None, None  # likewise: no finite value at all, binary32's error
    return _storage_findings(ext, storage, elementwise, saturation)


@lru_cache(maxsize=256)
def _superfp_storage_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    storage: str,
    elementwise: bool,
) -> tuple[str | None, str | None]:
    label = _superfp_label(man_bits, exp_bits, normal_binades, bias)
    if (
        _width_error(label, exp_bits, man_bits, 0) is not None
        or _superfp_regions_error(label, exp_bits, normal_binades) is not None
    ):
        return None, None  # as for binaryK: `check_superfp` has said so already
    ext = _superfp_extent(label, man_bits, exp_bits, normal_binades, bias, saturation)
    if ext.top_field < 1 or ext.top_exp < _F32_MIN_NORMAL_EXP:
        return None, None
    return _storage_findings(ext, storage, elementwise, saturation)


@lru_cache(maxsize=1)
def _library_frames() -> tuple[str, ...]:
    """The source trees a per-call warning looks past to name its caller.

    A storage check runs where the dtype is known, inside the op, and the op
    is reached from a wrapper, a layer's math hook or an autograd Function's
    backward -- at different depths, some of them under torch's own frames
    (`Module.__call__`, `Function.apply`). So rather than count frames it
    names the first one outside mptorch and torch, which is the line that made
    the call: the location worth reading, and the one `warnings`' default
    once-per-location rule then keys on, so a training loop hears it once.
    """
    roots = [os.path.dirname(os.path.abspath(__file__))]
    torch_module = sys.modules.get("torch")
    if torch_module is not None and torch_module.__file__:
        roots.append(os.path.dirname(os.path.abspath(torch_module.__file__)))
    return tuple(os.path.join(root, "") for root in roots)


def _report_per_call(error: str | None, warning: str | None) -> None:
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
    storage: str,
    elementwise: bool = False,
) -> None:
    """Hold a binaryK format against the ``storage`` dtype a result is written in.

    ``storage`` is ``"float16"`` or ``"bfloat16"``. The cast still rounds in
    binary32 -- which :func:`check_binaryK` holds the format against, and
    this does not repeat -- and writing the result rounds it again. Which
    results can land off the dtype's grid depends on what was rounded:
    ``elementwise=True`` for a quantizer, whose inputs are already the dtype's
    values, and ``False`` for a GEMM, whose sums reach every value the format
    has. The note above gives both rules. Raises :exc:`ValueError` where
    nothing of the format survives being stored, and warns with
    :class:`FormatRangeWarning` where part of it does not.

    The dtype belongs to the call, not the format, so :mod:`mptorch.quant.ops`
    runs this per call on the format whose values the result holds, and the
    warning names the first frame outside mptorch and torch.
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
    storage: str,
    elementwise: bool = False,
) -> None:
    """:func:`check_binaryK_storage` for a superfp format -- see it for the contract."""
    _report_per_call(
        *_superfp_storage_findings(
            man_bits, exp_bits, normal_binades, bias, saturation, storage, elementwise
        )
    )


# --- format objects ----------------------------------------------------------
#
# One value type per simulated format, carrying exactly what that format's
# cast reads and nothing else (X1). They exist so a caller can name a format
# once -- `BinaryK(8, 4)` -- instead of spelling its five or six schema
# arguments at each of the places a GEMM, a quantizer or a layer takes one, and
# so the two halves of a mixed-precision configuration can differ where the
# kernel already lets them differ.
#
# They are frozen and they resolve their derived constants in `__post_init__`,
# because the thing that reads them memoizes on the value: re-deriving a
# format per call costs 2.3-6.4 us (finding P4), which is the whole reason
# `mptorch.quant.gemm`'s factories resolve once per layer. Frozen also makes
# them hashable, which is what the memo keys on.


@dataclass(frozen=True)
class Number:
    """Base class for every simulated number format.

    Subclassed by :class:`FloatFormat` today. Fixed-point, block floating
    point, block minifloats, logarithmic and tapered (posit) formats belong
    here too -- each needs a kernel first, so none of them is declared as an
    empty class in the meantime (see ``dev/gemm_roadmap.md``, X1 open
    decision 3).
    """


@dataclass(frozen=True)
class FloatFormat(Number):
    """Base class for the two floating-point families the kernels implement."""


@dataclass(frozen=True)
class BinaryK(FloatFormat):
    """An IEEE P3109 binaryK float: ``K`` total bits of which ``P`` are precision.

    P3109, the draft Standard for Arithmetic Formats for Machine Learning
    (Fitzgibbon, Wintersteiger and Sarnoff give an overview in
    arXiv:2606.04028), parameterizes its binary formats by bitwidth, precision
    -- the implicit leading bit included -- signedness and domain, and names
    them ``Binary<K>p<P><s|u><e|f>``. ``K``, ``P`` and ``is_signed`` are the
    first three, and ``saturation`` carries the domain (see
    :class:`SaturationMode`), so ``BinaryK(8, 4)`` is ``Binary8p4se`` and
    ``BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE)`` is ``Binary8p4sf``.
    :doc:`/concepts` maps the rest of the standard onto this class and says
    where MPTorch goes beyond it.

    ``bias`` defaults to P3109's -- ``2**(K-P-1)`` signed, ``2**(K-P)``
    unsigned, which encodes 1.0 at the middle code point -- and is resolved
    here rather than at each call site. Any other bias, like any ``subnormals``
    but ``SUBNORMALS``, gives a format outside the standard; that is how OCP's
    E4M3, ``BinaryK(8, 4, bias=7)``, is spelled. ``prng_bits`` is the width of
    the random mantissa used by ``RoundMode.SR`` and is ignored under every
    other rounding mode; the rounding mode itself is not a property of the
    format but of the operation (see :class:`mptorch.quant.SplitMac`), because
    the kernels take it as a template parameter shared by a multiply and its
    accumulate.
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
        # one rule, and `mptorch.quant.ops`' plain-integer wrappers call the
        # same function -- see `check_binaryK`
        check_binaryK(
            self.K,
            self.P,
            self.bias,
            self.is_signed,
            self.saturation,
            self.subnormals,
            self.prng_bits,
            stacklevel=4,  # __init__ -> __post_init__ -> check -> warn
        )

    @property
    def man_bits(self) -> int:
        """Mantissa bits, as the cast counts them (the implicit bit is not one)."""
        return self.P - 1


@dataclass(frozen=True)
class SuperFP(FloatFormat):
    """A supernormal float, as ``common/cast_superfp.h`` implements it.

    Deliberately not symmetric with :class:`BinaryK`. ``bias`` is required and
    has no default, because the cast has no derivation rule for one; there is
    a single ``normal_binades`` rather than one per end of the range; and there
    is no ``subnormals`` mode, because the format has no subnormals -- the
    supernormal region sits *below* the normals, repurposing that encoding
    space for implicit-mantissa powers of two, and everything below it flushes
    to zero.
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

        check_superfp(
            self.man_bits,
            self.exp_bits,
            self.normal_binades,
            self.bias,
            self.saturation,
            self.prng_bits,
            stacklevel=4,  # __init__ -> __post_init__ -> check -> warn
        )
