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

import warnings
from dataclasses import KW_ONLY, dataclass
from enum import Enum
from functools import lru_cache


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
    """A format whose range outruns binary32, so quantizing to it is partial.

    Warned by :class:`BinaryK`, :class:`SuperFP` and the plain-integer
    wrappers of :mod:`mptorch.quant.ops` when a format reaches above
    binary32's largest finite value, spaces its values finer than
    :math:`2^{-149}`, or has a smallest value below :math:`2^{-125}` -- the
    last because every cast places an input by its binary32 exponent field,
    which all binary32 subnormals share, so the bottom binade is rounded onto
    the wrong grid. In each case the format quantizes correctly over the part
    of it binary32 holds and some of its values are simply never returned.

    It is a warning rather than an error because such a format is still
    useful: a wide binaryK is a reasonable precision-only target, where the
    unreachable top of the range is exactly the point. Filter it with
    :func:`warnings.simplefilter` when that is what is meant.
    """


def _top_of_range(exp_bits: int, man_bits: int, bias: int, reserved: int) -> tuple[int, int]:
    """``make_normal_range_params``' top of the range, as (exponent field, exponent).

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
    return top_field, top_field - bias


def _width_error(label: str, exp_bits: int, man_bits: int, prng_bits: int) -> str | None:
    """The bounds that must hold before anything shifts by ``exp_bits``."""
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


def _range_findings(
    label: str,
    exp_bits: int,
    top_field: int,
    top_exp: int,
    bottom_exp: int,
    step_exp: int,
    min_bottom: int = _F32_MIN_SIMULABLE_EXP,
) -> tuple[str | None, str | None]:
    """The top and bottom of the range against binary32's, after the widths.

    ``bottom_exp`` is the exponent of the format's smallest positive value and
    ``step_exp`` that of its finest spacing; for a format whose lowest binade
    carries a mantissa the two differ by ``man_bits``. ``min_bottom`` is how
    low that smallest value may go, which is not the same for every format --
    see the caller.

    Only the first finding is returned, so a format that outruns binary32 at
    both ends says so about the top first and about the bottom once that is
    fixed. The error is the one case that stops the format working at all.
    """
    if top_field < 1 or top_exp < _F32_MIN_NORMAL_EXP:
        return (
            f"{label} has no finite normal value binary32 can hold: the largest "
            f"finite code sits at 2^{top_exp}, so every nonzero result overflows"
        ), None
    if top_exp > _F32_TOP_EXP:
        return None, (
            f"{label} reaches 2^{top_exp}, above binary32's 2^{_F32_TOP_EXP}: the cast "
            f"saturates at binary32's largest value on the format's grid instead, and "
            f"the codes above it are unreachable (needs bias >= {(1 << exp_bits) - 128})"
        )
    if step_exp < _F32_MIN_SUBNORMAL_EXP:
        return None, (
            f"{label} spaces its values 2^{step_exp} apart at the bottom, below "
            f"binary32's 2^{_F32_MIN_SUBNORMAL_EXP}: values that far down are not "
            f"binary32 values at all, so the bottom of the range is unreachable"
        )
    if bottom_exp < min_bottom:
        return None, (
            f"{label}'s smallest value is 2^{bottom_exp}, below 2^{min_bottom}: the "
            f"casts place an input by its binary32 exponent field, which every binary32 "
            f"subnormal shares, so this format's values below 2^{_F32_MIN_NORMAL_EXP} "
            f"are rounded onto the wrong grid and some are never returned. See "
            f'concepts.rst, "What float32 can carry".'
        )
    return None, None


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
    label = (
        f"BinaryK(K={K}, P={P}, bias={bias}"
        + ("" if is_signed else ", is_signed=False")
        + ("" if subnormals is SubnormalsMode.SUBNORMALS else f", {subnormals.name}")
        + ")"
    )
    error = _width_error(label, exp_bits, man_bits, prng_bits)
    if error is not None:
        return error, None
    # P3109's reserved codes: +infinity outside the finite domain, and NaN
    # above it in an unsigned format (`make_binaryK_params`)
    reserved = (saturation is not SaturationMode.SAT_FINITE) + (0 if is_signed else 1)
    top_field, top_exp = _top_of_range(exp_bits, man_bits, bias, reserved)
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
    return _range_findings(label, exp_bits, top_field, top_exp, bottom_exp, step_exp, min_bottom)


@lru_cache(maxsize=256)
def _superfp_findings(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    saturation: SaturationMode,
    prng_bits: int,
) -> tuple[str | None, str | None]:
    label = (
        f"SuperFP(man_bits={man_bits}, exp_bits={exp_bits}, "
        f"normal_binades={normal_binades}, bias={bias})"
    )
    error = _width_error(label, exp_bits, man_bits, prng_bits)
    if error is not None:
        return error, None
    binades = 1 << exp_bits
    if normal_binades >= binades:
        # no supernormal codes left, which puts the supernormal floor one
        # *above* the normal one; the cast tests supernormal-then-underflow,
        # so the format's lowest normal binade would flush to zero
        return (
            f"{label} leaves no supernormal codes: normal_binades must be at most "
            f"2**exp_bits - 1 = {binades - 1}, or the cast flushes the format's "
            f"lowest normal binade to zero"
        ), None
    # superfp spends no code on a NaN, signed or not (`make_superfp_params`)
    reserved = saturation is not SaturationMode.SAT_FINITE
    top_field, top_exp = _top_of_range(exp_bits, man_bits, bias, reserved)
    normal_cutoff = binades - bias - normal_binades
    # the smallest supernormal, `superfp_region_cutoffs`
    bottom_exp = normal_cutoff - (binades - normal_binades) * (1 << man_bits) + 1
    step_exp = min(bottom_exp, normal_cutoff - man_bits)
    return _range_findings(label, exp_bits, top_field, top_exp, bottom_exp, step_exp)


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
