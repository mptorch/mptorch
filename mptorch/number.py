__all__ = [
    "SaturationMode",
    "SubnormalsMode",
    "RoundMode",
    "AccumulateAlgorithm",
    "Number",
    "FloatFormat",
    "BinaryK",
    "SuperFP",
]

from dataclasses import KW_ONLY, dataclass
from enum import Enum


class SaturationMode(Enum):
    """
    Enum for what a format does with a value beyond its largest finite one.

    Mirrors ``SaturationMode`` in ``csrc/common/modes.h``. NaN inputs pass
    through every mode unchanged. ``SAT_FINITE`` also clamps an infinite
    input, so everything it returns is finite; the other two keep infinities.
    """

    #: Clamp everything, infinities included, to the largest finite value (top
    #: mantissa code included).
    SAT_FINITE = 0
    SAT_PROPAGATE = 1  #: Clamp to the largest finite value with the top mantissa code excluded.
    OVF_INF = 2  #: Overflow becomes :math:`\pm\infty`, as in IEEE 754 (the default).


class SubnormalsMode(Enum):
    """
    Enum for how a format handles the range below its smallest normal number.

    Mirrors ``SubnormalsMode`` in ``csrc/common/modes.h``.
    """

    SUBNORMALS = 0  #: Gradual underflow: exponent code 0 encodes subnormal numbers (the default).
    NORMALS = 1  #: No subnormals; values below the smallest normal flush to zero.
    EXTENDED_NORMALS = 2  #: No subnormals; exponent code 0 is one more binade of normal numbers.


class RoundMode(Enum):
    """
    Enum for floating-point rounding modes.

    Result :math:`y` is obtained from an input :math:`x` depending on the rounding mode.
    """

    RNE = 0  #: Round to nearest, ties to even
    RNA = 1  #: Round to nearest, ties to away from zero
    RU = 2  #: Return the smallest :math:`y` such that :math:`y \ge x`
    RD = 3  #: Return the largest :math:`y` such that :math:`y \le x`
    RZ = 4  #: Return the largest :math:`y` such that :math:`|y| \le |x|`
    RO = 5  #: Round to odd
    SR = 6  #: Stochastic Rounding


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
    """A binaryK float: ``K`` total bits of which ``P`` are precision.

    ``bias`` defaults to the middle of the exponent range -- ``2**(K-P-1)``
    signed, ``2**(K-P)`` unsigned -- and is resolved here rather than at each
    call site. ``prng_bits`` is the width of the random mantissa used by
    ``RoundMode.SR`` and is ignored under every other rounding mode; the
    rounding mode itself is not a property of the format but of the operation
    (see :class:`mptorch.quant.SplitMac`), because the kernels take it as a
    template parameter shared by a multiply and its accumulate.
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
