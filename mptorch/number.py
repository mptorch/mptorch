__all__ = [
    "SaturationMode",
    "SubnormalsMode",
    "RoundMode",
    "AccumulateAlgorithm",
]

from enum import Enum

SaturationMode = Enum("SaturationMode", [("SAT_FINITE", 0), ("SAT_PROPAGATE", 1), ("OVF_INF", 2)])

SubnormalsMode = Enum(
    "SubnormalsMode", [("SUBNORMALS", 0), ("NORMALS", 1), ("EXTENDED_NORMALS", 2)]
)


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
    ``dev/gemm_core_roadmap.md``).
    """

    NAIVE = 0  #: Quantize the running sum after every accumulation step.
    KAHAN = 1  #: Kahan-compensated summation (not yet implemented).
    BLOCK = 2  #: Two-level (FABSum-style) block summation (not yet implemented).
    TREE = 3  #: Pairwise tree-reduction summation (not yet implemented).
