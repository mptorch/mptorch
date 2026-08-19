__all__ = [
    "SaturationMode",
    "SubnormalsMode",
    "RoundMode",
]

from enum import Enum

SaturationMode = Enum(
    "SaturationMode", [("SAT_FINITE", 0), ("SAT_PROPAGATE", 1), ("OVF_INF", 2)]
)

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
