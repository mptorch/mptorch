"""Number formats, the modes for rounding to them, and checks that a float type can hold them.

The format classes are :class:`BinaryK` and :class:`SuperFP` (floating-point
formats), :class:`FixedPoint` and :class:`BlockFormat` (elements that share a
scale per block), with presets for the OCP MX and NVFP4 formats. Each is a
frozen (immutable) value.

The enums :class:`RoundMode`, :class:`SaturationMode`, :class:`SubnormalsMode`
and :class:`AccumulateAlgorithm` have the same names and integer values as the
C++ enums in ``csrc/common/modes.h``. The kernels receive an enum's ``.value``
through ``torch.ops``, so the two definitions must be kept in sync.
:class:`Domain` exists only in Python: a format object accepts it as
``domain=`` and records it in its :class:`SaturationMode`, which is all the
kernels see.

The ``check_*`` functions test a format against two floating-point types: the
*carrier*, the binary float (binary32 or binary64) that the cast rounds in,
and the *storage* dtype that the result is written to, when that is narrower
than the carrier. They raise :exc:`ValueError` when the format cannot work in
that type at all, and warn with :class:`FormatRangeWarning` when it works over
only part of its range.
"""

# One module per format family, plus the modules they share:
#
#   modes        the enums: rounding, saturation, subnormals, accumulation, and Domain
#   base         the base classes Number and FloatFormat, FormatRangeWarning, and the
#                helpers that raise or warn about a format from inside a call
#   binaryk      BinaryK and its carrier and storage checks
#   superfp      SuperFP and its carrier and storage checks
#   fixedpoint   FixedPoint and its carrier and storage checks
#   block        BlockFormat, ScaleRounding, and the code layout the block kernels decode
#   presets      the OCP MX and NVFP4 formats, and bfp
#   _carrier     the limits of the binary32 and binary64 carriers, and the range check
#                that BinaryK and SuperFP share
#   _storage     the checks that a narrower storage dtype can hold a format's results
#
# A new family gets a module of its own (its class, its checks and the code
# behind them), a line in this table, and its public names imported below.

from .base import FloatFormat, FormatRangeWarning, Number
from .binaryk import BinaryK
from .binaryk import check_binaryK as check_binaryK
from .binaryk import check_binaryK_carrier as check_binaryK_carrier
from .binaryk import check_binaryK_storage as check_binaryK_storage
from .block import BlockFormat
from .block import ScaleRounding as ScaleRounding
from .block import check_block_carrier as check_block_carrier
from .block import check_block_storage as check_block_storage
from .fixedpoint import FixedPoint
from .fixedpoint import check_fixedpoint as check_fixedpoint
from .fixedpoint import check_fixedpoint_carrier as check_fixedpoint_carrier
from .fixedpoint import check_fixedpoint_storage as check_fixedpoint_storage
from .modes import AccumulateAlgorithm, Domain, RoundMode, SaturationMode, SubnormalsMode
from .presets import (
    E2M1,
    E2M3,
    E3M2,
    E4M3,
    E5M2,
    E8M0,
    MXFP4_E2M1,
    MXFP6_E2M3,
    MXFP6_E3M2,
    MXFP8_E4M3,
    MXFP8_E5M2,
    NVFP4,
    bfp,
)
from .superfp import SuperFP
from .superfp import check_superfp as check_superfp
from .superfp import check_superfp_carrier as check_superfp_carrier
from .superfp import check_superfp_storage as check_superfp_storage

__all__ = [
    "SaturationMode",
    "Domain",
    "SubnormalsMode",
    "RoundMode",
    "AccumulateAlgorithm",
    "Number",
    "FloatFormat",
    "BinaryK",
    "SuperFP",
    "BlockFormat",
    "FixedPoint",
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
    "bfp",
]
