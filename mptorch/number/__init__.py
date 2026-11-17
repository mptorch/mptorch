"""Number formats, the modes that round into them, and what a float makes of them.

The enums here (:class:`RoundMode`, :class:`SaturationMode`,
:class:`SubnormalsMode`, :class:`AccumulateAlgorithm`) mirror the C++ enums in
``csrc/common/modes.h`` one to one, by name and by integer value. The kernels
receive ``.value`` across the ``torch.ops`` boundary, so the two must stay in
sync. :class:`Domain` exists in Python only: a format object folds it into
its :class:`SaturationMode` when it is built. :class:`BinaryK` and
:class:`SuperFP` are the frozen value types that describe a format.

The ``check_*`` functions report what a *carrier* (the binary float a cast
rounds in) and a *storage* dtype (the tensor a result is written to) can do
with a format. They raise :exc:`ValueError` where the format cannot work at
all, and warn with :class:`FormatRangeWarning` where it works over part of
its range.
"""

# One module per format family, over the two rules they share:
#
#   modes        the rounding, saturation, subnormals and accumulate modes, and Domain
#   base         Number and FloatFormat; FormatRangeWarning, and how a call reports a finding
#   binaryk      BinaryK, and what holds it to a carrier and a storage dtype
#   superfp      SuperFP, likewise
#   fixedpoint   FixedPoint, likewise
#   block        BlockFormat and ScaleRounding, and the code layout the block kernels decode
#   presets      the OCP MX and NVFP4 formats, and bfp
#   _carrier     what a carrier can hold: its bounds, and binaryK's and superfp's range rule
#   _storage     what a narrower dtype can store: the rule every storage check reads
#
# A new family is a module of its own (its class, its checks and its findings),
# a line above and its names below.

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
