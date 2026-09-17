"""MPTorch: mixed-precision arithmetic simulation for PyTorch.

Importing the package loads the compiled extension ``mptorch._C``, which
registers the ``torch.ops.mptorch`` operators, and re-exports the number
formats and modes of :mod:`mptorch.number`. The quantizers, matrix products
and layers live in :mod:`mptorch.quant`.
"""

import os

import torch

from . import _C  # type: ignore # noqa: F401

# Pin torch's CUDA architecture list to the GPU that is present, unless the
# environment already names one. Anything torch compiles later in this process
# (`torch.utils.cpp_extension.load`, a JIT-built extension) then targets that
# one architecture instead of torch's default of every architecture it knows,
# which is a large difference in compile time. The extension imported above is
# prebuilt, so this does not affect it.
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

from .number import (
    AccumulateAlgorithm,
    BinaryK,
    FloatFormat,
    FormatRangeWarning,
    Number,
    RoundMode,
    SaturationMode,
    SubnormalsMode,
    SuperFP,
)

__version__ = "0.4.0"

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
