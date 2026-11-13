"""MPTorch: mixed-precision arithmetic simulation for PyTorch.

Importing the package loads the compiled extension ``mptorch._C``, which
registers the ``torch.ops.mptorch`` operators, and re-exports the number
formats and modes of :mod:`mptorch.number`. The quantizers, matrix products
and layers live in :mod:`mptorch.quant`.
"""

import os
from importlib.metadata import version as _dist_version

import torch

from . import _C  # type: ignore # noqa: F401

# Pin torch's CUDA architecture list to the GPUs that are present, unless the
# environment already names one. Anything torch compiles later in this process
# (`torch.utils.cpp_extension.load`, a JIT-built extension) then targets those
# architectures instead of torch's default of every architecture it knows,
# which is a large difference in compile time. The extension imported above is
# prebuilt, so this does not affect it. Every visible GPU's architecture is
# named, not GPU 0's alone, or a JIT build could not run on the others of a
# machine with GPUs of two generations.
if torch.cuda.is_available() and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    _archs = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{a}.{b}" for a, b in sorted(_archs))
    del _archs

from .number import (
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
    AccumulateAlgorithm,
    BinaryK,
    BlockFormat,
    Domain,
    FixedPoint,
    FloatFormat,
    FormatRangeWarning,
    Number,
    RoundMode,
    SaturationMode,
    ScaleRounding,
    SubnormalsMode,
    SuperFP,
    bfp,
)

# The version is pyproject.toml's, which `uv sync` installs with the package.
__version__ = _dist_version("mptorch")

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
    "ScaleRounding",
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
