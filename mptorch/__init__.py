import os
import torch

if torch.cuda.is_available():
    (major, minor) = torch.cuda.get_device_capability(0)
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"

from .number import *

__version__ = "0.4.0"