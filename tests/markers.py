import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA device found."
)