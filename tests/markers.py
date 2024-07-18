import pytest
import torch

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason='No CUDA-capable device found.'
)

available_devices = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason='No CUDA-capable device found.'
    ))
]