import torch
from mptorch.quant import float_quantize


def test_32_to_32_quantization():
    for mode in ["nearest","stochastic"]:
        a = torch.tensor(3.0)
        out = float_quantize(a, 8, 23, mode)
        assert out == a