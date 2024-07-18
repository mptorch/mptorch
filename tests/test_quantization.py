import torch
from mptorch.quant import float_quantize


def test_32_to_32_quantization():
    for mode in ["nearest","stochastic"]:
        a = torch.tensor(3.0)
        out = float_quantize(a, 8, 23, mode)
        assert out == a
        if not torch.cuda.is_available():
            continue
        a = a.cuda()
        out = float_quantize(a, 8, 23, mode)
        assert out == a

def test_32_to_E2M23_quantization():
    for mode in ["nearest","stochastic"]:
        a = torch.tensor(1097.0)
        out = float_quantize(a, 2, 23, mode, True, False)
        assert out.item() == float("inf")
        out = float_quantize(a, 2, 23, mode, True, True)
        assert out.item() == 3.999999761581421
        if not torch.cuda.is_available():
            continue
        a = a.cuda()
        out = float_quantize(a, 2, 23, mode, True, False)
        assert out.item() == float("inf")
        out = float_quantize(a, 2, 23, mode, True, True)
        assert out.item() == 3.999999761581421

def test_32_to_E8M1_quantization():
    a = torch.tensor(3.1516)
    out = float_quantize(a, 8, 1, "nearest", True, False)
    assert out.item() == 3.0
    if not torch.cuda.is_available():
        return
    a = a.cuda()
    out = float_quantize(a, 8, 1, "nearest", True, False)
    assert out.item() == 3.0
