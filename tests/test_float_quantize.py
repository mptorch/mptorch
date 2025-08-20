import torch
from mptorch.quant import float_quantize
import pytest
from tests.markers import available_devices


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ["nearest", "stochastic"])
def test_32_to_32_quantization(device, mode):
    a = torch.tensor(3.0, device=device)
    out = float_quantize(a, exp=8, man=23, rounding=mode)
    assert out == a


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ["nearest", "stochastic"])
def test_32_to_E2M23_quantization(device, mode):
    a = torch.tensor(1097.0, device=device)
    out = float_quantize(
        a, exp=2, man=23, rounding=mode, subnormals=True, saturate=False
    )
    assert out.item() == float("inf")
    out = float_quantize(
        a, exp=2, man=23, rounding=mode, subnormals=True, saturate=True
    )
    assert out.item() == 3.999999761581421
    a = torch.tensor(8.0, device=device)
    out = float_quantize(
        a, exp=3, man=23, rounding="nearest", subnormals=False, saturate=False
    )
    assert out.item() == 8.0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", ["nearest"])
def test_32_to_E8M1_quantization(device, mode):
    a = torch.tensor(3.1516, device=device)
    out = float_quantize(
        a, exp=8, man=1, rounding=mode, subnormals=True, saturate=False
    )
    assert out.item() == 3.0
