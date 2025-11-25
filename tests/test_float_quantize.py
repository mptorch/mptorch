import torch
from mptorch.quant import float_quantize_v2
from mptorch.number import RoundMode, SubnormalsMode
import pytest
from tests.markers import available_devices


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR])
def test_32_to_32_quantization(device, mode):
    a = torch.tensor(3.0, device=device)
    out = float_quantize_v2(a, exp=8, man=23, rounding_mode=mode)
    assert out == a


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE, RoundMode.SR])
def test_32_to_E2M23_quantization(device, mode):
    a = torch.tensor(1097.0, device=device)
    out = float_quantize_v2(
        a,
        exp=2,
        man=23,
        rounding_mode=mode,
        subnormals_mode=SubnormalsMode.SUBNORMALS,
        saturate=False,
    )
    assert out.item() == float("inf")
    out = float_quantize_v2(
        a,
        exp=2,
        man=23,
        rounding_mode=mode,
        subnormals_mode=SubnormalsMode.SUBNORMALS,
        saturate=True,
    )
    assert out.item() == 3.999999761581421
    a = torch.tensor(8.0, device=device)
    out = float_quantize_v2(
        a,
        exp=3,
        man=23,
        rounding_mode=mode,
        subnormals_mode=SubnormalsMode.NORMALS,
        saturate=False,
    )
    assert out.item() == 8.0


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", [RoundMode.RNE])
def test_32_to_E8M1_quantization(device, mode):
    a = torch.tensor(3.1516, device=device)
    out = float_quantize_v2(
        a,
        exp=8,
        man=1,
        rounding_mode=mode,
        subnormals_mode=SubnormalsMode.SUBNORMALS,
        saturate=False,
    )
    assert out.item() == 3.0
