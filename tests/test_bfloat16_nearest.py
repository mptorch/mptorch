from mptorch.quant import float_quantize_v2
from mptorch.number import RoundMode
import pytest
from tests.markers import available_devices
from tests.quant import bits_to_float, assert_quant


@pytest.mark.parametrize("device", available_devices)
def test_bfloat16(device):
    def quant(x):
        return float_quantize_v2(
            x, exp=8, man=7, saturate=False, rounding_mode=RoundMode.RNE
        )
    # normal
    assert_quant(
        [[20.0625, 20.06251], [20.0625, 20.06251]],
        [[20.0, 20.125], [20.0, 20.125]],
        quant,
        device,
    )

    def b2f(b):
        return [bits_to_float(b)]

    # assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant, device) # min normal
    # assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant, device) # round to 0
    # assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant, device) # round to min
    # assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant, device) # max normal
    # assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant, device) # overflow
    # assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant, device) # underflow

    assert_quant([float("inf")], [float("inf")], quant, device)
    assert_quant([-float("inf")], [-float("inf")], quant, device)
