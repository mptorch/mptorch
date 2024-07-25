from mptorch.quant import binary8_quantize
import pytest
from tests.markers import available_devices
from tests.quant import bits_to_float, assert_quant

@pytest.mark.parametrize("device", available_devices)
def test_binary8p1(device):
    quant = lambda x: binary8_quantize(x, 1, "truncate", "saturate_maxfloat", True, True)
    # normal
    assert_quant([[1.5,4.0E-13],[134210000.0,-9.0E-07]], [[1.0,2.2737368E-13],[6.7108864e+07,-4.7683716e-07]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant, device) # round to 0
    assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant, device) # overflow
    assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant, device) # underflow
    # assert_quant([float('nan')], [float('nan')], quant) # NaN
    assert_quant([float('inf')], b2f(0b01011111000000000000000000000000), quant, device) # +max normal
    assert_quant([-float('inf')], b2f(0b11011111000000000000000000000000), quant, device) # -max normal

@pytest.mark.parametrize("device", available_devices)
def test_binary8p4(device):
    quant = lambda x: binary8_quantize(x, 4, "truncate", "saturate_maxfloat", True, True)
    # normal
    assert_quant([[1.5,165.65200],[0.0550651,-0.0685105]], [[1.5,160.0],[0.0546875,-0.0625]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111100000000000000000000000000), b2f(0b00111100000000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00111010100000000000000000000000), b2f(0b00111010100000000000000000000000), quant, device) # min subnormal
    assert_quant(b2f(0b00111010000000000000000000000000), [0.0], quant, device) # round to 0
    assert_quant(b2f(0b00111010000000000000000000000001), b2f(0b00111010100000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01000011011000000000000000000000), b2f(0b01000011011000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01000011011100000000000000000000), b2f(0b01000011011000000000000000000000), quant, device) # overflow
    assert_quant(b2f(0b00111001100000000000000000000000), [0.0], quant, device) # underflow
    # assert_quant([float('nan')], [float('nan')], quant) # NaN
    assert_quant([float('inf')], b2f(0b01000011011000000000000000000000), quant, device) # +max normal
    assert_quant([-float('inf')], b2f(0b11000011011000000000000000000000), quant, device) # -max normal