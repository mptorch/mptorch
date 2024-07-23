import torch
from mptorch.quant import float_quantize
import struct
import pytest
from tests.markers import available_devices

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def assert_quant(x_arr, expected_arr, quant_fn, device):
    x = torch.tensor(x_arr, dtype=torch.float32, device=device)
    expected = torch.tensor(expected_arr, dtype=torch.float32, device=device)
    assert expected.equal(quant_fn(x))

@pytest.mark.parametrize("device", available_devices)
def test_bfloat16(device):
    quant = lambda x: float_quantize(x, 8, 7, "nearest", True, False)
    # normal
    assert_quant([[20.0625,20.06251],[20.0625,20.06251]], [[20.0,20.125],[20.0,20.125]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    # assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant) # min normal
    # assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant) # round to 0
    # assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant) # round to min
    # assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # max normal
    # assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    # assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant) # underflow

    assert_quant([float('inf')], [float('inf')], quant, device)
    assert_quant([-float('inf')], [-float('inf')], quant, device)