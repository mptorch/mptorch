import torch
from mptorch.quant import p3109_quantize
import struct

def no_cuda():
    return not torch.cuda.is_available()

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def assert_quant(x_arr, expected_arr, quant_fn):
    x = torch.tensor(x_arr, dtype=torch.float32, device="cuda")
    expected = torch.tensor(expected_arr, dtype=torch.float32, device="cuda")
    assert expected.equal(quant_fn(x))

def test_p3109p1():
    if no_cuda():
        return

    quant = lambda x: p3109_quantize(x, 1, "troncate", "saturate", True, True)
    # normal
    assert_quant([[1.5,4.0E-13],[134210000.0,-9.0E-07]], [[1.0,2.2737368E-13],[6.7108864e+07,-4.7683716e-07]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant)
    assert_quant([-float('inf')], [-float('inf')], quant)

def test_p3109p4():
    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 4, "troncate", "saturate", True, True)
    # normal
    assert_quant([[1.5,165.65200],[0.0550651,-0.0685105]], [[1.5,160.0],[0.0546875,-0.0625]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111100000000000000000000000000), b2f(0b00111100000000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00111010100000000000000000000000), b2f(0b00111010100000000000000000000000), quant) # min sub_normal
    assert_quant(b2f(0b00111010000000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00111010000000000000000000000001), b2f(0b00111010100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01000011011000000000000000000000), b2f(0b01000011011000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01000011011100000000000000000000), b2f(0b01000011011000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00111001100000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant)
    assert_quant([-float('inf')], [-float('inf')], quant)