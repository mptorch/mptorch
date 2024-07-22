import torch
from mptorch.quant import binary8_quantize
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


def test_binary8p1_saturate_maxfloat():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 1, "nearest", "saturate_maxfloat", False, True)
    # normal
    assert_quant([[1.5,6.0E-5],[72057500037900000.0,0.01171875]], [[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00000000100000000000000000000000), b2f(0b00000000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00000000010000000000000000000001), b2f(0b00000000100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01111110100000000000000000000000), b2f(0b01111110100000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01111110110011100000000000000000), b2f(0b01111110100000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b01111110100000000000000000000000), quant)

def test_binary8p1_saturate_maxfloat2():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 1, "nearest", "saturate_maxfloat2", False, True)
    # normal 
    assert_quant([[1.5,6.0E-5],[72057500037900000.0,0.01171875]], [[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00000000100000000000000000000000), b2f(0b00000000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00000000010000000000000000000001), b2f(0b00000000100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01111111000000000000000000000000), b2f(0b01111111000000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01111111010110000000110000111000), b2f(0b01111111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b01111111000000000000000000000000), quant) # +inf

def test_binary8p1_saturate_infty():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 1, "nearest", "saturate_infty", False, True)
    # normal 
    assert_quant([[1.5,6.0E-5],[72057500037900000.0,0.01171875]], [[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00000000100000000000000000000000), b2f(0b00000000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00000000010000000000000000000001), b2f(0b00000000100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01111110100000000000000000000000), b2f(0b01111110100000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01111111010001110000001100000100), [float('inf')], quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf

def test_binary8p2_saturate_maxfloat():
    if no_cuda():
        return 

    quant = lambda x: binary8_quantize(x, 2, "nearest", "saturate_maxfloat", False, True)
    # normal 
    assert_quant([[1.5,1],[1.77e-15, 5242880]], [[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00011111100000000000000000000000), b2f(0b00011111100000000000000000000000), quant) # min value
    assert_quant(b2f(0b00011111010000000000000000000001), b2f(0b00011111100000000000000000000000), quant) # middle value - round to min
    assert_quant(b2f(0b0001111100000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b01011110110000000000000000000000), b2f(0b01011110110000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01011110111001110000001100000100), b2f(0b01011110110000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b01011110110000000000000000000000), quant) # +inf

def test_binary8p2_saturate_maxfloat2(): # no inf so we go to 0xfe as max
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 2, "nearest", "saturate_maxfloat2", False, True)
    # normal 
    assert_quant([[1.5,1],[1.77e-15, 5242880]], [[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000000000000000000000011111), b2f(0b00100000000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b01011111000000000000000000000000), quant) # +inf

def test_binary8p2_saturate_infty(): # any overflow goes to inf, not 0xfd
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 2, "nearest", "saturate_infty", False, True)
    # normal 
    assert_quant([[1.5,1],[1.77e-15, 5242880]], [[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000000000000000000000011111), b2f(0b00100000000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b01011110010000000000000000000000), b2f(0b01011110010000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01011111100001110000001100000100), [float('inf')], quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf

def test_binary8p8_saturate_maxfloat():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 8, "nearest", "saturate_maxfloat", False, True)
    # normal 
    assert_quant([[1.5,1.00390625],[1.4921875,1.01171875E0]] , [[1.5,1.0],[1.4921875,1.015625E0]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111111100000000000000000000000), b2f(0b00111111100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00111100000000000000000000000000), b2f(0b00111100000000000000000000000000), quant) # min subnormal
    assert_quant(b2f(0b00111011100000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00111011100000000000000000000001), b2f(0b00111100000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00111111111111010000000000000000), b2f(0b00111111111111010000000000000000), quant) # max normal
    assert_quant(b2f(0b00111111111111110000000000000000), b2f(0b00111111111111010000000000000000), quant) # overflow
    assert_quant(b2f(0b00111011000000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b00111111111111010000000000000000), quant) # +inf

def test_binary8p8_saturate_maxfloat2():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 8, "nearest", "saturate_maxfloat2", False, True)
    # normal 
    assert_quant([[1.5,1.00390625],[1.4921875,1.01171875E0]] , [[1.5,1.0],[1.4921875,1.015625E0]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111111100000000000000000000000), b2f(0b00111111100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00111100000000000000000000000000), b2f(0b00111100000000000000000000000000), quant) # min subnormal
    assert_quant(b2f(0b00111011100000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00111011100000000000000000000001), b2f(0b00111100000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00111111111111100000000000000000), b2f(0b00111111111111100000000000000000), quant) # max normal
    assert_quant(b2f(0b00111111111111111000000000000000), b2f(0b00111111111111100000000000000000), quant) # overflow
    assert_quant(b2f(0b00111011000000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b00111111111111100000000000000000), quant) # +inf

def test_binary8p8_saturate_infty():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 8, "nearest", "saturate_infty", False, True)
    # normal 
    assert_quant([[1.5,1.00390625],[1.4921875,1.01171875E0]] , [[1.5,1.0],[1.4921875,1.015625E0]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111111100000000000000000000000), b2f(0b00111111100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00111100000000000000000000000000), b2f(0b00111100000000000000000000000000), quant) # min subnormal
    assert_quant(b2f(0b00111011100000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00111011100000000000000000000001), b2f(0b00111100000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00111111111111010000000000000000), b2f(0b00111111111111010000000000000000), quant) # max normal
    assert_quant(b2f(0b00111111111111110000000000000000), [float('inf')], quant) # overflow
    assert_quant(b2f(0b00111011000000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf