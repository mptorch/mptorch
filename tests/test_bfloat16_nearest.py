import torch
from mptorch.quant import bfloat16_quantize
import struct
import numpy as np
import random

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


def test_bfloat16():
    if no_cuda():
        return
    
    quant = lambda x: bfloat16_quantize(x, "nearest")
    # normal
    assert_quant([[20.0625,20.06251],[20.0625,20.06251]], [[20.0,20.125],[20.0,20.125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    # assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant) # min normal
    # assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant) # round to 0
    # assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant) # round to min
    # assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # max normal
    # assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    # assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant)
    assert_quant([-float('inf')], [-float('inf')], quant)


def main():
    test_bfloat16()


if __name__ == "__main__":
    main()