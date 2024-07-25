import torch
import struct

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