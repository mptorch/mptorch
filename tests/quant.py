import struct

import torch
from mptorch.number import RoundMode, SubnormalsMode
from mptorch.quant import float_quantize_v2, superfp_quantize_v2


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


def make_float_quantize_rne(man, exp, *, subnormals=True, saturate=False):
    sub_mode = (
        SubnormalsMode.SUBNORMALS if subnormals else SubnormalsMode.NORMALS
    )
    return lambda x: float_quantize_v2(
        x,
        exp=exp,
        man=man,
        rounding_mode=RoundMode.RNE,
        subnormals_mode=sub_mode,
        saturate=saturate,
    )


def make_superfp_quantize_rne(man, exp, binades, *, saturate=False):
    return lambda x: superfp_quantize_v2(
        x,
        exp=exp,
        man=man,
        binades=binades,
        rounding_mode=RoundMode.RNE,
        saturate=saturate,
    )