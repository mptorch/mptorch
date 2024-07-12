import torch
from mptorch.quant import binary8_quantize
import struct
import numpy as np
import random

from gfloat import RoundMode, round_float
from gfloat.formats import *

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

def test_binary8_to_gfloat():
    if no_cuda():
        return
    
    for P in range(1, 8):
        fi = format_info_p3109(P)
        exp_bits = 8 - P
        man_bits = P - 1

        spec_exp = 1 if P == 1 else 0
        max_exp = (1 << (exp_bits - 1)) - 1
        min_exp = spec_exp - max_exp

        min_val_uival = (min_exp - man_bits - 2 + 127) << 23
        max_val_uival = (max_exp + 2 + 127) << 23

        i_uival = min_val_uival
        while i_uival <= max_val_uival:
            i_fval = bits_to_float(i_uival)

            result1 = binary8_quantize(torch.tensor(i_fval, dtype=torch.float32, device="cuda"), P, "nearest", "saturate", True, True)
            result2 = round_float(fi, i_fval, RoundMode.TiesToEven, True)
            result1 = result1.cpu() 

            expected = torch.tensor(result2, dtype=torch.float32, device="cpu")

            assert result1.equal(expected) or np.isnan(result1)
            i_uival += 16384 #8192

def test_binary8p1():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 1, "nearest", "saturate", True, True)
    # normal
    assert_quant([[1.5,4.0E-13],[134210000.0,-9.0E-07]], [[2.0,4.5474735E-13],[134217728.0,-9.5367432E-07]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant)
    assert_quant([-float('inf')], [-float('inf')], quant)

def test_binary8p2():
    if no_cuda():
        return
    
    quant = lambda x: binary8_quantize(x, 2, "nearest", "saturate", True, True)
    # normal
    assert_quant([[1.5,2.90E-08],[-1.1402E-05,-25000824.0]], [[1.5,2.9802322E-08],[-1.1444092E-05,-25165824.0]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00110000000000000000000000000000), b2f(0b00110000000000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00101111100000000000000000000000), b2f(0b00101111100000000000000000000000), quant) # min subnormal
    assert_quant(b2f(0b00101111000000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00101111000000000000000000000001), b2f(0b00101111100000000000000000000000), quant) # round to min
    # assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # max normal
    # assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    # assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant)
    assert_quant([-float('inf')], [-float('inf')], quant) 

def test_no_subnormal_case():
    if no_cuda():
        return

    quant = lambda x: binary8_quantize(x, 3, "nearest", "saturate", True, False)

    assert_quant(float(1.9E-5), float(1.907348633E-5), quant)
    assert_quant(float(2.3E-5), float(2.288818359E-5), quant)
    assert_quant(float(2.67E-5), float(2.670288086E-5), quant)
    assert_quant(float(9.7E-6), float(1.907348633E-5), quant) # minimum normalized value
    assert_quant(float(9.5E-6), float(0), quant) # to 0

def test_binary8_signed_nearest():
    if no_cuda():
        return
    
    # parameters :
    iterations = 1
    
    for P in range(1, 8):

        print(P)

        exp_bits = 8 - P
        man_bits = P - 1  

        spec_exp = 1 if P == 1 else 0

        max_exp = (1 << (exp_bits - 1)) - 1
        min_exp = spec_exp - max_exp

        min_val = np.uint32((min_exp - man_bits + 127) << 23)

        max_exp_bias = np.uint32((max_exp + 127) << 23) 
        max_man = (((1 << man_bits) - 1) & ~1) << (23 - man_bits)
        max_val = bits_to_float(max_exp_bias | max_man)

        # previous_fval = bits_to_float(np.uint32(0))
        i_fval = bits_to_float(min_val) + 2**(-man_bits)*2**(min_exp)
        previous_fval = bits_to_float(min_val)

        while i_fval <= max_val:
            for i in range(10):
                random_float = bits_to_float(random.randint(float_to_bits(previous_fval), float_to_bits(i_fval)))
                num_tensor = torch.full((iterations,), random_float, dtype=torch.float32, device="cuda")
                result = binary8_quantize(num_tensor, P, "nearest", "saturate", True, True)
                result1 = result.cpu() 

                distance = (random_float - previous_fval) / (i_fval - previous_fval)
                if(distance < 0.5):
                    assert result1 == previous_fval
                else:
                    assert result1 == i_fval

            previous_fval = i_fval
            exp_prev = min_exp if previous_fval == 0 else max(((float_to_bits(previous_fval) << 1 >> 24) - 127),min_exp)
            i_fval += 2**(-man_bits)*2**(exp_prev)

def main():
    test_binary8_to_gfloat()


if __name__ == "__main__":
    main()