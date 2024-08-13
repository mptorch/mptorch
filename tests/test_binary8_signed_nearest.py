import torch
from mptorch.quant import binary8_quantize
import numpy as np
import random
import pytest
from tests.markers import available_devices
from gfloat import RoundMode, round_float
from gfloat.formats import *
from tests.quant import bits_to_float, float_to_bits, assert_quant

@pytest.mark.parametrize("device", available_devices)
def test_binary8_to_gfloat(device): 
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

            result1 = binary8_quantize(torch.tensor(i_fval, dtype=torch.float32, device=device), P, "nearest", "saturate_maxfloat", True, True)
            result2 = round_float(fi, i_fval, RoundMode.TiesToEven, True)
            result1 = result1.cpu() 

            expected = torch.tensor(result2, dtype=torch.float32, device="cpu")

            assert result1.equal(expected) or np.isnan(result1)
            i_uival += 16384 #8192

@pytest.mark.parametrize("device", available_devices)
def test_binary8p1_saturate_maxfloat(device):
    quant = lambda x: binary8_quantize(x, 1, "nearest", "saturate_maxfloat", True, True)
    # normal
    assert_quant([[1.5,4.0E-13],[134210000.0,-9.0E-07]], [[2.0,4.5474735E-13],[134217728.0,-9.5367432E-07]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000100000000000000000000000), b2f(0b00100000100000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00100000000000000000000000000000), [0.0], quant, device) # round to 0
    assert_quant(b2f(0b00100000000000000000000000000001), b2f(0b00100000100000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01011111000000000000000000000000), quant, device) # overflow
    assert_quant(b2f(0b00011111100000000000000000000000), [0.0], quant, device) # underflow
    assert_quant([float('inf')], b2f(0b01011111000000000000000000000000), quant, device)
    assert_quant([-float('inf')], b2f(0b11011111000000000000000000000000), quant, device)

@pytest.mark.parametrize("device", available_devices)
def test_binary8p2_saturate_maxfloat(device):
    quant = lambda x: binary8_quantize(x, 2, "nearest", "saturate_maxfloat", True, True)
    # normal
    assert_quant([[1.5,2.90E-08],[-1.1402E-05,-25000824.0]], [[1.5,2.9802322E-08],[-1.1444092E-05,-25165824.0]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00110000000000000000000000000000), b2f(0b00110000000000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00101111100000000000000000000000), b2f(0b00101111100000000000000000000000), quant, device) # min subnormal
    assert_quant(b2f(0b00101111000000000000000000000000), [0.0], quant, device)                                   # round to 0
    assert_quant(b2f(0b00101111000000000000000000000001), b2f(0b00101111100000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01001111000000000000000000000000), b2f(0b01001111000000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01011111100000000000000000000000), b2f(0b01001111000000000000000000000000), quant, device) # overflow
    assert_quant(b2f(0b00101110100000000000000000000000), [0.0], quant, device)                                   # underflow
    assert_quant([float('inf')], b2f(0b01001111000000000000000000000000), quant, device)
    assert_quant([-float('inf')], b2f(0b11001111000000000000000000000000), quant, device) 

@pytest.mark.parametrize("device", available_devices)
def test_binary8p3_saturate_maxfloat(device):
    quant = lambda x: binary8_quantize(x, 3, "nearest", "saturate_maxfloat", True, True)
    # normal
    assert_quant([[1.5,13.0],[13.000001,-13.0]], [[1.5,12.0],[14.0,-12]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111000000000000000000000000000), b2f(0b00111000000000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00110111000000000000000000000000), b2f(0b00110111000000000000000000000000), quant, device) # min subnormal
    assert_quant(b2f(0b00110110100000000000000000000000), [0.0], quant, device)                                   # round to 0
    assert_quant(b2f(0b00110110100000000000000000000001), b2f(0b00110111000000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01000111010000000000000000000000), b2f(0b01000111010000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01000111011000000000000000000000), b2f(0b01000111010000000000000000000000), quant, device) # overflow
    assert_quant(b2f(0b00110110000000000000000000000000), [0.0], quant, device)                                   # underflow
    assert_quant([float('inf')], b2f(0b01000111010000000000000000000000), quant, device)
    assert_quant([-float('inf')], b2f(0b11000111010000000000000000000000), quant, device) 

@pytest.mark.parametrize("device", available_devices)
def test_binary8p3_saturate_maxfloat2(device):
    quant = lambda x: binary8_quantize(x, 3, "nearest", "saturate_maxfloat2", True, True)
    # normal
    assert_quant([[1.5,13.0],[13.000001,-13.0]], [[1.5,12.0],[14.0,-12]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111000000000000000000000000000), b2f(0b00111000000000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00110111000000000000000000000000), b2f(0b00110111000000000000000000000000), quant, device) # min subnormal
    assert_quant(b2f(0b00110110100000000000000000000000), [0.0], quant, device)                                   # round to 0
    assert_quant(b2f(0b00110110100000000000000000000001), b2f(0b00110111000000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01000111011000000000000000000000), b2f(0b01000111011000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01000111011100000000000000000000), b2f(0b01000111011000000000000000000000), quant, device) # overflow
    assert_quant(b2f(0b00110110000000000000000000000000), [0.0], quant, device)                                   # underflow
    assert_quant([float('inf')], b2f(0b01000111011000000000000000000000), quant, device)
    assert_quant([-float('inf')], b2f(0b11000111011000000000000000000000), quant, device) 

@pytest.mark.parametrize("device", available_devices)
def test_binary8p3_saturate_infty(device):
    quant = lambda x: binary8_quantize(x, 3, "nearest", "saturate_infty", True, True)
    # normal
    assert_quant([[1.5,13.0],[13.000001,-13.0]], [[1.5,12.0],[14.0,-12]], quant, device)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111000000000000000000000000000), b2f(0b00111000000000000000000000000000), quant, device) # min normal
    assert_quant(b2f(0b00110111000000000000000000000000), b2f(0b00110111000000000000000000000000), quant, device) # min subnormal
    assert_quant(b2f(0b00110110100000000000000000000000), [0.0], quant, device)                                   # round to 0
    assert_quant(b2f(0b00110110100000000000000000000001), b2f(0b00110111000000000000000000000000), quant, device) # round to min
    assert_quant(b2f(0b01000111010000000000000000000000), b2f(0b01000111010000000000000000000000), quant, device) # max normal
    assert_quant(b2f(0b01000111011000000000000000000000), [float('inf')], quant, device) # overflow
    assert_quant(b2f(0b00110110000000000000000000000000), [0.0], quant, device)                                   # underflow
    assert_quant([float('inf')], [float('inf')], quant, device)
    assert_quant([-float('inf')], [-float('inf')], quant, device) 

@pytest.mark.parametrize("device", available_devices)
def test_no_subnormal_case(device):
    quant = lambda x: binary8_quantize(x, 3, "nearest", "saturate_maxfloat", True, False)

    assert_quant(float(1.9E-5), float(1.907348633E-5), quant, device)
    assert_quant(float(2.3E-5), float(2.288818359E-5), quant, device)
    assert_quant(float(2.67E-5), float(2.670288086E-5), quant, device)
    assert_quant(float(9.7E-6), float(1.907348633E-5), quant, device) # minimum normalized value
    assert_quant(float(9.5E-6), float(0), quant, device) # to 0
    assert_quant(float(1.5258789E-5), float(1.907348633E-5), quant, device)
    assert_quant(float(2.288818359375E-5), float(2.288818359E-5), quant, device)

@pytest.mark.parametrize("device", available_devices)
def test_binary8_signed_nearest(device):
    # parameters :
    iterations = 1
    
    for P in range(1, 8):

        # print(P)

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
                num_tensor = torch.full((iterations,), random_float, dtype=torch.float32, device=device)
                result = binary8_quantize(num_tensor, P, "nearest", "saturate_maxfloat", True, True)
                result1 = result.cpu() 

                distance = (random_float - previous_fval) / (i_fval - previous_fval)
                if distance < 0.5:
                    assert result1 == previous_fval
                elif distance > 0.5:
                    assert result1 == i_fval

            previous_fval = i_fval
            exp_prev = min_exp if previous_fval == 0 else max(((float_to_bits(previous_fval) << 1 >> 24) - 127),min_exp)
            i_fval += 2**(-man_bits)*2**(exp_prev)