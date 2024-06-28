import torch
from mptorch.quant import p3109_quantize
import struct
import numpy as np
import random

from gfloat import RoundMode, round_float
from gfloat.formats import *

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

# def test_p3109_to_gfloat():
#     for P in range(1, 8):
#         fi = format_info_p3109(P)
#         exp_bits = 8 - P
#         man_bits = P - 1

#         spec_exp = 1 if P == 1 else 0
#         max_exp = (1 << (exp_bits - 1)) - 1
#         min_exp = spec_exp - max_exp

#         min_val_uival = (min_exp - man_bits - 2 + 127) << 23
#         max_val_uival = (max_exp + 2 + 127) << 23

#         i_uival = min_val_uival
#         while i_uival <= max_val_uival:
#             i_fval = bits_to_float(i_uival)

#             result1 = p3109_quantize(torch.tensor(i_fval, dtype=torch.float32, device="cuda"), P, "nearest", "saturate", True, True)
#             result2 = round_float(fi, i_fval, RoundMode.TiesToEven, True)
#             result1 = result1.cpu() 

#             assert result1 == result2 or np.isnan(result1)
#             i_uival += 8192 #8192

def test_p3109p1():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,4.0E-13],[134210000.0,-9.0E-07]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,4.5474735E-13],[134217728.0,-9.5367432E-07]] , dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00100000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00100000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00100000000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00100000000000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00100000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01011111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00011111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    ]

    for test in tests:
        result_t = p3109_quantize(test["value"], 1, "nearest", "saturate", True, True)
        print(result_t)
        print(test["expected"])
        assert result_t.equal(test["expected"])

def test_p3109p2():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,2.90E-08],[-1.1402E-05,-25000824.0]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5,2.9802322E-08],[-1.1444092E-05,-25165824.0]] , dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00110000000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00110000000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 2: min_subnormal", "value": torch.tensor([bits_to_float(0b00101111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00101111100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 3: round_to_0"  , "value": torch.tensor([bits_to_float(0b00101111000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: round_to_min", "value": torch.tensor([bits_to_float(0b00101111000000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00101111100000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01011111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00011111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    ]

    for test in tests:
        result_t = p3109_quantize(test["value"], 2, "nearest", "saturate", True, True)  
        print(result_t) 
        print(test["expected"])
        assert result_t.equal(test["expected"])

def test_p3109_signed_nearest():

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
                result = p3109_quantize(num_tensor, P, "nearest", "saturate", True, True)
                result1 = result.cpu() 

                distance = (random_float - previous_fval) / (i_fval - previous_fval)
                if(distance < 0.5):
                    assert result1 == previous_fval
                else:
                    assert result1 == i_fval

            previous_fval = i_fval
            exp_prev = min_exp if previous_fval == 0 else max(((float_to_bits(previous_fval) << 1 >> 24) - 127),min_exp)
            i_fval += 2**(-man_bits)*2**(exp_prev)