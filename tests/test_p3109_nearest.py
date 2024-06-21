import torch
from mptorch.quant import p3109_quantize
import struct
import math

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def test_p3109p1():
    # Test cases
    tests = [
        {"description": "CASE 0: normal", "value": torch.tensor([[1.5,4.0E-13],[134210000.0,-9.0E-07]], dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,4.5474735E-13],[134217728.0,-9.5367432E-07]], dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal", "value": torch.tensor([bits_to_float(0b00100000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00100000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 2: min_subnormal", "value": torch.tensor([bits_to_float((min_exp - man_bits + 127) << 23)], dtype=torch.float32), "expected": torch.tensor([bits_to_float((min_exp - man_bits + 127) << 23)], dtype=torch.float32)},
        # {"description": "CASE 3: round_to_0", "value": torch.tensor([bits_to_float(middle_num)], dtype=torch.float32), "expected": torch.tensor([0.0], dtype=torch.float32)},
        # {"description": "CASE 3bis: round_to_min", "value": torch.tensor([bits_to_float(middle_num + 1)], dtype=torch.float32), "expected": torch.tensor([bits_to_float((min_exp - man_bits + 127) << 23)], dtype=torch.float32)},
        # {"description": "CASE 4: max_normal", "value": torch.tensor([bits_to_float((max_exp + 127) << 23 | max_man)], dtype=torch.float32), "expected": torch.tensor([bits_to_float((max_exp + 127) << 23 | max_man)], dtype=torch.float32)},
        # {"description": "CASE 5: overflow", "value": torch.tensor([bits_to_float((max_exp + 127) << 23 | (((1 << man_bits) - 1) & ~1) + 1 << (23 - man_bits))], dtype=torch.float32), "expected": torch.tensor([bits_to_float((max_exp + 127) << 23 | max_man)], dtype=torch.float32)},
        # {"description": "CASE 6: underflow", "value": torch.tensor([bits_to_float((min_exp - man_bits - 2 + 127) << 23)], dtype=torch.float32), "expected": torch.tensor([0.0], dtype=torch.float32)},
        # {"description": "CASE 7: NaN", "value": torch.tensor([float('nan')], dtype=torch.float32), "expected": torch.tensor([float('nan')], dtype=torch.float32)},
        # {"description": "CASE 8: +inf", "value": torch.tensor([float('inf')], dtype=torch.float32), "expected": torch.tensor([float('inf')], dtype=torch.float32)},
        # {"description": "CASE 9: -inf", "value": torch.tensor([-float('inf')], dtype=torch.float32), "expected": torch.tensor([-float('inf')], dtype=torch.float32)}
    ]

    for test in tests:
        description = test["description"]
        value = test["value"]
        expected = test["expected"]
        result = p3109_quantize(value, 1, "nearest", True, True)
        print(description)
        print(value)
        print(f"Result: {result}, Expected: {expected}")
        print("Pass" if (math.isnan(result) and math.isnan(expected)) or result == expected else "Fail")

# def test_p3109p2():
#     # Test cases
#     tests = [
#         {"description": "CASE 0: normal", "value": 1.5, "expected": 1.5},
#         {"description": "CASE 1: min_normal", "value": bits_to_float(0b00100000100000000000000000000000), "expected": bits_to_float(0b00100000100000000000000000000000)},
#         # {"description": "CASE 2: min_subnormal", "value": bits_to_float((min_exp - man_bits + 127) << 23), "expected": bits_to_float((min_exp - man_bits + 127) << 23)},
#         # {"description": "CASE 3: round_to_0", "value": bits_to_float(middle_num), "expected": 0.0},
#         # {"description": "CASE 3bis: round_to_min", "value": bits_to_float(middle_num + 1), "expected": bits_to_float((min_exp - man_bits + 127) << 23)},
#         # {"description": "CASE 4: max_normal", "value": bits_to_float((max_exp + 127) << 23 | max_man), "expected": bits_to_float((max_exp + 127) << 23 | max_man)},
#         # {"description": "CASE 5: overflow", "value": bits_to_float((max_exp + 127) << 23 | (((1 << man_bits) - 1) & ~1) + 1 << (23 - man_bits)), "expected": bits_to_float((max_exp + 127) << 23 | max_man)},
#         # {"description": "CASE 6: underflow", "value": bits_to_float((min_exp - man_bits - 2 + 127) << 23), "expected": 0.0},
#         # {"description": "CASE 7: NaN", "value": float('nan'), "expected": float('nan')},
#         # {"description": "CASE 8: +inf", "value": float('inf'), "expected": float('inf')},
#         # {"description": "CASE 9: -inf", "value": -float('inf'), "expected": -float('inf')}
#     ]
#     for test in tests:
#         description = test["description"]
#         value = test["value"]
#         expected = test["expected"]
#         result = p3109_quantize(value, 1, "nearest", True, True)
#         print(description)
#         print(value)
#         print(f"Result: {result}, Expected: {expected}")
#         print("Pass" if (math.isnan(result) and math.isnan(expected)) or result == expected else "Fail")