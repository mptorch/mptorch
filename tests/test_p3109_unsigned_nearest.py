import torch
from mptorch.quant import p3109_quantize
import struct
import math
import numpy as np

from gfloat import RoundMode, round_float
from gfloat.formats import *

def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def test_p3109p1_SATURATE():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,6.0E-5],[72057500037900000.0,0.01171875]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]] , dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00000000010000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01011111100001110000001100000100)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    ]

    for test in tests:
        result_t = p3109_quantize(test["value"], 1, "nearest", True, True)
        print(result_t)
        print(test["expected"])
        assert result_t.equal(test["expected"])

def test_p3109p1_NO_OVERFLOW():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,6.0E-5],[72057500037900000.0,0.01171875]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]] , dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00000000010000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01111111010110000000110000111000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    ]

    for test in tests:
        result_t = p3109_quantize(test["value"], 1, "nearest", True, True)
        print(result_t)
        print(test["expected"])
        assert result_t.equal(test["expected"])

def test_p3109p1_OVERFLOW():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,6.0E-5],[72057500037900000.0,0.01171875]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]] , dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00000000010000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01011111100001110000001100000100)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    ]

    for test in tests:
        result_t = p3109_quantize(test["value"], 1, "nearest", True, True)
        print(result_t)
        print(test["expected"])
        assert result_t.equal(test["expected"])