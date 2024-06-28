import torch
from mptorch.quant import p3109_quantize
import struct


def bits_to_float(bits):
    s = struct.pack('>I', bits)
    return struct.unpack('>f', s)[0]

def float_to_bits(value):
    s = struct.pack('>f', value)
    return struct.unpack('>I', s)[0]

def test_p3109p1():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,4.0E-13],[134210000.0,-9.0E-07]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.0,2.2737368E-13],[6.7108864e+07,-4.7683716e-07]] , dtype=torch.float32, device="cuda")},
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
        result_t = p3109_quantize(test["value"], 1, "troncate", "saturate", True, True)
        print(result_t)
        print(test["expected"])
        assert result_t.equal(test["expected"])

def test_p3109p4():
    # Test cases
    tests = [
        {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,165.65200],[0.0550651,-0.0685105]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5,160.0],[0.0546875,-0.0625]] , dtype=torch.float32, device="cuda")},
        {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 2: min_subnormal", "value": torch.tensor([bits_to_float(0b00111010100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111010100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 3: round_to_0"  , "value": torch.tensor([bits_to_float(0b00111010000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: round_to_min", "value": torch.tensor([bits_to_float(0b00111010000000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111010100000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01000011011000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01000011011000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01000011011100000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01000011011000000000000000000000)], dtype=torch.float32, device="cuda")},
        {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00111001100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    ]

    for test in tests:
        result_t = p3109_quantize(test["value"], 4, "troncate", "saturate", True, True)  
        print(result_t) 
        print(test["expected"])
        assert result_t.equal(test["expected"])

def main():
    test_p3109p4()
    test_p3109p1()


if __name__ == "__main__":
    main()