import torch
from mptorch.quant import p3109_quantize
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


def test_p3109p1_SATURATE():
    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "saturate", False, True)
    # normal
    assert_quant([[1.5,6.0E-5],[72057500037900000.0,0.01171875]], [[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00000000100000000000000000000000), b2f(0b00000000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00000000010000000000000000000001), b2f(0b00000000100000000000000000000000), quant) # round to min
    assert_quant(b2f(0b01111110100000000000000000000000), b2f(0b01111110100000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01111110110011100000000000000000), b2f(0b01111110100000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant)
    
    # Test cases
    # tests = [
        # {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,6.0E-5],[72057500037900000.0,0.01171875]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]] , dtype=torch.float32, device="cuda")},
        # {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
        # {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00000000010000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0b00000000100000000000000000000000], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01111110110011100000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 1, "nearest", "saturate", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p1_NO_OVERFLOW():
    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
    # normal 
    assert_quant([[1.5,6.0E-5],[72057500037900000.0,0.01171875]], [[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00000000100000000000000000000000), b2f(0b00000000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00000000010000000000000000000001), b2f(0b00000000100000000000000000000000)) # round to min
    assert_quant(b2f(0b01111111000000000000000000000000), b2f(0b01111111000000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01111111010110000000110000111000), b2f(0b01111111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], b2f(0b0111111100000000000000000000000), quant) # +inf

    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,6.0E-5],[72057500037900000.0,0.01171875]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00000000010000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01111111010110000000110000111000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111111000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 1, "nearest", "no_overflow", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p1_OVERFLOW():
    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
    # normal 
    assert_quant([[1.5,6.0E-5],[72057500037900000.0,0.01171875]], [[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00000000100000000000000000000000), b2f(0b00000000100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00000000010000000000000000000001), b2f(0b00000000100000000000000000000000)) # round to min
    assert_quant(b2f(0b01111111000000000000000000000000), b2f(0b01111111000000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01111111010110000000110000111000), b2f(0b01111111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf

    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,6.0E-5],[72057500037900000.0,0.01171875]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[2.0,6.103515625e-5],[72057594037927936.0,0.0078125]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: round_to_min", "value": torch.tensor([bits_to_float(0b00000000010000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00000000100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: max_normal"  , "value": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01111110100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: overflow"    , "value": torch.tensor([bits_to_float(0b01111111010001110000001100000100)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 1, "nearest", "overflow", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p2_SATURATE(): # 0xfd is max bc there is inf
    if no_cuda():
        return 

    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
    # normal 
    assert_quant([[1.5,1],[1.77e-15, 5242880]], [[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000000000000000000000011111), b2f(0b00100000000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b01011110010000000000000000000000), b2f(0b01011110010000000000000000000000), quant) # max normal
    assert_quant(b2f(0b01011110010001110000001100000100), b2f(0b01011110010000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf

    # # Test cases
    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,1],[1.77e-15, 5242880]]                     , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: round_to_min" ,"value": torch.tensor([bits_to_float(0b00100000000000000000000000011111)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00100000000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: max_normal"  , "value": torch.tensor([bits_to_float(0b01011110010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011110010000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: overflow"    , "value": torch.tensor([bits_to_float(0b01011110010001110000001100000100)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011110010000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: +inf"        , "value": torch.tensor([float('inf')],                                      dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 2, "nearest", "saturate", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p2_NO_OVERFLOW(): # no inf so we go to 0xfe as max
    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
    # normal 
    assert_quant([[1.5,1],[1.77e-15, 5242880]], [[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000000000000000000000011111), b2f(0b00100000000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b01011110110000000000000000000000), b2f(0b01011110110000000000000000000000)) # max normal
    assert_quant(b2f(0b01011111000000000000000000000000), b2f(0b01011111000000000000000000000000), quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [b2f(0b01011111000000000000000000000000)], quant) # +inf


    # Test cases
    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,1],[1.77e-15, 5242880]]                     , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: round_to_min"  , "value": torch.tensor([bits_to_float(0b00100000000000000000000000011111)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00100000000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: max_normal"  , "value": torch.tensor([bits_to_float(0b01011110110000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011110110000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: overflow"    , "value": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: +inf"        , "value": torch.tensor([float('inf')],                                      dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011111000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 2, "nearest", "no_overflow", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p2_OVERFLOW(): # any overflow goes to inf, not 0xfd

    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
    # normal 
    assert_quant([[1.5,1],[1.77e-15, 5242880]], [[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00100000000000000000000000011111), b2f(0b00100000000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00000000010000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b01011110010000000000000000000000), b2f(0b01011110010000000000000000000000)) # max normal
    assert_quant(b2f(0b01011111000000000000000000000000), [float('inf')], quant) # overflow
    assert_quant(b2f(0b00000000000000001110000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf

    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,1],[1.77e-15, 5242880]]     , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5, 1],[bits_to_float(0b00100111000000000000000000000000), 4194304]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: round_to_min"  , "value": torch.tensor([bits_to_float(0b00100000000000000000000000011111)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00100000000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: round_to_0"  , "value": torch.tensor([bits_to_float(0b00000000010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: max_normal"  , "value": torch.tensor([bits_to_float(0b01011110010000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b01011110010000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: overflow"    , "value": torch.tensor([bits_to_float(0b01011111100001110000001100000100)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: underflow"   , "value": torch.tensor([bits_to_float(0b00000000000000001110000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: +inf"        , "value": torch.tensor([float('inf')],                                      dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 2, "nearest", "overflow", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])


def test_p3109p8_SATURATE():
    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
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
    assert_quant([float('inf')], [float('inf')], quant) # +inf



 #   tests = [
      #  {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,1.00390625],[1.4921875,1.01171875E0]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5,1.0],[1.4921875,1.015625E0]] , dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00111111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111100000000000000000000000)], dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 2: min_subnor"  , "value": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 3: round_to_0"  , "value": torch.tensor([bits_to_float(0b00111011100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 4: round_to_min", "value": torch.tensor([bits_to_float(0b00111011100000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 5: max_normal"  , "value": torch.tensor([bits_to_float(0b00111111111111010000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111111111010000000000000000)], dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 6: overflow"    , "value": torch.tensor([bits_to_float(0b00111111111111110000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111111111010000000000000000)], dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 7: underflow"   , "value": torch.tensor([bits_to_float(0b00111011000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
      #  {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
        # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
 #   ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 8, "nearest", "saturate", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p8_NO_OVERFLOW():

    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
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

    # # Test cases
    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,1.00390625],[1.4921875,1.01171875E0]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5,1.0],[1.4921875,1.015625E0]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00111111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: min_subnor"  , "value": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: round_to_0"  , "value": torch.tensor([bits_to_float(0b00111011100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: round_to_min", "value": torch.tensor([bits_to_float(0b00111011100000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: max_normal"  , "value": torch.tensor([bits_to_float(0b00111111111111100000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111111111100000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: overflow"    , "value": torch.tensor([bits_to_float(0b00111111111111111000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111111111100000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 7: underflow"   , "value": torch.tensor([bits_to_float(0b00111011000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111111111100000000000000000)], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 8, "nearest", "no_overflow", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])

def test_p3109p8_OVERFLOW():

    if no_cuda():
        return
    
    quant = lambda x: p3109_quantize(x, 1, "nearest", "no_overflow", False, True)
    # normal 
    assert_quant([[1.5,1.00390625],[1.4921875,1.01171875E0]] , [[1.5,1.0],[1.4921875,1.015625E0]], quant)

    b2f = lambda b: [bits_to_float(b)]

    assert_quant(b2f(0b00111111100000000000000000000000), b2f(0b00111111100000000000000000000000), quant) # min normal
    assert_quant(b2f(0b00111100000000000000000000000000), b2f(0b00111100000000000000000000000000), quant) # min subnormal
    assert_quant(b2f(0b00111011100000000000000000000000), [0.0], quant) # round to 0
    assert_quant(b2f(0b00111011100000000000000000000001), b2f(0b00111100000000000000000000000000), quant) # round to min
    assert_quant(b2f(0b00111111111111010000000000000000), b2f(0b00111111111111010000000000000000), quant) # max normal
    assert_quant(b2f(0b00111111111111110000000000000000), b2f([float('inf')]), quant) # overflow
    assert_quant(b2f(0b00111011000000000000000000000000), [0.0], quant) # underflow
    assert_quant([float('inf')], [float('inf')], quant) # +inf



    # tests = [
    #     {"description": "CASE 0: normal"      , "value": torch.tensor([[1.5,1.00390625],[1.4921875,1.01171875E0]]             , dtype=torch.float32, device="cuda"), "expected": torch.tensor([[1.5,1.0],[1.4921875,1.015625E0]] , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 1: min_normal"  , "value": torch.tensor([bits_to_float(0b00111111100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111100000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 2: min_subnor"  , "value": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 3: round_to_0"  , "value": torch.tensor([bits_to_float(0b00111011100000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0]                                              , dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 4: round_to_min", "value": torch.tensor([bits_to_float(0b00111011100000000000000000000001)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111100000000000000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 5: max_normal"  , "value": torch.tensor([bits_to_float(0b00111111111111010000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([bits_to_float(0b00111111111111010000000000000000)], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 6: overflow"    , "value": torch.tensor([bits_to_float(0b00111111111111110000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 7: underflow"   , "value": torch.tensor([bits_to_float(0b00111011000000000000000000000000)], dtype=torch.float32, device="cuda"), "expected": torch.tensor([0.0], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 7: NaN"         , "value": torch.tensor([float('nan')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('nan')], dtype=torch.float32, device="cuda")},
    #     {"description": "CASE 8: +inf"        , "value": torch.tensor([float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([float('inf')], dtype=torch.float32, device="cuda")},
    #     # {"description": "CASE 9: -inf"        , "value": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda"), "expected": torch.tensor([-float('inf')], dtype=torch.float32, device="cuda")}
    # ]

    # for test in tests:
    #     result_t = p3109_quantize(test["value"], 8, "nearest", "overflow", False, True)
    #     print(result_t)
    #     print(test["expected"])
    #     assert result_t.equal(test["expected"])