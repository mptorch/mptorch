from mptorch import FloatingPoint


def test_fp32():
    num1 = FloatingPoint(exp=8, man=23)
    assert num1.is_fp32


def test_not_fp32():
    num1 = FloatingPoint(exp=8, man=15)
    assert not num1.is_fp32


def test_fp16():
    num1 = FloatingPoint(exp=5, man=10)
    assert num1.is_fp16


def test_not_fp16():
    num1 = FloatingPoint(exp=1, man=15)
    assert not num1.is_fp16


def test_bfloat16():
    num1 = FloatingPoint(exp=8, man=7)
    assert num1.is_bfloat16


def test_not_bfloat16():
    num1 = FloatingPoint(exp=8, man=8)
    assert not num1.is_bfloat16
