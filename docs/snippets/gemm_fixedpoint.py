"""A dot product in fixed point: exact sums, saturation or wrap-around, and a
wide accumulator."""

import torch

from mptorch import FixedPoint, Overflow
from mptorch.quant import (
    FusedMac,
    QLinear,
    Quant,
    SplitMac,
    fixedpoint_gemm_formats,
    fixedpoint_matmul,
    qmm,
)

torch.manual_seed(0)
K = 16
# Operands on an 8-bit grid (4 fractional bits), as a quantizer would leave them.
grid = Quant(FixedPoint(8, 4))
a, b = grid(torch.randn(1, K) * 2), grid(torch.randn(K, 1) * 2)
mul, acc = FixedPoint(8, 2), FixedPoint(12, 3)


def by_hand(q_mul, q_acc):
    # Each sum is exact: float64 holds every one of these values.
    s = torch.zeros(1, dtype=torch.float64)
    for k in range(K):
        s = q_acc(s + q_mul(a[0, k] * b[k, 0]).double())
    return s


out = qmm(a, b, SplitMac(mul, acc))
s = by_hand(Quant(mul), Quant(acc))
print(f"float32 torch.matmul         : {(a @ b).item():.6f}")
same = torch.equal(out.double().flatten(), s)
print(f"FixedPoint(8, 2) products,   : {out.item():.6f}   by hand equal: {same}")
print("  FixedPoint(12, 3) sums")
flat = fixedpoint_matmul(a, b, mul_wl=8, mul_fl=2, acc_wl=12, acc_fl=3)
print(f"fixedpoint_matmul            : {flat.item():.6f}   equal: {torch.equal(flat, out)}")

# int8 sums of 100, 100, -100 and -60: the second partial sum overflows.
x = torch.tensor([[100.0, 100.0, -100.0, -60.0]])
ones = torch.ones(4, 1)
int8, int16 = FixedPoint(8, 0), FixedPoint(16, 0)
print(f"\nint8 sums, saturated         : {qmm(x, ones, SplitMac(int16, int8)).item():.0f}")
wrapped = qmm(x, ones, SplitMac(int16, int8, overflow=Overflow.WRAP))
print(f"int8 sums, wrapped around    : {wrapped.item():.0f}")
fused = qmm(x, ones, FusedMac(int8, overflow=Overflow.WRAP))
print(f"int8 fused steps, wrapped    : {fused.item():.0f}")

# A 32-bit accumulator has more significant bits than float32 holds.
wide = dict(mul_wl=16, mul_fl=8, acc_wl=32, acc_fl=8)
print(f"\nwide result dtype            : {fixedpoint_matmul(a, b, **wide).dtype}")
try:
    qmm(a, b, SplitMac(FixedPoint(16, 8), FixedPoint(32, 8)))
except ValueError as e:
    print(f"ValueError: {e}")

# A layer rounds such a result once, after the bias, with its output rounding.
formats = fixedpoint_gemm_formats(16, 8, acc_wl=32, acc_fl=8, output_quant=FixedPoint(16, 8))
layer = QLinear(K, 4, formats=formats)
print(f"\nlayer output dtype           : {layer(a).dtype}")
