"""A SplitMac whose products and sums are rounded in two format families."""

import torch

from mptorch import BinaryK, SuperFP
from mptorch.quant import Quant, SplitMac, binaryK_superfp_matmul, qmm

torch.manual_seed(0)
K = 8
a = torch.randn(1, K)
b = torch.randn(K, 1)
binary8p4 = BinaryK(8, 4)
# 12 bits: a 6-bit mantissa in the top 31 binades, from 2**-14 up, and 63
# powers of two below them, down to 2**-77.
superfp12 = SuperFP(6, 5, 31, 15)


def by_hand(q_mul, q_acc):
    s = torch.zeros(1)
    for k in range(K):
        s = q_acc(s + q_mul(a[0, k] * b[k, 0]))
    return s


# Binary8p4 products, each partial sum rounded to the SuperFP format.
mac = SplitMac(binary8p4, superfp12)
out = qmm(a, b, mac)
s = by_hand(Quant(binary8p4), Quant(superfp12))
flat = binaryK_superfp_matmul(
    a,
    b,
    mul_K=8,
    mul_P=4,
    acc_man_bits=6,
    acc_exp_bits=5,
    acc_normal_binades=31,
    acc_bias=15,
)
print(f"float32 torch.matmul           : {(a @ b).item():.6f}")
same = torch.equal(out.flatten(), s)
print(f"Binary8p4 products, SuperFP sum: {out.item():.6f}   by hand equal: {same}")
print(f"binaryK_superfp_matmul         : {flat.item():.6f}   equal: {torch.equal(flat, out)}")

# The reverse: SuperFP products summed in Binary16p11.
binary16p11 = BinaryK(16, 11)
rev = qmm(a, b, SplitMac(superfp12, binary16p11))
s = by_hand(Quant(superfp12), Quant(binary16p11))
same = torch.equal(rev.flatten(), s)
print(f"SuperFP products, Binary16p11  : {rev.item():.6f}   by hand equal: {same}")

# A palette in either slot has no cross-family operator to select.
try:
    SplitMac([binary8p4, BinaryK(8, 3)], superfp12)
except ValueError as e:
    print(f"\nValueError: {e}")
