"""A SplitMac dot product, reproduced step by step in Python."""

import torch

from mptorch import BinaryK
from mptorch.quant import Quant, binaryK_matmul

torch.manual_seed(0)
K = 8
a = torch.randn(1, K)
b = torch.randn(K, 1)
fmt = BinaryK(8, 4)
q = Quant(fmt)  # round to E4M3

# The op: every product rounded to E4M3, and the running sum rounded to
# E4M3 after every addition. The *operands* are read as they are.
out = binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=8, acc_P=4)

# The same thing by hand, in the same order.
s = torch.zeros(1)
for k in range(K):
    s = q(s + q(a[0, k] * b[k, 0]))

print(f"float32 torch.matmul : {(a @ b).item():.6f}")
print(f"binaryK_matmul       : {out.item():.6f}")
print(f"by hand              : {s.item():.6f}   equal: {torch.equal(out.flatten(), s)}")

# Leaving the accumulator unquantized (acc=None / accumulate_quant=False)
# keeps the running sum in float32; only the products are rounded.
out32 = binaryK_matmul(a, b, mul_K=8, mul_P=4, accumulate_quant=False)
s32 = torch.zeros(1)
for k in range(K):
    s32 = s32 + q(a[0, k] * b[k, 0])
same = torch.equal(out32.flatten(), s32)
print(f"\nE4M3 products, fp32 sum : {out32.item():.6f}   by hand equal: {same}")
