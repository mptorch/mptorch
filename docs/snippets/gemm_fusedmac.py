"""SplitMac vs FusedMac: two roundings per step, or one."""

import torch

from mptorch import BinaryK
from mptorch.quant import FusedMac, Quant, SplitMac, qmatmul

torch.manual_seed(0)
K = 8
a = torch.randn(1, K)
b = torch.randn(K, 1)
binary8p4 = BinaryK(8, 4)
q = Quant(binary8p4)

split = qmatmul(a, b, SplitMac(binary8p4, binary8p4))  # round(round(a*b) + s), twice per step
fused = qmatmul(a, b, FusedMac(binary8p4))  # round(a*b + s), once per step

# FusedMac by hand: the product-and-add is exact (float64 holds it), then
# rounded once to Binary8p4.
s = torch.zeros(1, dtype=torch.float64)
for k in range(K):
    s = q((a[0, k].double() * b[k, 0].double() + s).float()).double()

print(f"float32 torch.matmul           : {(a @ b).item():.6f}")
print(f"SplitMac(Binary8p4, Binary8p4) : {split.item():.6f}")
same = torch.equal(fused.flatten(), s.float())
print(f"FusedMac(Binary8p4)            : {fused.item():.6f}   by hand equal: {same}")

# FusedMac(None) is the unrounded fused step: float32 FMAs, in order.
unrounded = qmatmul(a, b, FusedMac(None))
print(f"FusedMac(None)                 : {unrounded.item():.6f}")
