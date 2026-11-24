"""A sum whose every step rounds to a format, and what the accumulate algorithms recover."""

import torch

from mptorch import AccumulateAlgorithm, BinaryK
from mptorch.quant import FusedMac, Quant, qmean, qsum

fmt = BinaryK(8, 4)  # next to 1 its values are 1/8 apart
x = torch.cat([torch.ones(1), torch.full((255,), 1 / 16)])  # 1 + 255/16 = 16.9375

print("torch.sum:               ", torch.sum(x).item())
# 1 + 1/16 is halfway between 1 and 1.125, and to nearest even it is 1: the sum stalls
print("each step to the format: ", qsum(x, mac=fmt).item())
kahan = FusedMac(fmt, accumulate_algorithm=AccumulateAlgorithm.KAHAN)
print("KAHAN:                   ", qsum(x, mac=kahan).item())
blocks = FusedMac(fmt, accumulate_algorithm=AccumulateAlgorithm.BLOCK, block_size=16, outer=fmt)
print("blocks of 16:            ", qsum(x, mac=blocks).item())
print("their mean, one division:", qmean(x, mac=blocks, quant=Quant(fmt)).item())
