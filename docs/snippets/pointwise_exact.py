"""What the carrier rounds in a pointwise op, and how to make it exact."""

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import Quant, qadd, qdiv

up = Quant(BinaryK(8, 4), RoundMode.RU)  # 1.0 and 1.125 are neighbours there
one = torch.tensor([1.0])

# 1 + 2**-30 rounded up is 1.125, but binary32 holds only 1.0 of the sum
near = torch.tensor([2.0**-30])
print("1 + 2**-30, binary32:", qadd(one, near, up).item())
print("1 + 2**-30, binary64:", qadd(one, near, up, carrier=torch.float64).item())

# 2**-60 is past binary64's precision too; rounded to odd, the step keeps a trace of it
far = torch.tensor([2.0**-60])
print("1 + 2**-60, binary64:", qadd(one, far, up, carrier=torch.float64).item())
print("1 + 2**-60, to odd:  ", qadd(one, far, up, carrier_rounding=RoundMode.RO).item())

# a quotient of float32 values in binary64 is exact for this format's rounding
x = torch.randn(10_000, generator=torch.Generator().manual_seed(0))
y = torch.randn(10_000, generator=torch.Generator().manual_seed(1))
fmt = Quant(BinaryK(16, 11), RoundMode.RU)
default = qdiv(x, y, fmt)
exact = qdiv(x, y, fmt, carrier=torch.float64)
print("quotients binary32 rounds up to the wrong value:", int((default != exact).sum()))
