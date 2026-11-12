"""FixedPoint: the multiples of one step that a word of bits holds."""

import torch

from mptorch import FixedPoint, RoundMode
from mptorch.quant import Quant

# 8 bits, 4 of them fractional: every value is a multiple of 1/16. Signed
# formats are two's complement by default, so the bottom is one step further
# from zero than the top.
f = FixedPoint(8, 4)
print(f)
print(f"  step {f.step}, range [{f.min_value}, {f.max_value}]")
sym = FixedPoint(8, 4, symmetric=True)
uns = FixedPoint(8, 4, is_signed=False)
print(f"  symmetric: [{sym.min_value}, {sym.max_value}]")
print(f"  unsigned:  [{uns.min_value}, {uns.max_value}]")

q = Quant(f)
# Inside the range: the nearest multiple of 1/16. Under RNE a tie goes to the
# even integer code: 0.09375 is 1.5 steps and becomes 2 steps.
x = torch.tensor([1.1, 3.3, 0.09375, -0.40625])
print("\ninside the range ", x.tolist())
print("quantized        ", q(x).tolist())
# Below the step: zero or the step. Half a step is the tie, and zero is the
# even code. A negative value that rounds to zero gives +0.0.
x = torch.tensor([0.03, 0.04, -0.03])
print("\nbelow the step   ", x.tolist())
print("quantized        ", q(x).tolist())
# Past the ends, infinities included: saturated, since there is no code for an
# infinity. A NaN passes through.
x = torch.tensor([9.0, -9.0, float("inf"), float("-inf"), float("nan")])
print("\npast the ends    ", x.tolist())
print("quantized        ", q(x).tolist())

# Every rounding mode applies.
x = torch.tensor([1.1, -1.1])
print("\nrounding modes of", x.tolist())
for mode in (RoundMode.RNE, RoundMode.RU, RoundMode.RD, RoundMode.RZ, RoundMode.RO):
    print(f"  {mode.name:<3}", Quant(f, mode)(x).tolist())

# An unsigned format rounds a negative input to zero, and fl may be negative:
# FixedPoint(8, -2) has a step of 4.
print("\nunsigned          ", Quant(uns)(torch.tensor([-1.0, 2.5])).tolist())
coarse = FixedPoint(8, -2)
print(f"{coarse}: step {coarse.step}, range [{coarse.min_value}, {coarse.max_value}]")
print("  10.0 and 1000.0 ->", Quant(coarse)(torch.tensor([10.0, 1000.0])).tolist())
