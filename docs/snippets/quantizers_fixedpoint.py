"""The fixed-point quantizer: the function, its in-place twin, and Quant."""

import torch

from mptorch import FixedPoint, RoundMode
from mptorch.quant import Quant, fixedpoint_quantize, fixedpoint_quantize_

torch.manual_seed(0)
torch.set_printoptions(precision=5)

x = torch.randn(6)
print("x                ", x)
print("wl=8, fl=4       ", fixedpoint_quantize(x, wl=8, fl=4))
# Two more fractional bits: a finer step, and a range of [-2, 2) instead of [-8, 8).
print("wl=8, fl=6       ", fixedpoint_quantize(x, wl=8, fl=6))
print(
    "SR, 8 random bits", fixedpoint_quantize(x, wl=8, fl=4, prng_bits=8, rounding_mode=RoundMode.SR)
)

# Quant calls the function, with the arguments taken from the format.
q = Quant(FixedPoint(8, 4))
print("\nQuant == function:", torch.equal(q(x), fixedpoint_quantize(x, wl=8, fl=4)))

# The in-place twin writes over its argument and returns it.
y = x.clone()
out = fixedpoint_quantize_(y, wl=8, fl=4)
print("in place:         ", out is y, torch.equal(y, fixedpoint_quantize(x, wl=8, fl=4)))

# The magnitude bits must fit the carrier's mantissa: 23 bits in binary32, 52 in
# binary64. A float64 tensor is rounded in binary64, which holds the wider format.
try:
    fixedpoint_quantize(x, wl=26, fl=20)
except ValueError as e:
    print("\nfloat32: ValueError:", e)
wide = fixedpoint_quantize(x.double(), wl=26, fl=20)
print("float64:", wide[:2].tolist())
