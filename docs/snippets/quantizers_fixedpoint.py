"""The fixed-point quantizer: the function, its in-place twin, and Quant."""

import warnings

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

# The magnitude bits must fit the carrier's precision: 24 bits in binary32, 53 in
# binary64. A float64 tensor is rounded in binary64, which holds the wider format.
try:
    fixedpoint_quantize(x, wl=26, fl=20)
except ValueError as e:
    print("\nfloat32: ValueError:", e)
wide = fixedpoint_quantize(x.double(), wl=26, fl=20)
print("float64:", wide[:2].tolist())

# A float16 or bfloat16 result is rounded once more when it is stored. Only an
# end of the range can miss the dtype's grid, and the call warns when one can:
# FixedPoint(16, 14)'s largest value, 2 - 2**-14, needs 15 bits and float16 has
# 11, so 3.0, which saturates to it, is stored as 2.0.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    h = fixedpoint_quantize(torch.tensor([1.0, 3.0], dtype=torch.float16), wl=16, fl=14)
print("\nfloat16 result: ", h.tolist())
for w in caught:
    print("  warning:", str(w.message).split(". ")[0])
# FixedPoint(8, 6)'s largest value, 1.984375, needs 7 bits: bfloat16 holds it.
b = fixedpoint_quantize(torch.tensor([1.0, 3.0], dtype=torch.bfloat16), wl=8, fl=6)
print("bfloat16 result:", b.tolist(), "(no warning)")

# CPU and CUDA return the same words in every deterministic mode.
if torch.cuda.is_available():
    x = torch.randn(1 << 16) * 4
    modes = [m for m in RoundMode if m is not RoundMode.SR]
    same = all(
        torch.equal(
            fixedpoint_quantize(x, wl=8, fl=4, rounding_mode=m),
            fixedpoint_quantize(x.cuda(), wl=8, fl=4, rounding_mode=m).cpu(),
        )
        for m in modes
    )
    print(f"\nCPU == CUDA on {x.numel()} values in {len(modes)} modes:", same)
