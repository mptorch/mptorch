"""SuperFP: precision at the top of the range, dynamic range below it."""

import torch

from mptorch import SuperFP
from mptorch.quant import Quant

# 3 mantissa bits, 4 exponent bits, bias 7 -- the same widths as E4M3 -- but
# only the top 2 binades keep their mantissa. The codes that would have
# encoded the other 14 binades' mantissas become 14 * 2**3 = 112 further
# powers of two below them.
f = SuperFP(man_bits=3, exp_bits=4, normal_binades=2, bias=7)
q = Quant(f)

max_exp = 2**f.exp_bits - 1 - f.bias  # 8: the top binade is [2**8, 2**9)
normal_lo = max_exp - f.normal_binades + 1  # 7
super_lo = normal_lo - (2**f.exp_bits - f.normal_binades) * 2**f.man_bits + 1  # -104
print(f"{f}")
lo, hi = 2.0**normal_lo, 2.0 ** (max_exp + 1)
print(f"  normal region      [2^{normal_lo}, 2^{max_exp + 1})  = [{lo:g}, {hi:g})")
print(f"  supernormal region [2^{super_lo}, 2^{normal_lo})  = powers of two only")
print(f"  below 2^{super_lo - 1} = {2.0 ** (super_lo - 1):g}: flushed to zero")

# In the normal region the mantissa is kept ...
x = torch.tensor([128.0, 144.0, 160.0, 176.0, 300.0, 448.0, 511.0])
print("\nnormal region     ", x.tolist())
print("quantized         ", q(x).tolist())
# ... below it only the power of two survives (3 is rounded to 4: round to
# nearest is taken on the exponent, and 3 = 2**1.58 is nearer to 2**2) ...
x = torch.tensor([1.0, 1.5, 3.0, 100.0, 2.0**-20, 3.0 * 2**-50])
print("\nsupernormal region", x.tolist())
print("quantized         ", q(x).tolist())
# ... and far below, the flush to zero.
x = torch.tensor([2.0**-104, 2.0**-105, 2.0**-106])
print("\nunderflow         ", x.tolist())
print("quantized         ", q(x).tolist())
