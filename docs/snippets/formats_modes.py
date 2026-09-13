"""What happens at the two ends of the range: subnormals and saturation."""

import torch

from mptorch import BinaryK, RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import Quant

torch.set_printoptions(precision=6, sci_mode=False)

# Small values, E4M3 with bias 7: the smallest normal is 2**-6. Under
# EXTENDED_NORMALS the binade below it holds normals as well, but its
# mantissa-zero code is still the zero -- so 2**-7 is not one of them, and
# that binade starts at 1.125 * 2**-7.
small = torch.tensor([2.0**-5, 2.0**-6, 2.0**-7, 2.0**-8, 2.0**-9, 2.0**-10])
print("input           ", small.tolist())
for mode in SubnormalsMode:
    q = Quant(BinaryK(8, 4, bias=7, subnormals=mode))
    print(f"{mode.name:<16}", q(small).tolist())

# Large values, E4M3 with bias 7: the largest finite value is 480.
large = torch.tensor([448.0, 480.0, 500.0, 1e6, float("inf"), -float("inf"), float("nan")])
print("\ninput           ", large.tolist())
for mode in SaturationMode:
    q = Quant(BinaryK(8, 4, bias=7, saturation=mode))
    print(f"{mode.name:<16}", q(large).tolist())

# As in P3109, rounding comes before saturation, so OVF_INF overflows to infinity
# even when rounding toward zero, where IEEE 754 would stop at 448.
q = Quant(BinaryK(8, 4, bias=7), RoundMode.RZ)
print(f"{'OVF_INF, RZ':<16}", q(large).tolist())

# An unsigned format has no sign bit, so negative inputs collapse to zero and
# the freed bit goes to the exponent (bias 16 instead of 8).
u = BinaryK(8, 4, is_signed=False)
print(f"\n{u}")
print("unsigned        ", Quant(u)(torch.tensor([-3.0, -0.001, 0.0, 3.0, 3000.0])).tolist())
