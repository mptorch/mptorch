"""What happens at the two ends of the range: subnormals and saturation."""

import torch

from mptorch import BinaryK, Domain, RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import Quant

torch.set_printoptions(precision=6, sci_mode=False)

# Small values, in E4M3's layout (bias 7): the smallest normal is 2**-6. Under
# EXTENDED_NORMALS the binade below it holds normals as well, but its
# mantissa-zero code is still the zero -- so 2**-7 is not one of them, and
# that binade starts at 1.125 * 2**-7.
small = torch.tensor([2.0**-5, 2.0**-6, 2.0**-7, 2.0**-8, 2.0**-9, 2.0**-10])
print("input               ", small.tolist())
for mode in SubnormalsMode:
    q = Quant(BinaryK(8, 4, bias=7, subnormals=mode))
    print(f"{mode.name:<20}", q(small).tolist())

# Large values, same layout: the top code, 480, is a number in the finite
# domain (SAT_FINITE) and infinity in the extended one (the other three),
# whose largest finite value is 448.
large = torch.tensor([448.0, 480.0, 500.0, 1e6, float("inf"), -float("inf"), float("nan")])
print("\ninput               ", large.tolist())
for mode in SaturationMode:
    q = Quant(BinaryK(8, 4, bias=7, saturation=mode))
    print(f"{mode.name:<20}", q(large).tolist())

# As in P3109, rounding comes before saturation, so SAT_NONE overflows to infinity
# even when rounding toward zero, where IEEE 754 would stop at 448.
q = Quant(BinaryK(8, 4, bias=7), RoundMode.RZ)
print(f"{'SAT_NONE, RZ':<20}", q(large).tolist())

# P3109's own spelling of the pair: SatFinite in the extended domain is
# SAT_FINITE_EXTENDED, and the two spellings make the same format.
f = BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE, domain=Domain.EXTENDED)
print(
    f"\n{f.saturation.name}, {f.domain.name}:",
    f == BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE_EXTENDED),
)

# An unsigned format has no sign bit, so negative inputs collapse to zero and
# the freed bit goes to the exponent (bias 16 instead of 8).
u = BinaryK(8, 4, is_signed=False)
print(f"\n{u}")
print("unsigned            ", Quant(u)(torch.tensor([-3.0, -0.001, 0.0, 3.0, 3000.0])).tolist())
