"""BinaryK: IEEE P3109's K-bit floating-point formats with P bits of precision."""

import torch

from mptorch import BinaryK, SaturationMode
from mptorch.quant import Quant

torch.manual_seed(0)

binary8p4 = BinaryK(8, 4)  # P3109 Binary8p4se: 1 sign, 4 exponent, 3 stored mantissa bits
binary8p3 = BinaryK(8, 3)  # P3109 Binary8p3se: 1 sign, 5 exponent, 2 stored mantissa bits


def describe(f: BinaryK) -> None:
    exp_bits = f.K - f.P if f.is_signed else f.K - f.P + 1
    top_binade = 2.0 ** (2**exp_bits - 1 - f.bias)
    max_extended = (2 - 2 ** (2 - f.P)) * top_binade  # the last code is infinity
    max_finite = (2 - 2 ** (1 - f.P)) * top_binade  # the last code is a number
    min_normal = 2.0 ** (1 - f.bias)
    min_subnormal = 2.0 ** (1 - f.bias - f.man_bits)
    print(f"{f}")
    print(f"  exponent bits {exp_bits}, mantissa bits {f.man_bits}, bias {f.bias}")
    print(f"  largest finite  {max_extended:g} (extended domain), {max_finite:g} (finite domain)")
    print(f"  smallest normal {min_normal:g}")
    print(f"  smallest subnormal {min_subnormal:g}")
    # The quantizer agrees with the formulas: each value is a fixed point in the
    # domain that has it, and 1e30 overflows -- to infinity in the extended
    # domain, to the largest finite value in the finite one.
    probe = torch.tensor([max_extended, max_finite, min_normal, min_subnormal, 1e30])
    for sat in (SaturationMode.OVF_INF, SaturationMode.SAT_FINITE):
        q = Quant(BinaryK(f.K, f.P, bias=f.bias, saturation=sat))
        print(f"  {sat.name:<11}{q(probe).tolist()}")


describe(binary8p4)
describe(binary8p3)

# Between two consecutive powers of two the representable values are evenly
# spaced: 2**man_bits of them per binade. Here is the binade [1, 2) of Binary8p4.
x = torch.arange(1.0, 2.0, 1 / 16)
print("\ninput           ", [f"{v:.4f}" for v in x.tolist()])
print("Binary8p4 (RNE) ", [f"{v:.4f}" for v in Quant(binary8p4)(x).tolist()])
