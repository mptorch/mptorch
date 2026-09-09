"""BinaryK: a K-bit floating-point format with P bits of precision."""

import torch

from mptorch import BinaryK, SaturationMode
from mptorch.quant import Quant

torch.manual_seed(0)

e4m3 = BinaryK(8, 4)  # 8 bits: 1 sign, 4 exponent, 3 stored mantissa bits
e5m2 = BinaryK(8, 3)  # 8 bits: 1 sign, 5 exponent, 2 stored mantissa bits


def describe(f: BinaryK) -> None:
    exp_bits = f.K - f.P if f.is_signed else f.K - f.P + 1
    max_finite = (2 - 2 ** (1 - f.P)) * 2.0 ** (2**exp_bits - 1 - f.bias)
    min_normal = 2.0 ** (1 - f.bias)
    min_subnormal = 2.0 ** (1 - f.bias - f.man_bits)
    print(f"{f}")
    print(f"  exponent bits {exp_bits}, mantissa bits {f.man_bits}, bias {f.bias}")
    print(f"  largest finite  {max_finite:g}")
    print(f"  smallest normal {min_normal:g}")
    print(f"  smallest subnormal {min_subnormal:g}")
    # The quantizer agrees with the formulas: each of these is a fixed point.
    q = Quant(BinaryK(f.K, f.P, bias=f.bias, saturation=SaturationMode.SAT_FINITE))
    probe = torch.tensor([max_finite, min_normal, min_subnormal, 1e30])
    print(f"  quantized       {q(probe).tolist()}")


describe(e4m3)
describe(e5m2)

# Between two consecutive powers of two the representable values are evenly
# spaced: 2**man_bits of them per binade. Here is the binade [1, 2) of E4M3.
x = torch.arange(1.0, 2.0, 1 / 16)
print("\ninput      ", [f"{v:.4f}" for v in x.tolist()])
print("E4M3 (RNE) ", [f"{v:.4f}" for v in Quant(e4m3)(x).tolist()])
