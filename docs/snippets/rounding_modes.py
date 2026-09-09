"""The seven rounding modes on the same inputs (E4M3: 3 mantissa bits)."""

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import Quant

torch.manual_seed(0)
f = BinaryK(8, 4)

# In [1, 2) the representable values are 1, 1.125, 1.25, ... (spacing 1/8),
# so 1.0625 and 1.1875 are exact ties and 1.1 / 1.3 are not.
x = torch.tensor([1.0625, 1.1875, 1.1, 1.3, -1.0625, -1.1, 2.71828, -2.71828])
print(f"{'input':<6}", [f"{v:8.4f}" for v in x.tolist()])
for mode in RoundMode:
    y = Quant(f, mode)(x)
    print(f"{mode.name:<6}", [f"{v:8.4f}" for v in y.tolist()])

# RoundMode.SR draws its random bits from the *format's* prng_bits. The format
# above has prng_bits=0, so its SR row is plain truncation (the RZ row): give
# the format some random bits and the same input rounds differently per draw.
f_sr = BinaryK(8, 4, prng_bits=8)
for draw in range(3):
    y = Quant(f_sr, RoundMode.SR)(x)
    print(f"SR #{draw} ", [f"{v:8.4f}" for v in y.tolist()])
