"""Quant: a format plus a rounding mode, as a callable."""

import torch

from mptorch import BinaryK, RoundMode, SuperFP
from mptorch.quant import Quant, binaryK_quantize

torch.manual_seed(0)
x = torch.randn(4, 4)

q = Quant(BinaryK(8, 4))  # rounding defaults to RoundMode.RNE
print(q)
print(q(x))

# A Quant is exactly the flat function call, spelled once.
same = torch.equal(q(x), binaryK_quantize(x, K=8, P=4, bias=8))
print("Quant(BinaryK(8, 4))(x) == binaryK_quantize(x, K=8, P=4):", same)

# Any format the kernels implement works, with any rounding mode.
q_sr = Quant(SuperFP(3, 4, 2, 7, prng_bits=8), RoundMode.SR)
print(q_sr(x))

# Formats and Quants are frozen values: equal by value, and usable as keys.
assert BinaryK(8, 4) == BinaryK(8, 4, bias=8)
cache = {BinaryK(8, 4): "E4M3", BinaryK(8, 3): "E5M2"}
print(cache[BinaryK(8, 4)], cache[BinaryK(8, 3)])
