"""float64 tensors: rounded in binary64, or in binary32 when asked."""

import math

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import Quant, binaryK_quantize

torch.manual_seed(0)

# A float64 tensor is rounded in binary64, so a format float32 cannot carry is
# reachable: 30 bits of precision, each value rounded to the nearest multiple
# of its binade's step, 2**(e - 29).
x = torch.tensor([math.pi, -math.e, 1 / 3, 1e-9], dtype=torch.float64)
q = binaryK_quantize(x, K=40, P=30)
for v, r in zip(x.tolist(), q.tolist(), strict=True):
    step = 2.0 ** (math.floor(math.log2(abs(v))) - 29)
    exact = r == round(v / step) * step
    print(f"{v!r:>21} -> {r!r:<21} the nearest multiple of 2^{round(math.log2(step))}: {exact}")
try:
    binaryK_quantize(x.float(), K=40, P=30)
except ValueError as e:
    print("float32:", str(e).split(": ")[0])

# The same function on float32, float64 and float64-in-binary32. The float64
# input is rounded once, directly; carrier="binary32" narrows it to float32
# first -- the arithmetic every dtype had before float64 had a carrier of its
# own -- and 2**-30 does not survive that.
t = torch.tensor([1.0625 + 2**-30, 1.1875 - 2**-30], dtype=torch.float64)
print("\nfloat64                 ", binaryK_quantize(t, K=8, P=4).tolist())
print("float64, carrier=binary32", binaryK_quantize(t, K=8, P=4, carrier="binary32").tolist())
y = torch.randn(10_000, dtype=torch.float64)
same = torch.equal(
    binaryK_quantize(y, K=8, P=4, carrier="binary32"),
    binaryK_quantize(y.float(), K=8, P=4).double(),
)
print("carrier='binary32' == narrow, round in float32, widen:", same)

# Quant takes the same keyword, for a layer's quantizer slots.
q32 = Quant(BinaryK(8, 4), carrier="binary32")
print(
    "Quant(..., carrier='binary32') agrees:",
    torch.equal(q32(y), binaryK_quantize(y.float(), 8, 4).double()),
)

# Stochastic rounding draws its random bits in the carrier too, below the
# format's mantissa: 52 bits to share in binary64, 23 in binary32. The input
# sits 0.3 + 2**-37 of the way from 1.0 to 1.125, which 40 bits resolve.
v = torch.full((1_000_000,), 1 + 0.3 * 0.125 + 2**-40, dtype=torch.float64)
sr = binaryK_quantize(v, K=8, P=4, prng_bits=40, rounding_mode=RoundMode.SR)
values, up = sorted(sr.unique().tolist()), (sr > 1).double().mean().item()
print(f"\nSR with 40 random bits: values {values}, P(up) = {up:.4f}")
try:
    binaryK_quantize(v.float(), K=8, P=4, prng_bits=40, rounding_mode=RoundMode.SR)
except ValueError as e:
    print("float32:", str(e).split(";")[0])
