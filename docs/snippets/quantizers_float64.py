"""An elementwise quantizer's carrier: binary64 for float64, and for any tensor naming it."""

import math

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import Quant, binaryK_quantize

F32, F64 = torch.float32, torch.float64
torch.manual_seed(0)

# A float64 tensor is rounded in binary64, so a format float32 cannot carry is
# reachable: 30 bits of precision, each value rounded to the nearest multiple
# of its binade's step, 2**(e - 29).
x = torch.tensor([math.pi, -math.e, 1 / 3, 1e-9], dtype=F64)
q = binaryK_quantize(x, K=40, P=30)
for v, r in zip(x.tolist(), q.tolist(), strict=True):
    step = 2.0 ** (math.floor(math.log2(abs(v))) - 29)
    exact = r == round(v / step) * step
    print(f"{v!r:>21} -> {r!r:<21} the nearest multiple of 2^{round(math.log2(step))}: {exact}")
try:
    binaryK_quantize(x.float(), K=40, P=30)
except ValueError as e:
    print("float32:", str(e).split(": ")[0])

# The float64 input is rounded once, directly. Its float32 neighbour has lost
# the 2**-30 and sits on the tie itself, whichever carrier rounds it:
# carrier=torch.float64 widens the float32 tensor, rounds it in binary64 and
# stores the result back as float32.
t = torch.tensor([1.0625 + 2**-30, 1.1875 - 2**-30], dtype=F64)
print("\nfloat64                        ", binaryK_quantize(t, K=8, P=4).tolist())
print("float32                        ", binaryK_quantize(t.float(), K=8, P=4).tolist())
t64 = binaryK_quantize(t.float(), K=8, P=4, carrier=F64)
print("float32, carrier=torch.float64 ", t64.tolist(), t64.dtype)

# That is the rule for a tensor's own values: a format binary32 carries rounds
# them to the same answer in either carrier. So for float32, float16 or
# bfloat16 values, carrier=torch.float64 changes what the format may be, and
# the random draws of stochastic rounding, not a deterministic answer.
y = torch.randn(10_000)
for dtype in (F32, torch.float16, torch.bfloat16):
    yd = y.to(dtype)
    wide = binaryK_quantize(yd, K=8, P=4, carrier=F64)
    same = torch.equal(wide, binaryK_quantize(yd, K=8, P=4))
    print(f"{str(dtype):<14} carrier=torch.float64 -> {wide.dtype}, same as binary32's: {same}")

# Quant takes the same keyword, for a layer's quantizer slots.
q64 = Quant(BinaryK(8, 4), carrier=F64)
print(
    "Quant(..., carrier=torch.float64) agrees:",
    torch.equal(q64(y), binaryK_quantize(y, 8, 4, carrier=F64)),
)

# Stochastic rounding draws its random bits in the carrier, below the format's
# mantissa: 52 bits to share in binary64, 23 in binary32. The input sits
# 0.3 + 2**-37 of the way from 1.0 to 1.125, which 40 bits resolve. A float32
# tensor is refused 40 bits in its own carrier, and given them by naming binary64.
v = torch.full((1_000_000,), 1 + 0.3 * 0.125 + 2**-40, dtype=F64)
sr = binaryK_quantize(v, K=8, P=4, prng_bits=40, rounding_mode=RoundMode.SR)
values, up = sorted(sr.unique().tolist()), (sr > 1).double().mean().item()
print(f"\nSR with 40 random bits: values {values}, P(up) = {up:.4f}")
v32 = torch.full((1_000_000,), 1 + 0.3 * 0.125)
try:
    binaryK_quantize(v32, K=8, P=4, prng_bits=40, rounding_mode=RoundMode.SR)
except ValueError as e:
    print("float32:", str(e).split(";")[0])
sr32 = binaryK_quantize(v32, K=8, P=4, prng_bits=40, rounding_mode=RoundMode.SR, carrier=F64)
print(
    f"float32, carrier=torch.float64: P(up) = {(sr32 > 1).double().mean().item():.4f}, {sr32.dtype}"
)

# The carrier is never narrower than the tensor.
try:
    binaryK_quantize(x, K=8, P=4, carrier=F32)
except ValueError as e:
    print("\ncarrier=torch.float32 on float64:", str(e).split(";")[0])
