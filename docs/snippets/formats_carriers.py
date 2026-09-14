"""Carriers: the float arithmetic a tensor is rounded in, and what each can hold."""

import math
import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning
from mptorch.quant import Quant, SplitMac, qmatmul

F32, F64 = torch.float32, torch.float64

# The carrier is the tensor's unless a call names one: binary64 for float64,
# binary32 for the rest. 1.0625 is the tie between 1.0 and 1.125 on E4M3's
# grid, and 1.1875 the tie between 1.125 and 1.25; 2**-30 either side of them is
# a bit float32 does not have, so the float32 value *is* the tie, and
# round-to-nearest-even takes over -- in either carrier, since binary64 cannot
# give a float32 tensor back bits it never had.
x = torch.tensor([1.0625 + 2**-30, 1.1875 - 2**-30], dtype=F64)
e4m3 = BinaryK(8, 4)
print("input                              ", x.tolist())
print("float64, rounded in binary64       ", Quant(e4m3)(x).tolist())
print("float32, rounded in binary32       ", Quant(e4m3)(x.float()).tolist())
print("float32, carrier=torch.float64     ", Quant(e4m3, carrier=F64)(x.float()).tolist())

# What each carrier makes of a format is checked when a tensor meets it, not
# when the format is built: here by a float32 tensor in each carrier, and a
# float64 one.
formats = [
    BinaryK(8, 4),  # E4M3: well inside both
    BinaryK(16, 8),  # smallest value 2**-134, below binary32's floor
    BinaryK(40, 30),  # 30 bits of precision, ten exponent bits
    BinaryK(64, 53),  # eleven exponent bits: past binary64 at the bottom too
]
calls = {
    "float32": lambda fmt: Quant(fmt)(torch.ones(4)),
    "float32, carrier=torch.float64": lambda fmt: Quant(fmt, carrier=F64)(torch.ones(4)),
    "float64": lambda fmt: Quant(fmt)(torch.ones(4, dtype=F64)),
}
print()
for fmt in formats:
    print(f"BinaryK({fmt.K}, {fmt.P})")
    for name, call in calls.items():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FormatRangeWarning)
            try:
                call(fmt)
                verdict = f"warns ({caught[0].message}" if caught else "ok"
            except ValueError as e:
                verdict = f"raises ({e}"
        print(f"  {name}: {verdict.split(': ')[0] + (')' if verdict != 'ok' else '')}")

# Only what no carrier can do raises when the format is built.
try:
    BinaryK(64, 54)
except ValueError as e:
    print("\nBinaryK(64, 54):", str(e).split(": ")[0])

# A format float32 cannot carry, exactly, on a float64 tensor: pi to 30 bits
# of precision is pi rounded to the nearest multiple of 2**-28.
pi = torch.tensor([math.pi], dtype=F64)
got = Quant(BinaryK(40, 30))(pi).item()
want = round(math.pi * 2**28) / 2**28
print(
    f"\npi in BinaryK(40, 30): {got!r}, round(pi * 2**28) / 2**28: {want!r}, equal: {got == want}"
)

# Where the carrier does move a float32 result: a product. (1 + 2**-13)**2 is
# 1 + 2**-12 + 2**-26, a hair above the tie between 1 and 1 + 2**-11 on a
# 12-bit grid. binary32 rounds the product to 24 bits first, onto the tie, and
# RNE takes it down; binary64 holds the hair and rounds up. The result is a
# float32 tensor either way.
one = torch.full((1, 1), 1 + 2**-13)
p12 = BinaryK(16, 12)
print()
for carrier in (None, F64):
    out = qmatmul(one, one, SplitMac(p12, p12, carrier=carrier))
    print(f"(1 + 2**-13)**2 in P=12, carrier={carrier}: {out.item()!r} ({out.dtype})")

# A carrier is never narrower than the tensor: rounding a float64 one in
# binary32 would round every input once before the format does.
try:
    Quant(e4m3, carrier=F32)(x)
except ValueError as e:
    print("\ncarrier=torch.float32 on float64:", str(e).split(";")[0])
