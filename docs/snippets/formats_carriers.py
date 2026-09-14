"""Carriers: the float arithmetic a tensor is rounded in, and what each can hold."""

import math
import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning
from mptorch.quant import Quant

# The carrier is the tensor's: binary64 for float64, binary32 for the rest.
# 1.0625 is the tie between 1.0 and 1.125 on E4M3's grid, and 1.1875 the tie
# between 1.125 and 1.25; 2**-30 either side of them is a bit float32 does not
# have, so narrowing lands on the tie and round-to-nearest-even takes over.
x = torch.tensor([1.0625 + 2**-30, 1.1875 - 2**-30], dtype=torch.float64)
e4m3 = BinaryK(8, 4)
print("input                        ", x.tolist())
print("float64, rounded in binary64 ", Quant(e4m3)(x).tolist())
print("float64, carrier='binary32'  ", Quant(e4m3, carrier="binary32")(x).tolist())
print("float32, rounded in binary32 ", Quant(e4m3)(x.float()).tolist())

# What each carrier makes of a format is checked when a tensor meets it, not
# when the format is built.
formats = [
    BinaryK(8, 4),  # E4M3: well inside both
    BinaryK(16, 8),  # smallest value 2**-134, below binary32's floor
    BinaryK(40, 30),  # 30 bits of precision, ten exponent bits
    BinaryK(64, 53),  # eleven exponent bits: past binary64 at the bottom too
]
print()
for fmt in formats:
    verdicts = []
    for dtype in (torch.float32, torch.float64):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FormatRangeWarning)
            try:
                Quant(fmt)(torch.ones(4, dtype=dtype))
                verdict = f"warns ({caught[0].message})" if caught else "ok"
            except ValueError as e:
                verdict = f"raises ({e})"
        verdicts.append(verdict.split(": ")[0] + (")" if verdict != "ok" else ""))
    print(f"BinaryK({fmt.K}, {fmt.P})")
    print(f"  float32: {verdicts[0]}")
    print(f"  float64: {verdicts[1]}")

# Only what no carrier can do raises when the format is built.
try:
    BinaryK(64, 54)
except ValueError as e:
    print("\nBinaryK(64, 54):", str(e).split(": ")[0])

# A format float32 cannot carry, exactly, on a float64 tensor: pi to 30 bits
# of precision is pi rounded to the nearest multiple of 2**-28.
pi = torch.tensor([math.pi], dtype=torch.float64)
got = Quant(BinaryK(40, 30))(pi).item()
want = round(math.pi * 2**28) / 2**28
print(
    f"\npi in BinaryK(40, 30): {got!r}, round(pi * 2**28) / 2**28: {want!r}, equal: {got == want}"
)

# carrier="binary64" insists on a float64 tensor.
try:
    Quant(e4m3, carrier="binary64")(torch.ones(4))
except ValueError as e:
    print("\ncarrier='binary64' on float32:", e)
