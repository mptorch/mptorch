"""What the carrier rounds: the nearest carrier step and the odd one."""

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import SplitMac, qmm

# Eight bits of precision (the neighbours of 1 are 1 - 2^-8 and 1 + 2^-7), and
# a range that holds the products 2^-30 and 2^-61 exactly.
binary15p8 = BinaryK(15, 8)


def witness(sign: int, gap: int, dtype: torch.dtype):
    """Two products, 1 and sign * 2^-gap, each a Binary15p8 value: only the
    sum, 1 + sign * 2^-gap, is inexact."""
    h = gap // 2
    a = torch.tensor([[1.0, sign * 2.0**-h]], dtype=dtype)
    b = torch.tensor([[1.0], [2.0 ** -(gap - h)]], dtype=dtype)
    return a, b


def run(a, b, mode, carrier_rounding=RoundMode.RNE):
    mac = SplitMac(binary15p8, binary15p8, rounding=mode, carrier_rounding=carrier_rounding)
    return qmm(a, b, mac).item()


# (mode, the sign of the gap, the correct rounding of the exact sum)
cases = [
    (RoundMode.RNE, +1, 1.0),
    (RoundMode.RU, +1, 1 + 2**-7),
    (RoundMode.RD, -1, 1 - 2**-8),
    (RoundMode.RO, +1, 1 + 2**-7),
]
print(f"{'exact':10} {'mode':4} {'correct':>9} | {'carrier':8} {'nearest':>9} {'odd':>9}")
for mode, sign, correct in cases:
    for gap, dtype in ((30, torch.float32), (30, torch.float64), (61, torch.float64)):
        a, b = witness(sign, gap, dtype)
        carrier = "binary32" if dtype is torch.float32 else "binary64"
        nearest, odd = run(a, b, mode), run(a, b, mode, RoundMode.RO)
        exact = f"1 {'+' if sign > 0 else '-'} 2^-{gap}"
        print(f"{exact:10} {mode.name:4} {correct:9.7f} | {carrier:8} {nearest:9.7f} {odd:9.7f}")

# The guarantee needs a format narrower than the carrier: under RNE at most
# 22 bits of precision in binary32. A wider format raises, per call.
a, b = witness(+1, 30, torch.float32)
try:
    qmm(a, b, SplitMac(BinaryK(30, 23), carrier_rounding=RoundMode.RO))
except ValueError as e:
    print(f"\nValueError: {e}")
