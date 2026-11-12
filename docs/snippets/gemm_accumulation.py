"""How the accumulation format decides the error of a long dot product."""

import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning, RoundMode
from mptorch.quant import FusedMac, SplitMac, qmatmul

torch.manual_seed(0)
M, K, N = 64, 4096, 64
# Entries of size ~1/4, so the dot products (std ~4) stay inside Binary8p4's
# range of 224 and the question is precision rather than overflow.
a = torch.randn(M, K) / 4
b = torch.randn(K, N) / 4
exact = (a.double() @ b.double()).float()

binary8p4 = BinaryK(8, 4)
binary8p4_sr = BinaryK(8, 4, prng_bits=8)
binary8p3 = BinaryK(8, 3)
# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so a float32 call with one warns (concepts, "What
# the carrier can hold"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

binary16p8 = BinaryK(16, 8)  # bfloat16's field widths: 8 exponent, 7 mantissa bits
# float32's own precision, over a narrower exponent range: no 8-exponent-bit
# binaryK fits inside binary32 at any bias, so this is as wide as it goes.
wide = BinaryK(31, 24)  # 7 exponent, 23 mantissa bits

configs = {
    "torch.matmul (float32)": None,
    "SplitMac(Binary8p4, acc=None)": SplitMac(binary8p4, None),
    "SplitMac(Binary8p4, 24-bit acc)": SplitMac(binary8p4, wide),
    "SplitMac(Binary8p4, Binary16p8)": SplitMac(binary8p4, binary16p8),
    "SplitMac(Binary8p4, Binary8p3)": SplitMac(binary8p4, binary8p3),
    "SplitMac(Binary8p4, Binary8p4)": SplitMac(binary8p4, binary8p4),
    "SplitMac(Binary8p4, Binary8p4), SR": SplitMac(
        binary8p4_sr, binary8p4_sr, rounding=RoundMode.SR
    ),
    "FusedMac(Binary8p4)": FusedMac(binary8p4),
    "FusedMac(Binary16p8)": FusedMac(binary16p8),
}
print(f"{'arithmetic':<36}{'rel. error':>12}{'max |err|':>12}")
for name, mac in configs.items():
    out = qmatmul(a, b, mac)
    err = out - exact
    print(f"{name:<36}{err.norm() / exact.norm():>12.2e}{err.abs().max():>12.2f}")
