"""How the accumulation format decides the error of a long dot product."""

import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning, RoundMode
from mptorch.quant import FusedMac, SplitMac, qmatmul

torch.manual_seed(0)
M, K, N = 64, 4096, 64
# Entries of size ~1/4, so the dot products (std ~4) stay inside E4M3's
# range of 240 and the question is precision rather than overflow.
a = torch.randn(M, K) / 4
b = torch.randn(K, N) / 4
exact = (a.double() @ b.double()).float()

e4m3 = BinaryK(8, 4)
e4m3_sr = BinaryK(8, 4, prng_bits=8)
e5m2 = BinaryK(8, 3)
# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so building one warns (concepts, "What float32 can
# carry"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

bf16 = BinaryK(16, 8)  # 8 exponent, 7 mantissa bits
# float32's own precision, over a narrower exponent range: no 8-exponent-bit
# binaryK fits inside binary32 at any bias, so this is as wide as it goes.
wide = BinaryK(31, 24)  # 7 exponent, 23 mantissa bits

configs = {
    "torch.matmul (float32)": None,
    "SplitMac(E4M3, acc=None)": SplitMac(e4m3, None),
    "SplitMac(E4M3, 24-bit acc)": SplitMac(e4m3, wide),
    "SplitMac(E4M3, bf16)": SplitMac(e4m3, bf16),
    "SplitMac(E4M3, E5M2)": SplitMac(e4m3, e5m2),
    "SplitMac(E4M3, E4M3)": SplitMac(e4m3, e4m3),
    "SplitMac(E4M3, E4M3), SR": SplitMac(e4m3_sr, e4m3_sr, rounding=RoundMode.SR),
    "FusedMac(E4M3)": FusedMac(e4m3),
    "FusedMac(bf16)": FusedMac(bf16),
}
print(f"{'arithmetic':<28}{'rel. error':>12}{'max |err|':>12}")
for name, mac in configs.items():
    out = qmatmul(a, b, mac)
    err = out - exact
    print(f"{name:<28}{err.norm() / exact.norm():>12.2e}{err.abs().max():>12.2f}")
