"""Stochastic rounding: unbiased in expectation, and why that matters."""

import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning, RoundMode
from mptorch.quant import Quant

torch.manual_seed(0)

# 1. The probability of rounding up follows the residual. 0.3 sits between
#    the Binary8p4 neighbours 0.28125 and 0.3125, 60% of the way up. Each of
#    P3109's three stochastic modes reads prng_bits random bits its own way.
x = torch.full((1_000_000,), 0.3)
lo, hi = 0.28125, 0.3125
modes = (RoundMode.SR, RoundMode.SR_B, RoundMode.SR_C)
print("P(round up) over 1e6 draws")
print("prng_bits" + "".join(f"{m.name:>9}" for m in modes))
for prng_bits in (1, 2, 3, 4, 8, 16):
    row = f"{prng_bits:>9}"
    for mode in modes:
        q = Quant(BinaryK(8, 4, prng_bits=prng_bits), mode)(x)
        row += f"{(q == hi).float().mean().item():>9.4f}"
    print(row)
print(f"{'exact':>9}" + f"{(0.3 - lo) / (hi - lo):>9.4f}" * 3)

# 2. Accumulating small increments. In Binary16p8 (bfloat16's field widths: 8
#    exponent, 7 mantissa bits) the spacing just above 1 is 2**-7 = 0.0078, so adding
#    0.001 to 1.0 and rounding to nearest gives back 1.0 -- forever. Rounding
#    stochastically moves up with probability 0.001 / 0.0078 each step.
# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so a float32 call with one warns (concepts, "What
# the carrier can hold"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

binary16p8 = BinaryK(16, 8)
n_steps, batch = 2000, 1000
for mode, bits in ((RoundMode.RNE, 0), (RoundMode.SR, 8)):
    q = Quant(BinaryK(16, 8, prng_bits=bits), mode)
    acc = torch.ones(batch)  # 1000 independent accumulators
    for _ in range(n_steps):
        acc = q(acc + 0.001)
    print(f"\n1.0 + 0.001 * {n_steps} = {1 + 0.001 * n_steps:.3f}")
    print(
        f"  {mode.name}: mean over {batch} accumulators = {acc.mean().item():.4f}"
        f"  (min {acc.min().item():.4f}, max {acc.max().item():.4f})"
    )
