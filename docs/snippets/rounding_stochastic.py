"""Stochastic rounding: unbiased in expectation, and why that matters."""

import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning, RoundMode
from mptorch.quant import Quant

torch.manual_seed(0)

# 1. The probability of rounding up follows the residual. 0.3 sits between
#    the E4M3 neighbours 0.28125 and 0.3125, 60% of the way up.
x = torch.full((1_000_000,), 0.3)
lo, hi = 0.28125, 0.3125
print("prng_bits  P(round up)  mean of 1e6 draws")
for prng_bits in (1, 2, 3, 4, 8, 16):
    q = Quant(BinaryK(8, 4, prng_bits=prng_bits), RoundMode.SR)(x)
    p_up = (q == hi).float().mean().item()
    print(f"{prng_bits:>9}  {p_up:>11.4f}  {q.mean().item():.6f}")
print(f"{'exact':>9}  {(0.3 - lo) / (hi - lo):>11.4f}  0.300000")

# 2. Accumulating small increments. In a bfloat16-like format (8 exponent,
#    7 mantissa bits) the spacing just above 1 is 2**-7 = 0.0078, so adding
#    0.001 to 1.0 and rounding to nearest gives back 1.0 -- forever. Rounding
#    stochastically moves up with probability 0.001 / 0.0078 each step.
# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so building one warns (concepts, "What float32 can
# carry"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

bf16 = BinaryK(16, 8)
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
