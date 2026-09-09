"""E4M3 and E5M2 as BinaryK formats, checked against torch's native FP8 dtypes."""

import torch

from mptorch import BinaryK
from mptorch.quant import Quant

# The OCP / NVIDIA FP8 formats. torch's own dtypes are the reference.
e4m3 = BinaryK(8, 4, bias=7)  # torch.float8_e4m3fn
e5m2 = BinaryK(8, 3, bias=15)  # torch.float8_e5m2

print(f"{'':<22}{'E4M3':>14}{'torch e4m3fn':>14}{'E5M2':>14}{'torch e5m2':>14}")
rows = {
    "largest finite": lambda f, fi: (
        (2 - 2 ** (2 - f.P)) * 2.0 ** (2 ** (f.K - f.P) - 1 - f.bias),
        fi.max,
    ),
    "smallest normal": lambda f, fi: (2.0 ** (1 - f.bias), fi.tiny),
    "smallest subnormal": lambda f, fi: (
        2.0 ** (1 - f.bias - f.man_bits),
        fi.smallest_normal * 2.0**-f.man_bits,
    ),
    "spacing at 1": lambda f, fi: (2.0**-f.man_bits, fi.eps),
}
for name, fn in rows.items():
    a, ta = fn(e4m3, torch.finfo(torch.float8_e4m3fn))
    b, tb = fn(e5m2, torch.finfo(torch.float8_e5m2))
    print(f"{name:<22}{a:>14g}{ta:>14g}{b:>14g}{tb:>14g}")

# Exhaustive check: every one of the 2**32 float32 bit patterns, rounded to
# nearest-even by both. Chunked so it fits a laptop GPU (or runs on a CPU).
# The two overflow conventions differ, so the count is split at the largest
# finite value the torch dtype has: below it the two must agree exactly.
device = "cuda" if torch.cuda.is_available() else "cpu"
chunk = 2**24
inside = {"E4M3": 0, "E5M2": 0}
beyond = {"E4M3": 0, "E5M2": 0}
first = {}
for start in range(0, 2**32, chunk):
    bits = torch.arange(start, start + chunk, dtype=torch.int64, device=device).to(torch.int32)
    x = bits.view(torch.float32)
    for name, fmt, dt in (("E4M3", e4m3, torch.float8_e4m3fn), ("E5M2", e5m2, torch.float8_e5m2)):
        ours = Quant(fmt)(x)
        theirs = x.to(dt).to(torch.float32)
        # NaN != NaN, so compare NaN-ness separately from values.
        differ = torch.isfinite(x) & (ours != theirs) & ~(torch.isnan(ours) & torch.isnan(theirs))
        in_range = x.abs() <= torch.finfo(dt).max
        inside[name] += int((differ & in_range).sum())
        beyond[name] += int((differ & ~in_range).sum())
        if (differ & ~in_range).any() and name not in first:
            i = int((differ & ~in_range).nonzero()[0])
            first[name] = (x[i].item(), ours[i].item(), theirs[i].item())

print(f"\nfinite float32 inputs on which BinaryK and torch disagree ({device}):")
for name, dt in (("E4M3", torch.float8_e4m3fn), ("E5M2", torch.float8_e5m2)):
    x0, ours, theirs = first[name]
    print(
        f"  {name}: |x| <= {torch.finfo(dt).max:g}: {inside[name]:,}   beyond it: {beyond[name]:,}"
        f"  (first at |x| = {abs(x0):g}: BinaryK {ours:g}, torch {theirs:g})"
    )
