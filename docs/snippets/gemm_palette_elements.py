"""A format for every output element: a dense prec_idx map over an eight-format palette."""

import torch

from mptorch import BinaryK, RoundMode, SubnormalsMode
from mptorch.quant import FusedMac, Palette, SplitMac, qmatmul

torch.manual_seed(0)
M, K, N = 4, 64, 5
a = torch.randn(M, K)
b = torch.randn(K, N)
exact = a @ b

# Eight BinaryK formats, from 3 to 24 bits of precision -- a palette holds at
# most eight, and its entries may differ in anything but the fields the
# kernel shares across the palette (sign, stochastic bits, saturation and
# subnormals).
formats = [
    BinaryK(8, 3),
    BinaryK(8, 4),
    BinaryK(10, 6),
    BinaryK(12, 8),
    BinaryK(16, 11),
    BinaryK(20, 15),
    BinaryK(24, 19),
    BinaryK(31, 24),
]
palette = Palette(formats)

# Choose each output element's format on its own: the narrowest whose
# single-format GEMM is within 0.1% of the exact dot product there, or the
# widest where none is. Any rule works -- a map is just an integer per element.
tol = 1e-3
errors = torch.stack([(qmatmul(a, b, SplitMac(f, f)) - exact).abs() / exact.abs() for f in formats])
fits = errors <= tol
prec_idx = torch.where(fits.any(0), fits.int().argmax(0), len(formats) - 1).to(torch.int32)
print("P of the format each output element runs in:")
print(torch.tensor([[formats[i].P for i in row] for row in prec_idx.tolist()]))

# One GEMM, every element in its own format.
out = qmatmul(a, b, SplitMac(palette, palette), prec_idx=prec_idx)
rel = (out - exact).abs() / exact.abs()
print(f"\nworst relative error {rel.max().item():.2e} (target {tol:.0e})")

# ... and each element is exactly what a single-format GEMM in its format gives.
same = torch.zeros(M, N, dtype=torch.bool)
for i, f in enumerate(formats):
    here = prec_idx == i
    same[here] = qmatmul(a, b, SplitMac(f, f))[here] == out[here]
print(f"all {M * N} elements equal their own format's GEMM: {bool(same.all())}")

# The multiply and the accumulate of each element can differ too: palette entry
# i pairs multiply format i with accumulate format i, so a map entry picks a
# pair -- here a 4-bit multiply into a 24-bit sum, or the other way round.
mul = Palette([BinaryK(8, 4), BinaryK(31, 24)])
acc = Palette([BinaryK(31, 24), BinaryK(8, 4)])
pairs = torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 0, 1, 1, 0], [1, 1, 0, 0, 1]])
split = qmatmul(a, b, SplitMac(mul, acc), prec_idx=pairs)
ok = all(
    torch.equal(split[pairs == i], qmatmul(a, b, SplitMac(mul[i], acc[i]))[pairs == i])
    for i in range(2)
)
print(f"\nper-element (multiply, accumulate) pairs match their own GEMM: {ok}")

# A fused step takes a palette the same way, and so do the shared fields: here
# every entry rounds stochastically with 8 random bits and no subnormals.
fused_formats = [
    BinaryK(8, 4, prng_bits=8, subnormals=SubnormalsMode.NORMALS),
    BinaryK(16, 11, prng_bits=8, subnormals=SubnormalsMode.NORMALS),
]
fused = qmatmul(a, b, FusedMac(fused_formats, rounding=RoundMode.SR), prec_idx=pairs)
print("stochastic fused palette:", tuple(fused.shape))

# A batched product takes one map per batch element, [B, M, N], so every
# element of every sample can run in its own format.
ab, bb = torch.randn(3, M, K), torch.randn(3, K, N)
per_sample = torch.randint(0, len(formats), (3, M, N))
batched = qmatmul(ab, bb, SplitMac(palette, palette), prec_idx=per_sample)
first = qmatmul(ab[0], bb[0], SplitMac(palette, palette), prec_idx=per_sample[0])
print("batched map, sample 0 equals its own call:", torch.equal(batched[0], first))

# The fields a palette shares must agree, and it says which entry does not.
try:
    Palette([BinaryK(8, 4), BinaryK(16, 11, subnormals=SubnormalsMode.NORMALS)])
except ValueError as e:
    print("\nValueError:", e)
