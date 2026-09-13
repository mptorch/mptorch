"""Palette: a different format per output element, selected by prec_idx."""

import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning
from mptorch.quant import Palette, QMatmul, SplitMac, matmul_formats, qmatmul

torch.manual_seed(0)
M, K, N = 6, 32, 4
a = torch.randn(M, K)
b = torch.randn(K, N)

# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so building one warns (concepts, "What float32 can
# carry"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

e4m3, e5m2, bf16 = BinaryK(8, 4), BinaryK(8, 3), BinaryK(16, 8)
palette = Palette([e4m3, e5m2, bf16])
mac = SplitMac(palette, palette)  # entry i: multiply and accumulate in format i

# One map entry per output row: rows 0-1 in E4M3, 2-3 in E5M2, 4-5 in bf16.
prec_idx = torch.tensor([[0], [0], [1], [1], [2], [2]])
out = qmatmul(a, b, mac, prec_idx=prec_idx)

# Each row equals the same row computed with that single format.
for fmt, rows in ((e4m3, [0, 1]), (e5m2, [2, 3]), (bf16, [4, 5])):
    single = qmatmul(a, b, SplitMac(fmt, fmt))
    print(
        f"rows {rows} == {fmt.K}/{fmt.P} single-format rows:", torch.equal(out[rows], single[rows])
    )

# A map can also be per column ([1, N]) or dense ([M, N]).
per_col = torch.tensor([[0, 1, 2, 0]])
dense = torch.randint(0, 3, (M, N))
print("per-column map:", tuple(qmatmul(a, b, mac, prec_idx=per_col).shape))
print("dense map:     ", tuple(qmatmul(a, b, mac, prec_idx=dense).shape))

# A palette that also has to differentiate needs one map per pass, because
# the three passes have three output shapes: [M, N], [M, K] and [K, N].
formats = matmul_formats(
    mac,
    prec_idx=prec_idx,  # forward, [M, N]-shaped (or [M, 1] / [1, N])
    agrad_prec_idx=torch.zeros(M, 1, dtype=torch.long),  # grad of a: [M, K]
    bgrad_prec_idx=torch.full((1, N), 2),  # grad of b: [K, N]
)
layer = QMatmul(formats)
a.requires_grad_(True)
layer(a, b).sum().backward()
print("grad through a palette:", tuple(a.grad.shape))

# Forgetting one is an error naming the argument, not a silent fallback.
try:
    qmatmul(a, b, mac, prec_idx=prec_idx).sum().backward()
except ValueError as e:
    print("ValueError:", e)
