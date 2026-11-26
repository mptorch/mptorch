"""Exact sums of FP8 products in the binary64 carrier."""

import dataclasses

import torch

from mptorch import E4M3, E5M2, SaturationMode
from mptorch.quant import FusedMac, Quant, qmatmul

F64 = torch.float64
gen = torch.Generator().manual_seed(0)
in_binary64 = FusedMac(None, carrier=F64)  # products and sums in binary64, none rounded


def values(fmt, *shape, binades):
    """Values of ``fmt`` spread over 2^-binades to 2^binades, saturating at its largest."""
    fmt = dataclasses.replace(fmt, saturation=SaturationMode.SAT_FINITE)
    x = torch.randn(*shape, generator=gen, dtype=F64)
    spread = torch.empty(*shape, dtype=F64).uniform_(-binades, binades, generator=gen)
    return Quant(fmt)(x * torch.exp2(spread))


def inexact(out, a, b, step):
    """How many of ``out``'s sums differ from the exact ones. ``a / step`` and
    ``b / step`` are integers, so Python's integers give the exact sums."""
    rows, cols = (a / step).long().tolist(), (b / step).long().t().tolist()
    sums = (out.double() / step**2).tolist()
    wrong = 0
    for r, row in zip(rows, sums, strict=True):
        for c, s in zip(cols, row, strict=True):
            wrong += s != sum(x * y for x, y in zip(r, c, strict=True))
    return f"{wrong:3d} of {out.numel()} sums inexact"


K = 4096
for name, fmt, step, binades in (
    ("E4M3", E4M3, 2.0**-9, 12),  # beyond both ends of E4M3's range
    ("E5M2", E5M2, 2.0**-16, 4),
    ("E5M2", E5M2, 2.0**-16, 20),
):
    a, b = values(fmt, 16, K, binades=binades), values(fmt, K, 16, binades=binades)
    exact64 = inexact(qmatmul(a, b, in_binary64), a, b, step)
    in_binary32 = inexact(qmatmul(a.float(), b.float(), FusedMac(None)), a, b, step)
    print(f"{name} over ±{binades:2d} binades, K = {K}: binary64 {exact64}, binary32 {in_binary32}")

# A float32 GEMM narrows its binary64 result to binary32 before an output
# rounding rounds it again. Exact sums stood in for by random float64 values:
x = torch.randn(1 << 24, generator=gen, dtype=F64)
x = x * torch.exp2(torch.empty(1 << 24, dtype=F64).uniform_(-5, 5, generator=gen))
once = Quant(E4M3)(x)
twice = Quant(E4M3)(x.float()).double()
print(f"rounded to E4M3 through binary32: {int((once != twice).sum())} of {x.numel():,} differ")
