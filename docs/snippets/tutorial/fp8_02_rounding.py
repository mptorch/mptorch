"""What rounding a Gaussian tensor to FP8 costs, mode by mode."""

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import Quant

torch.manual_seed(0)
x = torch.randn(1_000_000)
e4m3 = BinaryK(8, 4, bias=7, prng_bits=8)
e5m2 = BinaryK(8, 3, bias=15, prng_bits=8)

print(f"{'format, rounding':<20}{'mean error':>12}{'RMS error':>12}{'rel. RMS':>10}")
for fmt, label in ((e4m3, "E4M3"), (e5m2, "E5M2")):
    for mode in (RoundMode.RNE, RoundMode.RZ, RoundMode.RU, RoundMode.SR):
        err = Quant(fmt, mode)(x) - x
        print(
            f"{label + ', ' + mode.name:<20}{err.mean():>12.2e}{err.pow(2).mean().sqrt():>12.4f}"
            f"{(err.norm() / x.norm()):>10.4f}"
        )

# Where each format stops working: relative error across nine decades.
print(f"\n{'|x|':>10}{'E4M3 rel. err':>16}{'E5M2 rel. err':>16}")
for e in range(-7, 6):
    v = torch.full((10000,), 10.0**e) * (1 + 0.5 * torch.rand(10000))
    for_row = []
    for fmt in (e4m3, e5m2):
        q = Quant(fmt)(v)
        rel = ((q - v).abs() / v).mean().item()
        for_row.append("overflow" if torch.isinf(q).any() else f"{rel:.4f}")
    print(f"{10.0**e:>10.0e}{for_row[0]:>16}{for_row[1]:>16}")
