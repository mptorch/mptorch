"""ScaledQuant: scale a tensor into a format's range, round, and scale back."""

import torch

from mptorch import BinaryK, SaturationMode
from mptorch.quant import Quant, ScaledQuant

torch.manual_seed(0)
g = torch.randn(4096) * 1e-4  # a gradient-sized tensor, mostly below E5M2's smallest value
e5m2 = BinaryK(8, 3, bias=15, saturation=SaturationMode.SAT_FINITE)
e4m3 = BinaryK(8, 4, bias=7, saturation=SaturationMode.SAT_FINITE)


def report(name: str, y: torch.Tensor) -> None:
    zeros = (y == 0).float().mean().item()
    error = ((y - g).norm() / g.norm()).item()
    print(f"{name:<38} {zeros:6.1%} zero, relative error {error:.4f}")


report("Quant(e5m2)", Quant(e5m2)(g))
report("ScaledQuant(e5m2, target=57344.0)", ScaledQuant(e5m2, target=57344.0)(g))
report("  pow2=True", ScaledQuant(e5m2, target=57344.0, pow2=True)(g))
report("  margin=3", ScaledQuant(e5m2, target=57344.0, margin=3)(g))

# One scale per row: rows of very different sizes each use the whole range.
w = torch.randn(4, 256) * torch.tensor([[1e-5], [1e-3], [1.0], [1e2]])
for granularity in ("tensor", "row"):
    q = ScaledQuant(e4m3, target=448.0, granularity=granularity)(w)
    per_row = ((q - w).norm(dim=1) / w.norm(dim=1)).tolist()
    print(f"{granularity:>6}: relative error per row", "  ".join(f"{e:.3f}" for e in per_row))

# A float16 tensor is scaled in binary32: a scale of about 5.7e7 is past
# float16's range, and the result comes back as float16, finite.
h = (torch.randn(8) * 1e-3).half()
out = ScaledQuant(e5m2, target=57344.0)(h)
print(out.dtype, "finite:", out.isfinite().all().item())

# A tensor holding an infinity or a NaN is scaled by 1, and this saturating
# format rounds the infinity to its largest finite value.
print(ScaledQuant(e5m2, target=57344.0)(torch.tensor([1e-3, float("inf"), 2e-3])))
