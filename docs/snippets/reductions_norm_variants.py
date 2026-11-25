"""Norm variants: L1 and L-infinity deviations, and FISR and a table for 1/sqrt."""

import torch
import torch.nn.functional as F

from mptorch.quant import FISR, NormFormats, PWLTable, qlayernorm

torch.manual_seed(0)
x = torch.randn(64, 512)  # 64 rows of 512 Gaussian features
exact = F.layer_norm(x, (512,))

print(f"{'formats':<32}{'std of x_hat':>13}{'largest |x_hat - layer_norm|':>30}")
for label, f in (
    ("deviation='l1'", NormFormats(deviation="l1")),
    ("deviation='linf'", NormFormats(deviation="linf")),
    ("inverse=FISR()", NormFormats(inverse=FISR())),
    ("inverse=FISR(2)", NormFormats(inverse=FISR(2))),
    ("inverse=PWLTable.rsqrt()", NormFormats(inverse=PWLTable.rsqrt())),
):
    y = qlayernorm(x, 512, formats=f)
    print(f"{label:<32}{y.std().item():>13.4f}{(y - exact).abs().max().item():>30.2e}")

# an approximate inverse takes the exact gradient, straight through
x.requires_grad_(True)
g = torch.randn_like(x)
(approx,) = torch.autograd.grad(qlayernorm(x, 512, formats=NormFormats(inverse=FISR())), x, g)
(true,) = torch.autograd.grad(F.layer_norm(x, (512,)), x, g)
print("\nFISR's gradient is layer_norm's:", torch.equal(approx, true))
