"""Counting an FP8 recipe's numeric events: gradients that underflow, and a loss scale."""

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import E4M3, E5M2, BinaryK
from mptorch.quant import FusedMac, GemmFormats, Recipe, simulate

# E4M3 operands and E5M2 gradients, unscaled, every sum rounded to binary16
mac = FusedMac(BinaryK(16, 11))
fp8 = GemmFormats(
    fwd_mac=mac,
    igrad_mac=mac,
    wgrad_mac=mac,
    input_quant=E4M3,
    weight_quant=E4M3,
    igrad_quant=E5M2,
    wgrad_quant=E5M2,
)
recipe = Recipe(linear=fp8)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 10)
)
x, y = torch.randn(512, 64), torch.randint(0, 10, (512,))


def step(loss_scale):
    """One training step's events, under a loss scale."""
    model.zero_grad()
    with simulate(recipe, model, events=True) as sim:
        (F.cross_entropy(model(x), y) * loss_scale).backward()
    return sim.events()


for loss_scale in (1.0, 2.0**10):
    report = step(loss_scale)
    grads = [r for r in report.rows if r.role in ("igrad_quant", "wgrad_quant")]
    casts = sum(r.counts.casts for r in grads)
    lost = sum(r.counts.underflow for r in grads)
    print(f"loss scale {loss_scale:6g}: {lost:6d} of {casts} gradient casts underflow")

# the scaled step, site by site
print(report)
