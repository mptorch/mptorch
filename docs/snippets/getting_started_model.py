"""A model you already have, in 8-bit arithmetic: convert it, compare it, count it."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import convert, presets, profile

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 10))
qmodel, report = convert(model, presets.uniform(BinaryK(8, 4)))  # a converted copy
x = torch.randn(4, 32)
print(f"largest difference to the float32 model: {(qmodel(x) - model(x)).abs().max():.4f}")
print(profile(qmodel, x).summary())
