"""Converting a model from a recipe, and counting its products by format."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import (
    GemmFormats,
    Quant,
    Recipe,
    SplitMac,
    convert,
    convert_,
    presets,
    profile,
    unconvert_,
)

torch.manual_seed(0)
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(32, 10),
)

p8 = BinaryK(8, 4)
gemm = GemmFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
recipe = Recipe(gemm=gemm).override("7", linear=None)  # the classifier stays float32
print(recipe.plan(model))

qmodel, report = convert(model, recipe)  # a converted copy; `model` is unchanged
x = torch.randn(8, 3, 32, 32)
print(f"\nlargest difference to the float32 model: {(qmodel(x) - model(x)).abs().max():.4f}\n")

# MACs per arithmetic, priced at 1 for a Binary8p4 product and 4 for a float32 one
print(profile(qmodel, x, cost={p8: 1.0, None: 4.0}).table())

# In place: the layers keep their Parameters, so an optimizer built before still trains them.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
convert_(model, presets.fp8())
model(x).sum().backward()
optimizer.step()
converted_class = type(model[0]).__name__
print(f"\n{converted_class}, then {unconvert_(model)} layers back:", type(model[0]).__name__)
