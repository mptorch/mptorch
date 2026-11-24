"""Activations, embeddings and a model's own arithmetic, in a recipe."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import (
    ActivationFormats,
    GemmFormats,
    PointwiseFormats,
    Quant,
    Recipe,
    SplitMac,
    convert,
    simulate,
)


class Block(nn.Module):
    """A residual MLP: its sum and its scaling are the module's own code."""

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + 0.5 * self.fc2(self.act(self.fc1(x)))


torch.manual_seed(0)
model = nn.Sequential(nn.Embedding(100, 16), Block(16), nn.Linear(16, 4))
ids = torch.randint(0, 100, (2, 8))

p8, p16 = BinaryK(8, 4), BinaryK(16, 11)
recipe = Recipe(
    linear=GemmFormats.uniform(SplitMac(p8, p16), Quant(p8)),
    activation=ActivationFormats(out=Quant(p8), grad=Quant(p16)),  # GELU's result and gradient
    embedding=PointwiseFormats(quant=Quant(p8)),  # the rows gathered
)
converted, report = convert(model, recipe)  # the Embedding, the GELU and the Linears
print(report)

# the residual sum and the scaling are no module: only simulate reaches them
recipe = recipe.override("1", elementwise=PointwiseFormats.uniform(Quant(p16)))
with simulate(recipe, model) as sim:
    model(ids)
print()
print(sim.report())

with sim:
    hidden = model[1](model[0](ids))  # the block's output, its arithmetic rounded
moved = (hidden - converted[1](converted[0](ids))).abs().max()
print(f"\nrounding the block's sum and scaling moves its output by up to {moved:.2e}")
