"""Softmax and the norms of a model, converted and simulated, with a norm of its own."""

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import (
    NormFormats,
    Quant,
    Recipe,
    SoftmaxFormats,
    SplitMac,
    convert,
    qrmsnorm,
    register,
    simulate,
)


class RMSNorm(nn.Module):
    """A norm written out of elementwise ops, as Hugging Face's Llama writes it."""

    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RMSNorm(16)
        self.proj = nn.Linear(16, 16)
        self.ln = nn.LayerNorm(16)

    def forward(self, x):
        h = self.proj(self.norm(x))
        weights = F.softmax(h @ h.mT / 4, dim=-1)
        return self.ln(weights @ h)


# the norm's forward in mptorch's arithmetic, which convert and simulate both run
register(
    RMSNorm, kind="norm", forward=lambda m, f, x: qrmsnorm(x, m.weight.shape, m.weight, m.eps, f)
)

fmt = BinaryK(14, 8)  # bfloat16's precision, a range binary32 holds
mac, q = SplitMac(fmt, fmt), Quant(fmt)
recipe = Recipe(norm=NormFormats.uniform(mac, q), softmax=SoftmaxFormats.uniform(mac, q))
torch.manual_seed(0)
model, x = Block(), torch.randn(2, 5, 16)

converted, report = convert(model, recipe)
print(report)
print()
with simulate(recipe, model) as sim:  # reaches the functional softmax too
    model(x)
print(sim.report())
