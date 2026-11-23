"""Simulating a model's products while it runs, without converting it."""

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import GemmFormats, Quant, Recipe, SplitMac, convert, simulate


class Attention(nn.Module):
    """Self-attention as such code is often written: the projection's weight
    used through F.linear, and the two products through torch.matmul."""

    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = F.linear(x, self.qkv.weight, self.qkv.bias).chunk(3, dim=-1)
        scores = torch.matmul(q, k.mT) / q.shape[-1] ** 0.5  # site "matmul#0"
        return self.out(torch.matmul(scores.softmax(-1), v))  # site "matmul#1"


torch.manual_seed(0)
model = nn.Sequential(Attention(16), nn.Linear(16, 4))
x = torch.randn(2, 8, 16)
reference = model(x)

p8 = BinaryK(8, 4)
gemm = GemmFormats.uniform(SplitMac(p8, BinaryK(16, 11)), Quant(p8))
# every linear, and the attention's second product (scores @ v)
recipe = Recipe(linear=gemm).override(Attention, site="matmul#1", matmul=gemm)

with simulate(recipe, model) as sim:
    simulated = model(x)
print(sim.report())

# convert reaches the layers it can switch, and qkv's layer is never called
converted, _ = convert(model, recipe)
diff_converted = (converted(x) - reference).abs().max()
diff_simulated = (simulated - reference).abs().max()
print(f"\nto float32: converted {diff_converted:.4f}, simulated {diff_simulated:.4f}")

# the module to compile: it enters the simulation inside the compiled code
fast = torch.compile(sim.wrap(), backend="aot_eager", fullgraph=True)
print("compiled equals eager:", torch.equal(fast(x), simulated))
