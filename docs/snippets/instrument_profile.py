"""Counting a model's products, sums and elements by format, forward and backward."""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from mptorch import BinaryK
from mptorch.quant import AttentionFormats, GemmFormats, Recipe, SplitMac, presets, profile


class Block(nn.Module):
    """A pre-norm attention block, as a transformer writes one."""

    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.heads = heads

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return x + self.out(h.transpose(1, 2).reshape(b, n, d))


p8, p16 = BinaryK(8, 4), BinaryK(16, 11)
mac = SplitMac(p8, p16)
recipe = presets.uniform(p8, acc=p16).override("*", attention=AttentionFormats.uniform(mac))
torch.manual_seed(0)
p = profile(Block(), torch.randn(2, 6, 16), recipe=recipe, cost={p8: 1.0, None: 4.0})
print(p.table())

# A model built on the meta device holds no weights, and run=False computes nothing:
# vit_b_16's training step on a batch of 32, counted from its shapes alone, each pass
# under its own mac (Binary8p3 for the gradients).
with torch.device("meta"):
    vit = torchvision.models.vit_b_16()
grads = SplitMac(BinaryK(8, 3), p16)
recipe = Recipe(gemm=GemmFormats.uniform(mac, grad_mac=grads))
recipe = recipe.override("*", attention=AttentionFormats.uniform(mac))
x = torch.empty(32, 3, 224, 224, device="meta")
p = profile(vit, x, recipe=recipe, backward=True, run=False)
for step in ("fwd", "igrad", "wgrad"):
    macs = sum(r.macs for r in p.records if r.step == step)
    print(f"{step:>5}: {macs / 1e12:.2f} TMACs")
print(p.summary())
