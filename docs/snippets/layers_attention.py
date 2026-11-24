"""An attention block's arithmetic, given once, for its SDPA spelling and its eager one."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import AttentionFormats, GemmFormats, Quant, Recipe, SplitMac, convert, simulate


class Attention(nn.Module):
    """Self-attention, written as SDPA or as transformers' eager attention."""

    def __init__(self, sdpa: bool):
        super().__init__()
        self.sdpa = sdpa
        self.qkv = nn.Linear(16, 48)

    def forward(self, x):
        q, k, v = self.qkv(x).view(2, 5, 3, 4, 4).permute(2, 0, 3, 1, 4)
        if self.sdpa:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        mask = torch.full((5, 5), torch.finfo(x.dtype).min).triu(1)
        weights = torch.matmul(q, k.mT) * (1 / math.sqrt(4)) + mask
        return torch.matmul(F.softmax(weights, dim=-1), v)


fmt = BinaryK(8, 4)
mac = SplitMac(fmt, BinaryK(14, 8))
recipe = Recipe(linear=GemmFormats.uniform(mac, Quant(fmt))).attention_parts(
    "*", AttentionFormats.uniform(mac, Quant(fmt))
)

x = torch.randn(2, 5, 16)
torch.manual_seed(0)
sdpa = Attention(sdpa=True)
torch.manual_seed(0)
eager = Attention(sdpa=False)
with simulate(recipe, sdpa) as a:
    y_sdpa = sdpa(x)
with simulate(recipe, eager) as b:
    y_eager = eager(x)
print("SDPA and eager, the same words:", torch.equal(y_sdpa, y_eager))
print(b.report())

# nn.MultiheadAttention converts: its projections under the linear kind, its core under attention
model = nn.Sequential(nn.MultiheadAttention(16, 4, batch_first=True))
converted, report = convert(model, recipe, strict=True)
print(type(converted[0]).__name__, [r.action for r in report.rows])
