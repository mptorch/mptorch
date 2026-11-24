"""A Llama-style model: its RMSNorm registered, attention with grouped keys, and the
arithmetic of its blocks, in one recipe."""

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import (
    ActivationFormats,
    AttentionFormats,
    NormFormats,
    PointwiseFormats,
    Quant,
    SplitMac,
    presets,
    profile,
    qrmsnorm,
    register,
    simulate,
)


class RMSNorm(nn.Module):
    """Llama's norm, written out of elementwise ops as transformers writes it."""

    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads):
        super().__init__()
        self.heads, self.kv_heads, self.d = heads, kv_heads, dim // heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.d, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.d, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.d).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.kv_heads, self.d).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.kv_heads, self.d).transpose(1, 2)
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.o_proj(h.transpose(1, 2).reshape(b, n, -1))


class Layer(nn.Module):
    def __init__(self, dim=32, heads=4, kv_heads=2, hidden=64):
        super().__init__()
        self.input_norm, self.attn = RMSNorm(dim), Attention(dim, heads, kv_heads)
        self.post_norm = RMSNorm(dim)
        self.gate, self.up = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        x = x + self.attn(self.input_norm(x))
        h = self.post_norm(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class Llama(nn.Module):
    def __init__(self, vocab=100, dim=32, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(Layer(dim) for _ in range(layers))
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


# the norm as a whole, in mptorch's arithmetic, rather than its steps one by one
register(
    RMSNorm, kind="norm", forward=lambda m, f, x: qrmsnorm(x, m.weight.shape, m.weight, m.eps, f)
)

p8, p16 = BinaryK(8, 4), BinaryK(16, 11)
bf = BinaryK(14, 8)  # bfloat16's precision, in a range binary32 holds
mac, half = SplitMac(p8, p16), Quant(bf)
recipe = (
    presets.uniform(p8, acc=p16)
    .override("layers.*.attn", attention=AttentionFormats.uniform(mac))
    .override(
        "layers.*",  # inside the blocks only: the model's own mask and embedding stay exact
        norm=NormFormats.uniform(SplitMac(bf, bf), half),
        activation=ActivationFormats(out=half),
        elementwise=PointwiseFormats(quant=half),  # the residual sums and the SwiGLU product
    )
    .override("lm_head", linear=None)
)

torch.manual_seed(0)
model, ids = Llama(), torch.randint(0, 100, (2, 12))
with simulate(recipe, model) as sim:
    logits = model(ids)
kinds = sorted({(r.kind, r.action) for r in sim.report().rows})
print("kinds seen:", ", ".join(f"{k} ({a.split(':')[0]})" for k, a in kinds))

p = profile(model, ids, recipe=recipe, backward=True, run=False)
print(p.summary())
elements = {}
for e in p.elements:
    elements[e.kind] = elements.get(e.kind, 0) + e.elements
print("elements by kind:", elements)
