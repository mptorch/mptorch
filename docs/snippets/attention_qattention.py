"""Attention with each part rounded, its grouped-query form, and a fully masked query."""

import torch
import torch.nn.functional as F

from mptorch import BinaryK
from mptorch.quant import AttentionFormats, Quant, SplitMac, qattention

torch.manual_seed(0)
q = torch.randn(1, 4, 6, 16)  # [batch, heads, length, head_dim]
k, v = torch.randn(1, 2, 6, 16), torch.randn(1, 2, 6, 16)  # two key/value heads

fmt = BinaryK(8, 4)
# both products multiply in fmt and sum in BinaryK(14, 8); the operands, the
# probabilities and every step of the softmax are rounded to fmt
formats = AttentionFormats.uniform(SplitMac(fmt, BinaryK(14, 8)), Quant(fmt))

exact = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
rounded = qattention(q, k, v, formats, is_causal=True, enable_gqa=True)
print("largest difference from SDPA:", f"{(rounded - exact).abs().max().item():.4f}")

# enable_gqa repeats each key/value head over its group of query heads, word for word
repeated = qattention(
    q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), formats, is_causal=True
)
print("enable_gqa is the repeated heads:", torch.equal(rounded, repeated))

# a query that the mask leaves no key attends to nothing, as in PyTorch's kernels
keep = torch.ones(6, 6, dtype=torch.bool).tril()
keep[3] = False
out = qattention(q, k, v, formats, attn_mask=keep, enable_gqa=True)
print("fully masked query's output:", out[0, 0, 3, :4].tolist())
