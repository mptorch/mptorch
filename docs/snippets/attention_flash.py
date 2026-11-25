"""Flash attention: FA2's order, tile by tile, and an FP8 recipe in the style of FA3."""

import dataclasses

import torch
import torch.nn.functional as F

from mptorch import E4M3
from mptorch.quant import (
    AttentionFormats,
    FlashFormats,
    GemmFormats,
    Quant,
    ScaledQuant,
    qattention,
)

torch.manual_seed(0)
q, k, v = (torch.randn(1, 4, 256, 64) for _ in range(3))
exact = F.scaled_dot_product_attention(q.double(), k.double(), v.double(), is_causal=True)


def error(af):
    out = qattention(q, k, v, af, is_causal=True).double()
    return ((out - exact).norm() / exact.norm()).item()


# with nothing rounded, flash is SDPA to float32's precision
print(f"flash alone:          {error(AttentionFormats(flash=FlashFormats())):.1e}")

# FA3's FP8 recipe: q, k and v scaled into E4M3 (per tile under flash), the
# probabilities rounded to E4M3, the scores, the running state and the
# output left in binary32
qk = GemmFormats(input_quant=ScaledQuant(E4M3), weight_quant=ScaledQuant(E4M3))
pv = GemmFormats(input_quant=Quant(E4M3), weight_quant=ScaledQuant(E4M3))
standard = AttentionFormats(qk=qk, pv=pv)
flash = dataclasses.replace(standard, flash=FlashFormats(block_q=64, block_kv=64))
print(f"FP8, standard path:   {error(standard):.4f}")
print(f"FP8, flash:           {error(flash):.4f}")
