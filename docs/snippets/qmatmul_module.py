"""QMatmul: the module form, for quantizers that carry state."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import QMatmul, QMatmulFormats, Quant, Quantizer, SplitMac, matmul_formats

torch.manual_seed(0)


class QAttention(nn.Module):
    """One attention head with both matmuls in E4M3 arithmetic."""

    def __init__(self, dim: int, formats: QMatmulFormats):
        super().__init__()
        self.scale = dim**-0.5
        self.qk = QMatmul(formats)  # scores = q @ k^T
        self.pv = QMatmul(formats)  # out = p @ v

    def forward(self, q, k, v):
        scores = self.qk(q, k.transpose(-2, -1)) * self.scale  # k^T is a view: no copy
        p = torch.softmax(scores, dim=-1)
        return self.pv(p, v)


e4m3 = BinaryK(8, 4)
formats = matmul_formats(SplitMac(e4m3, None))  # E4M3 products, float32 sums
formats.a_quant = Quant(e4m3)  # round both operands to E4M3 on the way in
formats.b_quant = Quant(e4m3)
formats.agrad_quant = Quantizer(BinaryK(8, 3, prng_bits=8))  # stochastic E5M2 on the gradient
formats.bgrad_quant = formats.agrad_quant

head = QAttention(64, formats)
q, k, v = (torch.randn(2, 128, 64, requires_grad=True) for _ in range(3))
out = head(q, k, v)
out.sum().backward()
print(
    "output",
    tuple(out.shape),
    " grads:",
    tuple(q.grad.shape),
    tuple(k.grad.shape),
    tuple(v.grad.shape),
)

# The Quantizer is registered under the QMatmul that holds it, so any state a
# quantizer keeps travels with the model. (named_modules lists a shared module
# once: both heads hold the same formats, and both gradient slots the same
# Quantizer.)
print("\nsubmodules:")
for name, _ in head.named_modules():
    if name:
        print("  ", name)
