"""QLinear with operand quantizers: what each slot of QAffineFormats does."""

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import QAffineFormats, QLinear, Quant

torch.manual_seed(0)
e4m3, e5m2 = Quant(BinaryK(8, 4)), Quant(BinaryK(8, 3))

# Forward signals in E4M3, backward signals in E5M2; the matmuls themselves
# stay float32 (no *_math hook set).
formats = QAffineFormats(
    weight_quant=e4m3,
    input_quant=e4m3,
    bias_quant=e4m3,
    igrad_quant=e5m2,
    wgrad_quant=e5m2,
    bgrad_quant=e5m2,
)
layer = QLinear(16, 8, formats=formats)
x = torch.randn(4, 16, requires_grad=True)
out = layer(x)
g = torch.randn_like(out)
out.backward(g)

# The same thing written out from the equations, on the same tensors.
with torch.no_grad():
    qx, qw, qb = e4m3(x), e4m3(layer.weight), e4m3(layer.bias)
    y = F.linear(qx, qw, qb)  # y = Q(x) Q(W)^T + Q(b)
    grad_x = e5m2(g) @ qw  # dL/dx = Q(dL/dy) Q(W)
    grad_w = e5m2(g).T @ qx  # dL/dW = Q(dL/dy)^T Q(x)
    grad_b = e5m2(g).sum(0)  # dL/db = sum over the batch of Q(dL/dy)

print("forward     ", torch.equal(out, y))
print("input grad  ", torch.equal(x.grad, grad_x))
print("weight grad ", torch.equal(layer.weight.grad, grad_w))
print("bias grad   ", torch.equal(layer.bias.grad, grad_b))

# A QAffineFormats() with nothing set is a plain nn.Linear.
plain = QLinear(16, 8)
ref = nn.Linear(16, 8)
ref.load_state_dict(plain.state_dict())
print("\nno formats == nn.Linear:", torch.equal(plain(x), ref(x)))
