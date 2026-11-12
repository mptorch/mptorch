"""QLinear with operand quantizers: what each slot of QAffineFormats does."""

import torch
import torch.nn.functional as F
from torch import nn

from mptorch import BinaryK
from mptorch.quant import QAffineFormats, QLinear, Quant

torch.manual_seed(0)
q8p4, q8p3 = Quant(BinaryK(8, 4)), Quant(BinaryK(8, 3))

# Forward signals in Binary8p4, backward signals in Binary8p3; the matmuls themselves
# stay float32 (no *_math hook set).
formats = QAffineFormats(
    weight_quant=q8p4,
    input_quant=q8p4,
    bias_quant=q8p4,
    igrad_quant=q8p3,
    wgrad_quant=q8p3,
    bgrad_quant=q8p3,
)
layer = QLinear(16, 8, formats=formats)
x = torch.randn(4, 16, requires_grad=True)
out = layer(x)
g = torch.randn_like(out)
out.backward(g)

# The same thing written out from the equations, on the same tensors.
with torch.no_grad():
    qx, qw, qb = q8p4(x), q8p4(layer.weight), q8p4(layer.bias)
    y = F.linear(qx, qw, qb)  # y = Q(x) Q(W)^T + Q(b)
    grad_x = q8p3(g) @ qw  # dL/dx = Q(dL/dy) Q(W)
    grad_w = q8p3(g).T @ qx  # dL/dW = Q(dL/dy)^T Q(x)
    grad_b = q8p3(g).sum(0)  # dL/db = sum over the batch of Q(dL/dy)

print("forward     ", torch.equal(out, y))
print("input grad  ", torch.equal(x.grad, grad_x))
print("weight grad ", torch.equal(layer.weight.grad, grad_w))
print("bias grad   ", torch.equal(layer.bias.grad, grad_b))

# A QAffineFormats() with nothing set is a plain nn.Linear.
plain = QLinear(16, 8)
ref = nn.Linear(16, 8)
ref.load_state_dict(plain.state_dict())
print("\nno formats == nn.Linear:", torch.equal(plain(x), ref(x)))
