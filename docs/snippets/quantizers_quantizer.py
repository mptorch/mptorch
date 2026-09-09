"""Quantizer: a straight-through estimator with its own backward format."""

import torch

from mptorch import BinaryK
from mptorch.quant import Quantizer, binaryK_quantize

torch.manual_seed(0)
torch.set_printoptions(precision=5)

x = torch.randn(5, requires_grad=True)
g = torch.randn(5) * 0.01  # an incoming gradient, small enough to show E5M2's grid

# Forward in E4M3, backward in E5M2.
act = Quantizer(BinaryK(8, 4), BinaryK(8, 3))
y = act(x)
y.backward(g)
print("x             ", x.detach())
print("act(x)        ", y.detach())
print("grad_output   ", g)
print("x.grad (E5M2) ", x.grad)

# Backward left as None: the gradient passes through untouched.
x.grad = None
Quantizer(BinaryK(8, 4))(x).backward(g)
print("x.grad (pass) ", x.grad, "\n  identical to grad_output:", torch.equal(x.grad, g))

# The raw quantize function is not differentiable, and says so rather than
# returning a tensor whose gradient would silently vanish.
try:
    binaryK_quantize(x, K=8, P=4)
except RuntimeError as e:
    print("\nRuntimeError:", str(e).split(" Detach")[0])

# ... unless no gradient is being tracked, which is how a layer's autograd
# Function calls it.
with torch.no_grad():
    print("\nunder no_grad:", binaryK_quantize(x, K=8, P=4))
