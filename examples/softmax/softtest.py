import torch
from torch.nn import functional as F
from mptorch.quant import *

device = torch.device("cpu")

man = 2
exp = 5

err = 1/8

print("Starting tensor:")
y = torch.full((2,3), err)
x = torch.randn(2,3)
print(x)

print("\nUsing pytorch sm:")
sm = F.softmax(x, dim=1)
print(sm)

print("\nUsing quant format:")
q = QSoftMax(x, man, exp, 2, 3, 2, True)
print(q)

print("\nUsing binary32 format:")
a = QSoftMax(x, man, exp, 2, 3, 2, False)
print(a)

diff = torch.abs(torch.sub(a, q))
print("\nDifference:\n", diff)

print(torch.gt(y, diff))

# Testing
# torch.testing.assert_close(a, q, atol = err, rtol = 0)