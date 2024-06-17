import torch
from torch.nn import functional as F
from mptorch.quant import *

device = torch.device("cpu")

man = 2
exp = 5

err = 1/8

print("Starting tensor:")
y = torch.full((2, 2), err)
x = torch.tensor([[1.,2.],[3.,4.]])
print(x)
print(x.dim())

print("\nUsing pytorch sm:")
sm = F.softmax(x, dim=1)
print(sm)

print("\nUsing binary32 format:")
a = QSoftMax(x, man, exp, 1, False)
print(a)

#diff = torch.abs(torch.sub(sm, q))
#print("\nDifference between PyTorch and Quant:\n", diff)
#print(torch.gt(y, diff))

# Testing
# torch.testing.assert_close(sm, q, atol = err, rtol = 0)