import torch
from torch.nn import functional as F
from mptorch.quant import *

device = torch.device("cpu")

man = 2
exp = 5

err = 1/8

print("Starting tensor:")
y = torch.full((2, 2), err)
x = torch.tensor([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]])
z = torch.tensor([[1.,2.,3.],[4.,5.,6.]])
p = torch.tensor([[1.,2.],[3.,4.]])
r = torch.randn(2,4,3,2)

t = r

print(t)
#print(t.dim())

print("\nUsing pytorch sm:")
sm = F.softmax(t, dim=0)
print(sm)

print("\nUsing binary32 format:")
a = QSoftMax(t, man, exp, 0, False)
print(a)

diff = torch.abs(torch.sub(sm, a))
print("\nDifference between PyTorch and Quant:\n", diff)
#print(torch.gt(sm, a))

# Testing
torch.testing.assert_close(sm, a, atol = err, rtol = 0)