import torch
from mptorch.quant import *

device = torch.device("cpu")

x = torch.rand(2,3)
print(x)

print(QSoftMax(x))