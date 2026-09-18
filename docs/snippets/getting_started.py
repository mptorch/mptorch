"""
Thirty seconds with MPTorch: round a tensor, then train
a layer in that format.
"""

import torch

from mptorch import BinaryK
from mptorch.quant import QAffineFormats, QLinear, Quant

torch.manual_seed(0)

# A format is a value; a Quant is that format as a function
# on tensors.
e4m3 = BinaryK(8, 4)  # 8 bits: sign, 4 exponent, 3 mantissa
x = torch.randn(4)
print("x        ", x)
print("E4M3(x)  ", Quant(e4m3)(x))

# A layer takes one quantizer per signal. Anything left
# None stays binary32.
formats = QAffineFormats(input_quant=Quant(e4m3), weight_quant=Quant(e4m3))
layer = QLinear(16, 4, formats=formats)
out = layer(torch.randn(2, 16))
out.sum().backward()
print("output   ", out.shape, " weight.grad", layer.weight.grad.shape)
