"""QConv2d: the same slots, over a convolution."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import QAffineFormats, QConv2d, Quant, Quantizer

torch.manual_seed(0)
e4m3, e5m2 = BinaryK(8, 4), BinaryK(8, 3)

formats = QAffineFormats(
    weight_quant=Quant(e4m3),
    input_quant=Quant(e4m3),
    igrad_quant=Quant(e5m2),
    wgrad_quant=Quant(e5m2),
)
conv = QConv2d(3, 8, kernel_size=3, padding=1, formats=formats)
ref = nn.Conv2d(3, 8, kernel_size=3, padding=1)
ref.load_state_dict(conv.state_dict())

x = torch.randn(2, 3, 16, 16, requires_grad=True)
out = conv(x)
out.sum().backward()
print("output", tuple(out.shape), "weight grad", tuple(conv.weight.grad.shape))

# The convolution is torch's own (there is no custom conv arithmetic), so
# with the operands rounded by hand it is reproduced exactly.
with torch.no_grad():
    y = torch.nn.functional.conv2d(Quant(e4m3)(x), Quant(e4m3)(conv.weight), conv.bias, padding=1)
print("forward == conv2d(Q(x), Q(W)) + b:", torch.equal(out, y))

# How far a quantized layer is from the float32 one, on the same weights.
cos = torch.nn.functional.cosine_similarity
print("cos(out, float32 conv):", f"{cos(out.flatten(), ref(x).flatten(), dim=0):.5f}")

# Activations between layers are quantized with a Quantizer module, which
# also fixes what their gradient is rounded to.
net = nn.Sequential(conv, nn.ReLU(), Quantizer(e4m3, e5m2), QConv2d(8, 4, 3, formats=formats))
print("\n", net)
