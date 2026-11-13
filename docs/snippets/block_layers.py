"""QLinear on the block GEMM, with a 1D weight format and with 16 x 16 tiles."""

import dataclasses

import torch

from mptorch import NVFP4, BinaryK
from mptorch.quant import BlockQuant, QLinear, block_gemm_formats

torch.manual_seed(0)
acc = BinaryK(16, 11)
x = torch.randn(32, 128, requires_grad=True)

# 1D weight blocks: every hook packs what it multiplies along the dimension
# it reduces over, the weight along its input features in the forward and
# along its output features in the input gradient.
layer = QLinear(128, 64, formats=block_gemm_formats(NVFP4, NVFP4, acc=acc))
layer(x).sum().backward()
print("1D weight: forward and both gradients ran")

# 16 x 16 weight tiles: packed once per step by weight_quant, and the input
# gradient reads that packing transposed, so the forward and backward passes
# see the same quantized weight.
w16 = dataclasses.replace(NVFP4, block_rows=16)
formats = block_gemm_formats(NVFP4, w16, acc=acc)
formats.weight_quant = BlockQuant(w16, axis=1).pack
tiled = QLinear(128, 64, formats=formats)
tiled.load_state_dict(layer.state_dict())
x2 = x.detach().clone().requires_grad_()
tiled(x2).sum().backward()

# The same input gradient as re-packing the float weight along the other
# axis: a square tile quantizes it the same either way.
repack = QLinear(128, 64, formats=block_gemm_formats(NVFP4, w16, acc=acc))
repack.load_state_dict(layer.state_dict())
x3 = x.detach().clone().requires_grad_()
repack(x3).sum().backward()
print(
    "16 x 16: input gradient from the forward's packing == re-packed:",
    torch.equal(x2.grad, x3.grad),
)
