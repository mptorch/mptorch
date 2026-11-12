"""2D tiles: one scale per block_rows x block_size tile, and why a square one
quantizes a matrix the same along either axis."""

import dataclasses

import torch

from mptorch import NVFP4
from mptorch.quant import block_pack, block_quantize

torch.manual_seed(0)
w = torch.randn(64, 96)

w16 = dataclasses.replace(NVFP4, block_rows=16)  # NVFP4 with 16 x 16 weight tiles
p = block_pack(w, w16)
print("scales", tuple(p.scales.shape), " bytes/elem", round(p.nbytes / w.numel(), 3))

# Packed along either axis, a square tile covers the same 16 x 16 elements
# and takes the same scale, so the values agree; a 1D block does not.
print(
    "16 x 16 tiles, axis 0 == axis 1:",
    torch.equal(block_quantize(w, w16, 0), block_quantize(w, w16, 1)),
)
print(
    "1 x 16 blocks, axis 0 == axis 1:",
    torch.equal(block_quantize(w, NVFP4, 0), block_quantize(w, NVFP4, 1)),
)

# The transpose is free: .mT relabels the same codes
print("mT shares the codes:", p.mT.data.data_ptr() == p.data.data_ptr(), p.mT.shape, p.mT.axis)
