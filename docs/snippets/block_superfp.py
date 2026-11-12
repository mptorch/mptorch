"""Block formats over SuperFP: supernormal elements, and a superfp scale."""

import torch

from mptorch import E2M1, E8M0, MXFP4_E2M1, NVFP4, BlockFormat, SuperFP
from mptorch.quant import BlockPacked, block_quantize, block_unpack

# A 4-bit superfp element: one normal binade (4, 6) and, below it, the powers
# of two down to 1/8. Codes 0..7 decode to its whole value set:
sfp4 = BlockFormat(SuperFP(1, 2, 1, 1), E8M0, 32)
codes = torch.tensor([[0x10, 0x32, 0x54, 0x76] * 4], dtype=torch.uint8)
one = BlockPacked(
    codes, torch.tensor([[127]], dtype=torch.uint8), sfp4, (1, 32), 1, 1.0, torch.float32
)
print("SuperFP(1, 2, 1, 1):", block_unpack(one)[0, :8].tolist())
print("E2M1:               ", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

# The same budget of bits spent differently: E2M1 has uniform steps near the
# top of a block, superfp has range at the bottom. On a block with a wide
# spread of magnitudes, superfp keeps the small ones:
torch.manual_seed(0)
x = torch.randn(256, 32) * torch.logspace(-3, 0, 32)
for name, fmt in (("MXFP4_E2M1", MXFP4_E2M1), ("SuperFP(1, 2, 1, 1)", sfp4)):
    q = block_quantize(x, fmt)
    zeros = (q == 0).float().mean().item()
    err = ((q - x).norm() / x.norm()).item()
    print(f"{name:20s} relative error {err:.3f}, flushed to zero {zeros:.0%}")

# A superfp scale: cast to nearest even like E4M3, times a tensor scale, but
# reaching down to 2^-104 in one byte where E4M3 stops at 2^-9. A block of
# values around 2^-66 keeps them; under NVFP4's E4M3 scale it flushes to zero.
sfp_scaled = BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 16)
tiny = torch.tensor([[1.0, 3.0, -2.0, 6.0] * 4]) * 2.0**-66
for name, fmt in (("E4M3 scale (NVFP4)", NVFP4), ("SuperFP(3, 4, 2, 7) scale", sfp_scaled)):
    q = block_quantize(tiny, fmt, tensor_scale=1.0)
    print(f"{name:26s} decoded / 2^-66: {(q / 2.0**-66)[0, :4].tolist()}")
