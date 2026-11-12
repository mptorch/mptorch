"""Block formats: packing a tensor, what the bytes hold, and reading it back."""

import torch

from mptorch import MXFP4_E2M1, MXFP8_E4M3, NVFP4
from mptorch.quant import block_pack, block_quantize, block_unpack

torch.manual_seed(0)
x = torch.randn(4, 64)

# The OCP MX presets and NVFP4: bytes per element, codes and scales included
for name, fmt in (("MXFP8_E4M3", MXFP8_E4M3), ("MXFP4_E2M1", MXFP4_E2M1), ("NVFP4", NVFP4)):
    p = block_pack(x, fmt)
    err = (p.unpack() - x).abs().max().item()
    print(
        f"{name:11s} data {tuple(p.data.shape)} scales {tuple(p.scales.shape)}"
        f"  {p.nbytes / x.numel():.3f} B/elem  max error {err:.4f}"
    )

# One MXFP4 block of 32: the scale is 2^(floor(log2(amax)) - emax), here
# amax = 6 and E2M1's emax = 2, so 2^0 (E8M0 code 127); the elements round
# to E2M1's grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, two codes to a byte, low
# nibble first.
v = torch.tensor([[0.3, -1.0, 2.5, 6.0] * 8])
p = block_pack(v, MXFP4_E2M1)
print("\nscale code", p.scales.item(), " first bytes", [hex(b) for b in p.data[0, :2].tolist()])
print("decoded    ", block_unpack(p)[0, :4].tolist())

# block_quantize is pack + unpack fused, with no packed intermediate
print(
    "fused equal:",
    torch.equal(block_quantize(x, MXFP8_E4M3), block_unpack(block_pack(x, MXFP8_E4M3))),
)
