"""How a block's scale is chosen: the four ScaleRounding rules."""

import dataclasses

import torch

from mptorch import E2M1, MXFP4_E2M1, MXFP6_E2M3, NVFP4, BlockFormat, ScaleRounding, SuperFP
from mptorch.quant import block_pack, block_quantize

# Two MXFP4 blocks. The first one's largest magnitude, 3.8, is 7.6 at the
# scale 1/2, past E2M1's 6; the second one's, 3.3, is 6.6 there, which rounds
# to 6 anyway.
x = torch.zeros(2, 32)
x[0, :3] = torch.tensor([3.8, -1.3, 0.3])
x[1, :3] = torch.tensor([3.3, -1.3, 0.3])
for rule in ScaleRounding:
    fmt = dataclasses.replace(MXFP4_E2M1, scale_rounding=rule)
    scales = (2.0 ** (block_pack(x, fmt).scales[:, 0].double() - 127)).tolist()
    q = block_quantize(x, fmt)[:, :3].tolist()
    print(f"{rule.name:<9}  scales {scales}  blocks {q[0]}  {q[1]}")

# Each scale's default rule, and what the four cost over many blocks: the mean
# relative error of a block, over Gaussian blocks of every magnitude.
sfp = BlockFormat(E2M1, SuperFP(3, 4, 2, 7), 32)
print("\ndefaults:", MXFP4_E2M1.scale_rule.name, NVFP4.scale_rule.name, sfp.scale_rule.name)
torch.manual_seed(0)
x = torch.randn(20000, 32) * torch.exp2(torch.rand(20000, 1) * 30 - 15)
print(f"\n{'':<22}" + "".join(f"{r.name:>11}" for r in ScaleRounding))
for name, base in (
    ("MXFP4_E2M1", MXFP4_E2M1),
    ("MXFP6_E2M3", MXFP6_E2M3),
    ("E2M1, SuperFP scale", sfp),
):
    row = []
    for rule in ScaleRounding:
        if rule is ScaleRounding.OCP and base.has_tensor_scale:
            row.append(f"{'-':>11}")  # OCP is a power-of-two scale's rule
            continue
        q = block_quantize(x, dataclasses.replace(base, scale_rounding=rule))
        err = ((q - x).norm(dim=1) / x.norm(dim=1)).mean().item()
        row.append(f"{err:>11.4f}")
    print(f"{name:<22}" + "".join(row))
