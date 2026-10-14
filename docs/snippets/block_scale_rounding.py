"""How a block's scale is chosen: the ScaleRounding rules."""

import dataclasses

import torch

from mptorch import E2M1, MXFP4_E2M1, MXFP6_E2M3, NVFP4, BlockFormat, ScaleRounding, SuperFP
from mptorch.quant import block_pack, block_quantize, block_unpack

# The four rules that take one of the two scales around the one a block asks for.
R = ScaleRounding
BRACKET = (R.OCP, R.NEAREST, R.UP, R.SELECTIVE)

# Two MXFP4 blocks. The first one's largest magnitude, 3.8, is 7.6 at the
# scale 1/2, past E2M1's 6; the second one's, 3.3, is 6.6 there, which rounds
# to 6 anyway.
x = torch.zeros(2, 32)
x[0, :3] = torch.tensor([3.8, -1.3, 0.3])
x[1, :3] = torch.tensor([3.3, -1.3, 0.3])
for rule in BRACKET:
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
print(f"\n{'':<22}" + "".join(f"{r.name:>11}" for r in BRACKET))
for name, base in (
    ("MXFP4_E2M1", MXFP4_E2M1),
    ("MXFP6_E2M3", MXFP6_E2M3),
    ("E2M1, SuperFP scale", sfp),
):
    row = []
    for rule in BRACKET:
        if rule is ScaleRounding.OCP and base.has_tensor_scale:
            row.append(f"{'-':>11}")  # OCP is a power-of-two scale's rule
            continue
        q = block_quantize(x, dataclasses.replace(base, scale_rounding=rule))
        err = ((q - x).norm(dim=1) / x.norm(dim=1)).mean().item()
        row.append(f"{err:>11.4f}")
    print(f"{name:<22}" + "".join(row))

# Four Over Six, on NVFP4: a block may instead take the scale 1.5 times larger,
# under which its largest element lands on 4 and not on 6, if that gives it
# the smaller error. The block of the paper's Table 2, under each choice.
x = torch.zeros(1, 16)
x[0, :4] = torch.tensor([10.0, 20.0, 30.0, 40.0])
print()
for rule in (R.NEAREST, R.FOUR_OVER_SIX_MSE):
    p = block_pack(x, dataclasses.replace(NVFP4, scale_rounding=rule), tensor_scale=1.0)
    scale = p.scales.view(torch.float8_e4m3fn).float().item()
    print(f"{rule.name:<18} scale {scale:<5} block {block_unpack(p)[0, :4].tolist()}")

# Over many blocks: how often each measure takes the second scale, and the
# mean relative error of a block.
torch.manual_seed(0)
x = torch.randn(20000, 16) * torch.exp2(torch.rand(20000, 1) * 8 - 4)
nearest = block_pack(x, NVFP4)
print(f"\n{'NVFP4':<22}{'takes 4':>9}{'error':>9}")
for rule in (R.NEAREST, R.FOUR_OVER_SIX_MSE, R.FOUR_OVER_SIX_MAE, R.FOUR_OVER_SIX_ABS_MAX):
    fmt = dataclasses.replace(NVFP4, scale_rounding=rule)
    p = block_pack(x, fmt, tensor_scale=nearest.tensor_scale)
    took = (p.scales != nearest.scales).float().mean().item()
    err = ((block_unpack(p) - x).norm(dim=1) / x.norm(dim=1)).mean().item()
    print(f"{rule.name:<22}{took:>9.1%}{err:>9.4f}")
print("default tensor scale: NVFP4's block of the largest value takes the scale", end=" ")
print(NVFP4.top_scale, "and Four Over Six's", fmt.top_scale)

# On 2D tiles the block is the tile: one choice, from all 256 of its elements.
# A square tile makes it the same way whichever axis the matrix is packed along.
torch.manual_seed(0)
w = torch.randn(20, 20, 16, 16) * torch.exp2(torch.rand(20, 20, 1, 1) * 8 - 4)
w = w.permute(0, 2, 1, 3).reshape(320, 320)
tiled = dataclasses.replace(NVFP4, block_rows=16)
nearest = block_pack(w, tiled)
print(f"\n{'NVFP4, 16 x 16 tiles':<22}{'takes 4':>9}{'error':>9}   same along both axes")
for rule in (R.NEAREST, R.FOUR_OVER_SIX_MSE, R.FOUR_OVER_SIX_MAE, R.FOUR_OVER_SIX_ABS_MAX):
    fmt = dataclasses.replace(tiled, scale_rounding=rule)
    ts = nearest.tensor_scale
    p = block_pack(w, fmt, tensor_scale=ts)
    took = (p.scales != nearest.scales).float().mean().item()
    diff = (block_unpack(p) - w).reshape(20, 16, 20, 16).permute(0, 2, 1, 3).reshape(400, 256)
    ref = w.reshape(20, 16, 20, 16).permute(0, 2, 1, 3).reshape(400, 256)
    err = (diff.norm(dim=1) / ref.norm(dim=1)).mean().item()
    same = torch.equal(
        block_quantize(w, fmt, 0, tensor_scale=ts), block_quantize(w, fmt, 1, tensor_scale=ts)
    )
    print(f"{rule.name:<22}{took:>9.1%}{err:>9.4f}   {same}")
