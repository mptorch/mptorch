"""IEEE P3109's block operations on packed tensors, and mptorch's two reductions."""

import torch

from mptorch import MXFP4_E2M1, MXFP8_E4M3, BinaryK
from mptorch.quant import (
    block_dot,
    block_exp,
    block_pack,
    block_project,
    block_reduce_add,
    convert_to_block_max_abs_finite,
)

x = torch.tensor([[0.3, -1.0, 2.5, 6.0] * 8, [0.1, 0.2, -0.4, 0.8] * 8])

# BlockProject: the codes under scale codes the caller gives. Onto the scales
# block_pack chose, it is block_pack.
p = block_pack(x, MXFP4_E2M1)
print("block_pack's scales:", p.scales.tolist())
same = block_project(x, MXFP4_E2M1, p.scales)
print("onto them, block_pack's codes:", torch.equal(same.data, p.data))
halved = block_project(x, MXFP4_E2M1, p.scales + 1)  # each scale doubled
print("onto scales twice as large:", halved.unpack()[:, :4].tolist())

# A NaN that E2M1 has no code for, under a scale that is not NaN, raises.
y = x.clone()
y[0, 0] = float("nan")
try:
    block_project(y, MXFP4_E2M1, p.scales)
except RuntimeError as e:
    print("RuntimeError:", str(e).split("\n")[0])
nan_scale = p.scales.clone()
nan_scale[0, 0] = 0xFF  # E8M0's NaN: the whole block decodes to NaN
print("under a NaN scale:", block_project(y, MXFP4_E2M1, nan_scale).unpack()[0, :4].tolist())

# BlockExp, with the scales chosen by the format's rule, and onto x's own.
e = block_pack(x / 4, MXFP8_E4M3)
print("exp, new scales:", block_exp(e).unpack()[0, :4].tolist())
print("exp onto x's   :", block_exp(e, e.scales).unpack()[0, :4].tolist())

# ConvertToBlockMaxAbsFinite: each block's largest finite magnitude becomes about 1.
c = convert_to_block_max_abs_finite(x, MXFP8_E4M3)
print("scaled to 1:", c.scales.tolist(), c.unpack()[:, :4].tolist())

# mptorch's BlockReduceAdd and BlockDotProduct, with a GEMM's arithmetic.
q = block_pack(x, MXFP8_E4M3)
print("sums per block:", block_reduce_add(q).tolist())
print("in BinaryK(8, 4):", block_reduce_add(q, BinaryK(8, 4)).tolist())
print("dot products:", block_dot(q, q).tolist())
