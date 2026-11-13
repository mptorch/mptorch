"""The block GEMM: two packed operands, decoded in the kernel's tile loads."""

import warnings

import torch

from mptorch import MXFP4_E2M1, MXFP8_E4M3, BinaryK, FormatRangeWarning
from mptorch.quant import BlockMac, binaryK_matmul, block_matmul, block_pack, qmatmul

torch.manual_seed(0)
x, w = torch.randn(8, 256), torch.randn(32, 256)

xp = block_pack(x, MXFP8_E4M3)  # [8, 256], blocks along K
wp = block_pack(w, MXFP4_E2M1)  # [32, 256], blocks along K, as a weight is stored
acc = BinaryK(16, 11)  # the running sum rounded to binary16's precision

y = block_matmul(xp, wp.mT, acc=acc)  # x @ w.T; .mT reads w's codes transposed

# The same numbers as the flat binaryK GEMM on the decoded operands, with a
# binary32 multiply: binary32 itself as a binaryK format, the identity on
# every product here (the warning is that its top, 2^128, is past binary32's).
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FormatRangeWarning)
    flat = binaryK_matmul(
        xp.unpack(), wp.unpack(), trans_b=True, mul_K=32, mul_P=24, mul_bias=127, acc_K=16, acc_P=11
    )
print("equal to the flat GEMM on the decoded operands:", torch.equal(y, flat))
ref = x @ w.T
print(f"relative error against float32: {((y - ref).norm() / ref.norm()).item():.3f}")

# The differentiable spelling: qmatmul packs both operands in the forward
# and each gradient pass packs what it multiplies
a = torch.randn(8, 256, requires_grad=True)
b = torch.randn(256, 32, requires_grad=True)
out = qmatmul(a, b, BlockMac(MXFP8_E4M3, MXFP4_E2M1, acc))
out.sum().backward()
print("gradients:", tuple(a.grad.shape), tuple(b.grad.shape))
