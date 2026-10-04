"""One routine behind every entry point: a batched matmul, a layer's three
passes, a convolution and a block product, each equal to the GEMM of one mac
over explicit matrices."""

import warnings

import torch
import torch.nn.functional as F

from mptorch import MXFP8_E4M3, BinaryK, FormatRangeWarning
from mptorch.quant import (
    QConv2d,
    QLinear,
    SplitMac,
    binaryK_gemm_formats,
    block_matmul,
    block_pack,
    conv_formats,
    qmatmul,
)

torch.manual_seed(0)
mac = SplitMac(BinaryK(8, 4), BinaryK(12, 7))  # Binary8p4 products, a 12-bit running sum


def gemm(a, b):
    """The core routine on two explicit matrices."""
    return qmatmul(a, b, mac)


# A batched product with a broadcast operand: one launch, each slice the 2D GEMM.
a, b = torch.randn(3, 5, 64), torch.randn(64, 7)
y = qmatmul(a, b, mac)
print("batched matmul, slice by slice:", all(torch.equal(y[i], gemm(a[i], b)) for i in range(3)))

# A linear layer: three GEMMs, x W^T forward, G W and G^T x backward.
lin = QLinear(64, 16, bias=False, formats=binaryK_gemm_formats(8, 4, acc_K=12, acc_P=7))
x = torch.randn(10, 64, requires_grad=True)
out = lin(x)
g = torch.randn_like(out)
out.backward(g)
W = lin.weight.detach()
print(
    "linear layer, forward and both gradients:",
    torch.equal(out.detach(), gemm(x.detach(), W.T)),
    torch.equal(x.grad, gemm(g, W)),
    torch.equal(lin.weight.grad, gemm(g.T, x.detach())),
)

# A convolution: the GEMM of the flattened weight and the unfolded input,
# though the kernel never builds the unfolded input.
conv = QConv2d(3, 8, 3, stride=2, padding=1, bias=False, formats=conv_formats(mac))
img = torch.randn(2, 3, 12, 12)
cols = F.unfold(img, 3, padding=1, stride=2)  # [2, 27, 36]
want = gemm(conv.weight.detach().reshape(8, -1), cols).reshape(2, 8, 6, 6)
print("convolution forward:", torch.equal(conv(img).detach(), want))

# A block product: the GEMM of the decoded operands, with the multiply in
# binary32 (the float32 layout as a format; it warns that its top, 2^128, is
# past binary32's own).
pa, pb = block_pack(torch.randn(6, 64), MXFP8_E4M3), block_pack(b, MXFP8_E4M3, 0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FormatRangeWarning)
    decoded = qmatmul(pa.unpack(), pb.unpack(), SplitMac(BinaryK(32, 24, bias=127), BinaryK(12, 7)))
print("block product:", torch.equal(block_matmul(pa, pb, acc=BinaryK(12, 7)), decoded))
