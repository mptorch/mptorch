"""QConv2d whose three convolutions run in custom arithmetic."""

import torch
import torch.nn.functional as F

from mptorch import AccumulateAlgorithm, BinaryK, RoundMode
from mptorch.quant import FusedMac, QConv2d, Quant, SplitMac, conv_formats, qmatmul

torch.manual_seed(0)
e4m3 = BinaryK(8, 4)

# E4M3 products summed in a 12-bit accumulator, for the forward convolution
# and both gradients; operand quantizers are layered on as for QLinear.
mac = SplitMac(e4m3, BinaryK(12, 7))
formats = conv_formats(mac)
formats.input_quant = Quant(e4m3)
formats.weight_quant = Quant(e4m3)

conv = QConv2d(3, 8, kernel_size=3, stride=2, padding=1, formats=formats)
x = torch.randn(4, 3, 16, 16, requires_grad=True)
out = conv(x)
out.sum().backward()
print("output", tuple(out.shape), "input grad", tuple(x.grad.shape))

# The forward is the GEMM of the same mac over the unfolded input, bit for
# bit, although no unfolded input (9x the input here) was ever built.
with torch.no_grad():
    qx, qw = Quant(e4m3)(x), Quant(e4m3)(conv.weight)
    cols = F.unfold(qx, 3, padding=1, stride=2)  # [4, 27, 64]
    ref = qmatmul(qw.reshape(8, -1), cols, mac).reshape(out.shape) + conv.bias.view(1, 8, 1, 1)
print("forward == qmatmul(Q(W), unfold(Q(x))) + b:", torch.equal(out, ref))

# Any accumulate algorithm and rounding mode, any geometry: here a fused
# multiply-add with Kahan summation, rounded stochastically, over a grouped,
# dilated convolution.
fused = FusedMac(BinaryK(12, 7, prng_bits=8), rounding=RoundMode.SR,
                 accumulate_algorithm=AccumulateAlgorithm.KAHAN)  # fmt: skip
grouped = QConv2d(4, 8, 3, padding=2, dilation=2, groups=2, formats=conv_formats(fused))
y = grouped(torch.randn(2, 4, 10, 10))
print("grouped, dilated, KAHAN under SR:", tuple(y.shape))

# A palette picks a format per output element; a [Cout, 1] map is one per
# output channel. Each pass has its own map, shaped like its own result.
pal = SplitMac([e4m3, BinaryK(8, 5)], BinaryK(12, 7))
per_channel = conv_formats(
    pal,
    prec_idx=torch.tensor([[0], [1]] * 4),  # the forward's 8 output channels
    igrad_prec_idx=torch.zeros(3, 1, dtype=torch.int64),  # the 3 input channels
    wgrad_prec_idx=torch.zeros(8, 27, dtype=torch.int64),  # the weight, [8, 3*3*3]
)
mixed = QConv2d(3, 8, 3, padding=1, formats=per_channel)
print("per-channel palette:", tuple(mixed(torch.randn(1, 3, 6, 6)).shape))
