"""Block floating point: fixed-point elements sharing a power-of-two scale."""

import torch

from mptorch import MXFP8_E4M3, BlockFormat, FixedPoint, RoundMode, bfp
from mptorch.quant import block_matmul, block_pack, block_quantize, fixedpoint_quantize

torch.manual_seed(0)
torch.set_printoptions(precision=6)

f = bfp(8)  # FixedPoint(8, 6, symmetric=True) elements, an E8M0 scale, blocks of 16
print("element:", f.elem)
print(f"elem_max {f.elem_max}, emax {f.emax}, {f.bytes_per_block} bytes of codes a block")

# A block is the elementwise fixed-point cast of each element over the
# block's scale, 2**floor(log2(amax)), times that scale.
x = torch.randn(1, 16) * 3
scale = torch.exp2(torch.floor(torch.log2(x.abs().max())))
print("\nscale           ", scale.item())
print("block_quantize  ", block_quantize(x, f)[0, :4])
print("fixedpoint * s  ", (fixedpoint_quantize(x / scale, 8, 6, symmetric=True) * scale)[0, :4])
print(
    "equal in every deterministic mode:",
    all(
        torch.equal(
            block_quantize(x, f, rounding=m),
            fixedpoint_quantize(x / scale, 8, 6, symmetric=True, rounding_mode=m) * scale,
        )
        for m in RoundMode
        if not m.is_stochastic
    ),
)

# The codes are sign-magnitude integers: 1.5 is 96 (= 1.5 * 2**6), -0.5 is
# 0x80 | 32. OCP's MXINT8 is two's complement, with one more code, -2.
v = torch.tensor([[1.5, -0.5, 1.0, 0.25] + [0.0] * 28])
print("\nbfp(8, 32) codes", block_pack(v, bfp(8, 32)).data[0, :4].tolist())

# Any signed, symmetric FixedPoint element works, under any scale; a two's
# complement or unsigned one has no sign-magnitude code for all its values.
e4m3 = BlockFormat(FixedPoint(8, 6, symmetric=True), MXFP8_E4M3.elem, 16)
print("\nan E4M3 scale:  ", e4m3.scale_rule.name, "rounding")
try:
    BlockFormat(FixedPoint(8, 6), f.scale, 16)
except ValueError as e:
    print("two's complement:", str(e).split(": ")[0])

# The block GEMM multiplies codes of up to 8 bits, so bfp(2) to bfp(8), in any
# pairing; wider ones pack, unpack and quantize only.
a, b = torch.randn(4, 64), torch.randn(64, 3)
pa, pb = block_pack(a, bfp(8, 32), axis=-1), block_pack(b, MXFP8_E4M3, axis=-2)
print("\nbfp8 x MXFP8:\n", block_matmul(pa, pb))
print("their decoded product, in float64:\n", (pa.unpack().double() @ pb.unpack().double()).float())
