"""Block formats in binary64: float64 tensors, the widening carrier, and formats
only binary64 holds."""

import warnings

import torch

from mptorch import E8M0, NVFP4, BinaryK, BlockFormat, FormatRangeWarning
from mptorch.quant import block_matmul, block_pack, block_quantize

torch.manual_seed(0)
x = torch.randn(4, 64, dtype=torch.float64)

# A float64 tensor packs and decodes in binary64, and remembers it.
p = block_pack(x, NVFP4)
print("packed in", p.carrier, "-> unpacks to", p.unpack().dtype)
print("its tensor scale is a float64 quotient:", p.tensor_scale)

# carrier=torch.float64 on a float32 tensor is the float64 call on the widened
# tensor, narrowed back once. (NVFP4's decoded values are binary64 products a
# float32 result rounds a second time, which a FormatRangeWarning says.)
warnings.simplefilter("ignore", FormatRangeWarning)
x32 = x.float()
same = torch.equal(
    block_quantize(x32, NVFP4, carrier=torch.float64),
    block_quantize(x32.double(), NVFP4).float(),
)
print("float32, carrier=torch.float64 == the float64 call, narrowed:", same)

# An element with nine exponent bits (2^-255 .. 2^256) is a format of float64
# tensors: binary32 refuses it, and says which carrier holds it.
e9m2 = BlockFormat(BinaryK(12, 3), E8M0, 32)
try:
    block_quantize(x32, e9m2)
except ValueError as e:
    print("binary32 refuses it:", str(e).split("; ")[-1])
tiny = x * 2.0**-200
q = block_quantize(tiny, e9m2)
print("binary64: relative error", (torch.linalg.norm(q - tiny) / torch.linalg.norm(tiny)).item())

# The block GEMM of two binary64 packings is a binary64 product.
w = torch.randn(64, 8, dtype=torch.float64)
y = block_matmul(block_pack(x, NVFP4), block_pack(w, NVFP4, 0), acc=BinaryK(32, 27))
print("block_matmul:", y.dtype, tuple(y.shape))
