"""The block GEMM's arithmetic: exact products, OCP's two-level sum, and where
NVFP4's tensor scales are applied."""

import torch

from mptorch import MXFP8_E4M3, NVFP4, AccumulateAlgorithm, BinaryK
from mptorch.quant import BlockPacked, block_matmul, block_pack

torch.manual_seed(0)
M, K, N = 16, 256, 12
a = torch.randn(M, K) * torch.logspace(-3, 3, K)  # magnitudes that change along K
b = torch.randn(K, N)

# 1. Under an E8M0 scale a decoded MX value is an element (four significant
#    bits for E4M3) times a power of two, so a product of two is exact.
pa, pb = block_pack(a, MXFP8_E4M3), block_pack(b, MXFP8_E4M3, 0)
A, B = pa.unpack(), pb.unpack()
exact = A.double()[:, :, None] * B.double()[None, :, :]
print(
    "MXFP8 products exact in binary32:",
    torch.equal((A[:, :, None] * B[None, :, :]).double(), exact),
)


# 2. OCP MX defines the dot product of two blocks as 2^(X_A + X_B) times the
#    sum of the element products, and a longer one as the sum of those. Rebuilt
#    from the codes, with binary32 sums: the elements decoded under a scale of
#    1 (E8M0 code 127), each block's sum scaled, the scaled sums added.
def elements(p):
    ones = torch.full_like(p.scales, 127)
    return BlockPacked(p.data, ones, p.fmt, p.shape, p.axis, 1.0, p.dtype).unpack()


eA, eB = elements(pa), elements(pb)
xA = torch.exp2(pa.scales.float() - 127)  # [M, K / 32]
xB = torch.exp2(pb.scales.float() - 127).T  # b is packed along K: [K / 32, N]
total = torch.zeros(M, N)
for blk in range(K // 32):
    s = torch.zeros(M, N)
    for k in range(32 * blk, 32 * blk + 32):
        s = s + eA[:, k, None] * eB[None, k, :]
    total = total + xA[:, blk, None] * xB[None, blk, :] * s
two_level = block_matmul(pa, pb, accumulate_algorithm=AccumulateAlgorithm.BLOCK, block_size=32)
naive = block_matmul(pa, pb)
print("BLOCK with block_size=32 == OCP's two-level sum:", torch.equal(two_level, total))
print(f"NAIVE, one running sum, differs in {int((naive != total).sum())} of {M * N} elements")

# 3. NVFP4's scale is an E4M3 value times a float32 tensor scale. By default a
#    decoded element carries both, e * (s * S_t), so the decode and the
#    product can round; tensor_scale_epilogue=True leaves S_t out of the loop
#    and multiplies the result by S_tA * S_tB once, as NVIDIA's GEMMs do.
qa, qb = block_pack(a, NVFP4), block_pack(b, NVFP4, 0)


def unit(p):  # the elements times their block scales, tensor scale 1
    return BlockPacked(p.data, p.scales, p.fmt, p.shape, p.axis, 1.0, p.dtype)


want = (unit(qa).unpack().double() * qa.tensor_scale) @ (
    unit(qb).unpack().double() * qb.tensor_scale
)  # the product of the operands' exact values
for name, epilogue in (("in the loop", False), ("as an epilogue", True)):
    y = block_matmul(qa, qb, tensor_scale_epilogue=epilogue)
    err = ((y.double() - want).norm() / want.norm()).item()
    print(f"NVFP4, tensor scales {name:<15} relative error {err:.1e}")

# Power-of-two tensor scales commute with every rounding: the two are equal.
ra = block_pack(a, NVFP4, tensor_scale=2.0**-8)
rb = block_pack(b, NVFP4, 0, tensor_scale=2.0**-9)
same = torch.equal(block_matmul(ra, rb), block_matmul(ra, rb, tensor_scale_epilogue=True))
print("power-of-two tensor scales, loop == epilogue:", same)

# Out of the loop, the accumulator sees the products before the tensor scales
# shrink them, so a narrow one needs their range.
acc = BinaryK(16, 11)
for name, epilogue in (("in the loop", False), ("as an epilogue", True)):
    y = block_matmul(qa, qb, acc=acc, tensor_scale_epilogue=epilogue)
    print(f"16-bit accumulator, tensor scales {name:<15} finite: {bool(y.isfinite().all())}")
