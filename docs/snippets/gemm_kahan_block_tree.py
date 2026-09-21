"""KAHAN, BLOCK and TREE on the operands of gemm_accumulation.py."""

import warnings

import torch

from mptorch import AccumulateAlgorithm, BinaryK, FormatRangeWarning
from mptorch.quant import FusedMac, SplitMac, qmatmul

NAIVE, KAHAN, BLOCK, TREE = AccumulateAlgorithm

torch.manual_seed(0)
M, K, N = 64, 4096, 64
a = torch.randn(M, K) / 4
b = torch.randn(K, N) / 4
# The reference is computed in float64, on the CPU.
exact = a.double() @ b.double()

warnings.simplefilter("ignore", FormatRangeWarning)
e4m3 = BinaryK(8, 4)
bf16 = BinaryK(16, 8)  # 8 exponent, 7 mantissa bits


def report(title, configs):
    print(f"{title:<42}{'rel. error':>12}{'max |err|':>12}")
    for name, mac in configs.items():
        err = qmatmul(a, b, mac).double() - exact
        print(f"  {name:<40}{err.norm() / exact.norm():>12.2e}{err.abs().max():>12.2e}")
    print()


# Nothing rounded but float32 itself: what each summation order is worth.
report(
    "float32 products and sums",
    {
        "NAIVE": FusedMac(None),
        "KAHAN": FusedMac(None, accumulate_algorithm=KAHAN),
        "BLOCK, 64 per block": FusedMac(None, accumulate_algorithm=BLOCK, block_size=64),
    },
)

# E4M3 products into a bfloat16-like accumulator, which NAIVE loses bits in at
# every one of the 4096 steps.
report(
    "E4M3 products, 8-bit-precision sums",
    {
        "NAIVE": SplitMac(e4m3, bf16),
        "KAHAN": SplitMac(e4m3, bf16, accumulate_algorithm=KAHAN),
        "BLOCK, 16 per block": SplitMac(e4m3, bf16, accumulate_algorithm=BLOCK, block_size=16),
        "BLOCK, 64 per block": SplitMac(e4m3, bf16, accumulate_algorithm=BLOCK, block_size=64),
        "BLOCK, 64 per block, outer=bf16": SplitMac(
            e4m3, bf16, accumulate_algorithm=BLOCK, block_size=64, outer=bf16
        ),
        "TREE, 16 per block": SplitMac(e4m3, bf16, accumulate_algorithm=TREE),
        "TREE, 256 per block": SplitMac(e4m3, bf16, accumulate_algorithm=TREE, block_size=256),
        "TREE, 256 per block, outer=bf16": SplitMac(
            e4m3, bf16, accumulate_algorithm=TREE, block_size=256, outer=bf16
        ),
        "the products' own error (acc=None)": SplitMac(e4m3, None),
    },
)

# The same with an accumulator as narrow as the products.
report(
    "E4M3 products, E4M3 sums",
    {
        "NAIVE": SplitMac(e4m3, e4m3),
        "KAHAN": SplitMac(e4m3, e4m3, accumulate_algorithm=KAHAN),
        "BLOCK, 4 per block": SplitMac(e4m3, e4m3, accumulate_algorithm=BLOCK, block_size=4),
        "BLOCK, 4 per block, outer=bf16": SplitMac(
            e4m3, e4m3, accumulate_algorithm=BLOCK, block_size=4, outer=bf16
        ),
        "TREE, 4 per block": SplitMac(e4m3, e4m3, accumulate_algorithm=TREE, block_size=4),
        "TREE, 4 per block, outer=bf16": SplitMac(
            e4m3, e4m3, accumulate_algorithm=TREE, block_size=4, outer=bf16
        ),
    },
)

# The three are binary32-only for now, and say so.
try:
    qmatmul(a.double(), b.double(), SplitMac(e4m3, bf16, accumulate_algorithm=KAHAN))
except ValueError as error:
    print("float64 operands:", str(error).split(", so they")[0])
