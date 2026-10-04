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
binary8p4 = BinaryK(8, 4)
binary16p8 = BinaryK(16, 8)  # bfloat16's field widths: 8 exponent, 7 mantissa bits


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

# Binary8p4 products into a Binary16p8 accumulator, which NAIVE loses bits in at
# every one of the 4096 steps.
report(
    "Binary8p4 products, Binary16p8 sums",
    {
        "NAIVE": SplitMac(binary8p4, binary16p8),
        "KAHAN": SplitMac(binary8p4, binary16p8, accumulate_algorithm=KAHAN),
        "BLOCK, 16 per block": SplitMac(
            binary8p4, binary16p8, accumulate_algorithm=BLOCK, block_size=16
        ),
        "BLOCK, 64 per block": SplitMac(
            binary8p4, binary16p8, accumulate_algorithm=BLOCK, block_size=64
        ),
        "BLOCK, 64 per block, outer=Binary16p8": SplitMac(
            binary8p4, binary16p8, accumulate_algorithm=BLOCK, block_size=64, outer=binary16p8
        ),
        "TREE, 16 per block": SplitMac(binary8p4, binary16p8, accumulate_algorithm=TREE),
        "TREE, 256 per block": SplitMac(
            binary8p4, binary16p8, accumulate_algorithm=TREE, block_size=256
        ),
        "TREE, 256 per block, outer=Binary16p8": SplitMac(
            binary8p4, binary16p8, accumulate_algorithm=TREE, block_size=256, outer=binary16p8
        ),
        "the products' own error (acc=None)": SplitMac(binary8p4, None),
    },
)

# The same with an accumulator as narrow as the products.
report(
    "Binary8p4 products, Binary8p4 sums",
    {
        "NAIVE": SplitMac(binary8p4, binary8p4),
        "KAHAN": SplitMac(binary8p4, binary8p4, accumulate_algorithm=KAHAN),
        "BLOCK, 4 per block": SplitMac(
            binary8p4, binary8p4, accumulate_algorithm=BLOCK, block_size=4
        ),
        "BLOCK, 4 per block, outer=Binary16p8": SplitMac(
            binary8p4, binary8p4, accumulate_algorithm=BLOCK, block_size=4, outer=binary16p8
        ),
        "TREE, 4 per block": SplitMac(
            binary8p4, binary8p4, accumulate_algorithm=TREE, block_size=4
        ),
        "TREE, 4 per block, outer=Binary16p8": SplitMac(
            binary8p4, binary8p4, accumulate_algorithm=TREE, block_size=4, outer=binary16p8
        ),
    },
)

# The three are binary32-only for now, and say so.
try:
    qmatmul(a.double(), b.double(), SplitMac(binary8p4, binary16p8, accumulate_algorithm=KAHAN))
except ValueError as error:
    print("float64 operands:", str(error).split(", so they")[0])
