"""The schema tier: one function per op, every argument spelled out."""

import warnings

import torch

from mptorch import FormatRangeWarning, RoundMode, SaturationMode
from mptorch.quant import (
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_mixed,
    superfp_matmul,
    superfp_matmul_fma,
)

# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so naming one warns (concepts, "What float32 can
# carry"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

torch.manual_seed(0)
a = torch.randn(4, 16)
b = torch.randn(3, 16)  # will be read transposed

# E4M3 multiply, bfloat16-like accumulate, no gradient tracking.
out = binaryK_matmul(a, b, trans_b=True, mul_K=8, mul_P=4, acc_K=16, acc_P=8)
cos = torch.nn.functional.cosine_similarity
print(
    "binaryK_matmul      ",
    tuple(out.shape),
    f"cos to float32: {cos(out.flatten(), (a @ b.T).flatten(), dim=0):.4f}",
)

# trans_* flags apply to the last two dimensions, with no copy for a view.
print(
    "== a @ b.T          ",
    torch.equal(out, binaryK_matmul(a, b.T, mul_K=8, mul_P=4, acc_K=16, acc_P=8)),
)

# Fused, saturating, round-to-zero.
out = binaryK_matmul_fma(
    a,
    b,
    trans_b=True,
    fma_K=8,
    fma_P=4,
    rounding_mode=RoundMode.RZ,
    saturation_mode=SaturationMode.SAT_FINITE,
)
print("binaryK_matmul_fma  ", tuple(out.shape))

# superfp needs its bias spelled out: the format has no default rule for one.
out = superfp_matmul(
    a, b, trans_b=True, mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=2, mul_bias=7
)
print("superfp_matmul      ", tuple(out.shape))
out = superfp_matmul_fma(
    a, b, trans_b=True, fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=2, fma_bias=7
)
print("superfp_matmul_fma  ", tuple(out.shape))

# The palette ops take per-entry sequences and a prec_idx map.
prec_idx = torch.tensor([[0], [1], [0], [1]])
out = binaryK_matmul_mixed(a, b, prec_idx, trans_b=True, mul_K=[8, 8], mul_P=[4, 3])
print("binaryK_matmul_mixed", tuple(out.shape))

# None of them carries a gradient; that is qmatmul's job.
try:
    binaryK_matmul(a.requires_grad_(True), b, trans_b=True, mul_K=8, mul_P=4)
except RuntimeError as e:
    print("\nRuntimeError:", str(e)[:120], "...")
