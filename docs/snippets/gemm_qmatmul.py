"""qmatmul: torch.matmul's contract, in the arithmetic you name."""

import torch

from mptorch import BinaryK
from mptorch.quant import QMatmulFormats, SplitMac, qbmm, qmatmul, qmm

torch.manual_seed(0)
e4m3 = BinaryK(8, 4)

# Every formats spelling, from the loosest to the most specific.
a = torch.randn(4, 8)
b = torch.randn(8, 3)
print("None (plain torch.matmul)  ", torch.equal(qmatmul(a, b), a @ b))
print(
    "BinaryK == SplitMac(f, f)  ",
    torch.equal(qmatmul(a, b, e4m3), qmatmul(a, b, SplitMac(e4m3, e4m3))),
)
print("QMatmulFormats() (nothing) ", torch.equal(qmatmul(a, b, QMatmulFormats()), a @ b))

# The operand rules are torch.matmul's: batches, broadcasting, 1D promotion.
cases = [
    ("2D @ 2D          ", (4, 8), (8, 3)),
    ("batched          ", (5, 4, 8), (5, 8, 3)),
    ("broadcast b      ", (5, 4, 8), (8, 3)),
    ("broadcast leading", (2, 1, 4, 8), (3, 8, 3)),
    ("1D @ 2D          ", (8,), (8, 3)),
    ("2D @ 1D          ", (4, 8), (8,)),
    ("1D @ 1D          ", (8,), (8,)),
]
for name, sa, sb in cases:
    x, y = torch.randn(*sa), torch.randn(*sb)
    out = qmatmul(x, y, e4m3)
    print(
        f"{name} {str(tuple(sa)):<14} @ {str(tuple(sb)):<12} -> {tuple(out.shape)}"
        f"   shape as torch: {out.shape == (x @ y).shape}"
    )

# qmm / qbmm are the rank-checked spellings, like torch.mm / torch.bmm.
print("\nqmm  ", torch.equal(qmm(a, b, e4m3), qmatmul(a, b, e4m3)))
x3, y3 = torch.randn(5, 4, 8), torch.randn(5, 8, 3)
print("qbmm ", torch.equal(qbmm(x3, y3, e4m3), qmatmul(x3, y3, e4m3)))
try:
    qbmm(a, b, e4m3)
except ValueError as e:
    print("qbmm on 2D:", e)

# It is differentiable in both operands. With no format the gradients are
# torch's own; with one, each gradient is a GEMM in that arithmetic.
a.requires_grad_(True)
b.requires_grad_(True)
qmatmul(a, b).sum().backward()
ga, gb = a.grad.clone(), b.grad.clone()
a.grad = b.grad = None
(a @ b).sum().backward()
print("\nno format: grads equal torch's:", torch.equal(ga, a.grad), torch.equal(gb, b.grad))

a.grad = b.grad = None
qmatmul(a, b, e4m3).sum().backward()
cos = torch.nn.functional.cosine_similarity
print("E4M3:      cos(grad_a, torch) =", f"{cos(a.grad.flatten(), ga.flatten(), dim=0):.4f}")
