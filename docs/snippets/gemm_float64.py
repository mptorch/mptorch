"""float64 GEMMs: every product, sum and rounding in binary64."""

import torch

from mptorch import BinaryK
from mptorch.quant import FusedMac, SplitMac, binaryK_matmul, qmatmul

torch.manual_seed(0)

# The product is rounded once. (1 + 2**-13)**2 = 1 + 2**-12 + 2**-26 is just
# above the tie between 1 and 1 + 2**-11 on a 12-bit grid. binary32 rounds the
# product to 24 bits before the format sees it -- onto the tie itself, which
# round-to-nearest-even then sends to 1 -- and binary64 keeps it whole.
p12 = SplitMac(BinaryK(16, 12), BinaryK(16, 12))
p12_32 = SplitMac(BinaryK(16, 12), BinaryK(16, 12), carrier="binary32")
one = torch.full((1, 1), 1 + 2**-13)
print("float32 operands            ", qmatmul(one, one, p12).item())
print("float64 operands            ", qmatmul(one.double(), one.double(), p12).item())
print("float64, carrier='binary32' ", qmatmul(one.double(), one.double(), p12_32).item())

# Formats only binary64 carries, against float64's own product.
a = torch.randn(64, 256, dtype=torch.float64)
b = torch.randn(256, 32, dtype=torch.float64)
exact = a @ b


def error(out: torch.Tensor) -> str:
    return f"{(torch.linalg.norm(out - exact) / torch.linalg.norm(exact)).item():.1e}"


wide = SplitMac(BinaryK(40, 30), BinaryK(48, 40))
p24 = BinaryK(31, 24)
macs = [
    ("SplitMac(P=30, P=40)", wide),
    ("FusedMac(P=40)", FusedMac(BinaryK(48, 40))),
    ("SplitMac(P=24, P=24)", SplitMac(p24, p24)),
    ("SplitMac(P=24, P=24), carrier='binary32'", SplitMac(p24, p24, carrier="binary32")),
    ("SplitMac(E4M3, E4M3)", SplitMac(BinaryK(8, 4), BinaryK(8, 4))),
]
print("\nrelative error against a @ b, float64 operands")
for name, mac in macs:
    print(f"  {name:<42}{error(qmatmul(a, b, mac))}")

# The wide formats are refused where binary32 would do the arithmetic.
try:
    qmatmul(a.float(), b.float(), wide)
except ValueError as e:
    print("\nfloat32 operands:", str(e).split(": ")[0])

# carrier="binary32" is float32's arithmetic on float64 operands, bit for bit,
# and so is the flat function's keyword.
e4m3 = SplitMac(BinaryK(8, 4), BinaryK(8, 4))
e4m3_32 = SplitMac(BinaryK(8, 4), BinaryK(8, 4), carrier="binary32")
narrowed = qmatmul(a.float(), b.float(), e4m3).double()
print(
    "\ncarrier='binary32' == the float32 call, widened:",
    torch.equal(qmatmul(a, b, e4m3_32), narrowed),
)
flat = binaryK_matmul(a, b, mul_K=8, mul_P=4, carrier="binary32")
print("binaryK_matmul(..., carrier='binary32') agrees:  ", torch.equal(flat, narrowed))
