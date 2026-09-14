"""binary64 GEMMs: every product, sum and rounding in binary64, on any operands."""

import torch

from mptorch import BinaryK
from mptorch.quant import FusedMac, SplitMac, binaryK_matmul, qmatmul

F32, F64 = torch.float32, torch.float64
torch.manual_seed(0)

# The product is rounded once. (1 + 2**-13)**2 = 1 + 2**-12 + 2**-26 is just
# above the tie between 1 and 1 + 2**-11 on a 12-bit grid. binary32 rounds the
# product to 24 bits before the format sees it -- onto the tie itself, which
# round-to-nearest-even then sends to 1 -- and binary64 keeps it whole, for
# float64 operands and for float32 ones that name it.
p12 = SplitMac(BinaryK(16, 12), BinaryK(16, 12))
p12_64 = SplitMac(BinaryK(16, 12), BinaryK(16, 12), carrier=F64)
one = torch.full((1, 1), 1 + 2**-13)
print("float32 operands                ", qmatmul(one, one, p12).item())
print("float64 operands                ", qmatmul(one.double(), one.double(), p12).item())
out = qmatmul(one, one, p12_64)
print("float32, carrier=torch.float64  ", out.item(), out.dtype)

# Formats only binary64 carries, against float64's own product.
a = torch.randn(64, 256, dtype=F64)
b = torch.randn(256, 32, dtype=F64)


def error(out: torch.Tensor, exact: torch.Tensor) -> str:
    return f"{(torch.linalg.norm(out.double() - exact) / torch.linalg.norm(exact)).item():.1e}"


wide = SplitMac(BinaryK(40, 30), BinaryK(48, 40))
p24 = BinaryK(31, 24)
macs = [
    ("SplitMac(P=30, P=40)", wide),
    ("FusedMac(P=40)", FusedMac(BinaryK(48, 40))),
    ("SplitMac(P=24, P=24)", SplitMac(p24, p24)),
    ("SplitMac(E4M3, E4M3)", SplitMac(BinaryK(8, 4), BinaryK(8, 4))),
]
print("\nrelative error against a @ b, float64 operands")
for name, mac in macs:
    print(f"  {name:<24}{error(qmatmul(a, b, mac), a @ b)}")

# Float32 operands, against the exact product of their values, in each carrier.
# A result stored in float32 holds at most 24 bits, so what binary64 buys is
# everything before the last rounding: sums that do not lose a bit per step, and
# a multiply format wider than the result.
a32, b32 = a.float(), b.float()
exact32 = a32.double() @ b32.double()
macs32 = [
    ("SplitMac(P=24, None)", lambda c: SplitMac(p24, None, carrier=c)),
    ("FusedMac(None)", lambda c: FusedMac(None, carrier=c)),
    ("SplitMac(P=30, P=24)", lambda c: SplitMac(BinaryK(40, 30), p24, carrier=c)),
    ("SplitMac(E4M3, E4M3)", lambda c: SplitMac(BinaryK(8, 4), BinaryK(8, 4), carrier=c)),
]
print(f"\nfloat32 operands            {'binary32':<10}carrier=torch.float64")
for name, build in macs32:
    row = []
    for carrier in (None, F64):
        try:
            row.append(error(qmatmul(a32, b32, build(carrier)), exact32))
        except ValueError:
            row.append("raises")
    print(f"  {name:<26}{row[0]:<10}{row[1]}")

# The wide formats are refused where binary32 does the arithmetic, and where a
# float32 result would have to store them.
try:
    qmatmul(a32, b32, wide)
except ValueError as e:
    print("\nfloat32 operands:", str(e).split(": ")[0])
try:
    qmatmul(a32, b32, SplitMac(BinaryK(40, 30), BinaryK(48, 40), carrier=F64))
except ValueError as e:
    print("float32, carrier=torch.float64:", str(e).split(": ")[0])

# carrier=torch.float64 is the float64 call on the widened operands, narrowed,
# and so is the flat function's keyword; the result keeps the operands' dtype.
# (E4M3's values are float16's too, so `.to` narrows them exactly.)
e4m3_64 = SplitMac(BinaryK(8, 4), BinaryK(8, 4), carrier=F64)
print()
for dtype in (F32, torch.float16):
    ad, bd = a.to(dtype), b.to(dtype)
    got = qmatmul(ad, bd, e4m3_64)
    same = torch.equal(got, qmatmul(ad.double(), bd.double(), e4m3_64).to(dtype))
    print(f"{str(dtype):<13} carrier=torch.float64 == the float64 call, narrowed: {same}")
flat = binaryK_matmul(a32, b32, mul_K=8, mul_P=4, carrier=F64)
agrees = torch.equal(flat, qmatmul(a32, b32, e4m3_64))
print("binaryK_matmul(..., carrier=torch.float64) agrees:", agrees)

# A carrier is never narrower than the operands.
try:
    qmatmul(a, b, SplitMac(BinaryK(8, 4), BinaryK(8, 4), carrier=F32))
except ValueError as e:
    print("\ncarrier=torch.float32 on float64:", str(e).split(";")[0])
