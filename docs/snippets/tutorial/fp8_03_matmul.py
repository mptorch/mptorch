"""One matmul, several arithmetics: what an FP8 tensor core does and does not round."""

import warnings

import torch

from mptorch import BinaryK, FormatRangeWarning, RoundMode
from mptorch.quant import FusedMac, Quant, SplitMac, qmatmul

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
M, K, N = 512, 1024, 512
x = torch.randn(M, K, device=device)
w = torch.randn(K, N, device=device) / K**0.5
exact = (x.double() @ w.double()).float()

e4m3 = Quant(BinaryK(8, 4, bias=7))
xq, wq = e4m3(x), e4m3(w)  # what an FP8 tensor core is fed

# An 8-exponent-bit binaryK reaches below 2**-126, where the casts cannot tell
# one input from another, so building one warns (concepts, "What float32 can
# carry"). Nothing here goes anywhere near that small.
warnings.simplefilter("ignore", FormatRangeWarning)

bf16 = BinaryK(16, 8)  # 8 exponent + 7 mantissa bits: holds an E4M3 x E4M3 product exactly
fp22 = BinaryK(23, 15)  # 8 exponent + 14 mantissa bits: a reduced-precision accumulator
fp16 = BinaryK(16, 11)
e4m3_fmt = BinaryK(8, 4, bias=7)


def rel(out):
    return f"{((out - exact).norm() / exact.norm()).item():.2e}"


print(f"{'operands':<10}{'arithmetic':<52}{'rel. error':>12}")
print(f"{'float32':<10}{'torch.matmul':<52}{rel(x @ w):>12}")
print(f"{'E4M3':<10}{'torch.matmul (float32 products and sums)':<52}{rel(xq @ wq):>12}")
fused32 = qmatmul(xq, wq, FusedMac(None))
print(f"{'E4M3':<10}{'FusedMac(None): sequential fp32 FMAs':<52}{rel(fused32):>12}")
configs = {
    "SplitMac(bf16, fp22): exact products, 14-bit sums": SplitMac(bf16, fp22),
    "SplitMac(bf16, fp16): exact products, fp16 sums": SplitMac(bf16, fp16),
    "SplitMac(bf16, bf16): exact products, bf16 sums": SplitMac(bf16, bf16),
    "SplitMac(bf16, bf16) with stochastic sums": SplitMac(
        BinaryK(16, 8, prng_bits=8), BinaryK(16, 8, prng_bits=8), rounding=RoundMode.SR
    ),
    "SplitMac(E4M3, fp22): products rounded to E4M3 too": SplitMac(e4m3_fmt, fp22),
    "FusedMac(E4M3): everything in E4M3": FusedMac(e4m3_fmt),
}
for name, mac in configs.items():
    print(f"{'E4M3':<10}{name:<52}{rel(qmatmul(xq, wq, mac)):>12}")
