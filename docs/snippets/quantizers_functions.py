"""The two elementwise quantization functions."""

import torch

from mptorch import RoundMode
from mptorch.quant import binaryK_quantize, superfp_quantize

torch.manual_seed(0)
torch.set_printoptions(precision=5)

x = torch.randn(6)
print("x           ", x)
print("E4M3, RNE   ", binaryK_quantize(x, K=8, P=4))
print("E4M3, RZ    ", binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.RZ))
print("E5M2, RNE   ", binaryK_quantize(x, K=8, P=3))
print("superfp     ", superfp_quantize(x, man_bits=3, exp_bits=4, normal_binades=2, bias=7))

# Every floating-point dtype torch trains in is accepted. The rounding is done
# in float32 and the result comes back in the input's dtype.
for dtype in (torch.float64, torch.float16, torch.bfloat16):
    y = binaryK_quantize(x.to(dtype), K=8, P=4)
    print(f"{str(dtype):<14}", y.dtype, y.tolist())

# Stochastic rounding takes its random bits from below the target mantissa,
# inside the storage dtype's own mantissa -- so the two have to fit together.
# bfloat16 has 7 mantissa bits: 3 for E4M3 leaves room for at most 4.
try:
    binaryK_quantize(x.to(torch.bfloat16), K=8, P=4, prng_bits=8, rounding_mode=RoundMode.SR)
except AssertionError as e:
    print("\nAssertionError:", e)

# CPU and CUDA give the same result under every deterministic rounding mode.
if torch.cuda.is_available():
    big = torch.randn(1_000_000)
    for mode in (RoundMode.RNE, RoundMode.RZ, RoundMode.RO):
        same = torch.equal(
            binaryK_quantize(big, K=8, P=4, rounding_mode=mode),
            binaryK_quantize(big.cuda(), K=8, P=4, rounding_mode=mode).cpu(),
        )
        print(f"CPU == CUDA under {mode.name}: {same}")
