"""The two elementwise quantization functions."""

import warnings

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

# Every floating-point dtype torch trains in is accepted. A float64 tensor is
# rounded in binary64, the others in binary32, and the result comes back in the
# input's dtype.
for dtype in (torch.float64, torch.float16, torch.bfloat16):
    y = binaryK_quantize(x.to(dtype), K=8, P=4)
    print(f"{str(dtype):<14}", y.dtype, y.tolist())

# The two carriers disagree where float32 cannot hold the input: 1.0625 is the
# tie between 1.0 and 1.125, and the 2**-30 above it is lost when a float64
# tensor is narrowed to float32 -- which is what carrier="binary32" asks for.
t = torch.tensor([1.0625 + 2**-30], dtype=torch.float64)
print(
    f"\n1.0625 + 2**-30: {binaryK_quantize(t, K=8, P=4).item()} in binary64,",
    f"{binaryK_quantize(t, K=8, P=4, carrier='binary32').item()} in binary32",
)

# Storing the result in float16 or bfloat16 rounds it once more, so the format
# has to fit that dtype too. The inputs are already float16 values, so only an
# edge of the format's range can miss: E5M2 as spelled here reaches 98304, and
# an input near float16's top rounds up to 65536 -- which float16 stores as
# infinity, whatever the format's saturation mode says.
x16 = torch.tensor([60000.0], dtype=torch.float16)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    y16 = binaryK_quantize(x16, K=8, P=3, bias=15, rounding_mode=RoundMode.RU)
y32 = binaryK_quantize(x16.float(), K=8, P=3, bias=15, rounding_mode=RoundMode.RU)
print(f"\nE5M2, RU, of 60000: {y32.item()} in float32, {y16.item()} in float16")
print(f"{caught[0].category.__name__}: {caught[0].message}")

# Stochastic rounding takes its random bits from below the target mantissa, in
# the carrier the rounding happens in, so the format's mantissa and its random
# bits share the carrier's: binary64's 52 bits for float64, binary32's 23 for
# the rest.
y = binaryK_quantize(x.double(), K=8, P=4, prng_bits=21, rounding_mode=RoundMode.SR)
print("\nfloat64, 3 + 21 bits:", y.tolist())
try:
    binaryK_quantize(x.to(torch.bfloat16), K=8, P=4, prng_bits=21, rounding_mode=RoundMode.SR)
except ValueError as e:
    print("bfloat16: ValueError:", e)

# CPU and CUDA give the same result under every deterministic rounding mode.
if torch.cuda.is_available():
    big = torch.randn(1_000_000)
    for mode in (RoundMode.RNE, RoundMode.RZ, RoundMode.RO):
        same = torch.equal(
            binaryK_quantize(big, K=8, P=4, rounding_mode=mode),
            binaryK_quantize(big.cuda(), K=8, P=4, rounding_mode=mode).cpu(),
        )
        print(f"CPU == CUDA under {mode.name}: {same}")
