"""The in-place quantizers: the same rounding, written over a tensor the caller owns."""

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    Quant,
    binaryK_quantize,
    binaryK_quantize_,
    superfp_quantize,
    superfp_quantize_,
)

# A weight quantized once at load: nothing else reads the unrounded values, so
# the result can take their place. The tensor handed in is the tensor returned.
w = torch.randn(256, 256)
expected = binaryK_quantize(w, K=8, P=4)
out = binaryK_quantize_(w, K=8, P=4)
print(
    "returns its argument:", out is w, "| equals the out-of-place result:", torch.equal(w, expected)
)

# Stochastic rounding too: an element's random word is keyed on its index, so
# under one seed the two spellings draw the same words.
x = torch.randn(10_000)
m3e4 = dict(man_bits=3, exp_bits=4, normal_binades=8, bias=7, prng_bits=8)
torch.manual_seed(0)
expected = superfp_quantize(x, **m3e4, rounding_mode=RoundMode.SR)
torch.manual_seed(0)
superfp_quantize_(x, **m3e4, rounding_mode=RoundMode.SR)
print("superfp, SR, same seed:", torch.equal(x, expected))

# A float64 tensor rounds in binary64 in place, like any dtype in its own
# carrier: 1.0625 + 2**-30 is above E4M3's tie and goes up, where its float32
# neighbour is the tie itself.
d = torch.tensor([1.0625 + 2**-30], dtype=torch.float64)
print("float64:", binaryK_quantize_(d.clone(), K=8, P=4).tolist(), end="  ")
print("float32:", binaryK_quantize_(d.float(), K=8, P=4).tolist())

# What the out-of-place functions handle by copying, these refuse: the kernel
# would round the copy and leave the caller's tensor as it was.
m = torch.randn(4, 4)
try:
    binaryK_quantize_(m.t(), K=8, P=4)
except RuntimeError as e:
    print("\na transposed view:", str(e).split(":")[0])
try:
    binaryK_quantize_(m.half(), K=8, P=4, carrier=torch.float64)
except ValueError as e:
    print("a wider carrier:", str(e).split(":")[0])
if torch.cuda.is_available():
    g = torch.randn(64, device="cuda")
    try:
        binaryK_quantize_(g[1:], K=8, P=4)
    except RuntimeError as e:
        print("a CUDA view off a 16-byte boundary:", str(e).split(";")[0].split(", so ")[1])
    binaryK_quantize_(g[16:], K=8, P=4)  # 64 bytes in: on the boundary
    print("g[16:] is rounded, g[:16] is not:", g[:16].mul(16).frac().ne(0).any().item())

# The write is one autograd knows about. a * s saves s for a's gradient, and
# rounding s before backward() is caught rather than differentiated against.
a = torch.randn(8, requires_grad=True)
s = torch.randn(8)
loss = (a * s).sum()
binaryK_quantize_(s, K=8, P=4)
try:
    loss.backward()
except RuntimeError as e:
    print("\nbackward after the write:", str(e).split(":")[0])

# Quant takes it as a keyword, and it is part of the value.
q = Quant(BinaryK(8, 4), inplace=True)
y = torch.randn(1000)
expected = Quant(BinaryK(8, 4))(y)
print("\nQuant(..., inplace=True):", q(y) is y, torch.equal(y, expected), q == Quant(BinaryK(8, 4)))

# What it saves is the result's allocation, not time: the kernel reads and
# writes the same bytes either way.
if torch.cuda.is_available():
    big = torch.randn(1 << 24, device="cuda")

    def peak(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.memory_allocated()
        r = fn()
        torch.cuda.synchronize()
        del r
        return (torch.cuda.max_memory_allocated() - start) / 2**20

    out_of_place = peak(lambda: binaryK_quantize(big, 8, 4))
    in_place = peak(lambda: binaryK_quantize_(big, 8, 4))
    print(
        f"\npeak above a 64 MiB float32 input: {out_of_place:.0f} MiB, in place {in_place:.0f} MiB"
    )
