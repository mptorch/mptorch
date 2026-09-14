"""A layer's carrier: binary64 for a float64 layer, and for a float32 one that names it."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import QLinear, Quant, binaryK_gemm_formats

torch.manual_seed(0)
F32, F64 = torch.float32, torch.float64

# A float64 reference whose weights and inputs are float32 values, so a float32
# layer holds them exactly and what separates it from the reference is its
# arithmetic alone.
reference = nn.Linear(256, 32, dtype=F64)
with torch.no_grad():
    for p in reference.parameters():
        p.copy_(p.float())
x = torch.randn(64, 256).double()
g = torch.randn(64, 32).double()  # the gradient of the output

ref_out = reference(x)
ref_out.backward(g)
assert reference.weight.grad is not None
ref_grad = reference.weight.grad


def run(formats, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """A QLinear of ``dtype`` with the reference's weights: its output and weight gradient."""
    layer = QLinear(256, 32, formats=formats, dtype=dtype)
    layer.load_state_dict(reference.state_dict())
    out = layer(x.to(dtype))
    out.backward(g.to(dtype))
    assert layer.weight.grad is not None
    return out.detach().double(), layer.weight.grad.double()


def rel(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"{(torch.linalg.norm(a - b) / torch.linalg.norm(b)).item():.1e}"


# Arithmetic only binary64 carries: 30-bit products into a 40-bit accumulator,
# in the forward product and both gradient products, on a float64 layer.
wide = binaryK_gemm_formats(40, 30, acc_K=48, acc_P=40)
out, grad = run(wide, F64)
print(f"float64 layer, 30/40-bit: output {rel(out, ref_out)}, weight grad {rel(grad, ref_grad)}")

# A float32 layer cannot have it in binary32, nor store a 40-bit result in
# binary64 -- each says so at the first call.
for carrier in (None, F64):
    try:
        run(binaryK_gemm_formats(40, 30, acc_K=48, acc_P=40, carrier=carrier), F32)
    except ValueError as e:
        print(f"float32 layer, carrier={carrier}:", str(e).split(": ")[0])

# What binary64 gives a float32 layer is everything before the stored result: a
# 30-bit multiply format under a 24-bit accumulator, which binary32 refuses.
out, grad = run(binaryK_gemm_formats(40, 30, acc_K=31, acc_P=24, carrier=F64), F32)
print(f"float32 layer, 30/24-bit, binary64: output {rel(out, ref_out)}, grad {rel(grad, ref_grad)}")


# And it changes the answer where the arithmetic sees bits it does not keep. An
# FP8 recipe -- E4M3 signals, E5M2 gradients, E4M3 arithmetic -- rounds every
# signal to 8 bits first, so each product and sum is exact in binary32 as in
# binary64 and the two carriers agree to the bit. A 22-bit arithmetic on
# unquantized operands does not: binary32 rounds each product and sum to 24
# bits before the formats round them to 22 and 23, where binary64 keeps 53.
def fp8(carrier: torch.dtype | None):
    formats = binaryK_gemm_formats(8, 4, carrier=carrier)
    formats.input_quant = formats.weight_quant = Quant(BinaryK(8, 4), carrier=carrier)
    formats.igrad_quant = formats.wgrad_quant = Quant(BinaryK(8, 3, bias=15), carrier=carrier)
    return formats


def bits22(carrier: torch.dtype | None):
    return binaryK_gemm_formats(26, 22, acc_K=27, acc_P=23, carrier=carrier)


print()
for label, build in (("FP8 recipe", fp8), ("22/23-bit arithmetic", bits22)):
    out32, grad32 = run(build(None), F32)
    out64, grad64 = run(build(F64), F32)
    print(
        f"float32 layer, {label}: error {rel(out32, ref_out)} in binary32, "
        f"{rel(out64, ref_out)} in binary64"
    )
    diff = f"output {rel(out64, out32)}, weight grad {rel(grad64, grad32)}"
    print(f"  binary64 against binary32: {diff}")
