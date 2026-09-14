"""A float64 layer computes in binary64: every quantizer and every product."""

import torch
from torch import nn

from mptorch import BinaryK
from mptorch.quant import QLinear, Quant, binaryK_gemm_formats

torch.manual_seed(0)
F64 = torch.float64

reference = nn.Linear(256, 32, dtype=F64)
x = torch.randn(64, 256, dtype=F64, requires_grad=True)


def run(formats) -> tuple[torch.Tensor, torch.Tensor]:
    """A QLinear with the reference's weights: its output and weight gradient."""
    layer = QLinear(256, 32, formats=formats, dtype=F64)
    layer.load_state_dict(reference.state_dict())
    out = layer(x)
    out.backward(torch.ones_like(out))
    assert layer.weight.grad is not None
    return out.detach(), layer.weight.grad


ref_out = reference(x)
ref_out.backward(torch.ones_like(ref_out))
assert reference.weight.grad is not None
ref_grad = reference.weight.grad


def rel(a: torch.Tensor, b: torch.Tensor) -> str:
    return f"{(torch.linalg.norm(a - b) / torch.linalg.norm(b)).item():.1e}"


# Arithmetic only binary64 carries: 30-bit products into a 40-bit accumulator,
# in the forward product and both gradient products.
wide = binaryK_gemm_formats(40, 30, acc_K=48, acc_P=40)
out, grad = run(wide)
print(f"30/40-bit arithmetic:  output {rel(out, ref_out)}, weight grad {rel(grad, ref_grad)}")

# The same layer in binary32 cannot have it, and says so at the first call.
try:
    run(binaryK_gemm_formats(40, 30, acc_K=48, acc_P=40, carrier="binary32"))
except ValueError as e:
    print("with carrier='binary32':", str(e).split(": ")[0])


# The carrier matters where the arithmetic sees bits it does not keep. An FP8
# recipe -- E4M3 signals, E5M2 gradients, E4M3 arithmetic -- rounds every
# signal to 8 bits first, so each product and sum is exact in binary32 as in
# binary64 and the two carriers agree to the bit. A 22-bit arithmetic on
# unquantized float64 operands does not: binary32 narrows each operand, and
# rounds each product and sum, to 24 bits before the formats round them to 22
# and 23, where binary64 keeps 53.
def fp8(carrier: str | None):
    formats = binaryK_gemm_formats(8, 4, carrier=carrier)
    formats.input_quant = formats.weight_quant = Quant(BinaryK(8, 4), carrier=carrier)
    formats.igrad_quant = formats.wgrad_quant = Quant(BinaryK(8, 3, bias=15), carrier=carrier)
    return formats


def bits22(carrier: str | None):
    return binaryK_gemm_formats(26, 22, acc_K=27, acc_P=23, carrier=carrier)


print()
for label, build in (("FP8 recipe", fp8), ("22/23-bit arithmetic", bits22)):
    x.grad = None
    out64, grad64 = run(build(None))
    x.grad = None
    out32, grad32 = run(build("binary32"))
    print(f"{label}: error {rel(out64, ref_out)} in binary64, {rel(out32, ref_out)} in binary32")
    diff = f"output {rel(out64, out32)}, weight grad {rel(grad64, grad32)}"
    print(f"  binary64 against binary32: {diff}")
