"""An FP8 training recipe for one layer: E4M3 forward, E5M2 backward, and scaling."""

import torch

from mptorch import BinaryK, SaturationMode
from mptorch.quant import QAffineFormats, QLinear, Quant, binaryK_gemm_formats

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

E4M3 = BinaryK(8, 4, bias=7)
E5M2 = BinaryK(8, 3, bias=15)


def fp8_formats(*, gemm_acc: BinaryK | None, loss_scale: float = 1.0) -> QAffineFormats:
    """Forward signals in E4M3, backward signals in E5M2, matmuls with the given accumulator.

    Products of two FP8 numbers are exact in a bfloat16-wide multiplier, so
    ``mul_K=16, mul_P=8`` rounds nothing a tensor core would keep; ``gemm_acc`` is what
    the running sum is rounded to (``None``: float32).
    """
    formats = binaryK_gemm_formats(
        mul_K=16,
        mul_P=8,
        acc_K=gemm_acc.K if gemm_acc else None,
        acc_P=gemm_acc.P if gemm_acc else None,
        accumulate_quant=gemm_acc is not None,
    )
    fwd, bwd = Quant(E4M3), Quant(E5M2)
    formats.input_quant, formats.weight_quant, formats.bias_quant = fwd, fwd, fwd
    formats.igrad_quant, formats.wgrad_quant, formats.bgrad_quant = bwd, bwd, bwd
    return formats


layer = QLinear(784, 128, formats=fp8_formats(gemm_acc=None), device=device)
ref = torch.nn.Linear(784, 128, device=device)
ref.load_state_dict(layer.state_dict())

x = torch.randn(64, 784, device=device, requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)
g = torch.randn(64, 128, device=device) * 1e-3  # a gradient of realistic size

layer(x).backward(g)
ref(x_ref).backward(g)
cos = torch.nn.functional.cosine_similarity
print("cosine similarity to the float32 layer")
print(f"  output       {cos(layer(x).flatten(), ref(x_ref).flatten(), dim=0):.5f}")
print(f"  input grad   {cos(x.grad.flatten(), x_ref.grad.flatten(), dim=0):.5f}")
print(f"  weight grad  {cos(layer.weight.grad.flatten(), ref.weight.grad.flatten(), dim=0):.5f}")

# Why gradients get E5M2 and not E4M3: their range. Rounded to E4M3 (smallest
# subnormal 2**-9 ~ 0.002) most of this gradient is simply zero.
for name, fmt in (("E4M3", E4M3), ("E5M2", E5M2)):
    zeroed = (Quant(fmt)(g) == 0).float().mean().item()
    print(f"\n{name}: {zeroed:.1%} of a gradient of size ~1e-3 rounds to zero")


# The usual remedy is a scale: multiply into range, round, divide back out.
# A quantizer slot takes any callable, so this is a few lines.
class ScaledQuant:
    """Per-tensor scaling to the format's largest finite value (like an FP8 'amax' scale)."""

    def __init__(self, fmt: BinaryK, max_finite: float):
        self.q = Quant(BinaryK(fmt.K, fmt.P, bias=fmt.bias, saturation=SaturationMode.SAT_FINITE))
        self.max_finite = max_finite

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        scale = self.max_finite / t.abs().amax().clamp_min(1e-30)
        return self.q(t * scale) / scale


for name, fmt, mx in (("E4M3", E4M3, 448.0), ("E5M2", E5M2, 57344.0)):
    sq = ScaledQuant(fmt, mx)
    zeroed = (sq(g) == 0).float().mean().item()
    print(
        f"scaled {name}: {zeroed:.1%} of the same gradient rounds to zero, "
        f"cos to unrounded {cos(sq(g).flatten(), g.flatten(), dim=0):.5f}"
    )
