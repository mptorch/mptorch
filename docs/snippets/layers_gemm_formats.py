"""QLinear whose matmuls themselves run in custom arithmetic."""

import torch

from mptorch import BinaryK, RoundMode
from mptorch.quant import (
    QLinear,
    Quant,
    binaryK_gemm_formats,
    binaryK_gemm_formats_fma,
    binaryK_matmul,
    superfp_gemm_formats,
    superfp_gemm_formats_fma,
)

torch.manual_seed(0)

# The factory sets the three math hooks -- forward, input-gradient and
# weight-gradient GEMMs -- to a binaryK core: E4M3 products, float32 sums.
formats = binaryK_gemm_formats(mul_K=8, mul_P=4, accumulate_quant=False)
print(
    "hooks set:",
    [h for h in ("fwd_math", "bwd_igrad_math", "bwd_wgrad_math") if getattr(formats, h)],
)
print("quantizers set:", [q for q in ("weight_quant", "input_quant") if getattr(formats, q)])

# Operand quantization is layered on separately -- the two are composable.
formats.input_quant = Quant(BinaryK(8, 4))
formats.weight_quant = Quant(BinaryK(8, 4))

layer = QLinear(256, 64, formats=formats)
x = torch.randn(32, 256, requires_grad=True)
out = layer(x)
out.sum().backward()

# The forward is exactly the raw op on the quantized operands ...
with torch.no_grad():
    qx, qw = Quant(BinaryK(8, 4))(x), Quant(BinaryK(8, 4))(layer.weight)
    ref = (
        binaryK_matmul(qx, qw, trans_b=True, mul_K=8, mul_P=4, accumulate_quant=False) + layer.bias
    )
print("\nforward == binaryK_matmul(Q(x), Q(W)^T) + b:", torch.equal(out, ref))

# ... and the gradient GEMMs run in the same arithmetic.
with torch.no_grad():
    g = torch.ones_like(out)
    ref_gx = binaryK_matmul(g, qw, mul_K=8, mul_P=4, accumulate_quant=False)
print("input grad == binaryK_matmul(dL/dy, Q(W)):", torch.equal(x.grad, ref_gx))

# Leading batch dims are folded into M for the GEMM and restored afterwards.
out3 = layer(torch.randn(2, 5, 256))
print("\n[2, 5, 256] ->", tuple(out3.shape))

# The four factories: split / fused, binaryK / superfp.

factories = {
    "binaryK_gemm_formats_fma": binaryK_gemm_formats_fma(
        fma_K=8, fma_P=4, rounding_mode=RoundMode.SR, fma_prng_bits=8
    ),
    "superfp_gemm_formats": superfp_gemm_formats(
        mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=2, mul_bias=7
    ),
    "superfp_gemm_formats_fma": superfp_gemm_formats_fma(
        fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=2, fma_bias=7
    ),
}
for name, f in factories.items():
    other = QLinear(256, 64, formats=f)
    other.load_state_dict(layer.state_dict())  # same weights, different arithmetic
    out = other(x)
    print(
        f"{name:<26} -> {tuple(out.shape)}, cos to float32 layer: "
        f"{torch.nn.functional.cosine_similarity(out.flatten(), layer(x).flatten(), dim=0):.4f}"
    )
