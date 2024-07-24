import torch
import pytest
from torch.nn import functional as F
from mptorch import FloatingPoint
from mptorch.quant import QLayerNorm
from mptorch.quant import QLayerNormFormats
from mptorch.quant import float_quantize
from mptorch.quant import functional as Q

fp_format = FloatingPoint(
    exp=8, man=23, subnormals=True, saturate=False
)

quant_fp = lambda x: float_quantize(
    x,
    exp=8,
    man=23,
    rounding="nearest",
    subnormals=True,
    saturate=False,
)

layer_formats = QLayerNormFormats(
    fwd_acc = fp_format,
    fwd_mul = fp_format,
    fwd_div = fp_format,
    fwd_sqrt = fp_format,

    bwd_acc = fp_format,
    bwd_mul = fp_format,
    bwd_div = fp_format,

    input_quant = quant_fp,
    output_quant = quant_fp,
    grad_quant = quant_fp,
    weight_quant = quant_fp,
    bias_quant = quant_fp,
)

@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("shape", [(30, 40, 50)])
@pytest.mark.parametrize("normalized_shape", [[40, 50], [50]])
def test_qlayer_norm(device, shape, normalized_shape):
    layer = torch.nn.LayerNorm(normalized_shape, 1e-5, False, False)
    qlayer = QLayerNorm(normalized_shape, layer_formats, 1e-5, False, False)

    x_ref = torch.rand(*shape, device=device, requires_grad=True)
    x_res = x_ref.clone().detach()
    x_res.requires_grad_(True)

    y_ref = layer.forward(x_ref)
    y_res = qlayer.forward(x_res)
    torch.testing.assert_close(y_res, y_ref, atol=2e-5, rtol=0.0)

    grad = torch.rand(*shape, device=device)
    y_ref.backward(grad)
    y_res.backward(grad)
    torch.testing.assert_close(x_res.grad, x_ref.grad, atol=2e-5, rtol=0.0)
