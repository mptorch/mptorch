import torch
import pytest
from torch.nn import functional as F
from mptorch import FloatingPoint
from mptorch.quant import QAffineFormats
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

layer_formats = QAffineFormats(
    fwd_mac=(fp_format),
    fwd_rnd="nearest",
    bwd_mac=(fp_format),
    bwd_rnd="nearest",
    weight_quant=quant_fp,
    input_quant=quant_fp,
    output_quant=quant_fp,
    grad_quant=quant_fp,
    bias_quant=quant_fp,
)

@pytest.mark.parametrize("normalized_shape", [[30, 40], [40]])
def test_qlayer_norm_forward(normalized_shape):
    a = torch.randn(20, 30, 40, device="cpu")
    ref = F.layer_norm(a, normalized_shape, weight=None, bias=None, eps=1e-5)
    res = Q.qlayernorm(a, normalized_shape, weight=None, bias=None, eps=1e-5, formats=layer_formats)

    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

@pytest.mark.parametrize("normalized_shape", [[30, 40], [40]])
def test_qlayer_norm_backward(normalized_shape):
    a = torch.randn(20, 30, 40, device="cpu")

    ref_x = torch.randn(20, 30, 40, device="cpu", requires_grad=True)
    res_x = ref_x.clone().detach()
    res_x.requires_grad_(True)

    ref_y = F.layer_norm(ref_x, normalized_shape, weight=None, bias=None, eps=1e-5)
    res_y = Q.qlayernorm(res_x, normalized_shape, weight=None, bias=None, eps=1e-5, formats=layer_formats)

    ref_y.backward(a)
    res_y.backward(a)

    torch.testing.assert_close(ref_x.grad, res_x.grad, atol=1e-5, rtol=0)