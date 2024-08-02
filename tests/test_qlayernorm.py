import torch
import mptorch
import pytest
from torch.nn import functional as F
from mptorch import FloatingPoint
from mptorch.quant import QLayerNorm
from mptorch.quant import QLayerNormFormats
from mptorch.quant import float_quantize
from tests.markers import available_devices

@pytest.fixture(scope="module")
def fp_format():
    return FloatingPoint(exp=8, man=23, subnormals=True, saturate=False)

@pytest.fixture(scope="module")
def quant_fp(fp_format):
    return lambda x: float_quantize(
        x,
        exp=fp_format.exp,
        man=fp_format.man,
        rounding="nearest",
        subnormals=True,
        saturate=False,
    )

@pytest.fixture(scope="module")
def norm_formats(fp_format, quant_fp):
    return QLayerNormFormats(
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

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(20, 30, 40)])
@pytest.mark.parametrize("normalized_shape", [[30, 40], [40]])
def test_qlayer_norm_custom(device, shape, normalized_shape, norm_formats):
    layer = torch.nn.LayerNorm(normalized_shape, 1e-5, True, True, device)
    qlayer = QLayerNorm(normalized_shape, norm_formats, 1e-5, True, True).to(device)

    x_ref = torch.rand(*shape, device=device, requires_grad=True)
    x_res = x_ref.clone().detach()
    x_res.requires_grad_(True)

    y_ref = layer.forward(x_ref)
    y_res = qlayer.forward(x_res)
    torch.testing.assert_close(y_res, y_ref, atol=1e-5, rtol=1e-5)

    grad = torch.rand(*shape, device=device)
    y_ref.backward(grad)
    y_res.backward(grad)
    torch.testing.assert_close(x_res.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(layer.bias.grad, qlayer.bias.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(layer.weight.grad, qlayer.weight.grad, atol=1e-4, rtol=1e-4)

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(20, 30, 40)])
@pytest.mark.parametrize("normalized_shape", [[30, 40], [40]])
def test_qlayer_norm_normal(device, shape, normalized_shape):
    layer = torch.nn.LayerNorm(normalized_shape, 1e-5, True, True, device)
    qlayer = QLayerNorm(normalized_shape, QLayerNormFormats(), 1e-5, True, True).to(device)

    x_ref = torch.rand(*shape, device=device, requires_grad=True)
    x_res = x_ref.clone().detach()
    x_res.requires_grad_(True)

    y_ref = layer.forward(x_ref)
    y_res = qlayer.forward(x_res)
    torch.testing.assert_close(y_res, y_ref, atol=1e-8, rtol=1e-8)

    grad = torch.rand(*shape, device=device)
    y_ref.backward(grad)
    y_res.backward(grad)
    torch.testing.assert_close(x_res.grad, x_ref.grad, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(layer.bias.grad, qlayer.bias.grad, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(layer.weight.grad, qlayer.weight.grad, atol=1e-8, rtol=1e-8)
