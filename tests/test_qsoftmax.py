import torch
import torch.nn
from mptorch import FloatingPoint
from mptorch.quant import QSoftmaxFormats
from mptorch.quant import float_quantize
from mptorch.quant import functional as Q
from mptorch.quant import QSoftmax
import pytest
from tests.markers import available_devices


@pytest.fixture(scope="module")
def fp_format():
    return FloatingPoint(exp=5, man=10, subnormals=True, saturate=False)

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
def formats_div(fp_format, quant_fp):
    return QSoftmaxFormats(
        fwd_exp=fp_format,
        fwd_off=fp_format,
        fwd_acc=fp_format,

        bwd_add=fp_format,
        bwd_mul=fp_format,

        output_quant=quant_fp,
        input_quant=quant_fp,
        grad_quant=quant_fp,
    )

@pytest.fixture(scope="module")
def formats_lse(fp_format, quant_fp):
    return QSoftmaxFormats(
        fwd_off=fp_format,
        fwd_lse=fp_format,

        bwd_add=fp_format,
        bwd_mul=fp_format,

        output_quant=quant_fp,
        input_quant=quant_fp,
        grad_quant=quant_fp,
    )

def test_softmax_formats(formats_lse, formats_div):
    assert not formats_lse.fwd_use_default_prec
    assert not formats_lse.bwd_use_default_prec
    assert not formats_div.fwd_use_default_prec
    assert not formats_div.bwd_use_default_prec
    assert formats_lse.use_lse
    assert not hasattr(formats_lse, "fwd_exp")
    assert not hasattr(formats_lse, "fwd_acc")
    assert not formats_div.use_lse
    assert hasattr(formats_div, "fwd_exp")
    assert hasattr(formats_div, "fwd_acc")

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(20, 30, 40)])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("fmt", ["div", "lse"])
def test_custom_softmax(device, shape, dim, fmt, formats_div, formats_lse):
    if fmt == "div":
        formats = formats_div
    else:
        formats = formats_lse
    
    layer = torch.nn.Softmax(dim)
    qlayer = QSoftmax(dim, formats)

    x_ref = torch.rand(*shape, device=device, requires_grad=True)
    x_res = x_ref.clone().detach()
    x_res.requires_grad_(True)

    y_ref = layer.forward(x_ref)
    y_res = qlayer.forward(x_res)
    torch.testing.assert_close(y_res, y_ref, atol=1e-3, rtol=0.0)

    grad = torch.rand(*shape, device=device)
    y_ref.backward(grad)
    y_res.backward(grad)
    torch.testing.assert_close(x_res.grad, x_ref.grad, atol=1e-3, rtol=0.0)

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("shape", [(20, 30, 40)])
@pytest.mark.parametrize("dim", [2])
def test_default_softmax(device, shape, dim):
    layer = torch.nn.Softmax(dim)
    qlayer = QSoftmax(dim, QSoftmaxFormats())

    x_ref = torch.rand(*shape, device=device, requires_grad=True)
    x_res = x_ref.clone().detach()
    x_res.requires_grad_(True)

    y_ref = layer.forward(x_ref)
    y_res = qlayer.forward(x_res)
    torch.testing.assert_close(y_res, y_ref, atol=1e-8, rtol=0.0)

    grad = torch.rand(*shape, device=device)
    y_ref.backward(grad)
    y_res.backward(grad)
    torch.testing.assert_close(x_res.grad, x_ref.grad, atol=1e-8, rtol=0.0)