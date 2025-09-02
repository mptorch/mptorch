import torch
import pytest
from mptorch import FloatingPoint
from mptorch.quant import QBatchNorm2d
from mptorch.quant import float_quantize
from tests.markers import available_devices


@pytest.fixture(scope="module")
def fp_format():
    return FloatingPoint(exp=8, man=22, subnormals=True, saturate=False)


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


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("batch_size", [100])
@pytest.mark.parametrize("num_features", [100])
def test_qbatchnorm2d_custom(device, batch_size, num_features, quant_fp):
    layer = torch.nn.BatchNorm2d(num_features, affine=True, device=device)
    qlayer = QBatchNorm2d(num_features, quant_fp, quant_fp).to(device)

    shape = (batch_size, num_features, 100, 100)
    x_ref = torch.randn(*shape, device=device, requires_grad=True)
    x_res = x_ref.clone().detach()
    x_res.requires_grad_(True)

    y_ref = layer.forward(x_ref)
    y_res = qlayer.forward(x_res)
    torch.testing.assert_close(y_res, y_ref, atol=1e-5, rtol=1e-5)

    grad = torch.rand(*shape, device=device)
    y_ref.backward(grad)
    y_res.backward(grad)
    torch.testing.assert_close(x_res.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

    torch.testing.assert_close(layer.bias.grad, qlayer.bias.grad, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(
        layer.weight.grad, qlayer.weight.grad, atol=1e-3, rtol=1e-3
    )
