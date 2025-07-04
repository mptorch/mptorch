import mptorch
import mptorch.quant as qt
import torch
import torch.nn as nn
from torch.testing import assert_close
import pytest
from tests.markers import available_devices

@pytest.fixture
def signal_q():
    man, exp = 23, 8
    return lambda x: qt.float_quantize(
        x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
    )

@pytest.fixture
def mac_format():
    man, exp = 23, 8
    return mptorch.FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("dilation", [2])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("out_padding", [0])
def test_qconvtranspose3d_custom_mm(groups, device, dilation, stride, out_padding, mac_format, signal_q):
    formats_q = qt.QAffineFormats(
        fwd_mac=mac_format,
        bwd_mac=mac_format,
        fwd_rnd="nearest",
        bwd_rnd="nearest",
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    x = torch.randn(1, 4, 12, 12, 12)
    m = nn.ConvTranspose3d(
        4,
        4,
        (3, 3, 3),
        groups=groups,
        bias=True,
        stride=stride,
        dilation=dilation,
        output_padding=out_padding,
    )
    qm = qt.QConvTranspose3d(
        4,
        4,
        (3, 3, 3),
        formats=formats_q,
        groups=groups,
        bias=True,
        stride=stride,
        dilation=dilation,
        output_padding=out_padding,
    )
    m = m.to(device)
    qm = qm.to(device)
    x = x.to(device)
    qx = x.clone().detach()
    m.weight.data = qm.weight.data.clone().detach()
    m.bias.data = qm.bias.data.clone().detach()

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()
    assert_close(m.bias.grad, qm.bias.grad, atol=0.0, rtol=1e-3)
    assert_close(m.weight.grad, qm.weight.grad, atol=0.0, rtol=1e-3)

    res_m = m(x)
    res_qm = qm(qx)
    assert res_m.shape == res_qm.shape
    assert_close(res_m, res_qm, atol=0.0, rtol=1e-2)

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("groups", [2])
@pytest.mark.parametrize("dilation", [2])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("out_padding", [0])
def test_qconvtranspose3d_default_mm(groups, device, dilation, stride, out_padding, signal_q):
    formats_q = qt.QAffineFormats(
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    x = torch.randn(1, 4, 12, 12, 12)
    m = nn.ConvTranspose3d(
        4,
        4,
        (3, 3, 3),
        groups=groups,
        bias=True,
        stride=stride,
        dilation=dilation,
        output_padding=out_padding,
    )
    qm = qt.QConvTranspose3d(
        4,
        4,
        (3, 3, 3),
        formats=formats_q,
        groups=groups,
        bias=True,
        stride=stride,
        dilation=dilation,
        output_padding=out_padding,
    )
    m = m.to(device)
    qm = qm.to(device)
    x = x.to(device)
    qx = x.clone().detach()
    m.weight.data = qm.weight.data.clone().detach()
    m.bias.data = qm.bias.data.clone().detach()

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()
    assert_close(m.bias.grad, qm.bias.grad, atol=0.0, rtol=1e-8)
    assert_close(m.weight.grad, qm.weight.grad, atol=0.0, rtol=1e-8)

    res_m = m(x)
    res_qm = qm(qx)
    assert res_m.shape == res_qm.shape
    assert_close(res_m, res_qm, atol=0.0, rtol=1e-8)