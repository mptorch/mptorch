import mptorch
import mptorch.quant as qt
import torch
import torch.nn as nn
from torch.testing import assert_close
import pytest
from tests.markers import available_devices


@pytest.fixture
def signal_q():
    man, exp = 12, 8
    return lambda x: qt.float_quantize(
        x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
    )


@pytest.fixture
def mac_format():
    man, exp = 12, 8
    return mptorch.FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "quant_fwd,quant_bwd", [(True, True), (False, True), (True, False)]
)
def test_qlinear_custom_mm(device, mac_format, signal_q, quant_fwd, quant_bwd):
    formats_q = qt.QAffineFormats(
        fwd_mac=(mac_format, mac_format) if quant_fwd else None,
        bwd_mac=(mac_format, mac_format) if quant_bwd else None,
        fwd_rnd="nearest",
        bwd_rnd="nearest",
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    x = torch.randn(11, 1034)
    m = torch.nn.Linear(1034, 542, bias=True)
    qm = qt.QLinear(1034, 542, formats=formats_q, bias=True)
    m = m.to(device)
    qm = qm.to(device)
    x = x.to(device)
    qx = x.clone().detach()
    m.weight.data = qm.weight.data.clone().detach()
    m.bias.data = qm.bias.data.clone().detach()

    res_m = m(x)
    res_qm = qm(qx)
    assert_close(res_m, res_qm, atol=1e-2, rtol=0.0)

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()
    assert_close(m.bias.grad, qm.bias.grad, atol=1e-3, rtol=0.0)
    assert_close(m.weight.grad, qm.weight.grad, atol=1e-3, rtol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_qlinear_default_mm(device, signal_q):
    formats_q = qt.QAffineFormats(
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    x = torch.randn(11, 1034)
    m = torch.nn.Linear(1034, 542, bias=True)
    qm = qt.QLinear(1034, 542, formats=formats_q, bias=True)
    m = m.to(device)
    qm = qm.to(device)
    x = x.to(device)
    qx = x.clone().detach()
    m.weight.data = qm.weight.data.clone().detach()
    m.bias.data = qm.bias.data.clone().detach()

    res_m = m(x)
    res_qm = qm(qx)
    assert_close(res_m, res_qm, atol=1e-3, rtol=0.0)

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()
    assert_close(m.bias.grad, qm.bias.grad, atol=1e-4, rtol=0.0)
    assert_close(m.weight.grad, qm.weight.grad, atol=1e-4, rtol=0.0)
