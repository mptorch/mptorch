from mptorch.quant import QAffineFormats
from mptorch import FloatingPoint, FixedPoint
from mptorch.quant.functional import qlinear, qmm, qmatmul
from mptorch.quant import QLinear
import torch
from torch.testing import assert_close
from unittest.mock import MagicMock, PropertyMock
import pytest


def test_scaling_format_deduction():
    fp_format = FloatingPoint(exp=3, man=4)
    formats = QAffineFormats(
        fwd_mac=fp_format,
        bwd_mac=fp_format,
        input_quant=(fp_format, "nearest"),
        output_quant=(fp_format, "nearest"),
        weight_quant=(fp_format, "nearest"),
        bias_quant=(fp_format, "nearest"),
        grad_quant=(fp_format, "nearest")
    )
    assert formats.input_scaled_format.exp == 3
    assert formats.input_scaled_format.man == 4
    assert formats.weight_scaled_format.exp == 3
    assert formats.weight_scaled_format.man == 4
    assert formats.grad_scaled_format.exp == 3
    assert formats.grad_scaled_format.man == 4


def test_scaling_format_deduction_non_float():
    fp_format = FixedPoint(3, 4)
    formats = QAffineFormats(
        fwd_mac=fp_format,
        bwd_mac=fp_format,
        input_quant=(fp_format, "nearest"),
        output_quant=(fp_format, "nearest"),
        weight_quant=(fp_format, "nearest"),
        bias_quant=(fp_format, "nearest"),
        grad_quant=(fp_format, "nearest")
    )
    assert formats.input_scaled_format is None
    assert formats.weight_scaled_format is None
    assert formats.grad_scaled_format is None


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("mm", [lambda x,w,fmt: qlinear(x,w,None,fmt), qmatmul, qmm])
def test_scaling_is_used(device, mm):
    mac_format = FloatingPoint(exp=5, man=10)
    input_scaled_format = MagicMock()
    weight_scaled_format = MagicMock()
    grad_scaled_format = MagicMock()
    normal_max_input = PropertyMock(name="normal_max_input", return_value=256.0)
    normal_max_weight = PropertyMock(name="normal_max_weight", return_value=256.0)
    normal_max_grad = PropertyMock(name="normal_max_grad", return_value=256.0)
    type(input_scaled_format).normal_max = normal_max_input
    type(weight_scaled_format).normal_max = normal_max_weight
    type(grad_scaled_format).normal_max = normal_max_grad
    formats = QAffineFormats(
        fwd_mac=mac_format,
        bwd_mac=mac_format,
        input_scaled_format=input_scaled_format,
        weight_scaled_format=weight_scaled_format,
        grad_scaled_format=grad_scaled_format
    )

    assert not formats.fwd_use_default_prec
    assert not formats.bwd_use_default_prec

    normal_max_input.assert_not_called()
    normal_max_weight.assert_not_called()
    normal_max_grad.assert_not_called()

    x = torch.rand(10, 10, dtype=torch.float32, device=device, requires_grad=True)
    w = torch.rand(10, 10, dtype=torch.float32, device=device)
    y = mm(x, w, formats)

    normal_max_input.assert_called_once()
    normal_max_weight.assert_called_once()
    normal_max_grad.assert_not_called()

    y.backward(torch.rand(10, 10, dtype=torch.float32, device=device))

    normal_max_input.assert_called_once()
    normal_max_weight.assert_called_once()
    normal_max_grad.assert_called_once()


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("exp,man", [(8,23), (5,10), (4,3)])
def test_qlinear_custom_mm_scaled(device, exp, man):
    mac_format = FloatingPoint(exp=8, man=23, subnormals=True, saturate=False)
    fp_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
    formats_q = QAffineFormats(
        fwd_mac=(mac_format,),
        bwd_mac=(mac_format,),
        fwd_rnd="nearest",
        bwd_rnd="nearest",
        weight_quant=(fp_format, "nearest"),
        grad_quant=(fp_format, "nearest"),
        output_quant=(fp_format, "nearest"),
        input_quant=(fp_format, "nearest"),
        bias_quant=(fp_format, "nearest")
    )
    x = torch.randn(11, 1034)
    m = torch.nn.Linear(1034, 542, bias=True)
    qm = QLinear(1034, 542, formats=formats_q, bias=True)
    m = m.to(device)
    qm = qm.to(device)
    x = x.to(device)
    qx = x.clone().detach()
    m.weight.data = qm.weight.data.clone().detach()
    m.bias.data = qm.bias.data.clone().detach()

    res_m = m(x)
    res_qm = qm(qx)
    assert_close(res_m, res_qm, atol=1e-1, rtol=0.0)

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()
    assert_close(m.bias.grad, qm.bias.grad, atol=1e-1, rtol=0.0)
    assert_close(m.weight.grad, qm.weight.grad, atol=1e-1, rtol=0.0)