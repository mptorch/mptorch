from mptorch.quant import QAffineFormats
from mptorch import FloatingPoint, FixedPoint
from mptorch.quant.functional import qlinear, qmm, qmatmul
from mptorch.quant import QLinear
import torch
from torch.testing import assert_close
from unittest.mock import MagicMock, PropertyMock
from tests.markers import available_devices
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
        grad_quant=(fp_format, "nearest"),
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
        grad_quant=(fp_format, "nearest"),
    )
    assert formats.input_scaled_format is None
    assert formats.weight_scaled_format is None
    assert formats.grad_scaled_format is None


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "mm", [lambda x, w, fmt: qlinear(x, w, None, fmt), qmatmul, qmm]
)
@pytest.mark.parametrize("use_scaling", [True, False])
def test_scaling_is_used(device, mm, use_scaling):
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
        input_scaled_format=input_scaled_format,
        weight_scaled_format=weight_scaled_format,
        grad_scaled_format=grad_scaled_format,
        use_scaling=use_scaling,
    )

    normal_max_input.assert_not_called()
    normal_max_weight.assert_not_called()
    normal_max_grad.assert_not_called()

    x = torch.rand(10, 10, dtype=torch.float32, device=device, requires_grad=True)
    w = torch.rand(10, 10, dtype=torch.float32, device=device)
    y = mm(x, w, formats)

    if use_scaling:
        normal_max_input.assert_called_once()
        normal_max_weight.assert_called_once()
    else:
        normal_max_input.assert_not_called()
        normal_max_weight.assert_not_called()
    normal_max_grad.assert_not_called()

    y.backward(torch.rand(10, 10, dtype=torch.float32, device=device))

    if use_scaling:
        normal_max_input.assert_called_once()
        normal_max_weight.assert_called_once()
        normal_max_grad.assert_called_once()
    else:
        normal_max_input.assert_not_called()
        normal_max_weight.assert_not_called()
        normal_max_grad.assert_not_called()


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("exp_1, man_1, exp_2, man_2", [(4, 3, 5, 2), (5, 10, 5, 10)])
def test_qlinear_custom_mm_scaled(device, exp_1, man_1, exp_2, man_2):
    fp_format_1 = FloatingPoint(exp=exp_1, man=man_1, subnormals=True, saturate=False)
    fp_format_2 = FloatingPoint(exp=exp_2, man=man_2, subnormals=True, saturate=False)
    formats_q = QAffineFormats(
        weight_quant=(fp_format_1, "nearest"),
        grad_quant=(fp_format_2, "nearest"),
        input_quant=(fp_format_1, "nearest"),
        use_scaling=True,
    )
    x = torch.randn(45, 431)
    m = torch.nn.Linear(431, 542, bias=True)
    qm = QLinear(431, 542, formats=formats_q, bias=True)
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
