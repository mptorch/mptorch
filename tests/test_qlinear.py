import mptorch
import mptorch.quant as qt
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
man, exp = 12, 8
signal_q = lambda x: qt.float_quantize(
    x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
)
mac_format = mptorch.FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
formats_q = qt.QAffineFormats(
    fwd_mac=(mac_format, mac_format),
    bwd_mac=(mac_format, mac_format),
    fwd_rnd="nearest",
    bwd_rnd="nearest",
    weight_quant=signal_q,
    grad_quant=signal_q,
    output_quant=signal_q,
    input_quant=signal_q,
    bias_quant=signal_q,
)


def test_qlinear_custom_mm():

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
    err_fwd = torch.max(torch.abs(res_m - res_qm)).item()
    assert err_fwd < 1e-2

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()

    err_grad_bias = torch.max(
        torch.abs(
            (m.bias.grad.view(qm.bias.grad.shape) - qm.bias.grad)
            / m.bias.grad.view(qm.bias.grad.shape)
        )
    ).item()

    err_grad_weight = torch.max(
        torch.abs((m.weight.grad.view(qm.weight.grad.shape) - qm.weight.grad))
    ).item()

    res_m = m(x)
    res_qm = qm(qx)
    assert err_grad_bias < 1e-3
    assert err_grad_weight < 1e-3


def test_qlinear_default_mm():
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
    err_fwd = torch.max(torch.abs(res_m - res_qm)).item()
    assert err_fwd < 1e-3

    res_m = m(x).mean()
    res_m.backward()
    res_qm = qm(qx).mean()
    res_qm.backward()

    err_grad_bias = torch.max(
        torch.abs(
            (m.bias.grad.view(qm.bias.grad.shape) - qm.bias.grad)
            / m.bias.grad.view(qm.bias.grad.shape)
        )
    ).item()

    err_grad_weight = torch.max(
        torch.abs((m.weight.grad.view(qm.weight.grad.shape) - qm.weight.grad))
    ).item()

    res_m = m(x)
    res_qm = qm(qx)
    assert err_grad_bias < 1e-4
    assert err_grad_weight < 1e-4
