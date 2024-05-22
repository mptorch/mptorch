import mptorch
import mptorch.quant as qt
import torch
import torch.nn as nn

seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = "cuda" if torch.cuda.is_available() else "cpu"
man, exp = 23, 8
signal_q = lambda x: qt.float_quantize(
    x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
)
mac_format = mptorch.FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
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

groups = 4


def test_qconv1d_custom_mm():

    x = torch.randn(1, 4, 240)
    m = nn.Conv1d(4, 4, 12, groups=groups, bias=True)
    qm = qt.QConv1d(4, 4, 12, formats=formats_q, groups=groups, bias=True)
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
    error = torch.max(torch.abs(res_m - res_qm) / torch.abs(res_m)).item()

    err_grad_bias = torch.max(
        torch.abs(
            (m.bias.grad.view(qm.bias.grad.shape) - qm.bias.grad)
            / m.bias.grad.view(qm.bias.grad.shape)
        )
    ).item()

    err_grad_weight = torch.max(
        torch.abs((m.weight.grad.view(qm.weight.grad.shape) - qm.weight.grad))
    ).item()
    assert err_grad_bias < 1e-4
    assert err_grad_weight < 1e-7

    res_m = m(x)
    res_qm = qm(qx)
    err_fwd = torch.max(torch.abs(res_m - res_qm) / torch.abs(res_m)).item()
    assert err_fwd < 1e-2


def test_qconv1d_default_mm():
    formats_q = qt.QAffineFormats(
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    x = torch.randn(1, 4, 160)
    m = nn.Conv1d(4, 4, 12, groups=groups, bias=True)
    qm = qt.QConv1d(4, 4, 12, formats=formats_q, groups=groups, bias=True)
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
    error = torch.max(torch.abs(res_m - res_qm) / torch.abs(res_m)).item()

    err_grad_bias = torch.max(
        torch.abs(
            (m.bias.grad.view(qm.bias.grad.shape) - qm.bias.grad)
            / m.bias.grad.view(qm.bias.grad.shape)
        )
    ).item()

    err_grad_weight = torch.max(
        torch.abs((m.weight.grad.view(qm.weight.grad.shape) - qm.weight.grad))
    ).item()
    assert err_grad_bias < 1e-8
    assert err_grad_weight < 1e-8

    res_m = m(x)
    res_qm = qm(qx)
    err_fwd = torch.max(torch.abs(res_m - res_qm) / torch.abs(res_m)).item()
    assert err_fwd < 1e-8
