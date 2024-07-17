import mptorch
import mptorch.quant as qt
import torch
import torch.nn as nn
from torch.testing import assert_close


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
dilation = 2
stride = 4
out_padding = 0


def test_qconvtranspose1d_custom_mm():

    x = torch.randn(1, 4, 60)
    m = nn.ConvTranspose1d(
        4,
        4,
        (12),
        groups=groups,
        bias=True,
        stride=stride,
        dilation=dilation,
        output_padding=out_padding,
    )
    qm = qt.QConvTranspose1d(
        4,
        4,
        (12),
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


def test_qconvtranspose1d_default_mm():
    formats_q = qt.QAffineFormats(
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    x = torch.randn(1, 4, 60)
    m = nn.ConvTranspose1d(
        4,
        4,
        (12),
        groups=groups,
        bias=True,
        stride=stride,
        dilation=dilation,
        output_padding=out_padding,
    )
    qm = qt.QConvTranspose1d(
        4,
        4,
        (12),
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
    err_fwd = torch.max(torch.abs(res_m - res_qm) / torch.abs(res_m)).item()
    assert res_m.shape == res_qm.shape
    assert_close(res_m, res_qm, atol=0.0, rtol=1e-8)
