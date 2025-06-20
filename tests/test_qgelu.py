import torch
from mptorch.quant.functional import qgelu
from mptorch.quant import QGELU
from mptorch.quant import QGELUFormats
from mptorch.quant import float_quantize
from torch.testing import assert_close
import pytest
from tests.markers import available_devices


@pytest.fixture
def signal_q():
    man, exp = 12, 8
    return lambda x: float_quantize(
        x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
    )


@pytest.fixture
def formats(signal_q):
    return QGELUFormats(
        input_quant=signal_q,
        inter_quant=signal_q,
        output_quant=signal_q,
        grad_quant=signal_q,
    )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_qgelu_custom(device, approximate, formats):
    layer = torch.nn.GELU(approximate=approximate)
    qlayer = QGELU(formats=formats)
    layer = layer.to(device)
    qlayer = qlayer.to(device)

    x = torch.randn(127, 31, requires_grad=True)
    x = x.to(device)
    x.retain_grad()  # this isn't supposed to be required?
    qx = x.clone().detach()
    qx = qx.requires_grad_(True)
    qx.retain_grad()

    expected = layer(x)
    actual = qlayer(qx)
    assert_close(expected, actual, atol=1e-2, rtol=0.0)

    expected = expected.mean()
    expected.backward()
    actual = actual.mean()
    actual.backward()
    assert_close(qx.grad, x.grad, atol=1e-2, rtol=0.0)
