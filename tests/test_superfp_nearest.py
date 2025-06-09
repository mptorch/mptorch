import mptorch
import mptorch.quant as qt
import torch
import torch.nn as nn
from torch.testing import assert_close
import pytest
from tests.markers import available_devices


@pytest.fixture
def signal_q():
    man, exp = 22, 8
    return lambda x: qt.superfp_quantize(
        x, exp=exp, man=man, binades=(1, 1), rounding="nearest", saturate=False
    )


@pytest.fixture
def mac_format():
    man, exp = 22, 8
    return mptorch.SuperNormalFloat(exp=exp, man=man, binades=(1, 1), saturate=False)


@pytest.mark.parametrize("device", available_devices)
def test_mm_default(device, signal_q):
    formats_q = qt.QAffineFormats(
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    a = torch.randn(78, 101).to(device)
    b = torch.randn(101, 145).to(device)

    c = torch.mm(a, b)
    qc = qt.mp_mm(a, b, formats_q)

    assert_close(c, qc, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("device", available_devices)
def test_bmm_default(device, signal_q):
    formats_q = qt.QAffineFormats(
        weight_quant=signal_q,
        grad_quant=signal_q,
        output_quant=signal_q,
        input_quant=signal_q,
        bias_quant=signal_q,
    )
    a = torch.randn(40, 78, 101).to(device)
    b = torch.randn(40, 101, 145).to(device)

    c = torch.bmm(a, b)
    qc = qt.mp_bmm(a, b, formats_q)

    assert_close(c, qc, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "quant_fwd,quant_bwd",
    [
        (True, True),
        (False, True),
        (True, False),
    ],
)
def test_mm_custom(device, mac_format, signal_q, quant_fwd, quant_bwd):
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

    a = torch.randn(78, 101).to(device)
    b = torch.randn(101, 145).to(device)

    c = torch.mm(a, b)
    qc = qt.mp_mm(a, b, formats_q, use_forward=quant_fwd)

    assert_close(c, qc, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize(
    "quant_fwd,quant_bwd",
    [
        (True, True),
        (False, True),
        (True, False),
    ],
)
def test_bmm_custom(device, mac_format, signal_q, quant_fwd, quant_bwd):
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

    a = torch.randn(50, 78, 101).to(device)
    b = torch.randn(50, 101, 145).to(device)

    c = torch.bmm(a, b)
    qc = qt.mp_bmm(a, b, formats_q, use_forward=quant_fwd)

    assert_close(c, qc, atol=1e-4, rtol=1e-5)
