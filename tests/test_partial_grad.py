"""
Tests for the case where a quantized layer is asked for only *some* of its
gradients -- a frozen block, or the first layer of a network.

These guard the narrowing done for findings P1/P2/P3 in
dev/gemm_perf_audit.md: backward now quantizes ``grad_output`` only for the
paths that will consume it, forward saves only the tensors backward reads,
and the GEMM ``fwd`` hooks fold the bias in place. Each is only sound because
of an exact claim about what backward touches, so the tests check the claims
rather than just the numbers: which quantizers ran, and what the graph node
actually holds.
"""

import functools

import pytest
import torch
from torch import nn

from mptorch.number import RoundMode
from mptorch.quant import (
    QAffineFormats,
    QConv1d,
    QConv2d,
    QConv3d,
    QLinear,
    binaryK_gemm_formats,
    binaryK_matmul,
)
from tests.markers import available_devices

# (input.requires_grad, weight/bias.requires_grad)
PATTERNS = [(True, True), (True, False), (False, True)]


class CountingQuant:
    """Identity quantizer that records how many times it was called."""

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return x.clone()


def counting_formats():
    quants = {
        name: CountingQuant()
        for name in (
            "weight_quant",
            "input_quant",
            "bias_quant",
            "wgrad_quant",
            "igrad_quant",
            "bgrad_quant",
        )
    }
    return QAffineFormats(**quants), quants


def _build(kind, device, bias, formats):
    """A quantized layer and its vanilla twin, sharing parameters."""
    if kind == "linear":
        q = QLinear(8, 6, bias=bias, formats=formats, device=device)
        v = nn.Linear(8, 6, bias=bias, device=device)
        x = torch.randn(4, 8, device=device)
    elif kind == "conv1d":
        q = QConv1d(3, 5, 3, padding=1, bias=bias, formats=formats, device=device)
        v = nn.Conv1d(3, 5, 3, padding=1, bias=bias, device=device)
        x = torch.randn(2, 3, 7, device=device)
    elif kind == "conv2d":
        q = QConv2d(3, 5, 3, padding=1, bias=bias, formats=formats, device=device)
        v = nn.Conv2d(3, 5, 3, padding=1, bias=bias, device=device)
        x = torch.randn(2, 3, 6, 6, device=device)
    else:
        q = QConv3d(3, 5, 3, padding=1, bias=bias, formats=formats, device=device)
        v = nn.Conv3d(3, 5, 3, padding=1, bias=bias, device=device)
        x = torch.randn(2, 3, 4, 4, 4, device=device)

    with torch.no_grad():
        q.weight.copy_(v.weight)
        if bias:
            assert q.bias is not None and v.bias is not None
            q.bias.copy_(v.bias)
    return q, v, x


def identity_formats():
    ident = {
        name: (lambda x: x)
        for name in (
            "weight_quant",
            "input_quant",
            "bias_quant",
            "wgrad_quant",
            "igrad_quant",
            "bgrad_quant",
        )
    }
    return QAffineFormats(**ident)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["linear", "conv1d", "conv2d", "conv3d"])
@pytest.mark.parametrize("rg_in, rg_param", PATTERNS)
def test_partial_grads_match_vanilla(device, kind, rg_in, rg_param):
    """
    With identity quantizers, whichever gradients are requested must still
    match the vanilla layer exactly, and the ones that are not requested must
    come back as None rather than as zeros or stale values.
    """
    q, v, x = _build(kind, device, True, identity_formats())
    q.weight.requires_grad_(rg_param)
    q.bias.requires_grad_(rg_param)

    x_q = x.clone().requires_grad_(rg_in)
    x_v = x.clone().requires_grad_(True)

    out_q = q(x_q)
    out_v = v(x_v)
    assert torch.allclose(out_q, out_v, atol=1e-5)

    g = torch.randn_like(out_v)
    out_q.backward(g)
    out_v.backward(g)

    assert (x_q.grad is not None) == rg_in
    assert (q.weight.grad is not None) == rg_param
    assert (q.bias.grad is not None) == rg_param

    if rg_in:
        assert torch.allclose(x_q.grad, x_v.grad, atol=1e-5)
    if rg_param:
        assert torch.allclose(q.weight.grad, v.weight.grad, atol=1e-5)
        assert torch.allclose(q.bias.grad, v.bias.grad, atol=1e-5)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["linear", "conv2d"])
@pytest.mark.parametrize("rg_in, rg_param", PATTERNS)
def test_grad_output_quantized_only_where_consumed(device, kind, rg_in, rg_param):
    """P1: a gradient quantizer runs exactly when its path runs."""
    formats, quants = counting_formats()
    q, _, x = _build(kind, device, True, formats)
    q.weight.requires_grad_(rg_param)
    q.bias.requires_grad_(rg_param)

    x_q = x.clone().requires_grad_(rg_in)
    q(x_q).backward(torch.randn(q(x_q).shape, device=device))

    assert quants["igrad_quant"].calls == (1 if rg_in else 0)
    assert quants["wgrad_quant"].calls == (1 if rg_param else 0)
    assert quants["bgrad_quant"].calls == (1 if rg_param else 0)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("kind", ["linear", "conv1d", "conv2d", "conv3d"])
@pytest.mark.parametrize("rg_in, rg_param", PATTERNS)
@pytest.mark.parametrize("bias", [True, False])
def test_forward_saves_only_what_backward_reads(device, kind, rg_in, rg_param, bias):
    """
    P2: the graph node holds the quantized input only for the weight-gradient
    path and the quantized weight only for the input-gradient path, and never
    holds a quantized bias.
    """
    q, _, x = _build(kind, device, bias, identity_formats())
    q.weight.requires_grad_(rg_param)
    if bias:
        q.bias.requires_grad_(rg_param)

    out = q(x.clone().requires_grad_(rg_in))
    saved = out.grad_fn.saved_tensors

    assert len(saved) == 2, "q_bias is never read in backward and must not be saved"
    assert (saved[0] is not None) == rg_param, "q_input is only for the weight gradient"
    assert (saved[1] is not None) == rg_in, "q_weight is only for the input gradient"


@pytest.mark.parametrize("device", available_devices)
def test_gemm_bias_fold_matches_out_of_place(device):
    """
    P3: folding the bias into the GEMM output in place is value-identical to
    the out-of-place add it replaced, and does not disturb the operands.
    """
    formats = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=10, acc_P=5)
    layer = QLinear(12, 7, bias=True, formats=formats, device=device)
    x = torch.randn(5, 12, device=device)

    w = layer.weight.detach().clone()
    b = layer.bias.detach().clone()

    got = layer(x)

    matmul = functools.partial(
        binaryK_matmul,
        mul_K=8,
        mul_P=4,
        acc_K=10,
        acc_P=5,
        rounding_mode=RoundMode.RNE,
    )
    expected = matmul(x, w, trans_a=False, trans_b=True) + b

    assert torch.equal(got, expected)
    assert torch.equal(layer.weight.detach(), w)
    assert torch.equal(layer.bias.detach(), b)
