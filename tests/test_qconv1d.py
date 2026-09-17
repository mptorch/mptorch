"""
``QConv1d`` against ``nn.Conv1d`` and against a hand-written baseline.

The three tiers every layer test module follows. Tier 1 (exact): identity
quantizers, and the quantized layer must match the vanilla layer's forward
and backward to within float32 noise, which validates the autograd plumbing
with no numerics in the way. Tier 2 (statistical): real ``binaryK_quantize``
quantizers on every tensor, and the outputs and gradients must stay close to
the vanilla layer's in cosine similarity, over float32, float16 and
bfloat16. Tier 3 (manual baseline): the expected result is recomputed by
hand from the formats' own quantizers, ``F.conv1d`` for the forward and
``torch.nn.grad.conv1d_input`` / ``conv1d_weight`` for the backward, so
the layer must quantize exactly the tensors it claims to and nowhere else.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad

from mptorch.number import RoundMode
from mptorch.quant import QAffineFormats, QConv1d, binaryK_quantize
from tests.markers import available_devices


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("bias", [True, False])
def test_qconv1d_tier1_exact(device, bias):
    """
    Tier 1: with identity quantizers on every tensor, ``QConv1d`` matches a
    vanilla ``nn.Conv1d`` in forward and in all three gradients.
    """
    dtype = torch.float32

    formats = QAffineFormats(
        weight_quant=lambda x: x,
        input_quant=lambda x: x,
        bias_quant=lambda x: x,
        igrad_quant=lambda x: x,
        wgrad_quant=lambda x: x,
        bgrad_quant=lambda x: x,
    )

    vanilla = nn.Conv1d(4, 8, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)
    q_layer = QConv1d(
        4, 8, kernel_size=3, padding=1, bias=bias, formats=formats, device=device, dtype=dtype
    )

    with torch.no_grad():
        q_layer.weight.copy_(vanilla.weight)
        if bias:
            assert vanilla.bias is not None and q_layer.bias is not None
            q_layer.bias.copy_(vanilla.bias)

    x_v = torch.randn(2, 4, 16, device=device, dtype=dtype, requires_grad=True)
    x_q = x_v.clone().detach().requires_grad_(True)

    out_v = vanilla(x_v)
    out_q = q_layer(x_q)

    assert torch.allclose(out_v, out_q, atol=1e-5), "Forward pass divergence!"

    loss_v = out_v.sum()
    loss_q = out_q.sum()

    loss_v.backward()
    loss_q.backward()

    assert x_v.grad is not None
    assert x_q.grad is not None
    assert vanilla.weight.grad is not None
    assert q_layer.weight.grad is not None
    assert torch.allclose(x_v.grad, x_q.grad, atol=1e-5), "Input grad divergence!"
    assert torch.allclose(vanilla.weight.grad, q_layer.weight.grad, atol=1e-5), (
        "Weight grad divergence!"
    )
    if bias:
        assert vanilla.bias is not None and vanilla.bias.grad is not None
        assert q_layer.bias is not None and q_layer.bias.grad is not None
        assert torch.allclose(vanilla.bias.grad, q_layer.bias.grad, atol=1e-5), (
            "Bias grad divergence!"
        )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_qconv1d_tier2_statistical(device, dtype, bias):
    """
    Tier 2: with a real K=8, P=4 quantizer on every tensor, the output and
    both gradients stay close to the vanilla layer's in cosine similarity.
    """

    def quant_fn(x):
        return binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.RNE)

    formats = QAffineFormats(
        weight_quant=quant_fn,
        input_quant=quant_fn,
        bias_quant=quant_fn,
        igrad_quant=quant_fn,
        wgrad_quant=quant_fn,
        bgrad_quant=quant_fn,
    )

    vanilla = nn.Conv1d(4, 8, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)
    q_layer = QConv1d(
        4, 8, kernel_size=3, padding=1, bias=bias, formats=formats, device=device, dtype=dtype
    )

    with torch.no_grad():
        q_layer.weight.copy_(vanilla.weight)
        if bias:
            assert vanilla.bias is not None and q_layer.bias is not None
            q_layer.bias.copy_(vanilla.bias)

    x_v = torch.randn(2, 4, 16, device=device, dtype=dtype, requires_grad=True)
    x_q = x_v.clone().detach().requires_grad_(True)

    out_v = vanilla(x_v)
    out_q = q_layer(x_q)

    cos_sim_out = F.cosine_similarity(
        out_v.flatten().float(), out_q.flatten().float(), dim=0
    ).item()
    assert cos_sim_out > 0.85, f"Forward Cosine Sim too low: {cos_sim_out}"

    g_out = torch.randn_like(out_v)

    out_v.backward(g_out)
    out_q.backward(g_out)

    assert x_v.grad is not None
    assert x_q.grad is not None
    assert vanilla.weight.grad is not None
    assert q_layer.weight.grad is not None

    cos_sim_igrad = F.cosine_similarity(
        x_v.grad.flatten().float(), x_q.grad.flatten().float(), dim=0
    ).item()
    cos_sim_wgrad = F.cosine_similarity(
        vanilla.weight.grad.flatten().float(), q_layer.weight.grad.flatten().float(), dim=0
    ).item()

    assert cos_sim_igrad > 0.80, f"Input Grad Cosine Sim too low: {cos_sim_igrad}"
    assert cos_sim_wgrad > 0.80, f"Weight Grad Cosine Sim too low: {cos_sim_wgrad}"


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_qconv1d_tier3_manual_baseline(device, dtype, bias):
    """
    Tier 3: the layer computes exactly what a hand-written script does with
    the same quantizers, ``F.conv1d`` forward and ``torch.nn.grad`` backward, so
    the custom Function quantizes exactly the tensors it claims to.
    """

    def quant_fn(x):
        return binaryK_quantize(x, K=8, P=4, rounding_mode=RoundMode.RNE)

    formats = QAffineFormats(
        weight_quant=quant_fn,
        input_quant=quant_fn,
        bias_quant=quant_fn,
        igrad_quant=quant_fn,
        wgrad_quant=quant_fn,
        bgrad_quant=quant_fn,
    )

    q_layer = QConv1d(
        4, 8, kernel_size=3, padding=1, bias=bias, formats=formats, device=device, dtype=dtype
    )

    w = q_layer.weight.clone().detach().requires_grad_(True)
    if bias:
        assert q_layer.bias is not None
        b = q_layer.bias.clone().detach().requires_grad_(True)
    else:
        b = None
    x_man = torch.randn(2, 4, 16, device=device, dtype=dtype, requires_grad=True)
    x_q_layer = x_man.clone().detach().requires_grad_(True)

    # Under no_grad because these quantize tensors that require grad and then
    # read only the values: nothing backprops through the quantizer here, and
    # the raw quantize ops raise on a requires_grad operand under grad mode
    # rather than hand back a tensor whose gradient would silently vanish
    # (csrc/autograd_ops.cpp).
    with torch.no_grad():
        qw = quant_fn(w)
        qx = quant_fn(x_man)
        qb = quant_fn(b) if bias else None
    out_man = F.conv1d(
        qx,
        qw,
        qb,
        stride=q_layer.stride,
        padding=q_layer.padding,
        dilation=q_layer.dilation,
        groups=q_layer.groups,
    )

    out_layer = q_layer(x_q_layer)

    # The forward must be bit-identical: same quantized operands, same op.
    assert torch.all(out_man == out_layer), "Manual Forward divergence!"

    g_out = torch.randn_like(out_man)

    q_igrad_out = quant_fn(g_out)
    q_wgrad_out = quant_fn(g_out)
    q_bgrad_out = quant_fn(g_out) if bias else None

    igrad_man = torch.nn.grad.conv1d_input(
        qx.shape,
        qw,
        q_igrad_out,
        stride=q_layer.stride,
        padding=q_layer.padding,
        dilation=q_layer.dilation,
        groups=q_layer.groups,
    )
    wgrad_man = torch.nn.grad.conv1d_weight(
        qx,
        qw.shape,
        q_wgrad_out,
        stride=q_layer.stride,
        padding=q_layer.padding,
        dilation=q_layer.dilation,
        groups=q_layer.groups,
    )
    if bias:
        assert q_bgrad_out is not None
        bgrad_man = q_bgrad_out.sum(dim=(0, 2))

    out_layer.backward(g_out)

    # The gradients are held to a small tolerance rather than to equality, a
    # margin for the convolution backward reducing in a different order
    # between two calls. A quantizer applied to the wrong tensor, or skipped,
    # moves a K=8, P=4 result far beyond 1e-3.
    assert x_q_layer.grad is not None
    assert q_layer.weight.grad is not None
    assert torch.allclose(x_q_layer.grad.detach(), igrad_man.detach(), atol=1e-3, rtol=1e-3), (
        "Manual Input Grad divergence!"
    )
    assert torch.allclose(q_layer.weight.grad.detach(), wgrad_man.detach(), atol=1e-3, rtol=1e-3), (
        "Manual Weight Grad divergence!"
    )
    if bias:
        assert q_layer.bias is not None
        assert q_layer.bias.grad is not None
        assert torch.allclose(
            q_layer.bias.grad.detach(), bgrad_man.detach(), atol=1e-3, rtol=1e-3
        ), "Manual Bias Grad divergence!"
