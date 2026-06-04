import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from tests.markers import available_devices
from mptorch.quant import QConv2d, QAffineFormats, binaryK_quantize
from mptorch.number import RoundMode

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("bias", [True, False])
def test_qconv_tier1_exact(device, bias):
    dtype = torch.float32
    
    formats = QAffineFormats(
        weight_quant=lambda x: x,
        input_quant=lambda x: x,
        bias_quant=lambda x: x,
        igrad_quant=lambda x: x,
        wgrad_quant=lambda x: x,
        bgrad_quant=lambda x: x,
    )
    
    vanilla = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)
    q_layer = QConv2d(4, 8, kernel_size=3, padding=1, bias=bias, formats=formats, device=device, dtype=dtype)
    
    with torch.no_grad():
        q_layer.weight.copy_(vanilla.weight)
        if bias:
            q_layer.bias.copy_(vanilla.bias)
            
    x_v = torch.randn(2, 4, 8, 8, device=device, dtype=dtype, requires_grad=True)
    x_q = x_v.clone().detach().requires_grad_(True)
    
    out_v = vanilla(x_v)
    out_q = q_layer(x_q)
    
    assert torch.allclose(out_v, out_q, atol=1e-5), "Forward pass divergence!"
    
    loss_v = out_v.sum()
    loss_q = out_q.sum()
    
    loss_v.backward()
    loss_q.backward()
    
    assert torch.allclose(x_v.grad, x_q.grad, atol=1e-5), "Input grad divergence!"
    assert torch.allclose(vanilla.weight.grad, q_layer.weight.grad, atol=1e-5), "Weight grad divergence!"
    if bias:
        assert torch.allclose(vanilla.bias.grad, q_layer.bias.grad, atol=1e-5), "Bias grad divergence!"

@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_qconv_tier2_statistical(device, dtype, bias):
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
    
    vanilla = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)
    q_layer = QConv2d(4, 8, kernel_size=3, padding=1, bias=bias, formats=formats, device=device, dtype=dtype)
    
    with torch.no_grad():
        q_layer.weight.copy_(vanilla.weight)
        if bias:
            q_layer.bias.copy_(vanilla.bias)
            
    x_v = torch.randn(2, 4, 8, 8, device=device, dtype=dtype, requires_grad=True)
    x_q = x_v.clone().detach().requires_grad_(True)
    
    out_v = vanilla(x_v)
    out_q = q_layer(x_q)
    
    cos_sim_out = F.cosine_similarity(out_v.flatten().float(), out_q.flatten().float(), dim=0).item()
    assert cos_sim_out > 0.85, f"Forward Cosine Sim too low: {cos_sim_out}"
    
    g_out = torch.randn_like(out_v)
    
    out_v.backward(g_out)
    out_q.backward(g_out)
    
    cos_sim_igrad = F.cosine_similarity(x_v.grad.flatten().float(), x_q.grad.flatten().float(), dim=0).item()
    cos_sim_wgrad = F.cosine_similarity(vanilla.weight.grad.flatten().float(), q_layer.weight.grad.flatten().float(), dim=0).item()
    
    assert cos_sim_igrad > 0.80, f"Input Grad Cosine Sim too low: {cos_sim_igrad}"
    assert cos_sim_wgrad > 0.80, f"Weight Grad Cosine Sim too low: {cos_sim_wgrad}"


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
def test_qconv_tier3_manual_baseline(device, dtype, bias):
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
    
    q_layer = QConv2d(4, 8, kernel_size=3, padding=1, bias=bias, formats=formats, device=device, dtype=dtype)
    
    w = q_layer.weight.clone().detach().requires_grad_(True)
    b = q_layer.bias.clone().detach().requires_grad_(True) if bias else None
    x_man = torch.randn(2, 4, 8, 8, device=device, dtype=dtype, requires_grad=True)
    x_q_layer = x_man.clone().detach().requires_grad_(True)
    
    qw = quant_fn(w)
    qx = quant_fn(x_man)
    qb = quant_fn(b) if bias else None
    out_man = F.conv2d(qx, qw, qb, stride=q_layer.stride, padding=q_layer.padding, dilation=q_layer.dilation, groups=q_layer.groups)
    
    out_layer = q_layer(x_q_layer)
    
    assert torch.all(out_man == out_layer), "Manual Forward divergence!"
    
    g_out = torch.randn_like(out_man)
    
    q_igrad_out = quant_fn(g_out)
    q_wgrad_out = quant_fn(g_out)
    q_bgrad_out = quant_fn(g_out) if bias else None
    
    igrad_man = torch.nn.grad.conv2d_input(qx.shape, qw, q_igrad_out, stride=q_layer.stride, padding=q_layer.padding, dilation=q_layer.dilation, groups=q_layer.groups)
    wgrad_man = torch.nn.grad.conv2d_weight(qx, qw.shape, q_wgrad_out, stride=q_layer.stride, padding=q_layer.padding, dilation=q_layer.dilation, groups=q_layer.groups)
    if bias:
        bgrad_man = q_bgrad_out.sum(dim=(0, 2, 3))
        
    out_layer.backward(g_out)
    
    assert torch.allclose(x_q_layer.grad.detach(), igrad_man.detach(), atol=1e-3, rtol=1e-3), "Manual Input Grad divergence!"
    assert torch.allclose(q_layer.weight.grad.detach(), wgrad_man.detach(), atol=1e-3, rtol=1e-3), "Manual Weight Grad divergence!"
    if bias:
        assert torch.allclose(q_layer.bias.grad.detach(), bgrad_man.detach(), atol=1e-3, rtol=1e-3), "Manual Bias Grad divergence!"
