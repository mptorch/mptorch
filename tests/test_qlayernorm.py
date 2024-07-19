import torch
from torch.nn import functional as F
from mptorch import FloatingPoint
from mptorch.quant import QAffineFormats
from mptorch.quant import float_quantize
from mptorch.quant import functional as Q

device = "cpu"
seed = 1234
torch.manual_seed(seed)

fp_format = FloatingPoint(
    exp=8, man=23, subnormals=True, saturate=False
)

quant_fp = lambda x: float_quantize(
    x,
    exp=8,
    man=23,
    rounding="nearest",
    subnormals=True,
    saturate=False,
)

layer_formats = QAffineFormats(
    fwd_mac=(fp_format),
    fwd_rnd="nearest",
    bwd_mac=(fp_format),
    bwd_rnd="nearest",
    weight_quant=quant_fp,
    input_quant=quant_fp,
    output_quant=quant_fp,
    grad_quant=quant_fp,
    bias_quant=quant_fp,
)

def test_qlayer_norm_forward():
    a = torch.randn(20, 30, 40, device=device)
    ref = F.layer_norm(a, [a.size(-2), a.size(-1)], weight=None, bias=None, eps=1e-5)
    res = Q.qlayernorm(a, [a.size(-2), a.size(-1)], weight=None, bias=None, eps=1e-5, formats=layer_formats)

    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

    
# def test_qlayer_norm_backward():
#     a = torch.randn(10, 30, 40, 20, device=device)

#     ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
#     res1 = ref1.clone().detach()
#     res1.requires_grad_(True)

#     ref2 = F.layer_norm(ref1, ref1.size(-1), )
#     res2 = Q.qlayernorm(res1, 1, layer_formats, weight=None, bias)

#     ref2.backward(a)
#     res2.backward(a)

#     torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)