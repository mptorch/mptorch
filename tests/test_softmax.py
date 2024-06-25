import torch
from torch.nn import functional as F
from mptorch import FloatingPoint
from mptorch.quant import (quant_softmax, quant_softmax_lse, QAffineFormats)

device = torch.device("cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)

fp_format = FloatingPoint(exp=8, man=23, subnormals=True, saturate=False)

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
    grad_quant=quant_fp,
    bias_quant=quant_fp,
)

# Testing mptorch quant_lse_softmax against pytorch softmax
def test_softmax_lse_dimneg2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-2)
	res = quant_softmax_lse(a, layer_formats, -2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dimneg1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-1)
	res = quant_softmax_lse(a, layer_formats, -1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = quant_softmax_lse(a, layer_formats, 0)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = quant_softmax_lse(a, layer_formats, 1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = quant_softmax_lse(a, layer_formats, 2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=3)
	res = quant_softmax_lse(a, layer_formats, 3)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)


# Testing mptorch quant_softmax against pytorch softmax
def test_softmax_dimneg2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-2)
	res = quant_softmax(a, layer_formats, -2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dimneg1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-1)
	res = quant_softmax(a, layer_formats,-1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = quant_softmax(a, layer_formats, 0)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = quant_softmax(a, layer_formats, 1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = quant_softmax(a, layer_formats, 2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim3():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=3)
	res = quant_softmax(a, layer_formats, 3)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)