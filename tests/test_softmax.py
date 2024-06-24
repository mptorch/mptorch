import torch
from torch.nn import functional as F
from mptorch import FloatingPoint
from mptorch.quant import quant_softmax
from mptorch.quant import quant_softmax_lse

device = torch.device("cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)

fp_format = FloatingPoint(exp=8, man=23, subnormals=True, saturate=False)

# Testing mptorch quant_lse_softmax against pytorch softmax
def test_softmax_lse_dimneg2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-2)
	res = quant_softmax(a, fp_format, -2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dimneg1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-1)
	res = quant_softmax_lse(a, fp_format, -1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = quant_softmax_lse(a, fp_format, 0)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = quant_softmax_lse(a, fp_format, 1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = quant_softmax_lse(a, fp_format, 2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=3)
	res = quant_softmax_lse(a, fp_format, 3)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)


# Testing mptorch quant_softmax against pytorch softmax
def test_softmax_dimneg2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-2)
	res = quant_softmax(a, fp_format, -2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dimneg1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=-1)
	res = quant_softmax(a, fp_format,-1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = quant_softmax(a, fp_format, 0)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = quant_softmax(a, fp_format, 1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = quant_softmax(a, fp_format, 2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim3():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=3)
	res = quant_softmax(a, fp_format, 3)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)