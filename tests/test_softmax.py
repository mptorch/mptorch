import torch
from torch.nn import functional as F
from mptorch.quant import quant_softmax
from mptorch.quant import quant_softmax_lse

device = torch.device("cpu")

torch.manual_seed(0)
torch.cuda.manual_seed(0)

man = 23
exp = 8

def test_softmax_lse_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = quant_softmax_lse(a, man, exp, 0)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = quant_softmax_lse(a, man, exp, 1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = quant_softmax_lse(a, man, exp, 2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = quant_softmax(a, man, exp, 0)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = quant_softmax(a, man, exp, 1)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = quant_softmax(a, man, exp, 2)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)