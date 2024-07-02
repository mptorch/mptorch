import torch
import torch.nn
from mptorch import FloatingPoint
from mptorch.quant import (float_softmax, float_softmax_lse, QAffineFormats, float_quantize)
from mptorch.quant import functional as Q

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
def test_softmax_lse_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = float_softmax_lse(a, 0, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = float_softmax_lse(a, 1, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = float_softmax_lse(a, 2, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim3():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=3)
	res = float_softmax_lse(a, 3, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)


# Testing mptorch quant_softmax against pytorch softmax
def test_softmax_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=0)
	res = float_softmax(a, 0, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=1)
	res = float_softmax(a, 1, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=2)
	res = float_softmax(a, 2, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim3():
	a = torch.randn(10, 30, 40, 20, device=device)
	ref = torch.softmax(a, dim=3)
	res = float_softmax(a, 3, layer_formats)
	torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

# Testing mptorch quant_softmax backward against pytorch
def test_softmax_backward_dim0():
	a = torch.randn(10, 30, 40, 20, device=device)

	ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
	res1 = ref1.clone().detach()
	res1.requires_grad_(True)

	ref2 = torch.softmax(ref1, dim=0)
	res2 = Q.qsoftmax(res1, 0, layer_formats, use_lse=True)

	ref2.backward(a)
	res2.backward(a)

	torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)

def test_softmax_backward_dim1():
	a = torch.randn(10, 30, 40, 20, device=device)

	ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
	res1 = ref1.clone().detach()
	res1.requires_grad_(True)

	ref2 = torch.softmax(ref1, dim=1)
	res2 = Q.qsoftmax(res1, 1, layer_formats, use_lse=True)

	ref2.backward(a)
	res2.backward(a)

	torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)

def test_softmax_backward_dim2():
	a = torch.randn(10, 30, 40, 20, device=device)

	ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
	res1 = ref1.clone().detach()
	res1.requires_grad_(True)

	ref2 = torch.softmax(ref1, dim=2)
	res2 = Q.qsoftmax(res1, 2, layer_formats)

	ref2.backward(a)
	res2.backward(a)

	torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)

def test_softmax_backward_dim3():
	a = torch.randn(10, 30, 40, 20, device=device)

	ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
	res1 = ref1.clone().detach()
	res1.requires_grad_(True)

	ref2 = torch.softmax(ref1, dim=3)
	res2 = Q.qsoftmax(res1, 3, layer_formats)

	ref2.backward(a)
	res2.backward(a)

	torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)