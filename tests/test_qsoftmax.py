import torch
import torch.nn
from mptorch import FloatingPoint
from mptorch.quant import QSoftmaxFormats
from mptorch.quant import float_softmax, float_quantize
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

formats_div = QSoftmaxFormats(
    fwd_trans=fp_format,
    fwd_add=fp_format,
    fwd_div=fp_format,
    bwd_add=fp_format,
    output_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
)

formats_lse = QSoftmaxFormats(
    fwd_trans=fp_format,
    fwd_add=fp_format,
    bwd_add=fp_format,
    output_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
)


def test_softmax_formats():
    assert formats_lse.use_lse
    assert formats_lse.fwd_div is None
    assert not formats_div.use_lse
    assert formats_div.fwd_div is not None

# Testing mptorch LSE-based softmax against pytorch softmax
def test_softmax_lse_dim0():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=0)
    res = float_softmax(a, 0, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim1():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=1)
    res = float_softmax(a, 1, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=2)
    res = float_softmax(a, 2, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim3():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=3)
    res = float_softmax(a, 3, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

# Testing mptorch division-based softmax against pytorch softmax
def test_softmax_dim0():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=0)
    res = float_softmax(a, 0, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim1():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=1)
    res = float_softmax(a, 1, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim2():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=2)
    res = float_softmax(a, 2, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim3():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=3)
    res = float_softmax(a, 3, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

# Testing mptorch backward against pytorch
def test_softmax_backward_dim0():
    a = torch.randn(10, 30, 40, 20, device=device)

    ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
    res1 = ref1.clone().detach()
    res1.requires_grad_(True)

    ref2 = torch.softmax(ref1, dim=0)
    res2 = Q.qsoftmax(res1, 0, formats_lse)

    ref2.backward(a)
    res2.backward(a)

    torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)

def test_softmax_backward_dim1():
    a = torch.randn(10, 30, 40, 20, device=device)

    ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
    res1 = ref1.clone().detach()
    res1.requires_grad_(True)

    ref2 = torch.softmax(ref1, dim=1)
    res2 = Q.qsoftmax(res1, 1, formats_lse)

    ref2.backward(a)
    res2.backward(a)

    torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)

def test_softmax_backward_dim2():
    a = torch.randn(10, 30, 40, 20, device=device)

    ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
    res1 = ref1.clone().detach()
    res1.requires_grad_(True)

    ref2 = torch.softmax(ref1, dim=2)
    res2 = Q.qsoftmax(res1, 2, formats_div)

    ref2.backward(a)
    res2.backward(a)

    torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)

def test_softmax_backward_dim3():
    a = torch.randn(10, 30, 40, 20, device=device)

    ref1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
    res1 = ref1.clone().detach()
    res1.requires_grad_(True)

    ref2 = torch.softmax(ref1, dim=3)
    res2 = Q.qsoftmax(res1, 3, formats_div)

    ref2.backward(a)
    res2.backward(a)

    torch.testing.assert_close(ref1.grad, res1.grad, atol=1e-5, rtol=0)