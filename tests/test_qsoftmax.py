import torch
import torch.nn
from mptorch import FloatingPoint
from mptorch.quant import QSoftmaxFormats
from mptorch.quant import float_quantize
from mptorch.quant import functional as Q
from mptorch.quant import QSoftmax

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
    fwd_exp=fp_format,
    fwd_off=fp_format,
    fwd_acc=fp_format,
    fwd_div=fp_format,

    bwd_add=fp_format,
    bwd_mul=fp_format,
    bwd_div=fp_format,

    output_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
)

formats_lse = QSoftmaxFormats(
    fwd_exp=fp_format,
    fwd_off=fp_format,
    fwd_lse=fp_format,

    bwd_add=fp_format,
    bwd_mul=fp_format,
    bwd_div=fp_format,

    output_quant=quant_fp,
    input_quant=quant_fp,
    grad_quant=quant_fp,
)


def test_softmax_formats():
    assert not formats_lse.fwd_use_default_prec
    assert not formats_lse.bwd_use_default_prec
    assert not formats_div.fwd_use_default_prec
    assert not formats_div.bwd_use_default_prec
    assert formats_lse.use_lse
    assert not hasattr(formats_lse, "fwd_div")
    assert not formats_div.use_lse
    assert hasattr(formats_div, "fwd_div")

# Testing mptorch LSE-based softmax against pytorch softmax
def test_softmax_lse_dim0():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=0)
    res = Q.qsoftmax(a, 0, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim1():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=1)
    res = Q.qsoftmax(a, 1, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim2():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=2)
    res = Q.qsoftmax(a, 2, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_lse_dim3():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=3)
    res = Q.qsoftmax(a, 3, formats_lse)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

# Testing mptorch division-based softmax against pytorch softmax
def test_softmax_dim0():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=0)
    res = Q.qsoftmax(a, 0, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim1():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=1)
    res = Q.qsoftmax(a, 1, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim2():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=2)
    res = Q.qsoftmax(a, 2, formats_div)
    torch.testing.assert_close(ref, res, atol=1e-5, rtol=0)

def test_softmax_dim3():
    a = torch.randn(10, 30, 40, 20, device=device)
    ref = torch.softmax(a, dim=3)
    res = Q.qsoftmax(a, 3, formats_div)
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

def test_softmax_default_prec():
    a1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
    a2 = a1.clone().detach()
    a2.requires_grad_(True)

    formats_default = QSoftmaxFormats() # default precision, no quantization
    assert formats_default.fwd_use_default_prec
    assert formats_default.bwd_use_default_prec

    b1 = torch.softmax(a1, dim=2)
    b2 = Q.qsoftmax(a2, 2, formats_default)

    x = torch.randn(10, 30, 40, 20, device=device)
    b1.backward(x)
    b2.backward(x)

    torch.testing.assert_close(b1, b2, atol=1e-5, rtol=0)
    torch.testing.assert_close(a2.grad, a1.grad, atol=1e-5, rtol=0)

def test_softmax_layer():
    man, exp = 12, 8
    quantize = lambda x: Q.float_quantize(
        x, exp=exp, man=man, rounding="nearest", subnormals=True, saturate=False
    )
    float_format = FloatingPoint(exp=exp, man=man, subnormals=True, saturate=False)
    qsoftmax_formats = QSoftmaxFormats(
        fwd_exp=float_format,
        fwd_off=float_format,
        fwd_acc=float_format,
        fwd_div=float_format,

        bwd_add=float_format,
        bwd_mul=float_format,
        bwd_div=float_format,

        output_quant=quantize,
        input_quant=quantize,
        grad_quant=quantize,
    )
    dim = 2
    layer = torch.nn.Softmax(dim)
    qlayer = QSoftmax(dim, qsoftmax_formats)

    a1 = torch.randn(10, 30, 40, 20, device=device, requires_grad=True)
    a2 = a1.clone().detach()
    a2.requires_grad_(True)

    b1 = layer.forward(a1)
    b2 = qlayer.forward(a2)

    x = torch.randn(10, 30, 40, 20, device=device)
    b1.backward(x)
    b2.backward(x)

    torch.testing.assert_close(b1, b2, atol=1e-3, rtol=0)
    torch.testing.assert_close(a2.grad, a1.grad, atol=1e-3, rtol=0)