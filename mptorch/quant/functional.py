import torch
import torch.nn as nn
from .quant_function import *
from mptorch import FloatingPoint

__all__ = ["qlinear", "qadd", "qmul", "qsqrt", "qdiv", "qpow", "qsum", "qmean"]

# Take inspiration for defining the custom derivation formulas from the PyTorch repository
# See: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml


class qlinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats
        qinput = formats.input_quant(input)
        qweight = formats.weight_quant(weight)

        if type(formats.fwd_add) == FloatingPoint:
            output = float_gemm(
                qinput,
                qweight.t(),
                man_add=formats.fwd_add.man,
                exp_add=formats.fwd_add.exp,
                man_mul=formats.fwd_mul.man,
                exp_mul=formats.fwd_mul.exp,
                rounding=formats.fwd_rnd,
                fma=formats.fwd_fma,
                subnormals=formats.fwd_add.subnormals,
                saturate=formats.fwd_add.saturate,
            )
        else:
            output = fxp_gemm(
                qinput,
                qweight.t(),
                wl_add=formats.fwd_add.wl,
                fl_add=formats.fwd_add.fl,
                wl_mul=formats.fwd_mul.wl,
                fl_mul=formats.fwd_mul.fl,
                symmetric=False,
                rounding=formats.fwd_rnd,
                fma=formats.fwd_fma,
            )
        if bias is not None:
            qbias = formats.bias_quant(bias)
            output += qbias.unsqueeze(0).expand_as(output)
            ctx.save_for_backward(qinput, qweight, qbias)
        else:
            ctx.save_for_backward(qinput, qweight, bias)
        if type(formats.fwd_add) == FloatingPoint:
            output = float_quantize(
                output.contiguous(),
                exp=formats.fwd_add.exp,
                man=formats.fwd_add.man,
                rounding=formats.fwd_rnd,
                subnormals=formats.fwd_add.subnormals,
                saturate=formats.fwd_add.saturate,
            )
        else:
            output = fixed_point_quantize(
                output.contiguous(),
                wl=formats.fwd_add.wl,
                fl=formats.fwd_add.fl,
                symmetric=False,
                rounding=formats.fwd_rnd,
            )
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qweight, qbias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        qgrad_output = ctx.formats.grad_quant(grad_output)

        if ctx.needs_input_grad[0]:
            if type(ctx.formats.bwd_add) == FloatingPoint:
                grad_input = float_gemm(
                    qgrad_output,
                    qweight,
                    man_add=ctx.formats.bwd_add.man,
                    exp_add=ctx.formats.bwd_add.exp,
                    man_mul=ctx.formats.bwd_mul.man,
                    exp_mul=ctx.formats.bwd_mul.exp,
                    rounding=ctx.formats.bwd_rnd,
                    fma=ctx.formats.bwd_fma,
                    subnormals=ctx.formats.bwd_add.subnormals,
                    saturate=ctx.formats.bwd_add.saturate,
                )
            else:
                grad_input = fxp_gemm(
                    qgrad_output,
                    qweight,
                    wl_add=ctx.formats.bwd_add.wl,
                    fl_add=ctx.formats.bwd_add.fl,
                    wl_mul=ctx.formats.bwd_mul.wl,
                    fl_mul=ctx.formats.bwd_mul.fl,
                    rounding=ctx.formats.bwd_rnd,
                    symmetric=False,
                    fma=ctx.formats.bwd_fma,
                )
        if ctx.needs_input_grad[1]:
            if type(ctx.formats.bwd_add) == FloatingPoint:
                grad_weight = float_gemm(
                    qgrad_output.t(),
                    qinput,
                    man_add=ctx.formats.bwd_add.man,
                    exp_add=ctx.formats.bwd_add.exp,
                    man_mul=ctx.formats.bwd_mul.man,
                    exp_mul=ctx.formats.bwd_mul.exp,
                    rounding=ctx.formats.bwd_rnd,
                    fma=ctx.formats.bwd_fma,
                    subnormals=ctx.formats.bwd_add.subnormals,
                    saturate=ctx.formats.bwd_add.saturate,
                )
            else:
                grad_weight = fxp_gemm(
                    qgrad_output.t(),
                    qinput,
                    wl_add=ctx.formats.bwd_add.wl,
                    fl_add=ctx.formats.bwd_add.fl,
                    wl_mul=ctx.formats.bwd_mul.wl,
                    fl_mul=ctx.formats.bwd_mul.fl,
                    rounding=ctx.formats.bwd_rnd,
                    symmetric=False,
                    fma=ctx.formats.bwd_fma,
                )
        if qbias is not None and ctx.needs_input_grad[2]:
            ones = torch.ones(1, qgrad_output.shape[0], device=grad_output.device)
            if type(ctx.formats.bwd_add) == FloatingPoint:
                grad_bias = float_gemm(
                    ones,
                    qgrad_output,
                    man_add=ctx.formats.bwd_add.man,
                    exp_add=ctx.formats.bwd_add.exp,
                    man_mul=ctx.formats.bwd_mul.man,
                    exp_mul=ctx.formats.bwd_mul.exp,
                    rounding=ctx.formats.bwd_rnd,
                    fma=ctx.formats.bwd_fma,
                    subnormals=ctx.formats.bwd_add.subnormals,
                    saturate=ctx.formats.bwd_add.saturate,
                ).reshape(-1)
            else:
                grad_bias = fxp_gemm(
                    ones,
                    qgrad_output,
                    wl_add=ctx.formats.bwd_add.wl,
                    fl_add=ctx.formats.bwd_add.fl,
                    wl_mul=ctx.formats.bwd_mul.wl,
                    fl_mul=ctx.formats.bwd_mul.fl,
                    rounding=ctx.formats.bwd_rnd,
                    symmetric=False,
                    fma=ctx.formats.bwd_fma,
                ).reshape(-1)

        if ctx.needs_input_grad[0]:
            qgrad_input = ctx.formats.grad_quant(grad_input)
        else:
            qgrad_input = None
        qgrad_weight = ctx.formats.grad_quant(grad_weight)
        if grad_bias is not None:
            qgrad_bias = ctx.formats.grad_quant(grad_bias)
        else:
            qgrad_bias = None
        return qgrad_input, qgrad_weight, qgrad_bias, None, None, None


class qadd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, fwd_quant, bwd_quant):
        ctx.save_for_backward(x, y)
        ctx.bwd_quant = bwd_quant
        z = x + y
        z = fwd_quant(z.contiguous())
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        grad_x = grad_z * torch.ones_like(x, device=x.device)
        grad_y = grad_z * torch.ones_like(y, device=y.device)

        grad_x = ctx.bwd_quant(grad_x.contiguous())
        grad_y = ctx.bwd_quant(grad_y.contiguous())
        return grad_x, grad_y, None, None


class qmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, fwd_quant, bwd_quant):
        ctx.save_for_backward(x, y)
        ctx.bwd_quant = bwd_quant
        z = x * y
        z = fwd_quant(z.contiguous())
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        grad_x = grad_z * y
        grad_y = grad_z * x
        grad_x = ctx.bwd_quant(grad_x.contiguous())
        grad_y = ctx.bwd_quant(grad_y.contiguous())
        return grad_x, grad_y, None, None


# see the following link for a discussion regarding numerical stability of
# backward propagation for division operations in PyTorch and for the basis
# of this implementation: https://github.com/pytorch/pytorch/issues/43414
class qdiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, fwd_quant, bwd_quant):
        z = fwd_quant(x / y)
        ctx.save_for_backward(x, y, z)
        ctx.bwd_quant = bwd_quant
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y, z = ctx.saved_tensors
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.bwd_quant(grad_z / y)
        if ctx.needs_input_grad[1]:
            grad_y = ctx.bwd_quant(-grad_z * z)
            grad_y = ctx.bwd_quant(grad_y / y)
        return grad_x, grad_y, None, None


class qpow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_quant, bwd_quant, n=2):
        ctx.n = n
        ctx.bwd_quant = bwd_quant
        y = fwd_quant(x**n)
        gy = bwd_quant(x ** (n - 1))
        ctx.save_for_backward(gy)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (gy,) = ctx.saved_tensors
        grad_x = ctx.bwd_quant(gy * ctx.n)
        grad_x = ctx.bwd_quant(grad_y * grad_x)
        return grad_x, None, None, None


class qsqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_quant, bwd_quant):
        ctx.bwd_quant = bwd_quant
        x = torch.sqrt(x)
        y = fwd_quant(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (y,) = ctx.saved_tensors
        grad_x = ctx.bwd_quant(y * 2)
        grad_x = ctx.bwd_quant(grad_y / grad_x)
        return grad_x, None, None


def qsum_kernel(x, dim, quant):
    shape = list(x.shape)
    shape[dim] = 1
    vs = torch.zeros(shape, device=x.device)
    vs = vs.transpose(0, dim).reshape(1, -1).transpose(0, 1)
    vx = torch.transpose(x, 0, dim).reshape(x.shape[dim], -1).transpose(0, 1)
    shape[dim] = x.shape[0]
    shape[0] = 1
    for k in range(x.shape[dim]):
        vs = quant(vs + vx[:, k : k + 1])
    return vs.transpose(0, 1).reshape(shape).transpose(0, dim)


class qsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quant, dim=0, keepdim=False):
        ctx.save_for_backward(x)

        if dim is None:
            dim = range(x.dim())
        elif isinstance(dim, tuple) == False:
            dim = (dim,)

        ctx.dim = dim
        sums = x
        for d in dim:
            sums = qsum_kernel(sums, d, quant)
        if keepdim is False:
            for _ in dim:
                idx = 0
                while sums.shape[idx] != 1:
                    idx += 1
                sums = torch.squeeze(sums, dim=idx)
        ctx.shape = sums.shape
        return sums

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        for d in ctx.dim:
            grad_output = torch.unsqueeze(grad_output, d)
        grad_x = grad_output * torch.ones_like(x, device=x.device)
        return grad_x, None, None, None


class qmean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fwd_quant, bwd_quant, dim=3, keepdim=False):
        ctx.bwd_quant = bwd_quant
        ctx.save_for_backward(x)

        if dim is None:
            dim = range(x.dim())
        elif isinstance(dim, tuple) == False:
            dim = (dim,)

        ctx.dim = dim
        sums = x
        numel = 1
        for d in dim:
            sums = qsum_kernel(sums, d, fwd_quant)
            numel *= x.shape[d]
        ctx.numel = numel
        sums = fwd_quant(sums / numel)
        if keepdim is False:
            for _ in dim:
                idx = 0
                while sums.shape[idx] != 1:
                    idx += 1
                sums = torch.squeeze(sums, dim=idx)
        ctx.shape = sums.shape
        return sums

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        for d in ctx.dim:
            grad_output = torch.unsqueeze(grad_output, d)
        grad_x = ctx.bwd_quant(
            grad_output * torch.ones_like(x, device=x.device) / ctx.numel
        )
        return grad_x, None, None, None, None
