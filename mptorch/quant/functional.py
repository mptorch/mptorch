import torch
import torch.nn as nn
from .quant_function import *
from mptorch import FloatingPoint

__all__ = [
    "qlinear",
    "qmatmul",
    "qadd",
    "qmul",
    "qsqrt",
    "qdiv",
    "qpow",
    "qsum",
    "qmean",
]

# Take inspiration for defining the custom derivation formulas from the PyTorch repository
# See: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml


class qlinear_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats
        qinput = formats.input_quant(input)
        qweight = formats.weight_quant(weight)

        output = float_bgemm(
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

        if bias is not None:
            qbias = formats.bias_quant(bias)
            output += qbias.view(
                -1, qbias.shape[-1]
            )  # broadcasting should be done automatically
            ctx.save_for_backward(qinput, qweight, qbias)
        else:
            ctx.save_for_backward(qinput, qweight, bias)
        output = float_quantize(
            output.contiguous(),
            exp=formats.fwd_add.exp,
            man=formats.fwd_add.man,
            rounding=formats.fwd_rnd,
            subnormals=formats.fwd_add.subnormals,
            saturate=formats.fwd_add.saturate,
        )
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qweight, qbias = ctx.saved_tensors
        qgrad_input, qgrad_weight, qgrad_bias = None, None, None
        qgrad_output = ctx.formats.grad_quant(grad_output)

        if ctx.needs_input_grad[0]:
            qgrad_input = float_bgemm(
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
            qgrad_input = ctx.formats.grad_quant(qgrad_input)

        if ctx.needs_input_grad[1]:
            qgrad_weight = float_bgemm(
                qgrad_output.transpose(-2, -1),
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
            qgrad_weight = ctx.formats.grad_quant(qgrad_weight)

        if qbias is not None and ctx.needs_input_grad[2]:
            # TODO: apply quantization on the sum (?!)
            qgrad_bias = qgrad_output.sum(
                dim=tuple([i for i in range(len(qgrad_output.shape) - 1)])
            ).reshape(qbias.shape)
            qgrad_bias = ctx.formats.grad_quant(qgrad_bias)

        return qgrad_input, qgrad_weight, qgrad_bias, None


def qlinear(input, weight, bias, formats):
    return qlinear_kernel.apply(input, weight, bias, formats)


class qmatmul_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, formats):
        ctx.formats = formats
        qinput = formats.input_quant(input)
        qother = formats.weight_quant(other)

        output = float_bgemm(
            qinput,
            qother,
            man_add=formats.fwd_add.man,
            exp_add=formats.fwd_add.exp,
            man_mul=formats.fwd_mul.man,
            exp_mul=formats.fwd_mul.exp,
            rounding=formats.fwd_rnd,
            fma=formats.fwd_fma,
            subnormals=formats.fwd_add.subnormals,
            saturate=formats.fwd_add.saturate,
        )

        ctx.save_for_backward(qinput, qother)
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qother = ctx.saved_tensors
        qgrad_input, qgrad_other = None, None
        qgrad_output = ctx.formats.grad_quant(grad_output)

        if ctx.needs_input_grad[0]:
            grad_input = float_bgemm(
                qgrad_output,
                qother.transpose(-2, -1),
                man_add=ctx.formats.bwd_add.man,
                exp_add=ctx.formats.bwd_add.exp,
                man_mul=ctx.formats.bwd_mul.man,
                exp_mul=ctx.formats.bwd_mul.exp,
                rounding=ctx.formats.bwd_rnd,
                fma=ctx.formats.bwd_fma,
                subnormals=ctx.formats.bwd_add.subnormals,
                saturate=ctx.formats.bwd_add.saturate,
            )
            qgrad_input = ctx.formats.grad_quant(grad_input)

        if ctx.needs_input_grad[1]:
            grad_other = float_bgemm(
                qinput.transpose(-2, -1),
                qgrad_output,
                man_add=ctx.formats.bwd_add.man,
                exp_add=ctx.formats.bwd_add.exp,
                man_mul=ctx.formats.bwd_mul.man,
                exp_mul=ctx.formats.bwd_mul.exp,
                rounding=ctx.formats.bwd_rnd,
                fma=ctx.formats.bwd_fma,
                subnormals=ctx.formats.bwd_add.subnormals,
                saturate=ctx.formats.bwd_add.saturate,
            )
            qgrad_other = ctx.formats.grad_quant(grad_other)

        return qgrad_input, qgrad_other, None


def qmatmul(input, other, formats):
    return qmatmul_kernel.apply(input, other, formats)


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
