import torch
import math
from .quant_function import *
from ..number import FloatType
from .quant_format import make_quant_function

__all__ = [
    "qlinear",
    "qmatmul",
    "qmm",
    "qadd",
    "qmul",
    "qsqrt",
    "qdiv",
    "qpow",
    "qsum",
    "qmean",
    "qlayernorm",
    "qsoftmax",
]

# Utility functions to perform tensor scaling operations in low precison (8-bit and
# below) DNN training
# See: https://arxiv.org/pdf/2309.17224
# NOTE: using this routine seems quite sensitive to how the margin term is picked,
# notably the empirical value suggested in the aformentioned reference leads to 
# divergence (i.e. grad_input computation in the linear layer backward pass is 
# unstable and produces NaNs right from the start); more investigation is needed 
# and this might need to be revisited in the case of supernormals
def compute_bias(x, cast_to: FloatType, margin=10):
    with torch.no_grad():
        (amax, _) = torch.max(torch.abs(x), dim=-1, keepdim=True)
        (amax, _) = torch.max(amax, dim=-2, keepdim=True)
    return torch.floor(torch.log2(cast_to.normal_max / amax)) - margin

def scale(x, x_scale):
    return x * torch.pow(2.0 * torch.ones_like(x), x_scale)

def unscale(x, x_scale):
    return x * torch.pow(2.0 * torch.ones_like(x), -x_scale)

# Take inspiration for defining the custom derivation formulas from the PyTorch repository
# See: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml

class qlinear_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats
        # NOTE: (optimization) precompute the LP weights and weight scale
        if (formats.weight_scaled_format is not None) and (formats.input_scaled_format is not None):
            ctx.weight_scale = compute_bias(weight, formats.weight_scaled_format)
            ctx.input_scale = compute_bias(input, formats.input_scaled_format)
        else:
            ctx.input_scale, ctx.weight_scale = 0, 0
        qinput = formats.input_quant(scale(input, ctx.input_scale))
        qweight = formats.weight_quant(scale(weight, ctx.weight_scale))
        # NOTE: investigate if the bias term needs to be scaled as well
        if bias is not None:
            qbias = formats.bias_quant(bias)
        else:
            qbias = None

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.nn.functional.linear(qinput, qweight)
        else:
            output = mp_bmm(qinput, qweight.t(), formats, use_forward=True)

        output = unscale(output, ctx.input_scale + ctx.weight_scale)

        if qbias is not None:
            output += qbias.view(
                -1, qbias.shape[-1]
            )  # broadcasting should be done automatically

        ctx.save_for_backward(qinput, qweight, qbias)
        qoutput = formats.output_quant(output)

        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qweight, qbias = ctx.saved_tensors
        qgrad_input, qgrad_weight, qgrad_bias = None, None, None

        if ctx.formats.grad_scaled_format is not None:
            grad_scale = compute_bias(grad_output, ctx.formats.grad_scaled_format)
        else:
            grad_scale = 0
        qgrad_output = ctx.formats.grad_quant(scale(grad_output, grad_scale))

        if ctx.needs_input_grad[0]:
            if ctx.formats.bwd_use_default_prec:
                qgrad_input = torch.matmul(qgrad_output, qweight)
            else:
                qgrad_input = mp_bmm(
                    qgrad_output,
                    qweight,
                    formats=ctx.formats,
                    use_forward=False,
                )

            qgrad_input = unscale(qgrad_input, grad_scale + ctx.weight_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_input = ctx.formats.grad_quant(qgrad_input)

        if ctx.needs_input_grad[1]:
            if ctx.formats.bwd_use_default_prec:
                qgrad_weight = torch.matmul(qgrad_output.transpose(-2, -1), qinput)
            else:
                qgrad_weight = mp_bmm(
                    qgrad_output.transpose(-2, -1),
                    qinput,
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_weight = unscale(qgrad_weight, grad_scale + ctx.input_scale)

            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_weight = ctx.formats.grad_quant(qgrad_weight)

        if qbias is not None and ctx.needs_input_grad[2]:
            if ctx.formats.bwd_use_default_prec:
                qgrad_bias = qgrad_output.sum(
                    dim=tuple([i for i in range(len(qgrad_output.shape) - 1)])
                ).reshape(qbias.shape)
            else:
                with torch.no_grad():
                    qgrad_bias = qsum(
                        qgrad_output,
                        dim=tuple([i for i in range(len(qgrad_output.shape) - 1)]),
                        quant=make_quant_function(ctx.formats.bwd_add, ctx.formats.bwd_rnd, ctx.formats.prng_bits),
                    ).reshape(qbias.shape)
            qgrad_bias = unscale(qgrad_bias, grad_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_bias = ctx.formats.grad_quant(qgrad_bias)

        return qgrad_input, qgrad_weight, qgrad_bias, None, None



def qlinear(input, weight, bias, formats):
    return qlinear_kernel.apply(input, weight, bias, formats)


class qmm_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, formats):
        ctx.formats = formats
        if (formats.weight_scaled_format is not None) and (formats.input_scaled_format is not None):
            ctx.weight_scale = compute_bias(other, formats.weight_scaled_format)
            ctx.input_scale = compute_bias(input, formats.input_scaled_format)
        else:
            ctx.input_scale, ctx.weight_scale = 0, 0

        qinput = formats.input_quant(scale(input, ctx.input_scale))
        qother = formats.weight_quant(scale(other, ctx.weight_scale))

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.mm(qinput, qother)
        else:
            output = mp_mm(
                qinput,
                qother,
                formats=formats,
                use_forward=True,
            )

        output = unscale(output, ctx.input_scale + ctx.weight_scale)

        ctx.save_for_backward(qinput, qother)
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qother = ctx.saved_tensors
        qgrad_input, qgrad_other = None, None

        if ctx.formats.grad_scaled_format is not None:
            grad_scale = compute_bias(grad_output, ctx.formats.grad_scaled_format)
        else:
            grad_scale = 0
        qgrad_output = ctx.formats.grad_quant(scale(grad_output, grad_scale))

        if ctx.needs_input_grad[0]:
            if ctx.formats.bwd_use_default_prec:
                grad_input = torch.mm(qgrad_output, qother.transpose(-2, -1))
            else:
                grad_input = mp_mm(
                    qgrad_output,
                    qother.transpose(-2, -1),
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_input = unscale(grad_input, grad_scale + ctx.weight_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_input = ctx.formats.grad_quant(grad_input)

        if ctx.needs_input_grad[1]:
            if ctx.formats.bwd_use_default_prec:
                grad_other = torch.mm(qinput.transpose(-2, -1), qgrad_output)
            else:
                grad_other = mp_mm(
                    qinput.transpose(-2, -1),
                    qgrad_output,
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_other = unscale(grad_other, grad_scale + ctx.input_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_other = ctx.formats.grad_quant(grad_other)

        return qgrad_input, qgrad_other, None


def qmm(input, other, formats):
    return qmm_kernel.apply(input, other, formats)


class qmatmul_kernel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, other, formats):
        ctx.formats = formats
        # NOTE: need to see how we do things for each matrix in
        # the batched version
        if (formats.weight_scaled_format is not None) and (formats.input_scaled_format is not None):
            ctx.weight_scale = compute_bias(other, formats.weight_scaled_format)
            ctx.input_scale = compute_bias(input, formats.input_scaled_format)
        else:
            ctx.input_scale, ctx.weight_scale = 0, 0
        qinput = formats.input_quant(scale(input, ctx.input_scale))
        qother = formats.weight_quant(scale(other, ctx.weight_scale))

        if formats.fwd_use_default_prec:
            with torch.no_grad():
                output = torch.matmul(qinput, qother)
        else:
            output = mp_bmm(
                qinput,
                qother,
                formats=formats,
                use_forward=True,
            )
        output = unscale(output, ctx.input_scale + ctx.weight_scale)

        ctx.save_for_backward(qinput, qother)
        qoutput = formats.output_quant(output)
        return qoutput

    @staticmethod
    def backward(ctx, grad_output):
        qinput, qother = ctx.saved_tensors
        qgrad_input, qgrad_other = None, None

        if ctx.formats.grad_scaled_format is not None:
            grad_scale = compute_bias(grad_output, ctx.formats.grad_scaled_format)
        else:
            grad_scale = 0
        qgrad_output = ctx.formats.grad_quant(scale(grad_output, grad_scale))

        if ctx.needs_input_grad[0]:
            if ctx.formats.bwd_use_default_prec:
                grad_input = torch.matmul(qgrad_output, qother.transpose(-2, -1))
                grad_input = torch.matmul(qgrad_output, qother.transpose(-2, -1))
            else:
                grad_input = mp_bmm(
                    qgrad_output,
                    qother.transpose(-2, -1),
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_input = unscale(grad_input, grad_scale + ctx.weight_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_input = ctx.formats.grad_quant(qgrad_input)

        if ctx.needs_input_grad[1]:
            if ctx.formats.bwd_use_default_prec:
                grad_other = torch.matmul(qinput.transpose(-2, -1), qgrad_output)
                grad_other = torch.matmul(qinput.transpose(-2, -1), qgrad_output)
            else:
                grad_other = mp_bmm(
                    qinput.transpose(-2, -1),
                    qgrad_output,
                    formats=ctx.formats,
                    use_forward=False,
                )
            qgrad_other = unscale(grad_other, grad_scale + ctx.input_scale)
            # NOTE: look if this needs to be redesigned (separate grad_quants ?!)
            qgrad_other = ctx.formats.grad_quant(qgrad_other)

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


# TODO: need CUDA-accelerated version of this routine
# and also of sums across multiple dimensions
def qsum_1d(x, dim, quant):
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


class qsum_kernel(torch.autograd.Function):
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
            sums = qsum_1d(sums, d, quant)
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


def qsum(x, dim, quant=lambda x: x):
    return qsum_kernel.apply(x, quant, dim)


# TODO: similar to sum, look into a CUDA-accelerated version of this routine
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


class qlayernorm_kernel(torch.autograd.Function):
    # TODO: implement GPU-accelerated version of functional layernorm
    pass


def qlayernorm(x, normalized_shape, weight, bias, eps, quant):
    pass


class qsoftmax_kernel(torch.autograd.Function):
    # TODO: implement GPU-accelerated version of functional softmax
    pass


def qsoftmax(x, dim, dtype, quant):
    pass
