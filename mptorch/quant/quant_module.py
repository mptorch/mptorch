import torch
import torch.nn as nn
from .quant_function import *
from mptorch import Number, FixedPoint, FloatingPoint
import numpy as np
import math

__all__ = [
    "Quantizer",
    "QLinear",
    "QConv2d",
    "QAvgPool2d",
    "QBatchNorm",
    "QBatchNorm1d",
    "QBatchNorm2d",
    "QAddFunction",
    "QMulFunction",
]

# Take inspiration for defining the custom derivation formulas from the PyTorch repository
# See: https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml


class Quantizer(nn.Module):
    def __init__(
        self,
        forward_number=None,
        backward_number=None,
        forward_rounding="stochastic",
        backward_rounding="stochastic",
    ):
        super(Quantizer, self).__init__()
        self.quantize = quantizer(
            forward_number, backward_number, forward_rounding, backward_rounding
        )

    def forward(self, x):
        return self.quantize(x)


class QAddFunction(torch.autograd.Function):
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


class QMulFunction(torch.autograd.Function):
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
class QDivFunction(torch.autograd.Function):
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


class QPowFunction(torch.autograd.Function):
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


class QSqrtFunction(torch.autograd.Function):
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


class QSumFunction(torch.autograd.Function):
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


class QMeanFunction(torch.autograd.Function):
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


class QLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats
        qinput = formats.input_quant(input)
        qweight = formats.param_quant(weight)

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
            qbias = formats.param_quant(bias)
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
        return output

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
            ones = torch.ones(qgrad_output.shape[0], 1, device=grad_output.device)
            if type(ctx.formats.bwd_add) == FloatingPoint:
                grad_bias = float_gemm(
                    qgrad_output.t(),
                    ones,
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
                    qgrad_output.t(),
                    ones,
                    wl_add=ctx.formats.bwd_add.wl,
                    fl_add=ctx.formats.bwd_add.fl,
                    wl_mul=ctx.formats.bwd_mul.wl,
                    fl_mul=ctx.formats.bwd_mul.fl,
                    rounding=ctx.formats.bwd_rnd,
                    symmetric=False,
                    fma=ctx.formats.bwd_fma,
                ).reshape(-1)

        return grad_input, grad_weight, grad_bias, None, None, None


class QLinear(nn.Module):
    def __init__(self, in_features, out_features, formats, bias=True):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.formats = formats
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data = self.formats.param_quant(self.weight.data)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias.data = self.formats.param_quant(self.bias.data)

    def quant_parameters(self):
        self.weight.data = self.formats.param_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.param_quant(self.bias.data)

    def forward(self, input):
        return QLinearFunction.apply(input, self.weight, self.bias, self.formats)


# TODO: will probably need to update this function a bit to allow for different
# kernel dimensions in the horizontal and vertical directions
class QConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        formats,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(QConv2d, self).__init__()
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.formats = formats
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        self.weight.data = self.formats.param_quant(self.weight.data)
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias.data = self.formats.param_quant(self.bias.data)

    def quant_parameters(self):
        self.weight.data = self.formats.param_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.param_quant(self.bias.data)

    def forward(self, input):
        batch, in_channel, in_height, in_width = input.shape
        num_filter, channel, kernel_h, kernel_w = self.weight.shape

        out_height = (in_height - kernel_h + 2 * self.padding) // self.stride + 1
        out_width = (in_width - kernel_w + 2 * self.padding) // self.stride + 1

        def tmp_matmul(a, b, bias, formats):
            assert len(a.shape) == 3
            assert len(b.shape) == 2
            assert a.shape[2] == b.shape[1]
            batch, m, k = a.shape
            n = b.shape[0]
            a = a.contiguous()
            b = b.contiguous()
            a = a.view(batch * m, k)
            return QLinearFunction.apply(a, b, bias, formats).view(batch, m, n)

        inp_unf = torch.nn.functional.unfold(
            input, self.kernel_size, stride=self.stride, padding=self.padding
        ).transpose(1, 2)
        out_unf = tmp_matmul(
            inp_unf, self.weight.view(self.weight.size(0), -1), self.bias, self.formats
        ).transpose(1, 2)
        return out_unf.view(batch, num_filter, out_height, out_width)


class QAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x, out_h, out_w, k_h, k_w, s_h, s_w, divisor, fwd_quant, bwd_quant
    ):
        ctx.divisor = divisor
        ctx.k_h = k_h
        ctx.k_w = k_w
        ctx.s_h = s_h
        ctx.s_w = s_w
        ctx.bwd_quant = bwd_quant
        ctx.save_for_backward(x)
        batch, in_channel, _, _ = x.shape
        y = torch.zeros((batch, in_channel, out_h, out_w), device=x.device)
        for h in range(out_h):
            for w in range(out_w):
                for m in range(k_h):
                    for n in range(k_w):
                        y[:, :, h, w] = fwd_quant(
                            y[:, :, h, w] + x[:, :, s_h * h + m, s_w * w + n]
                        )
        y = fwd_quant(y / divisor)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x,) = ctx.saved_tensors
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros(x.shape, device=grad_y.device)
            _, _, o_h, o_w = grad_y.shape
            for h in range(o_h):
                for w in range(o_w):
                    for kh in range(ctx.k_h):
                        for kw in range(ctx.k_w):
                            grad_x[
                                :, :, h * (ctx.s_h) + kh, w * (ctx.s_w) + kw
                            ] = ctx.bwd_quant(
                                grad_x[:, :, h * (ctx.s_h) + kh, w * (ctx.s_w) + kw]
                                + grad_y[:, :, h, w]
                            )
        grad_x = ctx.bwd_quant(grad_x / ctx.divisor)
        return grad_x, None, None, None, None, None, None, None, None, None


class QAvgPool2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        fwd_quant,
        bwd_quant,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        super(QAvgPool2d, self).__init__()
        self.fwd_quant = fwd_quant
        self.bwd_quant = bwd_quant
        if isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else:
            self.kernel_size = (kernel_size, kernel_size)
        if stride is not None:
            if isinstance(stride, tuple):
                self.stride = stride
            else:
                self.stride = (stride, stride)
        else:
            self.stride = self.kernel_size
        if isinstance(padding, tuple):
            self.padding = padding
        else:
            self.padding = (padding, padding)
        self.padF = torch.nn.ZeroPad2d(
            (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )
        # TODO: need to investigate how to handle these cases (ceil_mode and/or count_include_pad)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        if divisor_override is None:
            self.divisor = self.kernel_size[0] * self.kernel_size[1]
        else:
            self.divisor = divisor_override

    def forward(self, input):
        batch, in_channel, in_height, in_width = input.shape
        kernel_h, kernel_w = self.kernel_size
        if self.stride is None:
            self.stride = 1
        if self.ceil_mode:
            out_height = math.ceil(
                (in_height - kernel_h + 2 * self.padding[0]) / self.stride[0] + 1
            )
            out_width = math.ceil(
                (in_width - kernel_w + 2 * self.padding[1]) / self.stride[1] + 1
            )
        else:
            out_height = (in_height - kernel_h + 2 * self.padding[0]) // self.stride[
                0
            ] + 1
            out_width = (in_width - kernel_w + 2 * self.padding[1]) // self.stride[
                1
            ] + 1
        pinput = self.padF(input)
        return QAvgPool2dFunction.apply(
            pinput,
            out_height,
            out_width,
            kernel_h,
            kernel_w,
            self.stride[0],
            self.stride[1],
            self.divisor,
            self.fwd_quant,
            self.bwd_quant,
        )


class QMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_h, out_w, k_h, k_w, s_h, s_w, bwd_quant):
        ctx.bwd_quant = bwd_quant
        ctx.k_h = k_h
        ctx.k_w = k_w
        ctx.s_h = s_h
        ctx.s_w = s_w
        ctx.save_for_backward(x)
        batch, in_channel, _, _ = x.shape
        y = torch.zeros((batch, in_channel, out_h, out_w), device=x.device)
        for h in range(out_h):
            for w in range(out_w):
                y[:, :, h, w] = torch.amax(
                    x[:, :, s_h * h : s_h * h + k_h, s_w * w : s_w * w + k_w],
                    dim=(2, 3),
                )
        return y

    @staticmethod
    def backward(ctx, grad_y):
        (x,) = ctx.saved_tensors
        grad_x = torch.zeros_like(x, device=x.device)
        _, _, o_h, o_w = grad_y.shape
        for h in range(o_h):
            for w in range(o_w):
                for kh in range(ctx.k_h):
                    for kw in range(ctx.k_w):
                        tmp = x[:, :, h * ctx.s_h + kh, w * ctx.s_w + kw]
                        mask = tmp == torch.max(tmp)
                        grad_x[
                            :, :, h * ctx.s_h + kh, w * ctx.s_w + kw
                        ] = ctx.bwd_quant(
                            grad_x[:, :, h * ctx.s_h + kh, w * ctx.s_w + kw]
                            + grad_y[:, :, h, w] * mask
                        )
        return grad_x, None, None, None, None, None, None, None


def batch_norm(
    x, weight, bias, moving_mean, moving_var, eps, momentum, fwd_quant, bwd_quant
):
    if not torch.is_grad_enabled():
        x_hat = fwd_quant(
            fwd_quant(x - moving_mean)
            / fwd_quant(torch.sqrt(fwd_quant(moving_var + eps)))
        )
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            mean = QMeanFunction.apply(x, fwd_quant, bwd_quant, 2, False)
            var = QMeanFunction.apply(
                QPowFunction.apply(
                    QAddFunction.apply(x, -mean, fwd_quant, bwd_quant),
                    fwd_quant,
                    bwd_quant,
                    2,
                ),
                fwd_quant,
                bwd_quant,
                2,
                False,
            )
        else:
            mean = QMeanFunction.apply(x, fwd_quant, bwd_quant, (0, 2, 3), True)
            var = QMeanFunction.apply(
                QPowFunction.apply(
                    QAddFunction.apply(x, -mean, fwd_quant, bwd_quant),
                    fwd_quant,
                    bwd_quant,
                    2,
                ),
                fwd_quant,
                bwd_quant,
                (0, 2, 3),
                True,
            )
        x_hat = QDivFunction.apply(
            QAddFunction.apply(x, -mean, fwd_quant, bwd_quant),
            QSqrtFunction.apply(var + eps, fwd_quant, bwd_quant),
            fwd_quant,
            bwd_quant,
        )
        # moving mean and moving average do not have gradients that need to be recorded
        mfactor = fwd_quant(torch.tensor(1.0 - momentum, device=x.device))
        moving_mean = fwd_quant(momentum * moving_mean)
        diff_mean = fwd_quant(mfactor * mean)
        moving_mean = fwd_quant(moving_mean + diff_mean)
        moving_var = fwd_quant(momentum * moving_var)
        diff_var = fwd_quant(mfactor * var)
        moving_var = fwd_quant(moving_var + diff_var)

    y = QAddFunction.apply(
        QMulFunction.apply(weight, x_hat, fwd_quant, bwd_quant),
        bias,
        fwd_quant,
        bwd_quant,
    )
    return y, moving_mean.data, moving_var.data


class QBatchNorm(nn.Module):
    def __init__(self, num_features, num_dims, fwd_quant, bwd_quant):
        super().__init__()

        self.fwd_quant = fwd_quant
        self.bwd_quant = bwd_quant
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.weight = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape))

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)

        y, self.moving_mean, self.moving_var = batch_norm(
            x,
            self.weight,
            self.bias,
            self.moving_mean,
            self.moving_var,
            eps=1e-5,
            momentum=0.9,
            fwd_quant=self.fwd_quant,
            bwd_quant=self.bwd_quant,
        )
        return y


class QBatchNorm1d(QBatchNorm):
    def __init__(self, num_features, fwd_quant, bwd_quant):
        super().__init__(
            num_features, num_dims=2, fwd_quant=fwd_quant, bwd_quant=bwd_quant
        )


class QBatchNorm2d(QBatchNorm):
    def __init__(self, num_features, fwd_quant, bwd_quant):
        super().__init__(
            num_features, num_dims=4, fwd_quant=fwd_quant, bwd_quant=bwd_quant
        )
