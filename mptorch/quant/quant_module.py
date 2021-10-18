import torch
import torch.nn as nn 
from .quant_function import *
import numpy as np 
import math

__all__ = ["Quantizer", "QLinear", "QConv2d", "QAvgPool2d", "QBatchNorm", "QBatchNorm1d", "QBatchNorm2d", "QAddFunction", "QMulFunction"]

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
    def forward(ctx, x, y, exp=8, man=23, rounding='nearest'):
        ctx.save_for_backward(x, y)
        ctx.man = man
        ctx.exp = exp
        ctx.rounding = rounding
        z = x + y
        z = float_quantize(z.contiguous(), exp=exp, man=man, rounding=rounding)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        grad_x = grad_z * torch.ones_like(x, device=x.device)
        grad_y = grad_z * torch.ones_like(y, device=y.device) 
        # need to compute gradients in order of inputs
        grad_x = float_quantize(grad_x.contiguous(), exp=ctx.exp, man=ctx.man, 
            rounding=ctx.rounding)
        grad_y = float_quantize(grad_y.contiguous(), exp=ctx.exp, man=ctx.man, 
            rounding=ctx.rounding)
        return grad_x, grad_y, None, None, None

class QMulFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, exp=8, man=23, rounding='nearest'):
        ctx.save_for_backward(x, y)
        ctx.man = man
        ctx.exp = exp
        ctx.rounding = rounding
        z = x * y
        z = float_quantize(z.contiguous(), exp=exp, man=man, rounding=rounding)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y = ctx.saved_tensors
        grad_x = grad_z * y
        grad_y = grad_z * x
        grad_x = float_quantize(grad_x.contiguous(), exp=ctx.exp, man=ctx.man, 
            rounding=ctx.rounding)
        grad_y = float_quantize(grad_y.contiguous(), exp=ctx.exp, man=ctx.man, 
            rounding=ctx.rounding)
        return grad_x, grad_y, None, None, None

# see the following link for a discussion regarding numerical stability of 
# backward propagation for division operations in PyTorch and for the basis 
# of this implementation: https://github.com/pytorch/pytorch/issues/43414
class QDivFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, exp=8, man=23, rounding='nearest'):
        ctx.exp = exp
        ctx.man = man
        ctx.rounding = rounding
        z = float_quantize(x / y, exp=exp, man=man, rounding=rounding)
        ctx.save_for_backward(x, y, z)
        return z

    @staticmethod
    def backward(ctx, grad_z):
        x, y, z = ctx.saved_tensors
        grad_x = grad_y = None
        if ctx.needs_input_grad[0]:
            grad_x = float_quantize( grad_z / y, 
                        exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        if ctx.needs_input_grad[1]:                
            grad_y = float_quantize(-grad_z * z, 
                        exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
            grad_y = float_quantize(grad_y / y, 
                        exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        return grad_x, grad_y, None, None, None

class QPowFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, n=2, exp=8, man=23, rounding='nearest'):
        ctx.man = man
        ctx.exp = exp
        ctx.rounding = rounding
        ctx.n = n
        y = float_quantize(x**n, exp=exp, man=man, rounding=rounding)
        gy = float_quantize(x**(n-1), exp=exp, man=man, rounding=rounding)
        ctx.save_for_backward(gy)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        gy, = ctx.saved_tensors
        grad_x = float_quantize(gy * ctx.n, exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        grad_x = float_quantize(grad_y * grad_x, exp=ctx.exp, man=ctx.man, 
                        rounding=ctx.rounding)
        return grad_x, None, None, None, None

class QSqrtFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exp=8, man=23, rounding='nearest'):
        ctx.man = man
        ctx.exp = exp
        ctx.rounding = rounding
        x = torch.sqrt(x)
        y = float_quantize(x, exp=exp, man=man, rounding=rounding)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        y, = ctx.saved_tensors
        grad_x = float_quantize(y * 2, exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        grad_x = float_quantize(grad_y / grad_x, 
                    exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        return grad_x, None, None, None

def qsum_kernel(x, dim, exp=8, man=23, rounding='nearest'):
    shape = list(x.shape)
    shape[dim] = 1
    vs = torch.zeros(shape, device=x.device)
    vs = vs.transpose(0, dim).reshape(1, -1).transpose(0, 1)
    vx = torch.transpose(x, 0, dim).reshape(x.shape[dim], -1).transpose(0, 1)
    shape[dim] = x.shape[0]
    shape[0] = 1
    for k in range(x.shape[dim]):
        vs = float_quantize(vs + vx[:, k:k+1], exp=exp, man=man, rounding=rounding)
    return vs.transpose(0, 1).reshape(shape).transpose(0, dim)

class QSumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=0, keepdim=False, exp=8, man=23, rounding ='nearest'):
        ctx.exp = exp
        ctx.man = man
        ctx.rounding = rounding
        ctx.save_for_backward(x)

        if dim is None:
            dim = range(x.dim())
        elif isinstance(dim, tuple) == False:
            dim = (dim,)

        ctx.dim = dim
        sums = x
        for d in dim:
            sums = qsum_kernel(sums, d, exp=exp, man=man, rounding=rounding)
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
        x, = ctx.saved_tensors
        for d in ctx.dim:
            grad_output = torch.unsqueeze(grad_output, d)
        grad_x = grad_output * torch.ones_like(x, device=x.device)
        return grad_x, None, None, None, None, None

class QMeanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=3, keepdim=False, exp=8, man=23, rounding='nearest'):
        ctx.exp = exp
        ctx.man = man
        ctx.rounding = rounding
        ctx.save_for_backward(x)

        if dim is None:
            dim = range(x.dim())
        elif isinstance(dim, tuple) == False:
            dim = (dim,)

        ctx.dim = dim
        sums = x
        numel = 1
        for d in dim:
            sums = qsum_kernel(sums, d, exp=exp, man=man, rounding=rounding)
            numel *= x.shape[d]
        ctx.numel = numel
        sums = float_quantize(sums / numel, exp=exp, man=man, rounding=rounding)
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
        x, = ctx.saved_tensors
        for d in ctx.dim:
            grad_output = torch.unsqueeze(grad_output, d)
        grad_x = float_quantize(grad_output * torch.ones_like(x, device=x.device) / ctx.numel, 
                                exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        return grad_x, None, None, None, None, None

class QLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, exp=8, man=23, rounding='nearest'):
        ctx.exp = exp
        ctx.man = man
        ctx.rounding = rounding
        qweight = float_quantize(weight, exp=exp, man=man, rounding=rounding)
        output = quant_gemm(input, qweight.t(), man, exp)
        if bias is not None:
            qbias = float_quantize(bias, exp=exp, man=man, rounding=rounding)
            output += qbias.unsqueeze(0).expand_as(output)
            ctx.save_for_backward(input, qweight, qbias)
        else:
            ctx.save_for_backward(input, qweight, bias)
        output = float_quantize(output.contiguous(), exp=exp, man=man, rounding=rounding)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = quant_gemm(grad_output, weight, ctx.man, ctx.exp)
        if ctx.needs_input_grad[1]:
            grad_weight = quant_gemm(grad_output.t(), input, ctx.man, ctx.exp)
        if bias is not None and ctx.needs_input_grad[2]:
            ones = torch.ones(grad_output.shape[0], 1, device=grad_output.device)
            grad_bias = quant_gemm(grad_output.t(), ones, ctx.man, ctx.exp).reshape(-1)

        return grad_input, grad_weight, grad_bias, None, None, None

class QLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, exp=8, man=23, rounding='nearest'):
        super(QLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.exp = exp
        self.man = man
        self.rounding = rounding
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return QLinearFunction.apply(input, self.weight, self.bias, self.exp, self.man, self.rounding)


# TODO: will probably need to update this function a bit to allow for different
# kernel dimensions in the horizontal and vertical directions
class QConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, exp=8, man=23, rounding='nearest'):
        super(QConv2d, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.reset_parameters()
        self.man = man 
        self.exp = exp
        self.rounding = rounding

    def reset_parameters(self):
        n = self.in_channels
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        batch, in_channel, in_height, in_width = input.shape
        num_filter, channel, kernel_h, kernel_w = self.weight.shape 

        out_height = (in_height - kernel_h + 2 * self.padding) // self.stride + 1
        out_width  = (in_width  - kernel_w + 2 * self.padding) // self.stride + 1

        def tmp_matmul(a, b, bias, exp, man):
            assert len(a.shape) == 3
            assert len(b.shape) == 2
            assert a.shape[2] == b.shape[1]
            batch, m, k = a.shape
            n = b.shape[0]
            a = a.contiguous()
            b = b.contiguous()
            a = a.view(batch*m, k)
            return QLinearFunction.apply(a, b, bias, exp, man).view(batch, m, n)

        inp_unf = torch.nn.functional.unfold(
        input, self.kernel_size, stride=self.stride, padding=self.padding).transpose(1, 2)
        out_unf = tmp_matmul(inp_unf, self.weight.view(self.weight.size(0), -1), self.bias,
                         self.exp, self.man).transpose(1, 2)
        return out_unf.view(batch, num_filter, out_height, out_width)

class QAvgPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_h, out_w, k_h, k_w, s_h, s_w, divisor, 
                exp=8, man=23, rounding='nearest'):
        ctx.exp = exp
        ctx.man = man
        ctx.rounding = rounding
        ctx.divisor = divisor
        ctx.k_h = k_h
        ctx.k_w = k_w
        ctx.s_h = s_h
        ctx.s_w = s_w
        ctx.save_for_backward(x)
        batch, in_channel, _, _ = x.shape
        y = torch.zeros((batch, in_channel, out_h, out_w), device=x.device)
        for h in range(out_h):
            for w in range(out_w):
                for m in range(k_h):
                    for n in range(k_w):
                        y[:, :, h, w] = float_quantize(y[:, :, h, w] + 
                            x[:, :, s_h*h+m, s_w*w+n], exp=exp, man=man, rounding=rounding)
        y = float_quantize(y / divisor, exp=exp, man=man, rounding=rounding)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.zeros(x.shape, device=grad_y.device)
            _, _, o_h, o_w = grad_y.shape
            for h in range(o_h):
                for w in range(o_w):
                    for kh in range(ctx.k_h):
                        for kw in range(ctx.k_w):
                            grad_x[:, :, h*(ctx.s_h)+kh, w*(ctx.s_w)+kw] = float_quantize(
                                grad_x[:, :, h*(ctx.s_h)+kh, w*(ctx.s_w)+kw] + 
                                grad_y[:, :, h, w], exp=ctx.exp, man=ctx.man,
                                rounding=ctx.rounding)
        grad_x = float_quantize(grad_x / ctx.divisor, 
                                    exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        return grad_x, None, None, None, None, None, None, None, None, None, None

    
class QAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
          count_include_pad=True, divisor_override=None, exp=8, man=23, rounding='nearest'):
        super(QAvgPool2d, self).__init__()
        self.man = man 
        self.exp = exp
        self.rounding = rounding
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
        self.padF = torch.nn.ZeroPad2d((self.padding[1], self.padding[1], 
                                    self.padding[0], self.padding[0]))
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
            out_height = math.ceil((in_height - kernel_h + 2 * self.padding[0]) / self.stride[0] + 1)
            out_width = math.ceil((in_width - kernel_w + 2 * self.padding[1]) / self.stride[1] + 1)
        else:
            out_height = (in_height - kernel_h + 2 * self.padding[0]) // self.stride[0] + 1
            out_width = (in_width - kernel_w + 2 * self.padding[1]) // self.stride[1] + 1
        pinput = self.padF(input)
        return QAvgPool2dFunction.apply(pinput, out_height, out_width, 
            kernel_h, kernel_w, self.stride[0], self.stride[1], 
            self.divisor, self.exp, self.man, self.rounding)

class QMaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_h, out_w, k_h, k_w, s_h, s_w, 
                exp=8, man=23, rounding='nearest'):
        ctx.exp = exp
        ctx.man = man
        ctx.rounding = rounding
        ctx.k_h = k_h
        ctx.k_w = k_w
        ctx.s_h = s_h
        ctx.s_w = s_w
        ctx.save_for_backward(x)
        batch, in_channel, _, _ = x.shape
        y = torch.zeros((batch, in_channel, out_h, out_w), device=x.device)
        for h in range(out_h):
            for w in range(out_w):
                y[:, :, h, w] = torch.amax(x[:, :, s_h*h:s_h*h+k_h, s_w*w:s_w*w+k_w], dim=(2, 3))
        return y
    @staticmethod
    def backward(ctx, grad_y):
        x, = ctx.saved_tensors
        grad_x = torch.zeros_like(x, device=x.device)
        _, _, o_h, o_w = grad_y.shape
        for h in range(o_h):
            for w in range(o_w):
                for kh in range(ctx.k_h):
                    for kw in range(ctx.k_w):
                        tmp = x[:, :, h*ctx.s_h+kh, w*ctx.s_w+kw]
                        mask = (tmp == torch.max(tmp))
                        grad_x[:, :, h*ctx.s_h+kh, w*ctx.s_w+kw] = float_quantize(
                            grad_x[:, :, h*ctx.s_h+kh, w*ctx.s_w+kw] + grad_y[:, :, h, w] * mask, 
                            exp=ctx.exp, man=ctx.man, rounding=ctx.rounding)
        return grad_x, None, None, None, None, None, None, None, None, None

def batch_norm(x, weight, bias, moving_mean, moving_var, eps, momentum, 
               exp=8, man=23, rounding='nearest'):
    if not torch.is_grad_enabled():
        x_hat = float_quantize(float_quantize(x - moving_mean, exp=exp, man=man, rounding=rounding) 
        / float_quantize(torch.sqrt(float_quantize(moving_var + eps, exp=exp, man=man, 
        rounding=rounding)), exp=exp, man=man, rounding=rounding), exp=exp, man=man, 
        rounding=rounding)
    else:
        assert len(x.shape) in (2, 4)
        if len(x.shape) == 2:
            mean = QMeanFunction.apply(x, 2, False, exp, man, rounding)
            var = QMeanFunction.apply(QPowFunction.apply(
                QAddFunction.apply(x, -mean, exp, man, rounding), 
                2, exp, man, rounding), 2, False, exp, man, rounding)
        else:
            mean = QMeanFunction.apply(x, (0, 2, 3), True, exp, man, rounding)
            var = QMeanFunction.apply(QPowFunction.apply(
                    QAddFunction.apply(x, -mean, exp, man, rounding), 
                    2, exp, man, rounding), (0, 2, 3), True, exp, man, rounding)
        x_hat = QDivFunction.apply(QAddFunction.apply(x, -mean, exp, man, rounding), 
                                QSqrtFunction.apply(var + eps, exp, man, rounding),
                                exp, man, rounding)
        # moving mean and moving average do not have gradients that need to be recorded
        mfactor = float_quantize(torch.tensor(1.0 - momentum, device=x.device), 
                    exp=exp, man=man, rounding=rounding)
        moving_mean = float_quantize(momentum * moving_mean, exp=exp, man=man, rounding=rounding)
        diff_mean = float_quantize(mfactor * mean, exp=exp, man=man, rounding=rounding)
        moving_mean = float_quantize(moving_mean + diff_mean, exp=exp, man=man, rounding=rounding)
        moving_var = float_quantize(momentum * moving_var, exp=exp, man=man, rounding=rounding)
        diff_var = float_quantize(mfactor * var, exp=exp, man=man, rounding=rounding)
        moving_var = float_quantize(moving_var + diff_var, exp=exp, man=man, rounding=rounding)

    y = QAddFunction.apply(QMulFunction.apply(weight, x_hat, exp, man, rounding),
                        bias, exp, man, rounding)
    return y, moving_mean.data, moving_var.data

class QBatchNorm(nn.Module):
    def __init__(self, num_features, num_dims, exp=8, man=23, rounding='nearest'):
        super().__init__()

        self.exp = exp
        self.man = man
        self.rounding = rounding
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
            x, self.weight, self.bias, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9, exp=self.exp, man=self.man, rounding=self.rounding
        )
        return y

class QBatchNorm1d(QBatchNorm):
    def __init__(self, num_features, exp=8, man=23, rounding='nearest'):
        super().__init__(num_features, num_dims=2, exp=exp, man=man, rounding=rounding)

class QBatchNorm2d(QBatchNorm):
    def __init__(self, num_features, exp=8, man=23, rounding='nearest'):
        super().__init__(num_features, num_dims=4, exp=exp, man=man, rounding=rounding)