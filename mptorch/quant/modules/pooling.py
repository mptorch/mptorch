import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.common_types import _size_2_t
from typing import Callable, Optional

__all__ = ["QAvgPool2d"]


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
                            grad_x[:, :, h * (ctx.s_h) + kh, w * (ctx.s_w) + kw] = (
                                ctx.bwd_quant(
                                    grad_x[:, :, h * (ctx.s_h) + kh, w * (ctx.s_w) + kw]
                                    + grad_y[:, :, h, w]
                                )
                            )
        grad_x = ctx.bwd_quant(grad_x / ctx.divisor)
        return grad_x, None, None, None, None, None, None, None, None, None


class QAvgPool2d(nn.AvgPool2d):
    def __init__(
        self,
        kernel_size: _size_2_t,
        fwd_quant: Callable,
        bwd_quant: Callable,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super(QAvgPool2d).__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )

        self.fwd_quant = fwd_quant
        self.bwd_quant = bwd_quant
        if isinstance(padding, tuple):
            self.padding = padding
        else:
            self.padding = (padding, padding)
        self.padF = torch.nn.ZeroPad2d(
            (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        )

    def forward(self, input: Tensor) -> Tensor:
        _, _, in_height, in_width = input.shape
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
