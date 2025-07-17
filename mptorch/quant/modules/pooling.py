import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Callable

__all__ = ["QAvgPool2d"]

# TODO: interface changes and support for other layers


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
    r"""Applies a 2D average pooling over an input signal composed of several input planes. Performs
    the addition operations in the FWD and BWD passes using quantized operators given as parameters.

    In the simplest case, the output value of the layer with input size :math:`(N, C, H, W)`,
    output :math:`(N, C, H_\text{out}, W_\text{out})` and :attr:`kernel_size` :math:`(kH, kW)`
    can be precisely described as:

    .. math::

        \text{out}(N_i, C_j, h, w)  = \frac{1}{kH * kW} \sum_{m=0}^{kH-1} \sum_{n=0}^{kW-1}
                               \text{input}(N_i, C_j, \text{stride}[0] \times h + m, \text{stride}[1] \times w + n)

    If :attr:`padding` is non-zero, then the input is implicitly zero-padded on both sides
    for :attr:`padding` number of points.

    Note:
        When ceil_mode=True, sliding windows are allowed to go off-bounds if they start within the left padding
        or the input. Sliding windows that would start in the right padded region are ignored.

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Shape:
        - Input: :math:`(N, C, H_\text{in}, W_\text{in})` or :math:`(C, H_\text{in}, W_\text{in})`.
        - Output: :math:`(N, C, H_\text{out}, W_\text{out})` or :math:`(C, H_\text{out}, W_\text{out})`, where

          .. math::
              H_\text{out} = \left\lfloor\frac{H_\text{in}  + 2 \times \text{padding}[0] -
                \text{kernel_size}[0]}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_\text{out} = \left\lfloor\frac{W_\text{in}  + 2 \times \text{padding}[1] -
                \text{kernel_size}[1]}{\text{stride}[1]} + 1\right\rfloor

          Per the note above, if ``ceil_mode`` is True and :math:`(H_\text{out} - 1)\times \text{stride}[0]\geq H_\text{in}
          + \text{padding}[0]`, we skip the last window as it would start in the bottom padded region,
          resulting in :math:`H_\text{out}` being reduced by one.

          The same applies for :math:`W_\text{out}`.
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ):
        r"""
        Args:
            kernel_size: the size of the window
            fwd_quant: quantization function to use during FWD addition operations
            bwd_quant: quantization function to use during BWD addition operations
            stride: the stride of the window. Default value is :attr:`kernel_size`
            padding: implicit zero padding to be added on both sides
            ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
            count_include_pad: when True, will include the zero-padding in the averaging calculation
            divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used.
        """
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Performs the pooling operation over the :attr:`input` tensor.

        Args:
            input: the input tensor over which to perform the pooling operation.
                Must adhere to the input shape requirements.

        Returns:
            the result of the pooling operation.
        """
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
