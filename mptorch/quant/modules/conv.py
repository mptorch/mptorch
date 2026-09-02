from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as grad

from .format import QAffineFormats


class CustomArithConvNd(torch.autograd.Function):
    """
    Custom Autograd function for quantized Conv layers (1D, 2D, 3D).
    It delegates the forward and backward passes to the functions
    provided in the QAffineFormats configuration, falling back to
    PyTorch defaults if overrides are not provided.
    """

    @staticmethod
    def _default_bwd_igrad(
        grad_output: torch.Tensor,
        weight: torch.Tensor,
        input_size,
        stride,
        padding,
        dilation,
        groups,
        nd,
    ) -> torch.Tensor:
        if nd == 1:
            return grad.conv1d_input(
                input_size, weight, grad_output, stride, padding, dilation, groups
            )
        elif nd == 2:
            return grad.conv2d_input(
                input_size, weight, grad_output, stride, padding, dilation, groups
            )
        elif nd == 3:
            return grad.conv3d_input(
                input_size, weight, grad_output, stride, padding, dilation, groups
            )
        raise ValueError(f"Unsupported nd: {nd}")

    @staticmethod
    def _default_bwd_wgrad(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        weight_size,
        stride,
        padding,
        dilation,
        groups,
        nd,
    ) -> torch.Tensor:
        if nd == 1:
            return grad.conv1d_weight(
                input, weight_size, grad_output, stride, padding, dilation, groups
            )
        elif nd == 2:
            return grad.conv2d_weight(
                input, weight_size, grad_output, stride, padding, dilation, groups
            )
        elif nd == 3:
            return grad.conv3d_weight(
                input, weight_size, grad_output, stride, padding, dilation, groups
            )
        raise ValueError(f"Unsupported nd: {nd}")

    @staticmethod
    def forward(ctx, input, weight, bias, formats, stride, padding, dilation, groups, nd):
        ctx.formats = formats
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.nd = nd
        ctx.input_size = input.shape
        ctx.weight_size = weight.shape

        q_input = formats.input_quant(input) if formats.input_quant else input
        q_weight = formats.weight_quant(weight) if formats.weight_quant else weight
        q_bias = formats.bias_quant(bias) if formats.bias_quant and bias is not None else bias

        if formats.fwd_math is not None:
            output = formats.fwd_math(
                q_input,
                q_weight,
                q_bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                nd=nd,
            )
        else:
            if nd == 1:
                output = F.conv1d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            elif nd == 2:
                output = F.conv2d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            elif nd == 3:
                output = F.conv3d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            else:
                raise ValueError(f"Unsupported nd: {nd}")

        # Save only what backward will actually read. q_bias is never read
        # there, and each of the other two feeds exactly one gradient path:
        # q_input the weight gradient, q_weight the input gradient. So a
        # layer whose weight is frozen no longer pins a quantized copy of its
        # entire activation for the length of the backward pass -- on top of
        # the activation autograd is already holding. ctx.needs_input_grad is
        # populated before forward runs and is the same tuple backward sees.
        saved_input = q_input if ctx.needs_input_grad[1] else None
        saved_weight = q_weight if ctx.needs_input_grad[0] else None

        if (saved_input is None or isinstance(saved_input, torch.Tensor)) and (
            saved_weight is None or isinstance(saved_weight, torch.Tensor)
        ):
            ctx.save_for_backward(saved_input, saved_weight)
            ctx.saved_structs = None
        else:
            ctx.save_for_backward()
            ctx.saved_structs = (saved_input, saved_weight)

        ctx.needs_bias_grad = bias is not None

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        formats = ctx.formats
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        nd = ctx.nd
        input_size = ctx.input_size
        weight_size = ctx.weight_size

        if ctx.saved_structs is None:
            q_input, q_weight = ctx.saved_tensors
        else:
            q_input, q_weight = ctx.saved_structs

        grad_input = grad_weight = grad_bias = None

        # Quantize grad_output only for the paths that will consume it -- a
        # frozen layer, or the first layer of a network, used to pay a full
        # grad-output-sized quantize kernel and allocation for a tensor it
        # then discarded. Both quantizations stay ahead of both math calls,
        # in this order, so a stochastic quantizer draws from the global RNG
        # in exactly the sequence it did when these were unconditional; the
        # *_math hooks can consume randomness of their own.
        q_igrad_output = (
            (formats.igrad_quant(grad_output) if formats.igrad_quant else grad_output)
            if ctx.needs_input_grad[0]
            else None
        )
        q_wgrad_output = (
            (formats.wgrad_quant(grad_output) if formats.wgrad_quant else grad_output)
            if ctx.needs_input_grad[1]
            else None
        )

        if q_igrad_output is not None:
            bwd_igrad_math = (
                formats.bwd_igrad_math
                if formats.bwd_igrad_math
                else CustomArithConvNd._default_bwd_igrad
            )
            grad_input = bwd_igrad_math(
                q_igrad_output,
                q_weight,
                input_size=input_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                nd=nd,
            )

        if q_wgrad_output is not None:
            bwd_wgrad_math = (
                formats.bwd_wgrad_math
                if formats.bwd_wgrad_math
                else CustomArithConvNd._default_bwd_wgrad
            )
            grad_weight = bwd_wgrad_math(
                q_wgrad_output,
                q_input,
                weight_size=weight_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                nd=nd,
            )

        if ctx.needs_bias_grad and ctx.needs_input_grad[2]:
            q_bgrad_output = (
                formats.bgrad_quant(grad_output) if formats.bgrad_quant else grad_output
            )

            # Sum over N and spatial dimensions to get bias gradient
            # nd=1 -> [0, 2], nd=2 -> [0, 2, 3], nd=3 -> [0, 2, 3, 4]
            dims_to_sum = [0] + list(range(2, nd + 2))
            grad_bias = q_bgrad_output.sum(dim=dims_to_sum)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


# Alias for backwards compatibility
CustomArithConv2d = CustomArithConvNd


class QConv1d(nn.Conv1d):
    """
    Quantized version of torch.nn.Conv1d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,)
        else:
            padding = self.padding

        return CustomArithConvNd.apply(
            input,
            self.weight,
            self.bias,
            self.formats,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            1,
        )


class QConv2d(nn.Conv2d):
    """
    Quantized version of torch.nn.Conv2d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        return CustomArithConvNd.apply(
            input,
            self.weight,
            self.bias,
            self.formats,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            2,
        )


class QConv3d(nn.Conv3d):
    """
    Quantized version of torch.nn.Conv3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0, 0)
        else:
            padding = self.padding

        return CustomArithConvNd.apply(
            input,
            self.weight,
            self.bias,
            self.formats,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            3,
        )
