import torch
import torch.nn as nn
import torch.nn.grad as grad
import torch.nn.functional as F
from typing import Optional, Union, Tuple, Literal

from .format import QAffineFormats

class CustomArithConvNd(torch.autograd.Function):
    """
    Custom Autograd function for quantized Conv layers (1D, 2D, 3D).
    It delegates the forward and backward passes to the functions 
    provided in the QAffineFormats configuration, falling back to 
    PyTorch defaults if overrides are not provided.
    """
    @staticmethod
    def _default_bwd_igrad(grad_output: torch.Tensor, weight: torch.Tensor, 
                           input_size, stride, padding, dilation, groups, nd) -> torch.Tensor:
        if nd == 1:
            return grad.conv1d_input(input_size, weight, grad_output, stride, padding, dilation, groups)
        elif nd == 2:
            return grad.conv2d_input(input_size, weight, grad_output, stride, padding, dilation, groups)
        elif nd == 3:
            return grad.conv3d_input(input_size, weight, grad_output, stride, padding, dilation, groups)
        raise ValueError(f"Unsupported nd: {nd}")

    @staticmethod
    def _default_bwd_wgrad(grad_output: torch.Tensor, input: torch.Tensor, 
                           weight_size, stride, padding, dilation, groups, nd) -> torch.Tensor:
        if nd == 1:
            return grad.conv1d_weight(input, weight_size, grad_output, stride, padding, dilation, groups)
        elif nd == 2:
            return grad.conv2d_weight(input, weight_size, grad_output, stride, padding, dilation, groups)
        elif nd == 3:
            return grad.conv3d_weight(input, weight_size, grad_output, stride, padding, dilation, groups)
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
            output = formats.fwd_math(q_input, q_weight, q_bias, 
                                      stride=stride, padding=padding, 
                                      dilation=dilation, groups=groups, nd=nd)
        else:
            if nd == 1:
                output = F.conv1d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            elif nd == 2:
                output = F.conv2d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            elif nd == 3:
                output = F.conv3d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            else:
                raise ValueError(f"Unsupported nd: {nd}")
        
        if isinstance(q_input, torch.Tensor) and isinstance(q_weight, torch.Tensor) and (q_bias is None or isinstance(q_bias, torch.Tensor)):
            ctx.save_for_backward(q_input, q_weight, q_bias)
            ctx.saved_structs = None
        else:
            ctx.save_for_backward() 
            ctx.saved_structs = (q_input, q_weight, q_bias)

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
            q_input, q_weight, q_bias = ctx.saved_tensors
        else:
            q_input, q_weight, q_bias = ctx.saved_structs
        
        q_igrad_output = formats.igrad_quant(grad_output) if formats.igrad_quant else grad_output
        q_wgrad_output = formats.wgrad_quant(grad_output) if formats.wgrad_quant else grad_output
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            bwd_igrad_math = formats.bwd_igrad_math if formats.bwd_igrad_math else CustomArithConvNd._default_bwd_igrad
            grad_input = bwd_igrad_math(q_igrad_output, q_weight, 
                                        input_size=input_size, stride=stride, 
                                        padding=padding, dilation=dilation, groups=groups, nd=nd)
            
        if ctx.needs_input_grad[1]:
            bwd_wgrad_math = formats.bwd_wgrad_math if formats.bwd_wgrad_math else CustomArithConvNd._default_bwd_wgrad
            grad_weight = bwd_wgrad_math(q_wgrad_output, q_input, 
                                         weight_size=weight_size, stride=stride, 
                                         padding=padding, dilation=dilation, groups=groups, nd=nd)
            
        if ctx.needs_bias_grad and ctx.needs_input_grad[2]:
            q_bgrad_output = formats.bgrad_quant(grad_output) if formats.bgrad_quant else grad_output
            
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
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int], str] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = 'zeros',
        device=None,
        dtype=None,
        formats: Optional[QAffineFormats] = None
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,)
        else:
            padding = self.padding
            
        return CustomArithConvNd.apply(
            input, self.weight, self.bias, self.formats,
            self.stride, padding, self.dilation, self.groups, 1
        )


class QConv2d(nn.Conv2d):
    """
    Quantized version of torch.nn.Conv2d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = 'zeros',
        device=None,
        dtype=None,
        formats: Optional[QAffineFormats] = None
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding
            
        return CustomArithConvNd.apply(
            input, self.weight, self.bias, self.formats,
            self.stride, padding, self.dilation, self.groups, 2
        )


class QConv3d(nn.Conv3d):
    """
    Quantized version of torch.nn.Conv3d.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int], str] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = 'zeros',
        device=None,
        dtype=None,
        formats: Optional[QAffineFormats] = None
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0, 0)
        else:
            padding = self.padding
            
        return CustomArithConvNd.apply(
            input, self.weight, self.bias, self.formats,
            self.stride, padding, self.dilation, self.groups, 3
        )
