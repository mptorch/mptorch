import torch
import torch.nn.functional as F
from torch import nn

from .format import QAffineFormats


class CustomArithLinear(torch.autograd.Function):
    """
    Custom Autograd function for quantized linear layers.
    It delegates the forward and backward passes to the functions
    provided in the QAffineFormats configuration, falling back to
    PyTorch defaults if overrides are not provided.
    """

    @staticmethod
    def _default_bwd_igrad(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return grad_output.matmul(weight)

    @staticmethod
    def _default_bwd_wgrad(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        gO_flat = grad_output.reshape(-1, grad_output.shape[-1])
        i_flat = input.reshape(-1, input.shape[-1])
        return gO_flat.t().matmul(i_flat)

    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats

        q_input = formats.input_quant(input) if formats.input_quant else input
        q_weight = formats.weight_quant(weight) if formats.weight_quant else weight
        q_bias = formats.bias_quant(bias) if formats.bias_quant and bias is not None else bias

        if formats.fwd_math is not None:
            output = formats.fwd_math(q_input, q_weight, q_bias)
        else:
            output = F.linear(q_input, q_weight, q_bias)

        if (
            isinstance(q_input, torch.Tensor)
            and isinstance(q_weight, torch.Tensor)
            and (q_bias is None or isinstance(q_bias, torch.Tensor))
        ):
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

        if ctx.saved_structs is None:
            q_input, q_weight, _q_bias = ctx.saved_tensors
        else:
            q_input, q_weight, _q_bias = ctx.saved_structs

        q_igrad_output = formats.igrad_quant(grad_output) if formats.igrad_quant else grad_output
        q_wgrad_output = formats.wgrad_quant(grad_output) if formats.wgrad_quant else grad_output

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            bwd_igrad_math = (
                formats.bwd_igrad_math
                if formats.bwd_igrad_math
                else CustomArithLinear._default_bwd_igrad
            )
            grad_input = bwd_igrad_math(q_igrad_output, q_weight)

        if ctx.needs_input_grad[1]:
            bwd_wgrad_math = (
                formats.bwd_wgrad_math
                if formats.bwd_wgrad_math
                else CustomArithLinear._default_bwd_wgrad
            )
            grad_weight = bwd_wgrad_math(q_wgrad_output, q_input)

        if ctx.needs_bias_grad and ctx.needs_input_grad[2]:
            q_bgrad_output = (
                formats.bgrad_quant(grad_output) if formats.bgrad_quant else grad_output
            )
            grad_bias = q_bgrad_output.reshape(-1, q_bgrad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, grad_bias, None


class QLinear(nn.Linear):
    """
    Quantized version of torch.nn.Linear.
    It takes an optional QAffineFormats module to specify custom
    quantization and arithmetic for the forward and backward passes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomArithLinear.apply(input, self.weight, self.bias, self.formats)
