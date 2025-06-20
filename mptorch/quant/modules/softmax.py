import torch.nn as nn
import torch.nn.functional as F
import torch

from ..quant_format import QSoftmaxFormats
from ..functional import qsoftmax

__all__ = ["QSoftmax",]

class QSoftmax(nn.Softmax):
    r"""A quantized implementation of the Softmax activation function.

    This class extends PyTorch's :class:`torch.nn.Softmax` and allows one to specify if I/O
    signals and internal computations should be quantized during inference & training.
    This allows simulating the effect of custom precision in the internal forward and
    backward pass computations and is helpful in studying the effect of low precision
    compute during inference and training (not just data quantization).

    Args:
        dim (int): The dimension along which Softmax will be applied.
        formats (QSoftmaxFormats): Number formats specification during compute
            and quantization functions for signals during forward and back propagation
            (I/O activations and gradients).
    """

    def __init__(self,
                 dim: int,
                 formats: QSoftmaxFormats) -> None:
        super(QSoftmax, self).__init__(dim)
        self.formats = formats
        self.reset_quant_function()
    
    def reset_quant_function(self):
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)
    
    def quant_function(self, fwd_quant):
        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input):
        if self.formats.fwd_use_default_prec and self.formats.bwd_use_default_prec:
            return self.Qo(F.softmax(self.Qi(input), dim=self.dim))
        else:
            return qsoftmax(input, dim=self.dim, formats=self.formats)