from typing import Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

from ..quant_format import QSoftmaxFormats
from ..functional import qsoftmax

__all__ = [
    "QSoftmax",
]


class QSoftmax(nn.Softmax):
    r"""A quantized implementation of the Softmax activation function.

    This class extends PyTorch's :class:`torch.nn.Softmax` and allows one to specify if I/O
    signals and internal computations should be quantized during inference & training.
    This allows simulating the effect of custom precision in the internal forward and
    backward pass computations and is helpful in studying the effect of low precision
    compute during inference and training (not just data quantization).
    """

    def __init__(self, dim: int, formats: QSoftmaxFormats) -> None:
        r"""
        Args:
            dim: The dimension along which Softmax will be applied.
            formats: Number formats specification during compute
                and quantization functions for signals during forward and back propagation
                (I/O activations and gradients).
        """
        super(QSoftmax, self).__init__(dim)
        self.formats = formats
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized I/O signals in the module, depending if quantizers are
        specified in the associated :class:`QSoftmaxFormats` :attr:`formats` parameter.
        """
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

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
                return bwd_quant(grad_output)

        return round.apply

    def forward(self, input) -> torch.Tensor:
        r"""Performs the softmax operation on the input tensor. The use of
        quantized elementary operations (i.e., additions and multiplications) in
        the FWD and BWD passes are controlled through the :attr:`formats` argument
        to the module constructor.

        Args:
            input: the input tensor over which to perform the softmax operations.

        Returns:
            the result of the softmax operation.
        """
        if self.formats.fwd_use_default_prec and self.formats.bwd_use_default_prec:
            return self.Qo(F.softmax(self.Qi(input), dim=self.dim))
        else:
            return qsoftmax(input, dim=self.dim, formats=self.formats)
