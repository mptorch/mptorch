from typing import Callable, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant_format import QGELUFormats
from ..functional import qgelu

__all__ = ["QGELU"]


class QGELU(nn.GELU):
    r"""
    Applies the Gaussian Error Linear Units function to the input :math:`x`:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    When the approximate argument is ``'tanh'``, GELU is estimated with:

    .. math:: \text{GELU}(x) = 0.5 * x * (1 + \tanh(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))
    """

    def __init__(
        self, formats: QGELUFormats, approximate: Literal["none", "tanh"] = "none"
    ):
        r"""
        Args:
            x: the input tensor
            formats: configuration class for number formats and quantizers to use during
                forward and backward computations in GELU
            approximate: the GELU approximation algorithm to use:
                ``'none'`` | ``'tanh'``. Default: ``'none'``
        """
        super(QGELU, self).__init__(approximate)
        self.formats = formats
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized I/O signals in the module, depending if quantizers are
        specified in the associated :class:`QGELUFormats` :attr:`formats` parameter.
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

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Applies the GELU function on the input tensor element-wise.

        Args:
            input: the input tensor over which the GELU function is applied

        Returns:
            the result of applying the GELU function
        """
        if self.formats.inter_quant is not None:
            return qgelu(input, self.formats, self.approximate)
        else:
            return self.Qo(F.gelu(self.Qi(input), self.approximate))
