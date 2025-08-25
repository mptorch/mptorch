from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant_format import QLayerNormFormats
from ..functional import qlayernorm

__all__ = [
    "QLayerNorm",
]


class QLayerNorm(nn.LayerNorm):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The variance is calculated via the biased estimator, equivalent to
    ``torch.var(input, unbiased=False)``.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`\text{normalized_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias (Tensor):   the learnable bias of the module of shape
                :math:`\text{normalized_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        formats: QLayerNormFormats,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        r"""
        Args:
            normalized_shape (int or list or torch.Size): input shape from an expected input
                of size

                .. math::
                    [* \times \text{normalized_shape}[0] \times \text{normalized_shape}[1]
                        \times \ldots \times \text{normalized_shape}[-1]]

                If a single integer is used, it is treated as a singleton list, and this module will
                normalize over the last dimension which is expected to be of that specific size.
            eps: a value added to the denominator for numerical stability. Default: 1e-5
            elementwise_affine: a boolean value that when set to ``True``, this module
                has learnable per-element affine parameters initialized to ones (for weights)
                and zeros (for biases). Default: ``True``.
            bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
                :attr:`elementwise_affine` is ``True``). Default: ``True``.
        """
        super(QLayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine, bias
        )
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
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

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        if self.weight is not None:
            self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Performs the layernorm operation on the input tensor, using quantized elementary
        operations (e.g. additions and multiplications) in the FWD and BWD passes, as specified
        through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor over which to perform the layernorm operations.

        Returns:
            the output of the layernorm operation
        """
        if self.formats.fwd_use_default_prec and self.formats.bwd_use_default_prec:
            return self.Qo(
                F.layer_norm(
                    self.Qi(input),
                    list(self.normalized_shape),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.eps,
                )
            )
        else:
            return qlayernorm(
                input,
                normalized_shape=list(self.normalized_shape),
                weight=self.weight,
                bias=self.bias,
                eps=self.eps,
                formats=self.formats,
            )
