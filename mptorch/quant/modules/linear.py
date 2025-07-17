from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant_format import QAffineFormats
from ..functional import qlinear

__all__ = ["QLinear", "QLazyLinear"]


class QLinear(nn.Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y=xW^T + b`

    It is a subclass of :class:`torch.nn.Linear` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    and is helpful in studying the effect of low precision compute during inference
    and training (not just data quantization).

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        formats: number formats used during compute (addition and multiplication) and
            quantization functions for signals during forward and back propagation (I/O
            activations, weights, biases, and neural gradients)
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out_features}`.

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out_features}, \text{in_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in_features}}`
        bias (Tensor):   the learnable bias of the module of shape :math:`(\text{out_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in_features}}`

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        formats: QAffineFormats,
        bias: bool = True,
    ):
        super(QLinear, self).__init__(in_features, out_features, bias)
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

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

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
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor over which to perform the layer operations.
                Must adhere to the input shape requirements.

        Returns:
            the result of the :math:`xW^T + b` operation.
        """
        if self.formats.fwd_use_default_prec and self.formats.bwd_use_default_prec:
            return self.Qo(
                F.linear(self.Qi(input), self.Qw(self.weight), self.Qb(self.bias))
            )
        else:
            return qlinear(input, self.weight, self.bias, self.formats)


class QLazyLinear(torch.nn.modules.lazy.LazyModuleMixin, QLinear):
    r"""A linear module where `in_features` is inferred.

    In this module (an analogue to :class:`torch.nn.LazyLinear`), the
    ``in_features`` parameter of the quantized linear layer is inferred
    from the input's last dimension (i.e., ``input.shape[-1]``). The `weight`
    and `bias` layer parameters are of :class:`torch.nn.UninitializedParameter`
    class. They are initialized after the first call to ``forward`` and the
    module becomes a regular :class:`torch.nn.Linear` module.

    Args:
        out_features: size of each output sample
        formats: number formats used during compute (addition and multiplication) and
            quantization functions for signals during forward and back propagation (I/O
            activations, weights, biases, and neural gradients)
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
    """

    cls_to_become = QLinear
    weight: torch.nn.UninitializedParameter
    r"""
        The learnable weights of the module of shape
        :math:`(\text{out_features}, \text{in_features})`. The values are
        initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
        :math:`k = \frac{1}{\text{in_features}}`
    """
    bias: torch.nn.UninitializedParameter
    r"""
        The learnable bias of the module of shape :math:`(\text{out_features})`.
        If :attr:`bias` is ``True``, the values are initialized from
        :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
        :math:`k = \frac{1}{\text{in_features}}`
    """

    def __init__(
        self,
        out_features: int,
        formats: QAffineFormats,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # similar to nn.LazyLinear, bias is hardcoded to avoid
        # creating tensor that will soon be overwritten
        super().__init__(0, 0, formats=formats, bias=False)
        self.weight = torch.nn.UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = torch.nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self):
        r"""Resets parameter values in case parameters have been initialized"""
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input: torch.Tensor):
        r"""Initialize parameters according to the input
        batch properties.

        This adds an interface to isolate parameter initialization from the forward
        pass when doing parameter shape inference.

        Args:
            input: the input tensor on which to perform the layer operations. It's shape
                is used to determine the value of the :attr:`in_features` member variable.
        """
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
