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
        weight: the learnable weights of the module of shape
            :math:`(\text{out_features}, \text{in_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out_features})`.
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
    ) -> None:
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self):
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

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
            return self.Qo(
                F.linear(self.Qi(input), self.Qw(self.weight), self.Qb(self.bias))
            )
        else:
            return qlinear(input, self.weight, self.bias, self.formats)


class QLazyLinear(torch.nn.modules.lazy.LazyModuleMixin, QLinear):
    r"""A linear module where `in_features` is inferred.

    In this module (an analogue to :class:`torch.nn.LazyLinear`), the
    ``in_features`` parameter of the quantized linear layer is inferred
    from the input's last dimension (i.e., ``input.shape[-1]``).

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
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        # similar to nn.LazyLinear, bias is hardcoded to avoid
        # creating tensor that will soon be overwritten
        super().__init__(0, 0, formats=formats, bias=False)
        self.weight = torch.nn.UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        if bias:
            self.bias = torch.nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight.materialize((self.out_features, self.in_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()
