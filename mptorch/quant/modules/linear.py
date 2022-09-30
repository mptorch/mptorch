import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quant_format import QAffineFormats
from ..functional import qlinear

__all__ = ["QLinear"]


class QLinear(nn.Linear):
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
        if self.formats.use_default_prec:
            return self.Qo(
                F.linear(self.Qi(input), self.Qw(self.weight), self.Qb(self.bias))
            )
        else:
            return qlinear.apply(input, self.weight, self.bias, self.formats)
