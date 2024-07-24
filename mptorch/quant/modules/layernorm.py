import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant_format import QLayerNormFormats
from ..functional import qlayernorm

__all__ = ["QLayerNorm",]

class QLayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape,
                 formats: QLayerNormFormats,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True) -> None:
        super(QLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine, bias)
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()
    
    def reset_quant_function(self):
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
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

    def quant_parameters(self):
        if self.weight is not None:
            self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)
    
    def forward(self, input):
        if self.formats.fwd_use_default_prec and self.formats.bwd_use_default_prec:
            return self.Qo(
                F.layer_norm(self.Qi(input), 
                            list(self.normalized_shape), 
                            self.Qw(self.weight), 
                            self.Qb(self.bias), 
                            self.eps)
                )
        else:
            return qlayernorm(input, 
                            normalized_shape=list(self.normalized_shape),
                            weight=self.weight, 
                            bias=self.bias, 
                            eps=self.eps, 
                            formats=self.formats)