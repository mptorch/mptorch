import torch
import torch.nn as nn
import torch.nn.functional as F
from ..functional import qgelu

__all__ = ["QGELU"]

class QGELU(nn.GELU):
    def __init__(self, formats, approximate='none'):
        super(QGELU, self).__init__(approximate)
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
                    out = fwd_quant(out)
                    return out
                else:
                    return x
            @staticmethod
            def backward(ctx, grad_output):
                return self.formats.grad_quant(grad_output)

    def forward(self, input):
        if self.formats.inter_quant is not None:
            return qgelu(input, self.formats, self.approximate)
        else:
            return self.Qo(F.gelu(self.Qi(input), self.approximate))
