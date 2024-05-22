import torch.nn as nn
from ..quant_function import quantizer


class Quantizer(nn.Module):
    def __init__(
        self,
        forward_number=None,
        backward_number=None,
        forward_rounding="nearest",
        backward_rounding="nearest",
    ):
        super(Quantizer, self).__init__()
        self.quantize = quantizer(
            forward_number, backward_number, forward_rounding, backward_rounding
        )

    def forward(self, x):
        return self.quantize(x)
