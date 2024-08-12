import torch.nn as nn
from ..functional import qgelu

__all__ = ["QGELU",]

class QGELU(nn.GELU):
    def __init__(self, formats, approximate='none'):
        super(QGELU, self).__init__(approximate)
        self.formats = formats

    def forward(self, input):
        return qgelu(input, self.formats, self.approximate)