import torch.nn as nn 

from ..quant_format import QAffineFormats
from ..functional import qsum

__all__ = ["QSoftmax",]

class QSoftmax(nn.Softmax):
    def __init__(self,
                 dim: None,
                 formats) -> None:
        super(QSoftmax, self).__init__(dim)
        self.formats = formats
        # TODO: quant function initialization


    def forward(self, input):
        # TODO: call the functional verion of softmax in the forward pass
        # (the various operations need to use simulated kernels for exp and
        # sum operations, may need refactoring!)
        pass