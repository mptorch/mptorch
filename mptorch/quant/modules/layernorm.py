import torch.nn as nn

from ..quant_format import QAffineFormats
from ..functional import qmean, qadd, qpow, qdiv, qsqrt, qmul

__all__ = ["QLayerNorm",]

class QLayerNorm(nn.LayerNorm):
    def __init__(self,
                 normalized_shape,
                 formats: QAffineFormats,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True,
                 bias: bool = True) -> None:
        super(QLayerNorm, self).__init__(normalized_shape, eps, elementwise_affine, bias)
        self.formats = formats
        # TODO: parameter and quant function initialization

    
    def forward(self, input):
        # TODO: call the functional version of layernorm in the forward pass
        # (the various operations need to use the various simulated low precision 
        # kernels for sum and mean operations, which also need to be refactored 
        # for speed)
        pass