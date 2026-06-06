import torch.nn as nn
from typing import Callable, Optional


class QAffineFormats(nn.Module):
    """
    Configuration module that holds quantization functions and custom math
    operations for affine layers (e.g. Linear, Conv2d).

    By inheriting from nn.Module, if any of the assigned quantizers are
    themselves nn.Modules (e.g. for QAT/PTQ state like running max/min),
    PyTorch will automatically register them and include their state in
    the model's state_dict.
    """

    def __init__(
        self,
        weight_quant: Optional[Callable] = None,
        input_quant: Optional[Callable] = None,
        bias_quant: Optional[Callable] = None,
        wgrad_quant: Optional[Callable] = None,
        igrad_quant: Optional[Callable] = None,
        bgrad_quant: Optional[Callable] = None,
        fwd_math: Optional[Callable] = None,
        bwd_igrad_math: Optional[Callable] = None,
        bwd_wgrad_math: Optional[Callable] = None,
    ):
        super().__init__()

        self.weight_quant = weight_quant
        self.input_quant = input_quant
        self.bias_quant = bias_quant
        self.wgrad_quant = wgrad_quant
        self.igrad_quant = igrad_quant
        self.bgrad_quant = bgrad_quant

        self.fwd_math = fwd_math
        self.bwd_igrad_math = bwd_igrad_math
        self.bwd_wgrad_math = bwd_wgrad_math
