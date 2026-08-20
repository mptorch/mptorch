from collections.abc import Callable

from torch import nn


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
        weight_quant: Callable | None = None,
        input_quant: Callable | None = None,
        bias_quant: Callable | None = None,
        wgrad_quant: Callable | None = None,
        igrad_quant: Callable | None = None,
        bgrad_quant: Callable | None = None,
        fwd_math: Callable | None = None,
        bwd_igrad_math: Callable | None = None,
        bwd_wgrad_math: Callable | None = None,
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
