from typing import Literal
import torch
import torch.nn as nn
from ..quant_function import quantizer
from mptorch import Number


class Quantizer(nn.Module):
    r"""A quantization module that supports quantizing forward and backward process differently."""

    def __init__(
        self,
        forward_number: Number | None = None,
        backward_number: Number | None = None,
        forward_rounding: Literal["nearest", "stochastic"] = "nearest",
        backward_rounding: Literal["nearest", "stochastic"] = "nearest",
    ):
        r"""
        Args:
            forward_number: the number format used for forward quantization.
                    if is None, the quantization would be a identity mapping.
            backward_number: the number format used for backward quantization.
                    if is None, the quantization would be a identity mapping.
            forward_rounding: rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
            backward_rounding: rounding mode, \"stochastic\" or \"nearest\" (default: \"stochastic\")
        """
        super(Quantizer, self).__init__()
        self.quantize = quantizer(
            forward_number, backward_number, forward_rounding, backward_rounding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward pass call on the input tensor. Applies the forward quantization on the input and
        registers the backward quantization format.

        Args:
            x: the input tensor

        Returns:
            The quantized version of the input, as specified by the FWD number format and associated rounding mode
        """
        return self.quantize(x)
