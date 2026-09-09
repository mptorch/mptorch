"""Straight-through quantization: one format applied forward, another backward.

The elementwise quantize ops (``binaryK_quantize``, ``superfp_quantize``) are
not differentiable and do not pretend to be -- rounding to a coarse format has
a derivative of zero almost everywhere, so a gradient through it is a modelling
choice, not a fact about the op. This module is where that choice is made
explicit, and it is the reason those ops raise on a ``requires_grad`` operand
rather than silently dropping the gradient.

The estimator is the standard one: the forward rounds, the backward passes the
gradient through, optionally rounded to a *different* format. That asymmetry is
the point -- QAT typically wants a narrow forward format and a wider gradient
format -- and it is why this is a module wrapping two quantizers rather than a
straight-through kernel registered on the op, which could only ever express the
symmetric case.
"""

from collections.abc import Callable

import torch
from torch import nn

from mptorch.number import Number

from ..mac import Quant


class _StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, forward_quant, backward_quant):
        ctx.backward_quant = backward_quant
        return forward_quant(x) if forward_quant else x

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0]
        q = ctx.backward_quant
        return (q(grad) if q else grad), None, None


class Quantizer(nn.Module):
    """Quantize in the forward pass, quantize (or pass through) in the backward.

    ``forward_quant`` and ``backward_quant`` each take a format (wrapped in a
    :class:`mptorch.quant.Quant`), any callable ``Tensor -> Tensor``, or
    ``None`` to leave that direction alone::

        act = Quantizer(BinaryK(8, 4), BinaryK(8, 5))   # narrow fwd, wider bwd
        formats.input_quant = act                       # a module, so it is
                                                        # in the state_dict

    Being an ``nn.Module``, it sits in any ``*_quant`` slot of a
    ``QAffineFormats`` / ``QMatmulFormats`` and is registered with the parent
    model -- which matters as soon as the quantizer carries state of its own.
    """

    def __init__(
        self,
        forward_quant: Number | Callable | None = None,
        backward_quant: Number | Callable | None = None,
    ):
        super().__init__()
        self.forward_quant = (
            Quant(forward_quant) if isinstance(forward_quant, Number) else forward_quant
        )
        self.backward_quant = (
            Quant(backward_quant) if isinstance(backward_quant, Number) else backward_quant
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _StraightThrough.apply(x, self.forward_quant, self.backward_quant)
