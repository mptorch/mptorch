"""Straight-through quantization: one format applied forward, another backward.

The elementwise quantize ops (``binaryK_quantize``, ``superfp_quantize``) are
not differentiable and do not pretend to be: rounding to a coarse format has a
derivative of zero almost everywhere, so a gradient through it is a modelling
choice, not a fact about the op. This module is where that choice is made
explicit, and it is the reason those ops raise on a ``requires_grad`` operand
rather than silently dropping the gradient.

The estimator is the standard one: the forward rounds, the backward passes the
gradient through, optionally rounded to a *different* format. That asymmetry is
the point (QAT typically wants a narrow forward format and a wider gradient
format), and it is why this is a module wrapping two quantizers rather than a
straight-through kernel registered on the op, which could only ever express the
symmetric case.
"""

from collections.abc import Callable

import torch
from torch import nn

from mptorch.number import Number

from ..mac import Quant


class _StraightThrough(torch.autograd.Function):
    """The straight-through estimator as an autograd Function.

    ``forward`` applies ``forward_quant`` (or nothing) to ``x``; ``backward``
    applies ``backward_quant`` (or nothing) to the incoming gradient and
    passes it through unchanged otherwise. The two quantizers are not tensors
    and get no gradient, hence the two ``None``.
    """

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

    The module form of the straight-through estimator: the output is
    ``forward_quant(x)`` and the gradient reaching ``x`` is
    ``backward_quant(grad)``, with each direction left alone when its
    quantizer is ``None``. Being an ``nn.Module``, it sits in any ``*_quant``
    slot of a ``QAffineFormats`` / ``QMatmulFormats`` and is registered with
    the parent model, which matters as soon as the quantizer carries state of
    its own.

    Args:
        forward_quant (Number or Callable, optional): the forward's quantizer.
            A format is wrapped in a :class:`mptorch.quant.Quant` with
            round-to-nearest-even; any ``Tensor -> Tensor`` callable is used
            as is; ``None`` leaves the forward alone. Default: ``None``
        backward_quant (Number or Callable, optional): the backward's
            quantizer, in the same forms. Default: ``None``

    Shape:
        - Input: any shape.
        - Output: the same shape and dtype.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QAffineFormats, Quantizer
        >>> act = Quantizer(BinaryK(8, 4), BinaryK(8, 5))   # narrow fwd, wider bwd
        >>> x = torch.tensor([0.3], requires_grad=True)
        >>> y = act(x)
        >>> y
        tensor([0.3125], grad_fn=<_StraightThroughBackward>)
        >>> y.backward()
        >>> x.grad
        tensor([1.])
        >>> formats = QAffineFormats()
        >>> formats.input_quant = act   # a module, so it is in the state_dict
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
