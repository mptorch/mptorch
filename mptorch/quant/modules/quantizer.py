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

    This is the module form of the straight-through estimator. The output is
    ``forward_quant(x)``, and the gradient that reaches ``x`` is
    ``backward_quant(grad)``. A direction whose quantizer is ``None`` is left
    alone. Because it is an ``nn.Module``, it can sit in any ``*_quant`` slot
    of a ``QAffineFormats`` or ``QMatmulFormats`` and is registered with the
    parent model. That matters as soon as the quantizer carries state of its
    own.

    Args:
        forward_quant (Number or Callable, optional): the forward pass's
            quantizer. A format is wrapped in a :class:`mptorch.quant.Quant`
            with round-to-nearest-even. Any ``Tensor -> Tensor`` callable is
            used as it is. ``None`` leaves the forward pass alone. Default:
            ``None``
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

    # nn.Module declares forward as a writable Callable attribute, which ty
    # reads a method override as narrowing; every direct subclass does this.
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # ty: ignore[invalid-mutable-override]
        return _StraightThrough.apply(x, self.forward_quant, self.backward_quant)
