"""Straight-through quantization: one format applied in the forward pass, another in backward.

The elementwise quantize ops (``binaryK_quantize``, ``superfp_quantize``,
``fixedpoint_quantize``) are not differentiable and do not pretend to be.
Rounding to a coarse format has a derivative of zero almost everywhere, so
any gradient through it is a modelling choice, not a property of the op.
This module is where that choice is made explicitly. Because it exists, the
ops can raise an error on an operand that requires grad, rather than
silently dropping the gradient.

The estimator is the standard straight-through one: the forward pass
rounds, and the backward pass passes the gradient through unchanged, or
rounds it to a *different* format. That asymmetry is the point (QAT usually
wants a narrow forward format and a wider gradient format), and it is why
this is a module wrapping two quantizers, not a straight-through gradient
registered on the op, which could only express the symmetric case.
"""

from collections.abc import Callable

import torch
from torch import nn

from mptorch.number import Number

from ..mac import Quant
from ._opaque import _opaque_backward


class _StraightThrough(torch.autograd.Function):
    """The straight-through estimator as an autograd Function.

    ``forward`` applies ``forward_quant`` to ``x``, and ``backward`` applies
    ``backward_quant`` to the incoming gradient; a ``None`` quantizer passes
    its tensor through unchanged. The two quantizers are not tensors and get
    no gradient, hence the two ``None`` that ``backward`` returns.
    """

    @staticmethod
    def forward(ctx, x, forward_quant, backward_quant):
        # hidden from torch function modes, as in CustomArithLinear
        with torch._C.DisableTorchFunction():
            ctx.backward_quant = backward_quant
            return forward_quant(x) if forward_quant else x

    @staticmethod
    def backward(ctx, *grad_outputs):
        with _opaque_backward():  # as in forward, except under compile (_opaque.py)
            grad = grad_outputs[0]
            q = ctx.backward_quant
            return (q(grad) if q else grad), None, None


class Quantizer(nn.Module):
    """Quantize in the forward pass, and quantize (or pass through) in the backward pass.

    This is the module form of the straight-through estimator. The output is
    ``forward_quant(x)``, and the gradient that reaches ``x`` is
    ``backward_quant(grad)``. A pass whose quantizer is ``None`` is left
    unchanged. Because it is an ``nn.Module``, it can sit in any ``*_quant``
    slot of a ``QAffineFormats`` or ``QMatmulFormats`` and is registered with
    the parent model, which matters as soon as the quantizer carries state of
    its own.

    Args:
        forward_quant (Number or Callable, optional): the forward pass's
            quantizer. A format (a :class:`mptorch.Number`) is wrapped in a
            :class:`mptorch.quant.Quant` that rounds to nearest, ties to even.
            Any ``Tensor -> Tensor`` callable is used as it is. ``None``
            leaves the forward pass unchanged. Default: ``None``
        backward_quant (Number or Callable, optional): the backward pass's
            quantizer, in the same forms. Default: ``None``

    Shape:
        - Input: any shape.
        - Output: the same shape and dtype.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QAffineFormats, Quantizer
        >>> act = Quantizer(BinaryK(8, 4), BinaryK(8, 5))   # narrow forward, wider backward
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
        """Round ``x`` with ``forward_quant``; its gradient is rounded with ``backward_quant``."""
        return _StraightThrough.apply(x, self.forward_quant, self.backward_quant)
