"""The differentiable matmul: ``CustomArithMatmul`` and the ``QMatmul`` module.

The GEMM ops themselves have no autograd kernel and never will -- they are the
schema tier, and an op that quantizes its own arithmetic has no derivative the
dispatcher could infer. This is the layer that gives them one, and it is the
same split ``QLinear`` uses: a ``torch.autograd.Function`` that quantizes the
operands per ``formats`` and calls ``formats.fwd_math``, with a ``backward``
that quantizes the incoming gradient separately for each of the two gradient
paths before calling that path's own math hook.

What is different from ``CustomArithLinear`` is only what a symmetric op
implies: two operands with the same standing rather than an input and a
weight, so ``a``/``b``, ``a_quant``/``b_quant``, ``agrad_quant``/
``bgrad_quant`` -- and ``torch.matmul``'s operand rules, which mean a gradient
has to be reduced back over dimensions the forward broadcast, and the
dimension a 1D operand was promoted along has to come off again.
"""

import torch


def _promote(x: torch.Tensor | None, was_1d: bool, dim: int) -> torch.Tensor | None:
    """The dimension ``torch.matmul`` promoted, back onto a saved operand.

    ``a`` gains a leading row and ``b`` a trailing column, and both are
    removed from the result -- so the backward has to put them back before it
    can transpose anything. A view, and ``None`` for an operand this call has
    no gradient to compute from.
    """
    return x.unsqueeze(dim) if (was_1d and x is not None) else x


def _restore(grad: torch.Tensor, a_1d: bool, b_1d: bool) -> torch.Tensor:
    """The promoted dimensions back onto an incoming gradient.

    ``b`` first: on a scalar output (both operands 1D) there is no dimension
    -2 to insert at until ``b``'s has been.
    """
    if b_1d:
        grad = grad.unsqueeze(-1)
    if a_1d:
        grad = grad.unsqueeze(-2)
    return grad


def _reduce(grad: torch.Tensor, shape: tuple[int, ...], was_1d: bool, dim: int) -> torch.Tensor:
    """One operand's gradient, back in that operand's own shape.

    ``shape`` is the operand's *promoted* shape. ``sum_to_size`` folds away
    every batch dimension the forward broadcast -- including the ones a rank-2
    operand never had -- and the squeeze undoes the 1D promotion.
    """
    grad = grad.sum_to_size(shape)
    return grad.squeeze(dim) if was_1d else grad


class CustomArithMatmul(torch.autograd.Function):
    """``a @ b`` with per-operand quantization and custom dot-product math.

    Mirrors ``CustomArithLinear``: ``forward`` quantizes both operands and
    calls ``formats.fwd_math``; ``backward`` quantizes the incoming gradient
    once per gradient path that was actually requested, and calls that path's
    math hook. Any hook left ``None`` falls back to ``torch.matmul``, so a
    ``QMatmulFormats()`` with nothing set is plain ``torch.matmul`` with an
    extra frame.
    """

    @staticmethod
    def _default_fwd(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q_a, q_b)

    @staticmethod
    def _default_agrad(q_grad: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q_grad, q_b.transpose(-2, -1))

    @staticmethod
    def _default_bgrad(q_grad: torch.Tensor, q_a: torch.Tensor) -> torch.Tensor:
        return torch.matmul(q_a.transpose(-2, -1), q_grad)

    @staticmethod
    def forward(ctx, a, b, formats):
        ctx.formats = formats

        q_a = formats.a_quant(a) if formats.a_quant else a
        q_b = formats.b_quant(b) if formats.b_quant else b

        fwd_math = formats.fwd_math or CustomArithMatmul._default_fwd
        output = fwd_math(q_a, q_b)

        # Save only what backward reads, as CustomArithLinear does: q_b feeds
        # a's gradient and q_a feeds b's, so a call that needs one gradient
        # does not pin a quantized copy of both operands.
        ctx.save_for_backward(
            q_a if ctx.needs_input_grad[1] else None,
            q_b if ctx.needs_input_grad[0] else None,
        )
        ctx.promotion = (a.dim() == 1, b.dim() == 1)
        ctx.shapes = (a.shape, b.shape)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        formats = ctx.formats
        q_a, q_b = ctx.saved_tensors
        a_1d, b_1d = ctx.promotion
        a_shape, b_shape = ctx.shapes
        grad_a = grad_b = None

        grad = _restore(grad_outputs[0], a_1d, b_1d)
        q_a = _promote(q_a, a_1d, 0)
        q_b = _promote(q_b, b_1d, -1)

        # Quantize the gradient only for the paths that will consume it, and
        # both quantizations ahead of both math calls, in this order -- so a
        # stochastic quantizer draws from the global generator in the same
        # sequence however many gradients are requested.
        q_agrad = (
            (formats.agrad_quant(grad) if formats.agrad_quant else grad)
            if ctx.needs_input_grad[0]
            else None
        )
        q_bgrad = (
            (formats.bgrad_quant(grad) if formats.bgrad_quant else grad)
            if ctx.needs_input_grad[1]
            else None
        )

        if q_agrad is not None:
            assert q_b is not None  # saved iff a's gradient was asked for
            agrad_math = formats.bwd_agrad_math or CustomArithMatmul._default_agrad
            promoted = (1, *a_shape) if a_1d else tuple(a_shape)
            grad_a = _reduce(agrad_math(q_agrad, q_b), promoted, a_1d, -2)

        if q_bgrad is not None:
            assert q_a is not None  # saved iff b's gradient was asked for
            bgrad_math = formats.bwd_bgrad_math or CustomArithMatmul._default_bgrad
            promoted = (*b_shape, 1) if b_1d else tuple(b_shape)
            grad_b = _reduce(bgrad_math(q_bgrad, q_a), promoted, b_1d, -1)

        return grad_a, grad_b, None


class QMatmul(torch.nn.Module):
    """``torch.matmul`` with the arithmetic of one :class:`QMatmulFormats`.

    The module form of :func:`mptorch.quant.qmatmul`, for the same reason
    ``QLinear`` is a module: a ``QMatmulFormats`` is an ``nn.Module``, so a
    quantizer that carries state (a QAT observer) is registered in the parent
    model's ``state_dict`` when the matmul is held as a submodule. Two of
    these are what a quantized attention block calls per head.
    """

    def __init__(self, formats=None):
        super().__init__()
        # `formats` takes everything qmatmul takes -- None, a Number, a
        # SplitMac/FusedMac, or a QMatmulFormats. The conversion lives with
        # the functional entry point, and is imported here rather than at
        # module scope because that module imports this one.
        from ..matmul import as_matmul_formats

        self.formats = as_matmul_formats(formats)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return CustomArithMatmul.apply(a, b, self.formats)
