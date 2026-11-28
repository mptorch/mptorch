"""The differentiable matmul: ``CustomArithMatmul`` and the ``QMatmul`` module.

The GEMM ops themselves have no gradient: they are plain functions over the
kernels, and the dispatcher cannot derive a gradient for an op that
quantizes its own arithmetic. This module gives them one, split the same way
``QLinear`` is: a ``torch.autograd.Function`` quantizes the operands as
``formats`` says and calls ``formats.fwd_math``, and its ``backward``
quantizes the incoming gradient separately for each of the two gradient
paths before calling that path's own math hook.

The only differences from ``CustomArithLinear`` follow from the op being
symmetric. Its two operands play the same role, rather than being an input
and a weight, hence the names ``a`` and ``b``, ``a_quant`` and ``b_quant``,
``agrad_quant`` and ``bgrad_quant``. And it follows ``torch.matmul``'s
operand rules, so a gradient has to be summed back over the dimensions the
forward broadcast, and the dimension added to a 1D operand has to be removed
again.
"""

import torch

from .._opaque import _opaque_backward
from ._saved import load, quantize_operand, restore, save


def _promote(x: torch.Tensor | None, was_1d: bool, dim: int) -> torch.Tensor | None:
    """Put back onto a saved operand the dimension ``torch.matmul`` added to it.

    ``torch.matmul`` gives a 1D ``a`` a leading row dimension and a 1D ``b`` a
    trailing column dimension, and removes both from the result, so backward
    has to add them back before it can transpose anything. Returns a view, or
    ``None`` for an operand that was not saved because no gradient needs it.
    """
    return x.unsqueeze(dim) if (was_1d and x is not None) else x


def _restore(grad: torch.Tensor, a_1d: bool, b_1d: bool) -> torch.Tensor:
    """Put the dimensions that ``torch.matmul`` added and removed back onto an
    incoming gradient.

    ``b``'s column goes first: for a scalar output (both operands 1D) there is
    no dimension -2 at which to insert ``a``'s row until ``b``'s column is
    there.
    """
    if b_1d:
        grad = grad.unsqueeze(-1)
    if a_1d:
        grad = grad.unsqueeze(-2)
    return grad


def _reduce(grad: torch.Tensor, shape: tuple[int, ...], was_1d: bool, dim: int) -> torch.Tensor:
    """Return one operand's gradient in that operand's own shape.

    ``shape`` is the operand's shape after ``torch.matmul``'s 1D promotion.
    ``sum_to_size`` sums away every batch dimension the forward broadcast,
    including those a rank-2 operand never had, and the squeeze removes the
    dimension the 1D promotion added.
    """
    grad = grad.sum_to_size(shape)
    return grad.squeeze(dim) if was_1d else grad


#: What a ``None`` recompute flag means in ``CustomArithMatmul``, as
#: (a, b): the quantized copies of both are kept. A matmul's operands are
#: activations, and whether anything else keeps one alive depends on the
#: model. In attention's ``softmax(...) @ V``, ``a`` could be recomputed,
#: since softmax keeps it, but ``V`` could not: it is a Linear's output that
#: nothing else keeps (``_saved`` keeps a raw operand only where it is known
#: to stay alive anyway). It is a module constant for ``torch.compile``'s
#: sake, as in ``modules/linear.py``.
_MATMUL_RECOMPUTE_DEFAULT = (False, False)


class CustomArithMatmul(torch.autograd.Function):
    """``a @ b`` with per-operand quantization and custom dot-product math.

    It works as ``CustomArithLinear`` does. ``forward(a, b, formats)``
    quantizes both operands and calls ``formats.fwd_math``. ``backward``
    quantizes the incoming gradient once for each gradient that was actually
    requested, calls that gradient's math hook, and sums each result back to
    its operand's shape. Any hook left ``None`` falls back to
    ``torch.matmul``, so a ``QMatmulFormats()`` with nothing set is plain
    ``torch.matmul`` plus one extra function call. ``formats`` gets no
    gradient.
    """

    @staticmethod
    def _default_fwd(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        """The forward without a custom hook: ``a @ b``."""
        return torch.matmul(q_a, q_b)

    @staticmethod
    def _default_agrad(q_grad: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
        """``a``'s gradient without a custom hook: ``grad @ b^T``."""
        return torch.matmul(q_grad, q_b.transpose(-2, -1))

    @staticmethod
    def _default_bgrad(q_grad: torch.Tensor, q_a: torch.Tensor) -> torch.Tensor:
        """``b``'s gradient without a custom hook: ``a^T @ grad``."""
        return torch.matmul(q_a.transpose(-2, -1), q_grad)

    @staticmethod
    def forward(ctx, a, b, formats):
        # hidden from torch function modes, as in CustomArithLinear
        with torch._C.DisableTorchFunction():
            ctx.formats = formats
            recompute_a, recompute_b = _MATMUL_RECOMPUTE_DEFAULT

            # Save only what backward reads, as CustomArithLinear does: q_b
            # feeds a's gradient and q_a feeds b's, so a call that needs only
            # one gradient does not hold quantized copies of both operands. For
            # an operand that is read, what is saved is either its quantized
            # value or the raw operand, quantized again in backward (see
            # _saved). A quantizer result that is not a tensor (a packed block
            # format, which the square-tiled slots of block_matmul_formats
            # return) is kept as it is.
            q_a, saved_a, ctx.replay_a = quantize_operand(
                a,
                formats.a_quant,
                getattr(formats, "recompute_a", None),
                recompute_a,
                ctx.needs_input_grad[1],
            )
            q_b, saved_b, ctx.replay_b = quantize_operand(
                b,
                formats.b_quant,
                getattr(formats, "recompute_b", None),
                recompute_b,
                ctx.needs_input_grad[0],
            )

            fwd_math = formats.fwd_math or CustomArithMatmul._default_fwd
            output = fwd_math(q_a, q_b)

            save(ctx, saved_a, saved_b)
            ctx.promotion = (a.dim() == 1, b.dim() == 1)
            ctx.shapes = (a.shape, b.shape)
            return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        with _opaque_backward():  # as in forward, except under compile (_opaque.py)
            formats = ctx.formats
            saved_a, saved_b = load(ctx)
            a_1d, b_1d = ctx.promotion
            a_shape, b_shape = ctx.shapes
            grad_a = grad_b = None

            grad = _restore(grad_outputs[0], a_1d, b_1d)

            # Quantize the gradient only for the paths that use it, and run both
            # quantizations before both math calls, in this order, so that a
            # stochastic quantizer draws from the global random generator in the
            # same order whichever gradients are requested.
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
                # A recomputed operand is quantized again only here, so its
                # quantized copy exists only for this one GEMM.
                q_b = _promote(restore(saved_b, ctx.replay_b), b_1d, -1)
                assert q_b is not None  # saved exactly when a's gradient is requested
                agrad_math = formats.bwd_agrad_math or CustomArithMatmul._default_agrad
                promoted = (1, *a_shape) if a_1d else tuple(a_shape)
                grad_a = _reduce(agrad_math(q_agrad, q_b), promoted, a_1d, -2)
                del q_b

            if q_bgrad is not None:
                q_a = _promote(restore(saved_a, ctx.replay_a), a_1d, 0)
                assert q_a is not None  # saved exactly when b's gradient is requested
                bgrad_math = formats.bwd_bgrad_math or CustomArithMatmul._default_bgrad
                promoted = (*b_shape, 1) if b_1d else tuple(b_shape)
                grad_b = _reduce(bgrad_math(q_bgrad, q_a), promoted, b_1d, -1)

            return grad_a, grad_b, None


class QMatmul(torch.nn.Module):
    """``torch.matmul`` with the arithmetic of one :class:`QMatmulFormats`.

    This is the module form of :func:`mptorch.quant.qmatmul`. Like
    ``QLinear``, it is a module because its formats are one: a
    ``QMatmulFormats`` is an ``nn.Module``, so a quantizer that carries state
    (a QAT observer) is registered in the parent model's ``state_dict`` when
    the matmul is held as a submodule. A quantized attention block uses two
    of them: one for ``Q @ K^T`` and one for the probabilities times ``V``.
    For backward it keeps quantized copies of both operands, unless the
    ``recompute_a`` or ``recompute_b`` flag of its formats says to keep an
    operand raw and quantize it again there (worth it for the softmax
    probabilities, which softmax keeps alive anyway).

    Args:
        formats (None, Number, SplitMac, FusedMac, BlockMac or QMatmulFormats):
            any value :func:`mptorch.quant.qmatmul` accepts, converted once,
            here, by :func:`mptorch.quant.functional.matmul.as_matmul_formats`.
            Default: ``None`` (plain ``torch.matmul``)

    Shape:
        - Input: ``a`` and ``b`` in any shapes ``torch.matmul`` accepts.
        - Output: ``torch.matmul``'s output shape.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QMatmul, SplitMac
        >>> mm = QMatmul(SplitMac(BinaryK(8, 4), BinaryK(16, 11)))
        >>> mm(torch.randn(2, 4, 8), torch.randn(8, 3)).shape
        torch.Size([2, 4, 3])
    """

    #: ``mptorch.quant.simulate`` passes through every call made inside this
    #: layer: the layer already runs in mptorch's arithmetic.
    _mptorch_owned = True

    def __init__(self, formats=None):
        super().__init__()
        # The conversion is defined next to qmatmul, and imported here rather
        # than at module level because that module imports this one.
        from ..functional.matmul import as_matmul_formats

        self.formats = as_matmul_formats(formats)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply ``a`` and ``b`` as ``torch.matmul`` does, quantizing and computing as
        ``formats`` says."""
        return CustomArithMatmul.apply(a, b, self.formats)
