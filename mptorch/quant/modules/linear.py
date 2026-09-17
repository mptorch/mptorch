"""The quantized Linear layer: ``CustomArithLinear`` and the ``QLinear`` module.

``QLinear`` is a thin ``nn.Linear`` subclass whose ``forward`` delegates to a
``torch.autograd.Function``. The Function is where quantization happens, once
per operand and once per gradient path, so that the layer's ``formats`` can
round the input, the weight and the bias before the forward GEMM, round the
incoming gradient differently for the input-gradient and weight-gradient GEMMs,
and replace each of the three GEMMs with custom arithmetic. Any slot left
``None`` is plain PyTorch, so a ``QLinear`` with empty formats is exactly an
``nn.Linear``.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .format import QAffineFormats


class CustomArithLinear(torch.autograd.Function):
    """Autograd Function for a quantized linear layer.

    ``forward(input, weight, bias, formats)`` quantizes the three operands
    through ``formats``' ``*_quant`` slots and computes the output with
    ``formats.fwd_math`` or ``F.linear``. ``backward`` quantizes the output
    gradient separately for the input-gradient path (``igrad_quant``) and the
    weight-gradient path (``wgrad_quant``), then runs ``bwd_igrad_math`` and
    ``bwd_wgrad_math`` or their plain-matmul defaults, and sums the
    ``bgrad_quant``-quantized gradient over the leading dimensions for the
    bias. ``formats`` gets no gradient.
    """

    @staticmethod
    def _default_bwd_igrad(grad_output: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """The input gradient without a custom hook: ``grad_output @ weight``,
        with ``grad_output``'s leading dimensions kept."""
        return grad_output.matmul(weight)

    @staticmethod
    def _default_bwd_wgrad(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """The weight gradient without a custom hook: ``grad_output^T @ input``
        over every leading dimension flattened into the row dimension."""
        gO_flat = grad_output.reshape(-1, grad_output.shape[-1])
        i_flat = input.reshape(-1, input.shape[-1])
        return gO_flat.t().matmul(i_flat)

    @staticmethod
    def forward(ctx, input, weight, bias, formats):
        ctx.formats = formats

        q_input = formats.input_quant(input) if formats.input_quant else input
        q_weight = formats.weight_quant(weight) if formats.weight_quant else weight
        q_bias = formats.bias_quant(bias) if formats.bias_quant and bias is not None else bias

        if formats.fwd_math is not None:
            output = formats.fwd_math(q_input, q_weight, q_bias)
        else:
            output = F.linear(q_input, q_weight, q_bias)

        # Save only what backward will read. q_bias is never read there, and
        # each of the other two feeds exactly one gradient path: q_input the
        # weight gradient, q_weight the input gradient. So a layer whose
        # weight is frozen does not pin a quantized copy of its entire
        # activation for the length of the backward pass, on top of the
        # activation autograd already holds. ctx.needs_input_grad is
        # populated before forward runs and is the same tuple backward sees.
        saved_input = q_input if ctx.needs_input_grad[1] else None
        saved_weight = q_weight if ctx.needs_input_grad[0] else None

        # save_for_backward takes tensors only; a quantizer that returns some
        # other structure (a packed block format, say) goes on ctx directly.
        if (saved_input is None or isinstance(saved_input, torch.Tensor)) and (
            saved_weight is None or isinstance(saved_weight, torch.Tensor)
        ):
            ctx.save_for_backward(saved_input, saved_weight)
            ctx.saved_structs = None
        else:
            ctx.save_for_backward()
            ctx.saved_structs = (saved_input, saved_weight)

        ctx.needs_bias_grad = bias is not None

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        formats = ctx.formats

        if ctx.saved_structs is None:
            q_input, q_weight = ctx.saved_tensors
        else:
            q_input, q_weight = ctx.saved_structs

        grad_input = grad_weight = grad_bias = None

        # Quantize grad_output only for the paths that will consume it, so a
        # frozen layer, or the first layer of a network, does not pay a full
        # grad-output-sized quantize kernel and allocation for a tensor it
        # then discards. Both quantizations run ahead of both math calls, in
        # this order, so a stochastic quantizer draws from the global RNG in
        # the same sequence however many gradients are requested; the *_math
        # hooks may consume randomness of their own afterwards.
        q_igrad_output = (
            (formats.igrad_quant(grad_output) if formats.igrad_quant else grad_output)
            if ctx.needs_input_grad[0]
            else None
        )
        q_wgrad_output = (
            (formats.wgrad_quant(grad_output) if formats.wgrad_quant else grad_output)
            if ctx.needs_input_grad[1]
            else None
        )

        if q_igrad_output is not None:
            bwd_igrad_math = (
                formats.bwd_igrad_math
                if formats.bwd_igrad_math
                else CustomArithLinear._default_bwd_igrad
            )
            grad_input = bwd_igrad_math(q_igrad_output, q_weight)

        if q_wgrad_output is not None:
            bwd_wgrad_math = (
                formats.bwd_wgrad_math
                if formats.bwd_wgrad_math
                else CustomArithLinear._default_bwd_wgrad
            )
            grad_weight = bwd_wgrad_math(q_wgrad_output, q_input)

        if ctx.needs_bias_grad and ctx.needs_input_grad[2]:
            q_bgrad_output = (
                formats.bgrad_quant(grad_output) if formats.bgrad_quant else grad_output
            )
            # The bias gradient is the output gradient summed over every
            # leading (batch) dimension.
            grad_bias = q_bgrad_output.reshape(-1, q_bgrad_output.shape[-1]).sum(0)

        return grad_input, grad_weight, grad_bias, None


class QLinear(nn.Linear):
    """``torch.nn.Linear`` with per-tensor quantization and custom arithmetic.

    Holds the same parameters as ``nn.Linear`` and computes the same
    ``input @ weight.T + bias``, but through :class:`CustomArithLinear`, so
    the input, weight, bias and each gradient can be rounded by the
    ``formats``' quantizers and each GEMM can run in custom arithmetic
    (see :func:`mptorch.quant.binaryK_gemm_formats`). With ``formats`` left
    ``None``, or a ``QAffineFormats`` with every slot ``None``, the layer is
    numerically identical to ``nn.Linear``. ``formats`` is a submodule, so a
    stateful quantizer in it appears in the layer's ``state_dict``.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        bias (bool): whether the layer learns an additive bias.
            Default: ``True``
        device: as for ``nn.Linear``. Default: ``None``
        dtype: as for ``nn.Linear``. Default: ``None``
        formats (QAffineFormats, optional): the quantizers and math hooks.
            ``None`` builds an empty ``QAffineFormats()``. Default: ``None``

    Shape:
        - Input: ``(*, in_features)``, any number of leading dimensions.
        - Output: ``(*, out_features)``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QLinear, Quant, binaryK_gemm_formats
        >>> formats = binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=16, acc_P=11)
        >>> formats.input_quant = Quant(BinaryK(8, 4))
        >>> formats.weight_quant = Quant(BinaryK(8, 4))
        >>> layer = QLinear(256, 64, formats=formats)
        >>> layer(torch.randn(32, 256)).shape
        torch.Size([32, 64])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return CustomArithLinear.apply(input, self.weight, self.bias, self.formats)
