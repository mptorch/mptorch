"""The quantized convolutions: ``CustomArithConvNd`` and ``QConv1d/2d/3d``.

The three modules are thin ``nn.Conv1d``/``nn.Conv2d``/``nn.Conv3d`` subclasses
whose ``forward`` delegates to one ``torch.autograd.Function`` parameterized
on the number of spatial dimensions. The Function is where quantization
happens, once per operand and once per gradient path, exactly as
``CustomArithLinear`` does for Linear: the ``formats``' quantizers round the
input, weight and bias before the forward convolution and the incoming
gradient separately for the input-gradient and weight-gradient paths, and the
three math hooks may replace each convolution with custom arithmetic. Any
slot left ``None`` is plain PyTorch.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as grad

from .format import QAffineFormats


class CustomArithConvNd(torch.autograd.Function):
    """Autograd Function for a quantized convolution, 1D, 2D or 3D.

    ``forward(input, weight, bias, formats, stride, padding, dilation, groups,
    nd)`` quantizes the three operands through ``formats``' ``*_quant`` slots
    and computes the output with ``formats.fwd_math`` (called with the
    convolution parameters and ``nd`` as keywords) or ``F.conv{nd}d``.
    ``backward`` quantizes the output gradient separately for the
    input-gradient path (``igrad_quant``) and the weight-gradient path
    (``wgrad_quant``), runs ``bwd_igrad_math`` / ``bwd_wgrad_math`` or
    PyTorch's ``conv{nd}d_input`` / ``conv{nd}d_weight``, and sums the
    ``bgrad_quant``-quantized gradient over the batch and spatial dimensions
    for the bias. Only the first three inputs get a gradient.
    """

    @staticmethod
    def _default_bwd_igrad(
        grad_output: torch.Tensor,
        weight: torch.Tensor,
        input_size,
        stride,
        padding,
        dilation,
        groups,
        nd,
    ) -> torch.Tensor:
        """The input gradient without a custom hook: PyTorch's transposed
        convolution of ``grad_output`` with ``weight``, sized to ``input_size``."""
        if nd == 1:
            return grad.conv1d_input(
                input_size, weight, grad_output, stride, padding, dilation, groups
            )
        elif nd == 2:
            return grad.conv2d_input(
                input_size, weight, grad_output, stride, padding, dilation, groups
            )
        elif nd == 3:
            return grad.conv3d_input(
                input_size, weight, grad_output, stride, padding, dilation, groups
            )
        raise ValueError(f"Unsupported nd: {nd}")

    @staticmethod
    def _default_bwd_wgrad(
        grad_output: torch.Tensor,
        input: torch.Tensor,
        weight_size,
        stride,
        padding,
        dilation,
        groups,
        nd,
    ) -> torch.Tensor:
        """The weight gradient without a custom hook: PyTorch's correlation of
        ``input`` with ``grad_output``, sized to ``weight_size``."""
        if nd == 1:
            return grad.conv1d_weight(
                input, weight_size, grad_output, stride, padding, dilation, groups
            )
        elif nd == 2:
            return grad.conv2d_weight(
                input, weight_size, grad_output, stride, padding, dilation, groups
            )
        elif nd == 3:
            return grad.conv3d_weight(
                input, weight_size, grad_output, stride, padding, dilation, groups
            )
        raise ValueError(f"Unsupported nd: {nd}")

    @staticmethod
    def forward(ctx, input, weight, bias, formats, stride, padding, dilation, groups, nd):
        ctx.formats = formats
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.nd = nd
        ctx.input_size = input.shape
        ctx.weight_size = weight.shape

        q_input = formats.input_quant(input) if formats.input_quant else input
        q_weight = formats.weight_quant(weight) if formats.weight_quant else weight
        q_bias = formats.bias_quant(bias) if formats.bias_quant and bias is not None else bias

        if formats.fwd_math is not None:
            output = formats.fwd_math(
                q_input,
                q_weight,
                q_bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                nd=nd,
            )
        else:
            if nd == 1:
                output = F.conv1d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            elif nd == 2:
                output = F.conv2d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            elif nd == 3:
                output = F.conv3d(q_input, q_weight, q_bias, stride, padding, dilation, groups)
            else:
                raise ValueError(f"Unsupported nd: {nd}")

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
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        nd = ctx.nd
        input_size = ctx.input_size
        weight_size = ctx.weight_size

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
                else CustomArithConvNd._default_bwd_igrad
            )
            grad_input = bwd_igrad_math(
                q_igrad_output,
                q_weight,
                input_size=input_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                nd=nd,
            )

        if q_wgrad_output is not None:
            bwd_wgrad_math = (
                formats.bwd_wgrad_math
                if formats.bwd_wgrad_math
                else CustomArithConvNd._default_bwd_wgrad
            )
            grad_weight = bwd_wgrad_math(
                q_wgrad_output,
                q_input,
                weight_size=weight_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                nd=nd,
            )

        if ctx.needs_bias_grad and ctx.needs_input_grad[2]:
            q_bgrad_output = (
                formats.bgrad_quant(grad_output) if formats.bgrad_quant else grad_output
            )

            # The bias gradient sums over the batch and the spatial
            # dimensions, everything but the channel dimension 1:
            # nd=1 -> [0, 2], nd=2 -> [0, 2, 3], nd=3 -> [0, 2, 3, 4].
            dims_to_sum = [0] + list(range(2, nd + 2))
            grad_bias = q_bgrad_output.sum(dim=dims_to_sum)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


# The name the 2D-only version of this Function had, kept so existing imports
# still resolve; it is the same class.
CustomArithConv2d = CustomArithConvNd


class QConv1d(nn.Conv1d):
    """``torch.nn.Conv1d`` with per-tensor quantization and custom arithmetic.

    Holds the same parameters as ``nn.Conv1d`` and computes the same
    convolution, but through :class:`CustomArithConvNd`, so the input, weight,
    bias and each gradient can be rounded by the ``formats``' quantizers and
    each convolution can run in custom arithmetic. With ``formats`` left
    ``None``, or a ``QAffineFormats`` with every slot ``None``, the layer is
    numerically identical to ``nn.Conv1d``. A ``padding_mode`` other than
    ``"zeros"`` is applied with ``F.pad`` before the Function, as
    ``nn.Conv1d`` does, so the Function only ever sees zero padding.

    Args:
        in_channels (int): channels of the input.
        out_channels (int): channels of the output.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple): stride of the convolution. Default: ``1``
        padding (int, tuple or str): padding added to both sides of the
            input. Default: ``0``
        dilation (int or tuple): spacing between kernel elements.
            Default: ``1``
        groups (int): blocked connections from input to output channels.
            Default: ``1``
        bias (bool): whether the layer learns an additive bias.
            Default: ``True``
        padding_mode (str): ``"zeros"``, ``"reflect"``, ``"replicate"`` or
            ``"circular"``. Default: ``"zeros"``
        device: as for ``nn.Conv1d``. Default: ``None``
        dtype: as for ``nn.Conv1d``. Default: ``None``
        formats (QAffineFormats, optional): the quantizers and math hooks.
            ``None`` builds an empty ``QAffineFormats()``. Default: ``None``

    Shape:
        - Input: ``(N, C_in, L_in)`` or ``(C_in, L_in)``.
        - Output: ``(N, C_out, L_out)`` or ``(C_out, L_out)``, as for
          ``nn.Conv1d``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QAffineFormats, QConv1d, Quant
        >>> formats = QAffineFormats(weight_quant=Quant(BinaryK(8, 4)))
        >>> conv = QConv1d(3, 8, 3, formats=formats)
        >>> conv(torch.randn(1, 3, 16)).shape
        torch.Size([1, 8, 14])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0,)
        else:
            padding = self.padding

        return CustomArithConvNd.apply(
            input,
            self.weight,
            self.bias,
            self.formats,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            1,
        )


class QConv2d(nn.Conv2d):
    """``torch.nn.Conv2d`` with per-tensor quantization and custom arithmetic.

    The 2D twin of :class:`QConv1d`: the same parameters as ``nn.Conv2d``,
    the same convolution, computed through :class:`CustomArithConvNd` so the
    ``formats``' quantizers and math hooks apply. Numerically identical to
    ``nn.Conv2d`` with empty formats.

    Args:
        in_channels (int): channels of the input.
        out_channels (int): channels of the output.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple): stride of the convolution. Default: ``1``
        padding (int, tuple or str): padding added to all four sides of the
            input. Default: ``0``
        dilation (int or tuple): spacing between kernel elements.
            Default: ``1``
        groups (int): blocked connections from input to output channels.
            Default: ``1``
        bias (bool): whether the layer learns an additive bias.
            Default: ``True``
        padding_mode (str): ``"zeros"``, ``"reflect"``, ``"replicate"`` or
            ``"circular"``. Default: ``"zeros"``
        device: as for ``nn.Conv2d``. Default: ``None``
        dtype: as for ``nn.Conv2d``. Default: ``None``
        formats (QAffineFormats, optional): the quantizers and math hooks.
            ``None`` builds an empty ``QAffineFormats()``. Default: ``None``

    Shape:
        - Input: ``(N, C_in, H_in, W_in)`` or ``(C_in, H_in, W_in)``.
        - Output: ``(N, C_out, H_out, W_out)`` or ``(C_out, H_out, W_out)``,
          as for ``nn.Conv2d``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QAffineFormats, QConv2d, Quant
        >>> formats = QAffineFormats(weight_quant=Quant(BinaryK(8, 4)))
        >>> conv = QConv2d(3, 8, 3, formats=formats)
        >>> conv(torch.randn(1, 3, 16, 16)).shape
        torch.Size([1, 8, 14, 14])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0)
        else:
            padding = self.padding

        return CustomArithConvNd.apply(
            input,
            self.weight,
            self.bias,
            self.formats,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            2,
        )


class QConv3d(nn.Conv3d):
    """``torch.nn.Conv3d`` with per-tensor quantization and custom arithmetic.

    The 3D twin of :class:`QConv1d`: the same parameters as ``nn.Conv3d``,
    the same convolution, computed through :class:`CustomArithConvNd` so the
    ``formats``' quantizers and math hooks apply. Numerically identical to
    ``nn.Conv3d`` with empty formats.

    Args:
        in_channels (int): channels of the input.
        out_channels (int): channels of the output.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple): stride of the convolution. Default: ``1``
        padding (int, tuple or str): padding added to all six sides of the
            input. Default: ``0``
        dilation (int or tuple): spacing between kernel elements.
            Default: ``1``
        groups (int): blocked connections from input to output channels.
            Default: ``1``
        bias (bool): whether the layer learns an additive bias.
            Default: ``True``
        padding_mode (str): ``"zeros"``, ``"reflect"``, ``"replicate"`` or
            ``"circular"``. Default: ``"zeros"``
        device: as for ``nn.Conv3d``. Default: ``None``
        dtype: as for ``nn.Conv3d``. Default: ``None``
        formats (QAffineFormats, optional): the quantizers and math hooks.
            ``None`` builds an empty ``QAffineFormats()``. Default: ``None``

    Shape:
        - Input: ``(N, C_in, D_in, H_in, W_in)`` or ``(C_in, D_in, H_in, W_in)``.
        - Output: ``(N, C_out, D_out, H_out, W_out)`` or
          ``(C_out, D_out, H_out, W_out)``, as for ``nn.Conv3d``.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import QAffineFormats, QConv3d, Quant
        >>> formats = QAffineFormats(weight_quant=Quant(BinaryK(8, 4)))
        >>> conv = QConv3d(3, 8, 3, formats=formats)
        >>> conv(torch.randn(1, 3, 8, 16, 16)).shape
        torch.Size([1, 8, 6, 14, 14])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "replicate", "circular"] = "zeros",
        device=None,
        dtype=None,
        formats: QAffineFormats | None = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        if formats is None:
            self.formats = QAffineFormats()
        else:
            self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding_mode != "zeros":
            input = F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            padding = (0, 0, 0)
        else:
            padding = self.padding

        return CustomArithConvNd.apply(
            input,
            self.weight,
            self.bias,
            self.formats,
            self.stride,
            padding,
            self.dilation,
            self.groups,
            3,
        )
