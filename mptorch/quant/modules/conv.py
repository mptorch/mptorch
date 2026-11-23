"""The quantized convolutions: ``CustomArithConvNd`` and ``QConv1d/2d/3d``.

The three modules are thin ``nn.Conv1d``/``nn.Conv2d``/``nn.Conv3d`` subclasses
whose ``forward`` calls one ``torch.autograd.Function``, which takes the
number of spatial dimensions as an argument. The Function does the
quantizing, once per operand and once per gradient path, exactly as
``CustomArithLinear`` does for Linear. The quantizers in ``formats`` round
the input, the weight and the bias before the forward convolution, and round
the incoming gradient separately for the input-gradient and weight-gradient
paths, and the three math hooks can replace each convolution with custom
arithmetic. Any slot left ``None`` is plain PyTorch.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.grad as grad

from ._saved import load, quantize_operand, restore, save
from .format import QAffineFormats
from .linear import _from_float

#: What a ``None`` recompute flag means in ``CustomArithConvNd``, as
#: (input, weight): both are quantized again in backward rather than kept. The
#: weight stays alive anyway as the layer's parameter, and a convolution's
#: input usually does too, as the output a ReLU keeps for its own backward or
#: as the data batch (``_saved`` keeps a raw operand only where it stays alive
#: anyway). It is a module constant for ``torch.compile``'s sake, as in
#: ``linear.py``.
_CONV_RECOMPUTE_DEFAULT = (True, True)


class CustomArithConvNd(torch.autograd.Function):
    """Autograd Function for a quantized convolution, 1D, 2D or 3D.

    ``forward(input, weight, bias, formats, stride, padding, dilation, groups,
    nd, grad_enabled=False)`` quantizes the three operands through the
    ``*_quant`` slots of ``formats`` and computes the output with
    ``formats.fwd_math`` (called with the convolution parameters and ``nd`` as
    keywords), or ``F.conv{nd}d`` without one. ``grad_enabled`` is
    ``torch.is_grad_enabled()`` at the layer's call, which the forward, run
    with gradients off, cannot read; with it, ``F.conv{nd}d`` on the CPU runs
    the kernel ``nn.Conv{nd}d`` would run. ``backward`` quantizes the output
    gradient separately for the input-gradient path (``igrad_quant``) and the
    weight-gradient path (``wgrad_quant``), and runs ``bwd_igrad_math`` and
    ``bwd_wgrad_math``, or PyTorch's ``conv{nd}d_input`` and
    ``conv{nd}d_weight`` without them. For the bias, it sums the gradient
    quantized by ``bgrad_quant`` over the batch and spatial dimensions. Only
    the first three inputs get a gradient.
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
        if len(input_size) == nd + 1:
            # An unbatched input: torch.nn.grad's functions take a batched size
            # and gradient, so run the pass on a batch of one.
            return CustomArithConvNd._default_bwd_igrad(
                grad_output.unsqueeze(0),
                weight,
                (1, *input_size),
                stride,
                padding,
                dilation,
                groups,
                nd,
            ).squeeze(0)
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
        if input.dim() == nd + 1:
            # An unbatched input, as for the input gradient: a batch of one.
            return CustomArithConvNd._default_bwd_wgrad(
                grad_output.unsqueeze(0),
                input.unsqueeze(0),
                weight_size,
                stride,
                padding,
                dilation,
                groups,
                nd,
            )
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
    def forward(
        ctx, input, weight, bias, formats, stride, padding, dilation, groups, nd, grad_enabled=False
    ):
        # hidden from torch function modes, as in CustomArithLinear
        with torch._C.DisableTorchFunction():
            ctx.formats = formats
            ctx.stride = stride
            ctx.padding = padding
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.nd = nd
            ctx.input_size = input.shape
            ctx.weight_size = weight.shape

            recompute_input, recompute_weight = _CONV_RECOMPUTE_DEFAULT
            # Save only what backward will read, as CustomArithLinear does:
            # q_bias is never read there, q_input feeds only the weight
            # gradient, and q_weight only the input gradient. For an operand
            # that is read, what is saved is either its quantized value or the
            # raw operand, quantized again in backward (see _saved).
            q_input, saved_input, ctx.replay_input = quantize_operand(
                input,
                formats.input_quant,
                getattr(formats, "recompute_input", None),
                recompute_input,
                ctx.needs_input_grad[1],
            )
            q_weight, saved_weight, ctx.replay_weight = quantize_operand(
                weight,
                formats.weight_quant,
                getattr(formats, "recompute_weight", None),
                recompute_weight,
                ctx.needs_input_grad[0],
            )
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
                    conv = F.conv1d
                elif nd == 2:
                    conv = F.conv2d
                elif nd == 3:
                    conv = F.conv3d
                else:
                    raise ValueError(f"Unsupported nd: {nd}")
                # PyTorch's CPU convolution has kernels it runs only on a call
                # that is not recorded for backward (on an ARM CPU, a Winograd
                # kernel for 3x3 depthwise convolutions). A Function's forward
                # runs with gradients off, so the call below would run such a
                # kernel where a training nn.Conv2d runs another, whose sums
                # round differently. With gradients on whenever the layer's
                # call is recorded (grad_enabled: ctx.needs_input_grad is only
                # each input's requires_grad), it runs the kernel nn.Conv*d
                # runs, and the graph it records is dropped below (about 5 us
                # a call). No CUDA or MPS kernel depends on this, so they skip
                # it, and so does compiled code: PyTorch's own layers run a
                # compiled forward with gradients off too.
                if (
                    grad_enabled
                    and q_input.device.type == "cpu"
                    and any(ctx.needs_input_grad[:3])
                    and not torch.compiler.is_compiling()
                ):
                    with torch.enable_grad():
                        output = conv(q_input, q_weight, q_bias, stride, padding, dilation, groups)
                else:
                    output = conv(q_input, q_weight, q_bias, stride, padding, dilation, groups)
                if output.requires_grad or input.dim() == nd + 1:
                    # Detached, the result drops the graph recorded above. An
                    # unbatched input's result is also a view F.conv*d
                    # squeezed from a batched one, and autograd refuses an
                    # in-place op (an inplace ReLU after the layer) on a view
                    # a custom Function returns; detached, it is a tensor of
                    # its own.
                    output = output.detach()

            save(ctx, saved_input, saved_weight)
            ctx.needs_bias_grad = bias is not None

            return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        with torch._C.DisableTorchFunction():  # as in forward
            grad_output = grad_outputs[0]
            formats = ctx.formats
            stride = ctx.stride
            padding = ctx.padding
            dilation = ctx.dilation
            groups = ctx.groups
            nd = ctx.nd
            input_size = ctx.input_size
            weight_size = ctx.weight_size

            saved_input, saved_weight = load(ctx)

            grad_input = grad_weight = grad_bias = None

            # Quantize grad_output only for the paths that use it, and run both
            # quantizations before both math calls, as CustomArithLinear does.
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
                # A recomputed operand is quantized again only here, so its
                # quantized copy exists only for this one convolution.
                grad_input = bwd_igrad_math(
                    q_igrad_output,
                    restore(saved_weight, ctx.replay_weight),
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
                    restore(saved_input, ctx.replay_input),
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

                # The bias gradient sums over the batch and spatial dimensions,
                # that is, every dimension but the channels, dimension 1:
                # nd=1 -> [0, 2], nd=2 -> [0, 2, 3], nd=3 -> [0, 2, 3, 4]. For an
                # unbatched input the gradient has no batch dimension, and the
                # channels are dimension 0.
                if q_bgrad_output.dim() == nd + 1:
                    dims_to_sum = list(range(1, nd + 1))
                else:
                    dims_to_sum = [0] + list(range(2, nd + 2))
                grad_bias = q_bgrad_output.sum(dim=dims_to_sum)

            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None


# The name this Function had when it handled only 2D, kept so that existing
# imports still work; it is the same class.
CustomArithConv2d = CustomArithConvNd


class QConv1d(nn.Conv1d):
    """``torch.nn.Conv1d`` with a quantizer for each tensor and custom arithmetic.

    It holds the same parameters as ``nn.Conv1d`` and computes the same
    convolution, but through ``CustomArithConvNd``. The input, the
    weight, the bias and each gradient can therefore be rounded by the
    quantizers in ``formats``, and each convolution can run in custom
    arithmetic. With ``formats`` left ``None``, or with a ``QAffineFormats``
    whose slots are all ``None``, the layer is numerically identical to
    ``nn.Conv1d``. A ``padding_mode`` other than ``"zeros"`` is applied with
    ``F.pad`` before the Function, as ``nn.Conv1d`` does, so the Function
    only ever sees zero padding. For backward the layer keeps its input and
    weight raw and quantizes them again there when their quantizers are
    mptorch's stateless ones, unless the ``recompute_input`` and
    ``recompute_weight`` flags of ``formats`` say otherwise (see
    :class:`QAffineFormats`).

    Args:
        in_channels (int): channels of the input.
        out_channels (int): channels of the output.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple): stride of the convolution. Default: ``1``
        padding (int, tuple or str): padding added to both sides of the
            input. Default: ``0``
        dilation (int or tuple): spacing between kernel elements.
            Default: ``1``
        groups (int): the number of groups the channels are split into; each
            group of output channels sees only its own group of input
            channels. Default: ``1``
        bias (bool): whether the layer learns an additive bias.
            Default: ``True``
        padding_mode (str): ``"zeros"``, ``"reflect"``, ``"replicate"`` or
            ``"circular"``. Default: ``"zeros"``
        device: as for ``nn.Conv1d``. Default: ``None``
        dtype: as for ``nn.Conv1d``. Default: ``None``
        formats (QAffineFormats, optional): the quantizers and math hooks;
            :func:`mptorch.quant.conv_formats` builds hooks that run the three
            convolutions in custom arithmetic. ``None`` builds an empty
            ``QAffineFormats()``. Default: ``None``

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

    @classmethod
    def from_float(cls, conv: nn.Conv1d, formats: QAffineFormats | None = None) -> "QConv1d":
        """A ``QConv1d`` over ``conv``'s own parameters, with ``formats``, as
        :meth:`mptorch.quant.QLinear.from_float` makes a ``QLinear``.

        Raises:
            TypeError: unless ``conv`` is exactly an ``nn.Conv1d``.
        """
        return _from_float(conv, cls, nn.Conv1d, formats)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convolve ``input`` with the layer's weight, quantizing and computing as
        ``formats`` says."""
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
            torch.is_grad_enabled(),
        )


class QConv2d(nn.Conv2d):
    """``torch.nn.Conv2d`` with a quantizer for each tensor and custom arithmetic.

    The 2D counterpart of :class:`QConv1d`. It has the same parameters as
    ``nn.Conv2d`` and computes the same convolution, through
    ``CustomArithConvNd``, so the quantizers and math hooks in
    ``formats`` apply. With empty formats it is numerically identical to
    ``nn.Conv2d``.

    Args:
        in_channels (int): channels of the input.
        out_channels (int): channels of the output.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple): stride of the convolution. Default: ``1``
        padding (int, tuple or str): padding added to all four sides of the
            input. Default: ``0``
        dilation (int or tuple): spacing between kernel elements.
            Default: ``1``
        groups (int): the number of groups the channels are split into; each
            group of output channels sees only its own group of input
            channels. Default: ``1``
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

    @classmethod
    def from_float(cls, conv: nn.Conv2d, formats: QAffineFormats | None = None) -> "QConv2d":
        """A ``QConv2d`` over ``conv``'s own parameters, with ``formats``, as
        :meth:`mptorch.quant.QLinear.from_float` makes a ``QLinear``.

        Raises:
            TypeError: unless ``conv`` is exactly an ``nn.Conv2d``.
        """
        return _from_float(conv, cls, nn.Conv2d, formats)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convolve ``input`` with the layer's weight, quantizing and computing as
        ``formats`` says."""
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
            torch.is_grad_enabled(),
        )


class QConv3d(nn.Conv3d):
    """``torch.nn.Conv3d`` with a quantizer for each tensor and custom arithmetic.

    The 3D counterpart of :class:`QConv1d`. It has the same parameters as
    ``nn.Conv3d`` and computes the same convolution, through
    ``CustomArithConvNd``, so the quantizers and math hooks in
    ``formats`` apply. With empty formats it is numerically identical to
    ``nn.Conv3d``.

    Args:
        in_channels (int): channels of the input.
        out_channels (int): channels of the output.
        kernel_size (int or tuple): size of the convolving kernel.
        stride (int or tuple): stride of the convolution. Default: ``1``
        padding (int, tuple or str): padding added to all six sides of the
            input. Default: ``0``
        dilation (int or tuple): spacing between kernel elements.
            Default: ``1``
        groups (int): the number of groups the channels are split into; each
            group of output channels sees only its own group of input
            channels. Default: ``1``
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

    @classmethod
    def from_float(cls, conv: nn.Conv3d, formats: QAffineFormats | None = None) -> "QConv3d":
        """A ``QConv3d`` over ``conv``'s own parameters, with ``formats``, as
        :meth:`mptorch.quant.QLinear.from_float` makes a ``QLinear``.

        Raises:
            TypeError: unless ``conv`` is exactly an ``nn.Conv3d``.
        """
        return _from_float(conv, cls, nn.Conv3d, formats)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Convolve ``input`` with the layer's weight, quantizing and computing as
        ``formats`` says."""
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
            torch.is_grad_enabled(),
        )
