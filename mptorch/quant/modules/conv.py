from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from unfoldNd import unfoldNd

from ..functional import qlinear
from ..quant_format import QAffineFormats

__all__ = [
    "QConv1d",
    "QConv2d",
    "QConv3d",
    "QConvTranspose1d",
    "QConvTranspose2d",
    "QConvTranspose3d",
]

# TODO: support for Lazy variants


def tmp_matmul(
    X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, formats: QAffineFormats
) -> torch.Tensor:
    assert len(X.shape) == 3
    assert len(W.shape) == 2
    assert X.shape[2] == W.shape[1]
    batch, m, k = X.shape
    n = W.shape[0]
    X = X.contiguous()
    W = W.contiguous()
    X = X.view(batch * m, k)
    return qlinear(X, W, b, formats).view(batch, m, n)


class QConv1d(nn.Conv1d):
    r"""Applies a 1D convolution over an input signal composed of several input planes.

    It is a subclass of :class:`torch.nn.Conv1d` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    (which are performed using the `im2col` and `col2im` algorithms implemented in the
    `unfoldNd`_ library) and is helpful in studying the effect of low precision compute
    during inference and training (not just data quantization).

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, L)` and output :math:`(N, C_{\text{out}}, L_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_\text{in} - 1} \text{weight}(C_{\text{out}_j}, k)
        \star \text{input}(N_i, k)

    where :math:`\star` is the valid `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`L` is a length of signal sequence.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a one-element tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string `valid` or `same` or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.

    * :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`.
      For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`).

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_\text{in}, L_\text{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

    Shape:
        - Input: :math:`(N, C_\text{in}, L_\text{in})` or :math:`(C_\text{in}, L_\text{in})`
        - Output: :math:`(N, C_\text{out}, L_\text{out})` or :math:`(C_\text{out}, L_\text{out})`, where

          .. math::
              L_\text{out} = \left\lfloor\frac{L_\text{in} + 2 \times \text{padding} - \text{dilation}
                        \times (\text{kernel_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out_channels},
            \frac{\text{in_channels}}{\text{groups}}, \text{kernel_size})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{\text{groups}}{C_\text{in} * \text{kernel_size}}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``, then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{\text{groups}}{C_\text{in} * \text{kernel_size}}`

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _unfoldNd:
        https://github.com/f-dangel/unfoldNd

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        formats: QAffineFormats,
        stride: int | tuple[int] = 1,
        padding: str | int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        r"""
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            formats: Number formats used during compute (addition and multiplication) and
                quantization functions for signals during forward and back propagation (I/O
                activations, weights, biases, and neural gradients)
            stride: Stride of the convolution. Default: 1
            padding: Padding added to both sides of the input. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input
                channels to output channels. Default: 1
            bias: If ``True``, adds a learnable bias to the output. Default: ``True``
            padding_mode: ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        """
        super(QConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return bwd_quant(grad_output)

        return round.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor on which to perform the layer operations.
                Must adhere to the input shape requirements.

        Returns:
            the result of the 1D cross-correlation operation.
        """
        if self.formats.fwd_use_default_prec:
            if self.padding_mode != "zeros":
                return self.Qo(
                    F.conv1d(
                        F.pad(
                            self.Qi(input),
                            self._reversed_padding_repeated_twice,
                            mode=self.padding_mode,
                        ),
                        self.Qw(self.weight),
                        self.Qb(self.bias),
                        self.stride,
                        0,
                        self.dilation,
                        self.groups,
                    )
                )
            return self.Qo(
                F.conv1d(
                    self.Qi(input),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            )
        else:
            batch, in_channel, input_w = input.shape
            num_filter, _, kernel_w = self.weight.shape

            out_w = (
                input_w + 2 * self.padding[0] - self.dilation[0] * (kernel_w - 1) - 1
            ) // self.stride[0] + 1

            def group_conv(
                X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None
            ) -> torch.Tensor:
                if self.padding_mode != "zeros":
                    X_unf = unfoldNd(
                        torch.functional.pad(
                            X,
                            self._reversed_padding_repeated_twice,
                            mode=self.padding_mode,
                        ),
                        self.kernel_size,
                        stride=self.stride,
                        padding=0,
                        dilation=self.dilation,
                    ).transpose(1, 2)
                else:
                    X_unf = unfoldNd(
                        X,
                        self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    ).transpose(1, 2)
                Y_unf = tmp_matmul(
                    X_unf, W.view(W.size(0), -1), b, self.formats
                ).transpose(1, 2)
                return Y_unf.view(batch, -1, out_w)

            gr_channel = in_channel // self.groups
            gr_filter = num_filter // self.groups

            if self.bias is not None:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :],
                        self.bias[i * gr_filter : (i + 1) * gr_filter],
                    )
                    for i in range(self.groups)
                ]
            else:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :],
                        None,
                    )
                    for i in range(self.groups)
                ]
            return torch.cat(out_convs, 1)


class QConv2d(nn.Conv2d):
    r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    It is a subclass of :class:`torch.nn.Conv2d` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    (which are performed using the `im2col` and `col2im` algorithms implemented in the
    `unfoldNd`_ library) and is helpful in studying the effect of low precision compute
    during inference and training (not just data quantization).

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.

    * :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`.
      For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_\text{in}, L_\text{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Shape:
        - Input: :math:`(N, C_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in}, H_\text{in}, W_\text{in})`
        - Output: :math:`(N, C_\text{out}, H_\text{out}, W_\text{out})` or :math:`(C_\text{out}, H_\text{out}, W_\text{out})`, where

          .. math::
              H_\text{out} = \left\lfloor\frac{H_\text{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_\text{out} = \left\lfloor\frac{W_\text{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}},`
            :math:`\text{kernel_size[0]}, \text{kernel_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{\text{groups}}{C_\text{in} * \prod_{i=0}^{1}\text{kernel_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{\text{groups}}{C_\text{in} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _unfoldNd:
        https://github.com/f-dangel/unfoldNd

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        formats: QAffineFormats,
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        r"""
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            formats: Number formats used during compute (addition and multiplication) and
                quantization functions for signals during forward and back propagation (I/O
                activations, weights, biases, and neural gradients)
            stride: Stride of the convolution. Default: 1
            padding: Padding added to all four sides of
                the input. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input
                channels to output channels. Default: 1
            bias: If ``True``, adds a learnable bias to the
                output. Default: ``True``
            padding_mode: ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        """
        super(QConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.formats = formats
        # self.quant_parameters()
        self.params_are_quantized = False  # nazar
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return bwd_quant(grad_output)

        return round.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor on which to perform the layer operations.
                Must adhere to the input shape requirements.

        Returns:
            the result of the 2D cross-correlation operation.
        """
        if not self.params_are_quantized:  # nazar
            self.quant_parameters()
            self.params_are_quantized = True
        if self.formats.fwd_use_default_prec:
            if self.padding_mode != "zeros":
                return self.Qo(
                    F.conv2d(
                        F.pad(
                            self.Qi(input),
                            self._reversed_padding_repeated_twice,
                            mode=self.padding_mode,
                        ),
                        self.Qw(self.weight),
                        self.Qb(self.bias),
                        self.stride,
                        (0, 0),
                        self.dilation,
                        self.groups,
                    )
                )
            return self.Qo(
                F.conv2d(
                    self.Qi(input),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            )
        else:
            batch, in_channel, in_h, in_w = input.shape
            num_filter, _, kernel_h, kernel_w = self.weight.shape

            out_h = (
                in_h + 2 * self.padding[0] - self.dilation[0] * (kernel_h - 1) - 1
            ) // self.stride[0] + 1
            out_w = (
                in_w + 2 * self.padding[1] - self.dilation[1] * (kernel_w - 1) - 1
            ) // self.stride[1] + 1

            def group_conv(
                X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None
            ) -> torch.Tensor:
                if self.padding_mode != "zeros":
                    X_unf = unfoldNd(
                        torch.functional.pad(
                            X,
                            self._reversed_padding_repeated_twice,
                            mode=self.padding_mode,
                        ),
                        self.kernel_size,
                        stride=self.stride,
                        padding=(0, 0),
                        dilation=self.dilation,
                    ).transpose(1, 2)
                else:
                    X_unf = unfoldNd(
                        X,
                        self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    ).transpose(1, 2)
                Y_unf = tmp_matmul(
                    X_unf, W.view(W.size(0), -1), b, self.formats
                ).transpose(1, 2)
                return Y_unf.view(batch, -1, out_h, out_w)

            gr_channel = in_channel // self.groups
            gr_filter = num_filter // self.groups

            if self.bias is not None:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :],
                        self.bias[i * gr_filter : (i + 1) * gr_filter],
                    )
                    for i in range(self.groups)
                ]
            else:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :],
                        None,
                    )
                    for i in range(self.groups)
                ]
            return torch.cat(out_convs, 1)


class QConv3d(nn.Conv3d):
    r"""Applies a 3D convolution over an input signal composed of several input
    planes.

    It is a subclass of :class:`torch.nn.Conv3d` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    (which are performed using the `im2col` and `col2im` algorithms implemented in the
    `unfoldNd`_ library) and is helpful in studying the effect of low precision compute
    during inference and training (not just data quantization).

    In the simplest case, the output value of the layer with input size :math:`(N, C_{\text{in}}, D, H, W)`
    and output :math:`(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})` can be
    precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
                                \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 3D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`D` is a depth of input planes in pixels, :math:`H` is a height
    of input planes in pixels, and :math:`W` is width in pixels.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or an int / a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    * :attr:`groups` controls the connections between inputs and outputs.

    * :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`.
      For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        When `groups == in_channels` and `out_channels == K * in_channels`,
        where `K` is a positive integer, this operation is also known as a "depthwise convolution".

        In other words, for an input of size :math:`(N, C_\text{in}, L_\text{in})`,
        a depthwise convolution with a depthwise multiplier `K` can be performed with the arguments
        :math:`(C_\text{in}=C_\text{in}, C_\text{out}=C_\text{in} \times \text{K}, ..., \text{groups}=C_\text{in})`.

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Shape:
        - Input: :math:`(N, C_\text{in}, D_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in}, D_\text{in}, H_\text{in}, W_\text{in})`
        - Output: :math:`(N, C_\text{out}, D_\text{out}, H_\text{out}, W_\text{out})` or :math:`(C_\text{out}, D_\text{out}, H_\text{out}, W_\text{out})`,
          where

          .. math::
              D_\text{out} = \left\lfloor\frac{D_\text{in} + 2 \times \text{padding}[0] - \text{dilation}[0]
                    \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              H_\text{out} = \left\lfloor\frac{H_\text{in} + 2 \times \text{padding}[1] - \text{dilation}[1]
                    \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

          .. math::
              W_\text{out} = \left\lfloor\frac{W_\text{in} + 2 \times \text{padding}[2] - \text{dilation}[2]
                    \times (\text{kernel_size}[2] - 1) - 1}{\text{stride}[2]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out_channels}, \frac{\text{in_channels}}{\text{groups}},`
                         :math:`\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{in} * \prod_{i=0}^{2}\text{kernel_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{in} * \prod_{i=0}^{2}\text{kernel_size}[i]}`

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _unfoldNd:
        https://github.com/f-dangel/unfoldNd

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        formats: QAffineFormats,
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        r"""
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            formats: Number formats used during compute (addition and multiplication) and
                quantization functions for signals during forward and back propagation (I/O
                activations, weights, biases, and neural gradients)
            stride: Stride of the convolution. Default: 1
            padding: Padding added to all four sides of
                the input. Default: 0
            dilation: Spacing between kernel elements. Default: 1
            groups: Number of blocked connections from input
                channels to output channels. Default: 1
            bias: If ``True``, adds a learnable bias to the
                output. Default: ``True``
            padding_mode: ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        """
        super(QConv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return bwd_quant(grad_output)

        return round.apply

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor on which to perform the layer operations.
                Must adhere to the input shape requirements.

        Returns:
            the result of the 3D cross-correlation operation.
        """
        if self.formats.fwd_use_default_prec:
            if self.padding_mode != "zeros":
                return self.Qo(
                    F.conv3d(
                        F.pad(
                            self.Qi(input),
                            self._reversed_padding_repeated_twice,
                            mode=self.padding_mode,
                        ),
                        self.Qw(self.weight),
                        self.Qb(self.bias),
                        self.stride,
                        (0, 0, 0),
                        self.dilation,
                        self.groups,
                    )
                )
            return self.Qo(
                F.conv3d(
                    self.Qi(input),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            )
        else:
            batch, in_channel, in_d, in_h, in_w = input.shape
            num_filter, _, kernel_d, kernel_h, kernel_w = self.weight.shape

            out_d = (
                in_d + 2 * self.padding[0] - self.dilation[0] * (kernel_d - 1) - 1
            ) // self.stride[0] + 1
            out_h = (
                in_h + 2 * self.padding[1] - self.dilation[1] * (kernel_h - 1) - 1
            ) // self.stride[1] + 1
            out_w = (
                in_w + 2 * self.padding[2] - self.dilation[2] * (kernel_w - 1) - 1
            ) // self.stride[2] + 1

            def group_conv(
                X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None
            ) -> torch.Tensor:
                if self.padding_mode != "zeros":
                    X_unf = unfoldNd(
                        torch.functional.pad(
                            X,
                            self._reversed_padding_repeated_twice,
                            mode=self.padding_mode,
                        ),
                        self.kernel_size,
                        stride=self.stride,
                        padding=(0, 0, 0),
                        dilation=self.dilation,
                    ).transpose(1, 2)
                else:
                    X_unf = unfoldNd(
                        X,
                        self.kernel_size,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                    ).transpose(1, 2)
                Y_unf = tmp_matmul(
                    X_unf, W.view(W.size(0), -1), b, self.formats
                ).transpose(1, 2)
                return Y_unf.view(batch, -1, out_d, out_h, out_w)

            gr_channel = in_channel // self.groups
            gr_filter = num_filter // self.groups

            if self.bias is not None:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :, :],
                        self.bias[i * gr_filter : (i + 1) * gr_filter],
                    )
                    for i in range(self.groups)
                ]
            else:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :, :],
                        None,
                    )
                    for i in range(self.groups)
                ]
            return torch.cat(out_convs, 1)


class QConvTranspose1d(nn.ConvTranspose1d):
    r"""Applies a 1D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv1d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    It is a subclass of :class:`torch.nn.ConvTranspose1d` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    (which are performed using the `im2col` and `col2im` algorithms implemented in the
    `unfoldNd`_ library) and is helpful in studying the effect of low precision compute
    during inference and training (not just data quantization).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`.
      For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`).

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`torch.nn.Conv1d` and a :class:`torch.nn.ConvTranspose1d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`torch.nn.Conv1d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Shape:
        - Input: :math:`(N, C_\text{in}, L_\text{in})` or :math:`(C_\text{in}, L_\text{in})`
        - Output: :math:`(N, C_\text{out}, L_\text{out})` or :math:`(C_\text{out}, L_\text{out})`, where

          .. math::
              L_\text{out} = (L_\text{in} - 1) \times \text{stride} - 2 \times \text{padding} + \text{dilation}
                        \times (\text{kernel_size} - 1) + \text{output_padding} + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}},`
                         :math:`\text{kernel_size})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{out} * \text{kernel_size}}`
        bias (Tensor):   the learnable bias of the module of shape (`out_channels`).
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{out} * \text{kernel_size}}`

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

    .. _unfoldNd:
        https://github.com/f-dangel/unfoldNd
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        formats: QAffineFormats,
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] = 0,
        output_padding: int | tuple[int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int] = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        r"""
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            formats: Number formats used during compute (addition and multiplication) and
                quantization functions for signals during forward and back propagation (I/O
                activations, weights, biases, and neural gradients)
            stride: Stride of the convolution. Default: 1
            padding: ``dilation * (kernel_size - 1) - padding`` zero-padding
                will be added to both sides of the input. Default: 0
            output_padding: Additional size added to one side
                of the output shape. Default: 0
            groups: Number of blocked connections from input channels to output channels. Default: 1
            bias: If ``True``, adds a learnable bias to the output. Default: ``True``
            dilation: Spacing between kernel elements. Default: 1
        """
        super(QConvTranspose1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return bwd_quant(grad_output)

        return round.apply

    def forward(
        self, input: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor on which to perform the layer operations.
                Must adhere to the input shape requirements.
            output_size: specifies how the output should be padded

        Returns:
            the result of the 1D transposed convolution operation.
        """
        num_spatial_dims = 1
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        if self.formats.fwd_use_default_prec:
            if self.padding_mode != "zeros":
                raise ValueError(
                    "Only `zeros` padding mode is supported for QConvTranspose2d"
                )
            return self.Qo(
                F.conv_transpose1d(
                    self.Qi(input),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.stride,
                    self.padding,
                    output_padding,
                    self.groups,
                    self.dilation,
                )
            )
        else:
            batch, in_channel, in_w = input.shape
            num_filter, _, kernel_w = self.weight.shape

            nin_w = (in_w - 1) * self.stride[0] + 1
            nW_w = (kernel_w - 1) * self.dilation[0] + 1
            padding_w = nW_w - 1 - self.padding[0]

            def group_conv(
                X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None
            ) -> torch.Tensor:
                nX = torch.zeros(
                    (X.shape[0], X.shape[1], nin_w),
                    dtype=X.dtype,
                    device=X.device,
                )
                nX[:, :, 0 : nin_w : self.stride[0]] = X
                dW = torch.zeros(
                    (W.shape[0], W.shape[1], nW_w), dtype=W.dtype, device=W.device
                )
                dW[:, :, 0 : nW_w : self.dilation[0]] = W
                nW = torch.flip(dW, [2]).transpose(0, 1)

                X_unf = unfoldNd(
                    nX,
                    (nW.shape[2]),
                    padding=(padding_w),
                ).transpose(1, 2)
                Y_unf = tmp_matmul(
                    X_unf, nW.contiguous().view(nW.size(0), -1), b, self.formats
                ).transpose(1, 2)

                out_w = nin_w + 2 * padding_w - (nW.shape[2] - 1)

                if self.output_padding == (0):
                    return Y_unf.view(batch, -1, out_w)
                else:
                    Y_out = Y_unf.view(batch, -1, out_w)
                    Y_padded = torch.zeros(
                        (
                            batch,
                            Y_out.shape[1],
                            out_w + output_padding[0],
                        ),
                        device=Y_out.device,
                        dtype=Y_out.dtype,
                    )
                    Y_padded[:, :, 0:out_w] = Y_out
                    return Y_padded

            gr_channel = in_channel // self.groups
            gr_filter = num_filter // self.groups

            if self.bias is not None:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :],
                        self.bias[i * gr_filter : (i + 1) * gr_filter],
                    )
                    for i in range(self.groups)
                ]
            else:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :],
                        None,
                    )
                    for i in range(self.groups)
                ]
            return torch.cat(out_convs, 1)


class QConvTranspose2d(nn.ConvTranspose2d):
    r"""Applies a 2D transposed convolution operator over an input image
    composed of several input planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    It is a subclass of :class:`torch.nn.ConvTranspose2d` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    (which are performed using the `im2col` and `col2im` algorithms implemented in the
    `unfoldNd`_ library) and is helpful in studying the effect of low precision compute
    during inference and training (not just data quantization).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`.
      For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimensions
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`torch.nn.Conv2d` and a :class:`torch.nn.ConvTranspose2d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`torch.nn.Conv2d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Shape:
        - Input: :math:`(N, C_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in}, H_\text{in}, W_\text{in})`
        - Output: :math:`(N, C_\text{out}, H_\text{out}, W_\text{out})` or :math:`(C_\text{out}, H_\text{out}, W_\text{out})`, where

        .. math::
              H_\text{out} = (H_\text{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1
        .. math::
              W_\text{out} = (W_\text{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}},`
                         :math:`\text{kernel_size[0]}, \text{kernel_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{out} * \prod_{i=0}^{1}\text{kernel_size}[i]}`

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

    .. _unfoldNd:
        https://github.com/f-dangel/unfoldNd
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        formats: QAffineFormats,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        r"""
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            formats: Number formats used during compute (addition and multiplication) and
                quantization functions for signals during forward and back propagation (I/O
                activations, weights, biases, and neural gradients)
            stride: Stride of the convolution. Default: 1
            padding: ``dilation * (kernel_size - 1) - padding`` zero-padding
                will be added to both sides of the input. Default: 0
            output_padding: Additional size added to one side
                of the output shape. Default: 0
            groups: Number of blocked connections from input channels to output channels. Default: 1
            bias: If ``True``, adds a learnable bias to the output. Default: ``True``
            dilation: Spacing between kernel elements. Default: 1
        """
        super(QConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return bwd_quant(grad_output)

        return round.apply

    def forward(
        self, input: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor on which to perform the layer operations.
                Must adhere to the input shape requirements.
            output_size: specifies how the output should be padded

        Returns:
            the result of the 2D transposed convolution operation.
        """
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        if self.formats.fwd_use_default_prec:
            if self.padding_mode != "zeros":
                raise ValueError(
                    "Only `zeros` padding mode is supported for QConvTranspose2d"
                )
            return self.Qo(
                F.conv_transpose2d(
                    self.Qi(input),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.stride,
                    self.padding,
                    output_padding,
                    self.groups,
                    self.dilation,
                )
            )
        else:
            batch, in_channel, in_h, in_w = input.shape
            num_filter, _, kernel_h, kernel_w = self.weight.shape

            nin_h = (in_h - 1) * self.stride[0] + 1
            nin_w = (in_w - 1) * self.stride[1] + 1
            nW_h = (kernel_h - 1) * self.dilation[0] + 1
            nW_w = (kernel_w - 1) * self.dilation[1] + 1
            padding_h = nW_h - 1 - self.padding[0]
            padding_w = nW_w - 1 - self.padding[1]

            def group_conv(
                X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None
            ) -> torch.Tensor:
                nX = torch.zeros(
                    (X.shape[0], X.shape[1], nin_h, nin_w),
                    dtype=X.dtype,
                    device=X.device,
                )
                nX[:, :, 0 : nin_h : self.stride[0], 0 : nin_w : self.stride[1]] = X
                dW = torch.zeros(
                    (W.shape[0], W.shape[1], nW_h, nW_w), dtype=W.dtype, device=W.device
                )
                dW[:, :, 0 : nW_h : self.dilation[0], 0 : nW_w : self.dilation[1]] = W
                nW = torch.flip(dW, [2, 3]).transpose(0, 1)

                X_unf = unfoldNd(
                    nX,
                    (nW.shape[2], nW.shape[3]),
                    padding=(padding_h, padding_w),
                ).transpose(1, 2)
                Y_unf = tmp_matmul(
                    X_unf, nW.contiguous().view(nW.size(0), -1), b, self.formats
                ).transpose(1, 2)

                out_h = nin_h + 2 * padding_h - (nW.shape[2] - 1)
                out_w = nin_w + 2 * padding_w - (nW.shape[3] - 1)

                if self.output_padding == (0, 0):
                    return Y_unf.view(batch, -1, out_h, out_w)
                else:
                    Y_out = Y_unf.view(batch, -1, out_h, out_w)
                    Y_padded = torch.zeros(
                        (
                            batch,
                            Y_out.shape[1],
                            out_h + output_padding[0],
                            out_w + output_padding[1],
                        ),
                        device=Y_out.device,
                        dtype=Y_out.dtype,
                    )
                    Y_padded[:, :, 0:out_h, 0:out_w] = Y_out
                    return Y_padded

            gr_channel = in_channel // self.groups
            gr_filter = num_filter // self.groups

            if self.bias is not None:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :],
                        self.bias[i * gr_filter : (i + 1) * gr_filter],
                    )
                    for i in range(self.groups)
                ]
            else:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :],
                        None,
                    )
                    for i in range(self.groups)
                ]
            return torch.cat(out_convs, 1)


class QConvTranspose3d(nn.ConvTranspose3d):
    r"""Applies a 3D transposed convolution operator over an input image
    composed of several input planes. The transposed convolution operator multiplies
    each input value element-wise by a learnable kernel, and sums over the outputs
    from all input feature planes.

    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or
    a deconvolution (although it is not an actual deconvolution operation as it does
    not compute a true inverse of convolution). For more information, see the visualizations
    `here`_ and the `Deconvolutional Networks`_ paper.

    It is a subclass of :class:`torch.nn.ConvTranspose3d` and allows one to specify if I/O
    signals should be quantized during inference & training (needed for instance
    in QAT and PTQ methods), but also the precision(s) to be used in internal GEMM
    computations (addition and multiplication, fused or not). This allows simulating
    the effect of custom precision during GEMM calls in the forward and backward pass
    (which are performed using the `im2col` and `col2im` algorithms implemented in the
    `unfoldNd`_ library) and is helpful in studying the effect of low precision compute
    during inference and training (not just data quantization).

    * :attr:`stride` controls the stride for the cross-correlation.

    * :attr:`padding` controls the amount of implicit zero padding on both
      sides for ``dilation * (kernel_size - 1) - padding`` number of points. See note
      below for details.

    * :attr:`output_padding` controls the additional size added to one side
      of the output shape. See note below for details.

    * :attr:`dilation` controls the spacing between the kernel points; also known as the à trous algorithm.
      It is harder to describe, but the link `here`_ has a nice visualization of what :attr:`dilation` does.

    * :attr:`in_channels` and :attr:`out_channels` must both be divisible by :attr:`groups`.
      For example,

        * At groups=1, all inputs are convolved to all outputs.
        * At groups=2, the operation becomes equivalent to having two conv
          layers side by side, each seeing half the input channels
          and producing half the output channels, and both subsequently
          concatenated.
        * At groups= :attr:`in_channels`, each input channel is convolved with
          its own set of filters (of size
          :math:`\frac{\text{out_channels}}{\text{in_channels}}`).

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`output_padding`
    can either be:

        - a single ``int`` -- in which case the same value is used for the depth, height and width dimensions
        - a ``tuple`` of three ints -- in which case, the first `int` is used for the depth dimension,
          the second `int` for the height dimension and the third `int` for the width dimension

    Note:
        The :attr:`padding` argument effectively adds ``dilation * (kernel_size - 1) - padding``
        amount of zero padding to both sizes of the input. This is set so that
        when a :class:`torch.nn.Conv3d` and a :class:`torch.nn.ConvTranspose3d`
        are initialized with same parameters, they are inverses of each other in
        regard to the input and output shapes. However, when ``stride > 1``,
        :class:`torch.nn.Conv3d` maps multiple input shapes to the same output
        shape. :attr:`output_padding` is provided to resolve this ambiguity by
        effectively increasing the calculated output shape on one side. Note
        that :attr:`output_padding` is only used to find output shape, but does
        not actually add zero-padding to output.

    Shape:
        - Input: :math:`(N, C_\text{in}, D_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in}, D_\text{in}, H_\text{in}, W_\text{in})`
        - Output: :math:`(N, C_\text{out}, D_\text{out}, H_\text{out}, W_\text{out})` or
          :math:`(C_\text{out}, D_\text{out}, H_\text{out}, W_\text{out})`, where

        .. math::
              D_\text{out} = (D_\text{in} - 1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) + \text{output_padding}[0] + 1
        .. math::
              H_\text{out} = (H_\text{in} - 1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) + \text{output_padding}[1] + 1
        .. math::
              W_\text{out} = (W_\text{in} - 1) \times \text{stride}[2] - 2 \times \text{padding}[2] + \text{dilation}[2]
                        \times (\text{kernel_size}[2] - 1) + \text{output_padding}[2] + 1


    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{in_channels}, \frac{\text{out_channels}}{\text{groups}},`
                         :math:`\text{kernel_size[0]}, \text{kernel_size[1]}, \text{kernel_size[2]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{out} * \prod_{i=0}^{2}\text{kernel_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
                         If :attr:`bias` is ``True``, then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{\text{groups}}{C_\text{out} * \prod_{i=0}^{2}\text{kernel_size}[i]}`

    .. _`here`:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md

    .. _`Deconvolutional Networks`:
        https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf

    .. _unfoldNd:
        https://github.com/f-dangel/unfoldNd
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        formats: QAffineFormats,
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        output_padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int, int] = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ):
        r"""
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            formats: Number formats used during compute (addition and multiplication) and
                quantization functions for signals during forward and back propagation (I/O
                activations, weights, biases, and neural gradients)
            stride: Stride of the convolution. Default: 1
            padding: ``dilation * (kernel_size - 1) - padding`` zero-padding
                will be added to both sides of the input. Default: 0
            output_padding: Additional size added to one side
                of the output shape. Default: 0
            groups: Number of blocked connections from input channels to output channels. Default: 1
            bias: If ``True``, adds a learnable bias to the output. Default: ``True``
            dilation: Spacing between kernel elements. Default: 1
        """
        super(QConvTranspose3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self.formats = formats
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        r"""Sets a straight-through estimator-like function to all the
        quantized signals in the module (these can be the weight, bias, input,
        and/or output signals), depending if quantizers are specified in the
        associated :class:`QAffineFormats` :attr:`formats` parameter.
        """
        self.Qw = self.quant_function(
            self.formats.weight_quant, self.formats.grad_quant
        )
        self.Qb = self.quant_function(self.formats.bias_quant, self.formats.grad_quant)
        self.Qi = self.quant_function(self.formats.input_quant, self.formats.grad_quant)
        self.Qo = self.quant_function(
            self.formats.output_quant, self.formats.grad_quant
        )

    def quant_parameters(self):
        r"""Quantizes the module parameters :attr:`weight` and :attr:`bias` using
        the quantization functions specified in :attr:`formats.weight_quant` and
        :attr:`formats.bias_quant`, respectively."""
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(
        self,
        fwd_quant: Callable[[torch.Tensor], torch.Tensor],
        bwd_quant: Callable[[torch.Tensor], torch.Tensor],
    ):
        r"""Defines a straight-through estimator-like function (see *Estimating or
        Propagating Gradients Through Stochastic Neurons for Conditional Computation*
        (https://arxiv.org/abs/1308.3432)) that applies potentially different
        quantization functions in the forward and backward passes through the
        input and output gradient signals, respectively.

        Args:
            fwd_quant: the quantization function to apply during the forward pass
                through the input signal
            bwd_quant: the quantization function to apply during the backward pass
                through the output gradient signal
        """

        class round(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                if x is not None:
                    out = fwd_quant(x)
                    return out
                else:
                    return x

            @staticmethod
            def backward(ctx, grad_output):
                return bwd_quant(grad_output)

        return round.apply

    def forward(
        self, input: torch.Tensor, output_size: list[int] | None = None
    ) -> torch.Tensor:
        r"""Describes the computations that get performed at every call of the module. The use
        of quantized elementary operations (i.e., additions and multiplications) in the FWD
        and BWD passes are controlled through the :attr:`formats` argument to the module constructor.

        Args:
            input: the input tensor on which to perform the layer operations.
                Must adhere to the input shape requirements.
            output_size: specifies how the output should be padded

        Returns:
            the result of the 3D transposed convolution operation.
        """
        num_spatial_dims = 3
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            num_spatial_dims,
            self.dilation,
        )
        if self.formats.fwd_use_default_prec:
            if self.padding_mode != "zeros":
                raise ValueError(
                    "Only `zeros` padding mode is supported for QConvTranspose2d"
                )
            return self.Qo(
                F.conv_transpose3d(
                    self.Qi(input),
                    self.Qw(self.weight),
                    self.Qb(self.bias),
                    self.stride,
                    self.padding,
                    output_padding,
                    self.groups,
                    self.dilation,
                )
            )
        else:
            batch, in_channel, in_d, in_h, in_w = input.shape
            num_filter, _, kernel_d, kernel_h, kernel_w = self.weight.shape

            nin_d = (in_d - 1) * self.stride[0] + 1
            nin_h = (in_h - 1) * self.stride[1] + 1
            nin_w = (in_w - 1) * self.stride[2] + 1
            nW_d = (kernel_d - 1) * self.dilation[0] + 1
            nW_h = (kernel_h - 1) * self.dilation[1] + 1
            nW_w = (kernel_w - 1) * self.dilation[2] + 1
            padding_d = nW_d - 1 - self.padding[0]
            padding_h = nW_h - 1 - self.padding[1]
            padding_w = nW_w - 1 - self.padding[2]

            def group_conv(
                X: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None
            ) -> torch.Tensor:
                nX = torch.zeros(
                    (X.shape[0], X.shape[1], nin_d, nin_h, nin_w),
                    dtype=X.dtype,
                    device=X.device,
                )
                nX[
                    :,
                    :,
                    0 : nin_d : self.stride[0],
                    0 : nin_h : self.stride[1],
                    0 : nin_w : self.stride[2],
                ] = X
                dW = torch.zeros(
                    (W.shape[0], W.shape[1], nW_d, nW_h, nW_w),
                    dtype=W.dtype,
                    device=W.device,
                )
                dW[
                    :,
                    :,
                    0 : nW_d : self.dilation[0],
                    0 : nW_h : self.dilation[1],
                    0 : nW_w : self.dilation[2],
                ] = W
                nW = torch.flip(dW, [2, 3, 4]).transpose(0, 1)

                X_unf = unfoldNd(
                    nX,
                    (nW.shape[2], nW.shape[3], nW.shape[4]),
                    padding=(padding_d, padding_h, padding_w),
                ).transpose(1, 2)
                Y_unf = tmp_matmul(
                    X_unf, nW.contiguous().view(nW.size(0), -1), b, self.formats
                ).transpose(1, 2)

                out_d = nin_d + 2 * padding_d - (nW.shape[2] - 1)
                out_h = nin_h + 2 * padding_h - (nW.shape[3] - 1)
                out_w = nin_w + 2 * padding_w - (nW.shape[4] - 1)

                if self.output_padding == (0, 0, 0):
                    return Y_unf.view(batch, -1, out_d, out_h, out_w)
                else:
                    Y_out = Y_unf.view(batch, -1, out_d, out_h, out_w)
                    Y_padded = torch.zeros(
                        (
                            batch,
                            Y_out.shape[1],
                            out_d + output_padding[0],
                            out_h + output_padding[1],
                            out_w + output_padding[2],
                        ),
                        device=Y_out.device,
                        dtype=Y_out.dtype,
                    )
                    Y_padded[:, :, 0:out_d, 0:out_h, 0:out_w] = Y_out
                    return Y_padded

            gr_channel = in_channel // self.groups
            gr_filter = num_filter // self.groups

            if self.bias is not None:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :, :],
                        self.bias[i * gr_filter : (i + 1) * gr_filter],
                    )
                    for i in range(self.groups)
                ]
            else:
                out_convs = [
                    group_conv(
                        input[:, i * gr_channel : (i + 1) * gr_channel, :, :, :],
                        self.weight[i * gr_filter : (i + 1) * gr_filter, :, :, :, :],
                        None,
                    )
                    for i in range(self.groups)
                ]
            return torch.cat(out_convs, 1)
