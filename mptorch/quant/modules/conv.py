import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from unfoldNd import unfoldNd

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Union, Optional, List

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


def tmp_matmul(X: Tensor, W: Tensor, b: Optional[Tensor], formats: QAffineFormats):
    assert len(X.shape) == 3
    assert len(W.shape) == 2
    assert X.shape[2] == W.shape[1]
    batch, m, k = X.shape
    n = W.shape[0]
    X = X.contiguous()
    W = W.contiguous()
    X = X.view(batch * m, k)
    return qlinear.apply(X, W, b, formats).view(batch, m, n)


class QConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        formats: QAffineFormats,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
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
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self) -> None:
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
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
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input: Tensor) -> Tensor:
        if self.formats.use_default_prec:
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

            def group_conv(X: Tensor, W: Tensor, b: Optional[Tensor]):
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        formats: QAffineFormats,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
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
        self.quant_parameters()
        self.reset_quant_function()

    def reset_quant_function(self):
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self) -> None:
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
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
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input: Tensor) -> Tensor:
        if self.formats.use_default_prec:
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

            def group_conv(X: Tensor, W: Tensor, b: Optional[Tensor]):
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        formats: QAffineFormats,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
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
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self) -> None:
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
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
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input: Tensor) -> Tensor:
        if self.formats.use_default_prec:
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

            def group_conv(X: Tensor, W: Tensor, b: Optional[Tensor]):
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        formats: QAffineFormats,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
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
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self) -> None:
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
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
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
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
        if self.formats.use_default_prec:
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

            def group_conv(X: Tensor, W: Tensor, b: Optional[Tensor]):
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        formats: QAffineFormats,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
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
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self) -> None:
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
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
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
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
        if self.formats.use_default_prec:
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

            def group_conv(X: Tensor, W: Tensor, b: Optional[Tensor]):
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
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        formats: QAffineFormats,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
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
        self.Qw = self.quant_function(self.formats.weight_quant)
        self.Qb = self.quant_function(self.formats.bias_quant)
        self.Qi = self.quant_function(self.formats.input_quant)
        self.Qo = self.quant_function(self.formats.output_quant)

    def quant_parameters(self) -> None:
        self.weight.data = self.formats.weight_quant(self.weight.data)
        if self.bias is not None:
            self.bias.data = self.formats.bias_quant(self.bias.data)

    def quant_function(self, fwd_quant):
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
                return self.formats.grad_quant(grad_output)

        return round.apply

    def forward(self, input: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
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
        if self.formats.use_default_prec:
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

            def group_conv(X: Tensor, W: Tensor, b: Optional[Tensor]):
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
