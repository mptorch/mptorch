"""Activation modules that round as an ``ActivationFormats`` says.

Each class is its torch class with a ``formats`` attribute: ``None`` runs the
torch module's own forward, ``inplace`` included, and an
:class:`~mptorch.quant.ActivationFormats` runs the matching
``mptorch.quant.pointwise`` op, out of place even when ``inplace`` is set (a
rounded result is a new tensor, so the input is never written).
:class:`QSoftmax` takes a :class:`~mptorch.quant.SoftmaxFormats` instead.
:func:`~mptorch.quant.convert_` turns torch's activation modules into these.
"""

from typing import Any

import torch
from torch import nn

# Module references rather than names: `pointwise` and `norm` import this package
# for `_opaque`, and the names are read when a forward runs.
from .. import norm, pointwise


class QGELU(nn.GELU):
    """:class:`torch.nn.GELU` that rounds as ``formats`` says (:func:`~mptorch.quant.qgelu`)."""

    _mptorch_owned = True

    def __init__(self, approximate: str = "none", formats: Any = None) -> None:
        super().__init__(approximate)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qgelu(input, self.formats, self.approximate)


class QSiLU(nn.SiLU):
    """:class:`torch.nn.SiLU` that rounds as ``formats`` says (:func:`~mptorch.quant.qsilu`)."""

    _mptorch_owned = True

    def __init__(self, inplace: bool = False, formats: Any = None) -> None:
        super().__init__(inplace)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qsilu(input, self.formats)


class QSigmoid(nn.Sigmoid):
    """:class:`torch.nn.Sigmoid` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qsigmoid`)."""

    _mptorch_owned = True

    def __init__(self, formats: Any = None) -> None:
        super().__init__()
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qsigmoid(input, self.formats)


class QTanh(nn.Tanh):
    """:class:`torch.nn.Tanh` that rounds as ``formats`` says (:func:`~mptorch.quant.qtanh`)."""

    _mptorch_owned = True

    def __init__(self, formats: Any = None) -> None:
        super().__init__()
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qtanh(input, self.formats)


class QHardswish(nn.Hardswish):
    """:class:`torch.nn.Hardswish` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qhardswish`)."""

    _mptorch_owned = True

    def __init__(self, inplace: bool = False, formats: Any = None) -> None:
        super().__init__(inplace)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qhardswish(input, self.formats)


class QHardsigmoid(nn.Hardsigmoid):
    """:class:`torch.nn.Hardsigmoid` that rounds as ``formats`` says
    (:func:`~mptorch.quant.qhardsigmoid`)."""

    _mptorch_owned = True

    def __init__(self, inplace: bool = False, formats: Any = None) -> None:
        super().__init__(inplace)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qhardsigmoid(input, self.formats)


class QReLU(nn.ReLU):
    """:class:`torch.nn.ReLU` that rounds as ``formats`` says (:func:`~mptorch.quant.qrelu`)."""

    _mptorch_owned = True

    def __init__(self, inplace: bool = False, formats: Any = None) -> None:
        super().__init__(inplace)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The activation, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return pointwise.qrelu(input, self.formats)


class QSoftmax(nn.Softmax):
    """:class:`torch.nn.Softmax` that rounds as ``formats``, a
    :class:`~mptorch.quant.SoftmaxFormats`, says (:func:`~mptorch.quant.qsoftmax`)."""

    _mptorch_owned = True

    def __init__(self, dim: int | None = None, formats: Any = None) -> None:
        super().__init__(dim)
        self.formats = formats

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The softmax, rounded as ``formats`` says (torch's with ``None``)."""
        if self.formats is None:
            return super().forward(input)
        return norm.qsoftmax(input, self.dim, self.formats)
