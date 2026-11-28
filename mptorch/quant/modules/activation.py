"""Activation modules that round as an ``ActivationFormats`` says.

Each class is its torch class with a ``formats`` attribute: ``None`` runs the
torch module's own forward, ``inplace`` included, and an
:class:`~mptorch.quant.ActivationFormats` runs the matching
``mptorch.quant.functional.activation`` op, out of place even when
``inplace`` is set (a rounded result is a new tensor, so the input is never
written); :class:`QPWLActivation` evaluates a table. :class:`QSoftmax` is
with the norms, in ``modules/norm.py``. :func:`~mptorch.quant.convert_` turns
torch's activation modules into these.
"""

from typing import Any

import torch
from torch import nn

from ..functional.activation import qgelu, qhardsigmoid, qhardswish, qrelu, qsigmoid, qsilu, qtanh
from ..functional.pwl import qpwl


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
        return qgelu(input, self.formats, self.approximate)


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
        return qsilu(input, self.formats)


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
        return qsigmoid(input, self.formats)


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
        return qtanh(input, self.formats)


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
        return qhardswish(input, self.formats)


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
        return qhardsigmoid(input, self.formats)


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
        return qrelu(input, self.formats)


class QPWLActivation(nn.Module):
    """A piecewise-linear table as a module (:func:`~mptorch.quant.qpwl`), for
    a function of your own; torch's activations take a table through their
    ``formats`` (:class:`~mptorch.quant.ActivationFormats`' ``table``).

    Args:
        table (PWLTable): the table.
        quant, grad_quant, derivative, carrier: as for :func:`~mptorch.quant.qpwl`.
    """

    _mptorch_owned = True

    def __init__(
        self,
        table: Any,
        *,
        quant: Any = None,
        grad_quant: Any = None,
        derivative: Any = None,
        carrier: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.table, self.quant, self.grad_quant = table, quant, grad_quant
        self.derivative, self.carrier = derivative, carrier

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The table at ``input``, rounded as ``quant`` says."""
        return qpwl(
            input,
            self.table,
            quant=self.quant,
            grad_quant=self.grad_quant,
            derivative=self.derivative,
            carrier=self.carrier,
        )

    def extra_repr(self) -> str:
        return self.table.label()
