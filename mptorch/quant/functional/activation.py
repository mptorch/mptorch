"""Activations that round their results to a simulated format, as an ``ActivationFormats`` says.

``qgelu``, ``qsilu``, ``qsigmoid``, ``qtanh``, ``qhardswish``, ``qhardsigmoid``
and ``qrelu`` compute in a carrier and round once, with the slots, carriers
and backward of :mod:`mptorch.quant.functional.pointwise`'s ops: a slot left
``None`` is torch's own computation, and the backward is torch's derivative of
the unrounded activation, rounded by the gradient slot. The three gated
activations (GELU, SiLU, hardswish) can round their gate as well. With a table
(``ActivationFormats(table=)``) an activation evaluates a piecewise-linear
table of ``mptorch.quant.functional.pwl`` instead of torch's function.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .._opaque import _opaque_backward
from ..mac import _as_quant
from ..ops._checks import _checked_carrier
from ..ops._spec import _round_result, _stored
from .pointwise import _SAME, _carrier_dtype, _checked_slot, _label, _with_section
from .pwl import PWLTable, _rounded

__all__ = [
    "ActivationFormats",
    "qgelu",
    "qsilu",
    "qsigmoid",
    "qtanh",
    "qhardswish",
    "qhardsigmoid",
    "qrelu",
]


@dataclass(frozen=True, kw_only=True)
class ActivationFormats:
    """The formats of an activation: its gate's rounding, its result's and its gradient's.

    It is the value of a recipe's ``activation`` kind, and of the ``formats``
    of :class:`~mptorch.quant.QGELU` and its siblings. A slot left ``None``
    does not round; with every slot ``None`` an activation is torch's.

    Args:
        inner: rounds the gate of the three gated activations, computed in
            the carrier: GELU's ``erf(x / sqrt(2))`` (or, approximated,
            ``tanh(sqrt(2/pi) * (x + 0.044715 * x**3))``), SiLU's
            ``sigmoid(x)`` and hardswish's ``hardsigmoid(x)``. The result is
            then ``0.5 * x * (1 + gate)`` or ``x * gate`` in the carrier.
            Sigmoid, tanh, hardsigmoid and ReLU have no gate and ignore it.
            A format means ``Quant(fmt)``. Default: ``None``, torch's function
            in the carrier
        out: rounds the result. A format means ``Quant(fmt)``.
            Default: ``None``
        grad: rounds the input's gradient: torch's derivative of the
            unrounded activation, computed in the carrier, times the incoming
            gradient. A format means ``Quant(fmt)``. Default: ``None``
        carrier (torch.dtype, optional): as for :class:`PointwiseFormats`.
            Default: ``None``
        table (PWLTable, optional): evaluate this piecewise-linear table in
            the carrier in place of the activation's function
            (:class:`~mptorch.quant.PWLTable`), ``out`` rounding the result.
            The gradient stays the activation's own, the derivative of the
            function the table approximates. Default: ``None``

    Raises:
        TypeError, ValueError: as for :class:`PointwiseFormats`, and for a
            ``table`` that is not a ``PWLTable``, a table of 1/sqrt, or a table
            together with ``inner`` (a table has no gate).
    """

    inner: Any = None
    out: Any = None
    grad: Any = None
    carrier: torch.dtype | None = None
    table: Any = None

    def __post_init__(self) -> None:
        carrier = _checked_carrier(self.carrier)
        for name in ("inner", "out", "grad"):
            slot = _as_quant(getattr(self, name), name)
            _checked_slot(slot, carrier, name)
            object.__setattr__(self, name, slot)
        if self.table is not None:
            if not isinstance(self.table, PWLTable):
                raise TypeError(
                    "ActivationFormats' table is a PWLTable or None, "
                    f"got {type(self.table).__name__}"
                )
            if self.table.reduction is not None:
                raise ValueError(
                    "this is a table of 1/sqrt, which a NormFormats takes as its inverse: an "
                    "activation's table has no reduction"
                )
            if self.inner is not None:
                raise ValueError(
                    "ActivationFormats has both a table and an inner rounding: a table replaces "
                    "the whole function, which then has no gate to round -- set one of them"
                )

    @classmethod
    def uniform(
        cls, quant: Any, *, grad: Any = _SAME, carrier: torch.dtype | None = None
    ) -> ActivationFormats:
        """One rounding for the gate, the result and the gradient; ``grad``
        replaces the gradient's (``None`` leaves it unrounded)."""
        return cls(inner=quant, out=quant, grad=quant if grad is _SAME else grad, carrier=carrier)

    @property
    def _rounds(self) -> bool:
        return (
            self.inner is not None
            or self.out is not None
            or self.grad is not None
            or self.table is not None
        )

    def label(self) -> str:
        """A one-line summary, for reports."""
        slots = {"inner": self.inner, "out": self.out, "grad": self.grad}
        if self.inner is not None and self.inner == self.out == self.grad:
            parts = [f"quant {_label(self.inner)}"]
        else:
            parts = [f"{n} {_label(q)}" for n, q in slots.items() if q is not None]
        if self.table is not None:
            parts.insert(0, self.table.label())
        if self.carrier is not None:
            parts.append(f"carrier {str(self.carrier).removeprefix('torch.')}")
        return ", ".join(parts) or "torch"


# --- the activations -------------------------------------------------------------

# GELU's constants, as ATen's GeluKernel names them
_SQRT1_2 = math.sqrt(0.5)
_BETA = math.sqrt(2.0) * (2.0 / math.sqrt(math.pi)) * 0.5
_KAPPA = 0.044715

_GATED = ("gelu", "silu", "hardswish")


def _act_plain(act: str, x: torch.Tensor, approximate: str) -> torch.Tensor:
    if act == "gelu":
        return F.gelu(x, approximate=approximate)
    if act == "silu":
        return F.silu(x)
    if act == "sigmoid":
        return torch.sigmoid(x)
    if act == "tanh":
        return torch.tanh(x)
    if act == "hardswish":
        return F.hardswish(x)
    if act == "hardsigmoid":
        return F.hardsigmoid(x)
    return torch.relu(x)


def _gate(act: str, x: torch.Tensor, approximate: str) -> torch.Tensor:
    if act == "gelu":
        if approximate == "tanh":
            return torch.tanh(_BETA * (x + _KAPPA * (x * x * x)))
        return torch.erf(x * _SQRT1_2)
    if act == "silu":
        return torch.sigmoid(x)
    return F.hardsigmoid(x)


def _combine(act: str, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if act == "gelu":
        return x * 0.5 * (1 + gate)
    return x * gate


def _act_grad(act: str, g: torch.Tensor, x: torch.Tensor, approximate: str) -> torch.Tensor:
    """torch's backward op for ``act``, which is the gradient autograd gives."""
    aten = torch.ops.aten
    if act == "gelu":
        return aten.gelu_backward(g, x, approximate=approximate)
    if act == "silu":
        return aten.silu_backward(g, x)
    if act == "sigmoid":
        return aten.sigmoid_backward(g, torch.sigmoid(x))
    if act == "tanh":
        return aten.tanh_backward(g, torch.tanh(x))
    if act == "hardswish":
        return aten.hardswish_backward(g, x)
    if act == "hardsigmoid":
        return aten.hardsigmoid_backward(g, x)
    return aten.threshold_backward(g, x, 0)


def _act_forward(ctx, act: str, x: torch.Tensor, f: ActivationFormats, approximate: str):
    ctx.save_for_backward(x)
    ctx.formats, ctx.approximate = f, approximate
    if f.table is not None:
        if f.table.name is not None and f.table.name != act:
            raise ValueError(
                f"q{act} was given a table of {f.table.name}: an activation evaluates a table of "
                "its own function, or of one of yours (a table with no name)"
            )
        return _rounded(f.table, x, f.out, f.carrier, x.dtype)
    if f.inner is None and f.out is None:
        return _act_plain(act, x, approximate)
    dtype = x.dtype
    xc = x.to(_carrier_dtype(f.carrier, dtype))
    if f.inner is not None and act in _GATED:
        gate = _round_result(f.inner, _gate(act, xc, approximate), xc.dtype)
        t = _combine(act, xc, gate)
    else:
        t = _act_plain(act, xc, approximate)
    return _stored(t, dtype) if f.out is None else _round_result(f.out, t, dtype)


def _act_backward(ctx, act: str, g: torch.Tensor):
    (x,) = ctx.saved_tensors
    f: ActivationFormats = ctx.formats
    if f.grad is None:
        return _act_grad(act, g, x, ctx.approximate), None, None
    c = _carrier_dtype(f.carrier, g.dtype)
    gx = _act_grad(act, g.to(c), x.to(c), ctx.approximate)
    return _round_result(f.grad, gx, g.dtype), None, None


class _QGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "gelu", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "gelu", grads[0])


class _QSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "silu", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "silu", grads[0])


class _QSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "sigmoid", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "sigmoid", grads[0])


class _QTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "tanh", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "tanh", grads[0])


class _QHardswish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "hardswish", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "hardswish", grads[0])


class _QHardsigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "hardsigmoid", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "hardsigmoid", grads[0])


class _QReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, formats, approximate):
        with torch._C.DisableTorchFunction():
            return _act_forward(ctx, "relu", x, formats, approximate)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _act_backward(ctx, "relu", grads[0])


_ACTIVATIONS = {
    "gelu": _QGELU,
    "silu": _QSiLU,
    "sigmoid": _QSigmoid,
    "tanh": _QTanh,
    "hardswish": _QHardswish,
    "hardsigmoid": _QHardsigmoid,
    "relu": _QReLU,
}


def _activation(
    act: str, x: torch.Tensor, formats: ActivationFormats | None, approximate: str = "none"
) -> torch.Tensor:
    if formats is not None and not isinstance(formats, ActivationFormats):
        raise TypeError(
            f"q{act}'s formats is an ActivationFormats or None, got {type(formats).__name__}"
        )
    if formats is None or not formats._rounds:
        with torch._C.DisableTorchFunction():  # an op, not the calls it makes
            return _act_plain(act, x, approximate)
    return _ACTIVATIONS[act].apply(x, formats, approximate)


_ACT_DOC = """
    Args:
        x (Tensor): a float tensor.
        formats (ActivationFormats, optional): the roundings: ``inner`` (the
            gate of GELU, SiLU and hardswish), ``out`` and ``grad``.
            Default: ``None``, torch's function
"""


def qgelu(
    x: torch.Tensor, formats: ActivationFormats | None = None, approximate: str = "none"
) -> torch.Tensor:
    """``F.gelu(x, approximate=approximate)``, rounded as ``formats`` says.

    Computed in the carrier: torch's ``F.gelu``, or, when ``formats.inner``
    rounds the gate, ``0.5 * x * (1 + gate)`` over the rounded gate
    (``erf(x / sqrt(2))``, or ``tanh(sqrt(2/pi) * (x + 0.044715 * x**3))``
    under ``approximate="tanh"``). ``formats.out`` then rounds the result,
    and the gradient is torch's (``gelu_backward``) in the carrier, rounded
    by ``formats.grad``.
    """
    if approximate not in ("none", "tanh"):
        raise ValueError(f"approximate is 'none' or 'tanh', got {approximate!r}")
    return _activation("gelu", x, formats, approximate)


def qsilu(x: torch.Tensor, formats: ActivationFormats | None = None) -> torch.Tensor:
    """``F.silu(x)``, rounded as ``formats`` says: torch's ``F.silu`` in the
    carrier, or ``x * sigmoid(x)`` over the rounded gate when
    ``formats.inner`` is set; the gradient is ``silu_backward``'s."""
    return _activation("silu", x, formats)


def qsigmoid(x: torch.Tensor, formats: ActivationFormats | None = None) -> torch.Tensor:
    """``torch.sigmoid(x)``, computed in the carrier and rounded by
    ``formats.out``; the gradient is ``sigmoid_backward``'s."""
    return _activation("sigmoid", x, formats)


def qtanh(x: torch.Tensor, formats: ActivationFormats | None = None) -> torch.Tensor:
    """``torch.tanh(x)``, computed in the carrier and rounded by
    ``formats.out``; the gradient is ``tanh_backward``'s."""
    return _activation("tanh", x, formats)


def qhardswish(x: torch.Tensor, formats: ActivationFormats | None = None) -> torch.Tensor:
    """``F.hardswish(x)``, rounded as ``formats`` says: torch's
    ``F.hardswish`` in the carrier, or ``x * hardsigmoid(x)`` over the
    rounded gate when ``formats.inner`` is set; the gradient is
    ``hardswish_backward``'s."""
    return _activation("hardswish", x, formats)


def qhardsigmoid(x: torch.Tensor, formats: ActivationFormats | None = None) -> torch.Tensor:
    """``F.hardsigmoid(x)``, computed in the carrier and rounded by
    ``formats.out``; the gradient is ``hardsigmoid_backward``'s."""
    return _activation("hardsigmoid", x, formats)


def qrelu(x: torch.Tensor, formats: ActivationFormats | None = None) -> torch.Tensor:
    """``torch.relu(x)``, rounded by ``formats.out``. ReLU is exact, so that
    is the format's rounding of each positive element; the gradient is
    ``threshold_backward``'s, rounded by ``formats.grad``."""
    return _activation("relu", x, formats)


for _f in (qgelu, qsilu, qsigmoid, qtanh, qhardswish, qhardsigmoid, qrelu):
    _with_section(_f, _ACT_DOC)
