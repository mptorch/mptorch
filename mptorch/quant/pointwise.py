"""Pointwise ops and activations that round their results to a simulated format.

Each op computes in a carrier, binary32 or binary64, and then rounds once
(``docs/source/kernels/pointwise.rst`` is the user's account):

* ``qadd``, ``qsub``, ``qmul``, ``qdiv``: two tensors of one float dtype, or a
  tensor and a Python number, broadcast as torch broadcasts them;
* ``qsqrt``, ``qrsqrt``, ``qexp`` and ``qpow`` (a number exponent);
* the activations ``qgelu``, ``qsilu``, ``qsigmoid``, ``qtanh``,
  ``qhardswish``, ``qhardsigmoid`` and ``qrelu``;
* ``qembedding``, a gather, whose rows are rounded.

What a step rounds to is its *slot*: a :class:`~mptorch.quant.Quant`, another
callable, or ``None``. A step whose slot is ``None`` is torch's own
computation in the tensor's dtype, so with every slot ``None`` an op is
exactly torch's function. A step with a slot computes in the carrier: float32
for float32, float16 and bfloat16 tensors, which are widened first, and
float64 for float64 tensors or under ``carrier=torch.float64``. The slot then
rounds the carrier's result, and the result is stored in the tensor's dtype.
So the carrier rounds first, to nearest, and the format second: the result is
the format's rounding of the exact value when the carrier's result is exact,
and the documentation's table says when that is. ``carrier_rounding=
RoundMode.RO`` makes the elementary ops' carrier step exact for every format
within its bounds (:func:`_odd_reason`).

With a ``Quant`` slot of a binaryK, superfp or fixed-point format, on the CPU
and CUDA, ``qadd``, ``qsub``, ``qmul``, ``qdiv`` and ``qsqrt`` run the step and
the rounding as one fused op (``_fused_op``, ``csrc/common/pointwise.h``),
which gives the two passes' words; ``mptorch.quant.pwl`` does the same for a
table.

Each op is an autograd Function. Its backward is the derivative of the
unrounded function, torch's own formula, computed in the carrier and rounded
once by the gradient slot; the forward roundings pass the gradient through
unchanged. With no gradient slot it is torch's formula in the gradient's
dtype, which is the gradient torch's autograd gives.
"""

import inspect
import math
import operator
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from mptorch.number import BinaryK, FixedPoint, Number, RoundMode, SuperFP
from mptorch.number.base import _report_per_call

from .mac import Quant, _as_quant
from .modules._opaque import _opaque_backward
from .ops._checks import _call_carrier, _checked_carrier
from .ops._odd import _odd_findings
from .ops._spec import _round_result, _stored

__all__ = [
    "PointwiseFormats",
    "ActivationFormats",
    "qadd",
    "qsub",
    "qmul",
    "qdiv",
    "qsqrt",
    "qrsqrt",
    "qexp",
    "qpow",
    "qgelu",
    "qsilu",
    "qsigmoid",
    "qtanh",
    "qhardswish",
    "qhardsigmoid",
    "qrelu",
    "qembedding",
]

_RNE, _RO = RoundMode.RNE, RoundMode.RO

# The default of the `uniform` constructors' gradient arguments: the forward's.
_SAME: Any = object()

Slot = Callable[[torch.Tensor], torch.Tensor] | None


# --- the formats ---------------------------------------------------------------


def _checked_rounding(carrier_rounding: RoundMode) -> RoundMode:
    if carrier_rounding is not _RNE and carrier_rounding is not _RO:
        raise ValueError(
            "carrier_rounding is RoundMode.RNE (the carrier's nearest step) or RoundMode.RO "
            f"(rounded to odd), got {carrier_rounding!r}"
        )
    return carrier_rounding


def _checked_slot(q: Any, carrier: torch.dtype | None, name: str) -> None:
    """Check one slot of a call: ``None``, a ``Quant`` that rounds in the
    op's carrier (its own ``carrier`` unset, or the op's), or a callable."""
    if q is None:
        return
    if isinstance(q, Quant):
        if q.carrier is not None and q.carrier is not carrier:
            raise ValueError(
                f"{name} is a Quant with carrier={q.carrier}, and the op's carrier is "
                f"{carrier}: a pointwise op computes and rounds in the one carrier its "
                "carrier argument names -- leave the Quant's carrier unset, and pass the "
                "carrier to the op"
            )
        return
    if isinstance(q, Number):
        raise TypeError(
            f"{name} is a format, {q!r}: pass Quant(fmt), which rounds to it, or put the "
            "format in a PointwiseFormats, which turns it into one when it is built"
        )
    if not callable(q):
        raise TypeError(f"{name} takes a Quant, a callable or None, got {type(q).__name__}")


def _label(q: Any) -> str:
    from .recipe import _quant_label  # imported here: recipe imports this module

    return _quant_label(q)


@dataclass(frozen=True, kw_only=True)
class PointwiseFormats:
    """The formats of a pointwise op: its result's rounding and its gradients'.

    It is the value of a recipe's ``elementwise`` kind, which rounds a
    model's additions, subtractions, products, quotients, square roots,
    reciprocal square roots, exponentials and powers, and of its
    ``embedding`` kind, which rounds an embedding's rows and its weight's
    gradient. The elementary ops take the same four values as keywords. A
    slot left ``None`` does not round; with both ``None`` an op is torch's
    function.

    Args:
        quant: rounds the result. A format means ``Quant(fmt)``.
            Default: ``None``
        grad_quant: rounds each gradient. A format means ``Quant(fmt)``.
            Default: ``None``
        carrier (torch.dtype, optional): where the op computes and rounds:
            ``None`` for the tensors' own carrier (binary32, or binary64 for
            float64 tensors), ``torch.float64`` for binary64 whatever their
            dtype. A ``Quant`` in a slot leaves its own ``carrier`` unset.
            Default: ``None``
        carrier_rounding (RoundMode): ``RoundMode.RNE``, the carrier's
            nearest result, or ``RoundMode.RO``, the exact result rounded to
            odd, which makes the rounding of an addition, subtraction,
            product, quotient or square root the format's rounding of the
            exact value (``docs/source/kernels/pointwise.rst`` states the
            bounds). Default: ``RoundMode.RNE``

    Raises:
        TypeError: for a module in a slot (a recipe shares its formats, and
            with them a module's state), or a slot that is not callable.
        ValueError: for a ``Quant`` whose own carrier is set and differs, an
            unknown carrier, or a ``carrier_rounding`` other than RNE or RO.
    """

    quant: Any = None
    grad_quant: Any = None
    carrier: torch.dtype | None = None
    carrier_rounding: RoundMode = _RNE

    def __post_init__(self) -> None:
        carrier = _checked_carrier(self.carrier)
        _checked_rounding(self.carrier_rounding)
        for name in ("quant", "grad_quant"):
            slot = _as_quant(getattr(self, name), name)
            _checked_slot(slot, carrier, name)
            object.__setattr__(self, name, slot)

    @classmethod
    def uniform(
        cls,
        quant: Any,
        *,
        grad_quant: Any = _SAME,
        carrier: torch.dtype | None = None,
        carrier_rounding: RoundMode = _RNE,
    ) -> "PointwiseFormats":
        """One rounding for the result and the gradients; ``grad_quant``
        replaces the gradients' (``None`` leaves them unrounded)."""
        return cls(
            quant=quant,
            grad_quant=quant if grad_quant is _SAME else grad_quant,
            carrier=carrier,
            carrier_rounding=carrier_rounding,
        )

    @property
    def _keywords(self) -> dict[str, Any]:
        return {"carrier": self.carrier, "carrier_rounding": self.carrier_rounding}

    def label(self) -> str:
        """A one-line summary, for reports."""
        parts = []
        if self.quant is not None and self.quant == self.grad_quant:
            parts.append(f"quant {_label(self.quant)}")
        else:
            parts += [
                f"{n} {_label(q)}"
                for n, q in (("out", self.quant), ("grad", self.grad_quant))
                if q is not None
            ]
        if self.carrier is not None:
            parts.append(f"carrier {str(self.carrier).removeprefix('torch.')}")
        if self.carrier_rounding is _RO:
            parts.append("RO")
        return ", ".join(parts) or "torch"


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
            from .pwl import PWLTable  # imported here: pwl imports this module

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
    ) -> "ActivationFormats":
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


# --- the carrier ---------------------------------------------------------------


def _carrier_dtype(carrier: torch.dtype | None, dtype: torch.dtype) -> torch.dtype:
    """The dtype a call on ``dtype`` tensors computes in: float64 for binary64,
    float32 for binary32. ``carrier=torch.float32`` on float64 raises."""
    wide, _ = _call_carrier(carrier, dtype)
    return torch.float64 if wide else torch.float32


# The precision of each tensor dtype a pointwise op takes, for the bounds of
# the round-to-odd step.
_PRECISION = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}


def _boundary_bits(P: int, prng_bits: int, mode: RoundMode) -> tuple[int, str]:
    """How many significant bits a rounding boundary of a P-bit format has
    under ``mode``: its values for the directed modes, the midpoints between
    them for the nearest ones, the ``prng_bits`` random bits below them for
    StochasticA, and one bit more for StochasticB and C."""
    if mode in (RoundMode.RNE, RoundMode.RNA):
        return P + 1, "P + 1"
    if mode is RoundMode.SR:
        return P + prng_bits, "P + prng_bits"
    if mode.is_stochastic:
        return P + prng_bits + 1, "P + prng_bits + 1"
    return P, "P"


def _precision(fmt: Any) -> int | None:
    """The precision of a format whose rounding boundaries the round-to-odd
    step's bounds can be checked against, or ``None``."""
    if isinstance(fmt, BinaryK):
        return fmt.P
    if isinstance(fmt, SuperFP):
        return fmt.man_bits + 1
    if isinstance(fmt, FixedPoint):
        return fmt.wl - 1 if fmt.is_signed else fmt.wl
    return None


def _odd_reason(
    op: str, quant: Any, wide: bool, dtype: torch.dtype, fused: bool = False
) -> str | None:
    """Why ``carrier_rounding=RoundMode.RO`` cannot give ``op`` the format's
    rounding of the exact value, or ``None`` when it can.

    A call that runs a fused op (``_fused_op``) rounds the exact result to
    odd in the carrier, in either carrier and for float64 operands too
    (``csrc/common/carrier_odd_div_sqrt.h``), so only the format's own bounds
    apply to it (``_odd_findings``). Otherwise the step is computed in
    float64, rounded to odd at 53 bits:
    the product and the sum (with the error 2Sum finds) exactly, the quotient
    and the root to nearest. A float64 quotient of operands of p bits is on
    the exact quotient's side of every number of at most 53 - p bits, and a
    float64 root on the exact root's side of every number of at most 25 bits
    (one within an ulp, as torch's CPU root is, too: the exact root is more
    than 2^-51 from such a number, relatively, and an ulp is at most 2^-52),
    so each holds every bit a cast reads when the format's rounding
    boundaries are no wider (``_boundary_bits``). Under binary32 that value
    is rounded to odd again at 24 bits, which both bounds allow, and the cast
    then needs ``_odd_findings``' bound for binary32.
    """
    if dtype is torch.float64 and not fused:
        return (
            "carrier_rounding=RoundMode.RO takes float32, float16 and bfloat16 operands, whose "
            "exact results float64 holds or rounds to odd; float64 operands are not supported"
        )
    if op not in ("add", "sub", "mul", "div", "sqrt"):
        return (
            f"carrier_rounding=RoundMode.RO: q{op} has no exact carrier step (its carrier result "
            "is a library function's); carrier=torch.float64 is the closest it comes"
        )
    P = _precision(getattr(quant, "fmt", None)) if isinstance(quant, Quant) else None
    if P is None:
        return (
            "carrier_rounding=RoundMode.RO needs the result's slot to be a Quant of a BinaryK, "
            f"SuperFP or FixedPoint format, whose rounding boundaries it checks; got {quant!r}"
        )
    assert isinstance(quant, Quant)
    prng_bits = getattr(quant.fmt, "prng_bits", 0)
    found32, found64 = _odd_findings([("result", P, prng_bits)], quant.rounding)
    found = found64 if wide else found32
    if found:
        return found[0][0]
    if wide and op in ("div", "sqrt") and not fused:
        k, spelled = _boundary_bits(P, prng_bits, quant.rounding)
        limit = 53 - _PRECISION[dtype] if op == "div" else 25
        if k > limit:
            what = f"quotient of {dtype} operands" if op == "div" else "root"
            return (
                f"carrier_rounding=RoundMode.RO: the result format's {spelled} is {k} under "
                f"RoundMode.{quant.rounding.name}, and a binary64 {what} is rounded exactly only "
                f"when it is at most {limit}"
            )
    return None


def _checked_call(
    op: str,
    quant: Any,
    grad_quant: Any,
    carrier: torch.dtype | None,
    carrier_rounding: RoundMode,
    dtype: torch.dtype,
    fused: bool = False,
) -> bool:
    """Check one call's slots and carrier; return whether it rounds to odd.
    ``fused`` says that the call runs a fused op (``_fused_op``)."""
    carrier = _checked_carrier(carrier)
    _checked_slot(quant, carrier, "quant")
    _checked_slot(grad_quant, carrier, "grad_quant")
    rounding = _checked_rounding(carrier_rounding)
    if quant is None and grad_quant is None:
        return False
    wide = _carrier_dtype(carrier, dtype) is torch.float64  # raises for a narrower carrier
    if rounding is _RNE or quant is None:
        return False
    reason = _odd_reason(op, quant, wide, dtype, fused)
    if reason is not None:
        raise ValueError(reason)
    return True


# --- the round-to-odd step ---------------------------------------------------------


def _odd_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a + b`` in float64, rounded to odd: the nearest sum, moved one step
    toward the exact sum where 2Sum finds an error and the sum's word is even.
    There is no product in it, so a compiler fusing multiply-adds leaves it
    as it is."""
    s = a + b
    bb = s - a
    err = (a - (s - bb)) + (b - bb)
    even = (s.view(torch.int64) & 1) == 0
    fix = (err != 0) & even & torch.isfinite(s)
    toward = torch.where(err > 0, math.inf, -math.inf).to(s.dtype)
    return torch.where(fix, torch.nextafter(s, toward), s)


def _odd_binary32(v: torch.Tensor) -> torch.Tensor:
    """A float64 value rounded to odd at 53 bits, rounded to odd again at 24
    bits, as a float32 tensor: the nearest float32, moved one step toward
    ``v`` where it is inexact and its word is even. Two roundings to odd, the
    second at least two bits narrower, are one. A finite value past the
    largest float32 becomes the largest (the nearest is infinity, whose word
    is even, and the step toward ``v`` lands there)."""
    f = v.float()
    back = f.double()
    even = (f.view(torch.int32) & 1) == 0
    fix = (back != v) & even
    toward = torch.where(v > back, math.inf, -math.inf).to(torch.float32)
    return torch.where(fix, torch.nextafter(f, toward), f)


def _odd_operand(v: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """An operand of the round-to-odd step as float64. A number is first
    rounded to the operands' dtype, so that both operands have its precision,
    which the step's bounds rest on."""
    if isinstance(v, torch.Tensor):
        return v.double()
    return torch.full((), v, dtype=dtype, device=device).double()


def _odd_step(op: str, a: Any, b: Any, ref: torch.Tensor, wide: bool) -> torch.Tensor:
    """The exact result of ``op`` rounded to odd in the carrier (``_odd_reason``)."""
    a64 = _odd_operand(a, ref.dtype, ref.device)
    if op == "sqrt":
        v = torch.sqrt(a64)
    else:
        b64 = _odd_operand(b, ref.dtype, ref.device)
        if op == "add":
            v = _odd_sum(a64, b64)
        elif op == "sub":
            v = _odd_sum(a64, -b64)
        elif op == "mul":
            v = a64 * b64
        else:
            v = a64 / b64
    return v if wide else _odd_binary32(v)


# --- the elementary ops ----------------------------------------------------------

_PLAIN: dict[str, Callable[[Any, Any], torch.Tensor]] = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


def _in_carrier(v: Any, c: torch.dtype, ref: torch.Tensor, op: str) -> Any:
    """An operand in the carrier dtype ``c``. A number stays one, which torch
    converts to the carrier, except as a divisor or dividend: torch's CUDA
    division by a CPU number multiplies by its reciprocal, so it becomes a
    0-dim tensor of the carrier on the operand's device."""
    if isinstance(v, torch.Tensor):
        return v.to(c)
    if op == "div":
        return torch.full((), v, dtype=c, device=ref.device)
    return v


# --- the fused ops -----------------------------------------------------------------

# Each family's fused pointwise op (``csrc/common/pointwise.h``), by the name
# of its flat quantizer: the carrier step and the slot's rounding in one pass,
# word for word the two passes below, stochastic rounding included. The
# devices that have their kernels.
_FUSED: dict[str, Callable[..., torch.Tensor]] = {
    "binaryK_quantize": torch.ops.mptorch.binaryK_pointwise.default,
    "superfp_quantize": torch.ops.mptorch.superfp_pointwise.default,
    "fixedpoint_quantize": torch.ops.mptorch.fixedpoint_pointwise.default,
}
_FUSED_DEVICES = frozenset(("cpu", "cuda"))
# The op codes: the tensor first, and, for a number first, the op with its
# operands in their order (n + x and n * x are x + n and x * n).
_CODE = {"add": 0, "sub": 1, "mul": 2, "div": 3, "sqrt": 4}
_CODE_NUMBER_FIRST = {"add": 0, "sub": 5, "mul": 2, "div": 6}


def _fused_op(quant: Any, a: Any, b: Any = None) -> Callable[..., torch.Tensor] | None:
    """The fused op of a call whose result slot is ``quant``, or ``None``
    where the two passes run: a slot that is not a ``Quant`` of a binaryK,
    superfp or fixed-point format, a device without the op's kernels,
    tensors on two devices (torch takes a CPU 0-dim tensor beside a CUDA
    one), or an ``int`` operand that a float64 does not hold (torch converts
    it to the carrier once). ``b`` is ``None`` for a unary op."""
    if not isinstance(quant, Quant) or quant._spec is None:
        return None
    fused = _FUSED.get(quant._spec.name)
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    ref = a if ta else b
    if fused is None or ref.device.type not in _FUSED_DEVICES:
        return None
    if ta and tb and a.device != b.device:
        return None
    for v in (a, b):
        if isinstance(v, int) and abs(v) > 2**53:
            return None
    return fused


def _report_fused(spec: Any, wide: bool, dtype: torch.dtype) -> None:
    """What ``_round_result`` reports for a slot's spec: its finding in the
    carrier (``wide`` for binary64), and on storing a computed result in
    ``dtype``."""
    found = spec.findings[wide]
    if found is not None:
        _report_per_call(*found)
    stored = spec.result_stored[wide].get(dtype)
    if stored is not None:
        _report_per_call(*stored)


def _fused_forward(fused, op: str, a, b, quant: Quant, carrier, odd: bool) -> torch.Tensor:
    """The carrier step and the slot's rounding in one fused op, with the
    checks ``_round_result`` makes for the carrier and the stored dtype.
    Under ``carrier=torch.float64`` the operands are widened first and the
    result narrowed with one rounding, as in the two passes. Under RO a
    number becomes a 0-dim tensor of the operands' dtype, as in
    ``_odd_operand``, so that both operands have that dtype's precision."""
    spec = quant._spec
    assert spec is not None  # _fused_op checked it
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    ref = a if ta else b
    dtype = ref.dtype
    wide = _carrier_dtype(carrier, dtype) is torch.float64
    _report_fused(spec, wide, dtype)
    if odd and op != "sqrt" and not (ta and tb):
        if ta:
            b, tb = torch.full((), b, dtype=dtype, device=ref.device), True
        else:
            a, ta = torch.full((), a, dtype=dtype, device=ref.device), True
    if wide and dtype is not torch.float64:
        a = a.double() if ta else a
        b = b.double() if tb else b
    rounding = (_RO if odd else _RNE).value
    if op == "sqrt":
        out = fused(a, None, None, _CODE[op], rounding, *spec.args)
    elif ta and tb:
        out = fused(a, b, None, _CODE[op], rounding, *spec.args)
    elif ta:
        out = fused(a, None, float(b), _CODE[op], rounding, *spec.args)
    else:
        out = fused(b, None, float(a), _CODE_NUMBER_FIRST[op], rounding, *spec.args)
    return _stored(out, dtype)


def _binary_forward(ctx, op: str, a, b, quant, grad_quant, carrier, odd: bool) -> torch.Tensor:
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    ref = a if ta else b
    dtype = ref.dtype
    ctx.op_inputs = (
        None if ta else a,
        None if tb else b,
        a.shape if ta else None,
        b.shape if tb else None,
    )
    ctx.grad_quant, ctx.carrier = grad_quant, carrier
    ctx.save_for_backward(
        a if ta and op in ("mul", "div") else None, b if tb and op in ("mul", "div") else None
    )
    if quant is None:
        return _PLAIN[op](a, b)
    fused = _fused_op(quant, a, b)
    if fused is not None:
        return _fused_forward(fused, op, a, b, quant, carrier, odd)
    c = _carrier_dtype(carrier, dtype)
    if odd:
        t = _odd_step(op, a, b, ref, c is torch.float64)
    else:
        t = _PLAIN[op](_in_carrier(a, c, ref, op), _in_carrier(b, c, ref, op))
    return _round_result(quant, t, dtype)


def _summed(t: torch.Tensor | None, shape: torch.Size | None) -> torch.Tensor | None:
    """A gradient summed over the dimensions its operand was broadcast along."""
    if t is None or shape is None or t.shape == shape:
        return t
    return t.sum_to_size(shape)


def _binary_grads(op: str, g, a, b, need_a: bool, need_b: bool):
    """torch's formulas for the gradients of ``a op b`` (``derivatives.yaml``)."""
    ga = gb = None
    if op == "add":
        ga, gb = g, g
    elif op == "sub":
        ga, gb = g, (-g if need_b else None)
    elif op == "mul":
        ga = g * b if need_a else None
        gb = g * a if need_b else None
    else:
        ga = g / b if need_a else None
        gb = -g * ((a / b) / b) if need_b else None
    return (ga if need_a else None), (gb if need_b else None)


def _binary_backward(ctx, op: str, g: torch.Tensor):
    na, nb, sa, sb = ctx.op_inputs
    sa_t, sb_t = ctx.saved_tensors
    a = na if sa is None else sa_t
    b = nb if sb is None else sb_t
    need_a = ctx.needs_input_grad[0] and sa is not None
    need_b = ctx.needs_input_grad[1] and sb is not None
    q = ctx.grad_quant
    if q is None:
        ga, gb = _binary_grads(op, g, a, b, need_a, need_b)
        return _summed(ga, sa), _summed(gb, sb), None, None, None, None
    c = _carrier_dtype(ctx.carrier, g.dtype)
    gc = g.to(c)
    ac = None if a is None else _in_carrier(a, c, g, op)
    bc = None if b is None else _in_carrier(b, c, g, op)
    ga, gb = _binary_grads(op, gc, ac, bc, need_a, need_b)
    ga, gb = _summed(ga, sa), _summed(gb, sb)
    return (
        None if ga is None else _round_result(q, _fresh(ga, gc), g.dtype),
        None if gb is None else _round_result(q, _fresh(gb, gc), g.dtype),
        None,
        None,
        None,
        None,
    )


def _fresh(t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """``t``, or a copy of it when it is the gradient ``g`` itself (an
    addition's, or a subtraction's first): the incoming gradient may be read
    elsewhere, and the other operand's gradient may be the same tensor, so a
    rounding in place must not write it."""
    return t.clone() if t is g else t


class _QAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():  # hidden from torch function modes
            return _binary_forward(ctx, "add", a, b, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():  # as in forward, except under compile (_opaque.py)
            return _binary_backward(ctx, "add", grads[0])


class _QSub(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _binary_forward(ctx, "sub", a, b, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _binary_backward(ctx, "sub", grads[0])


class _QMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _binary_forward(ctx, "mul", a, b, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _binary_backward(ctx, "mul", grads[0])


class _QDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _binary_forward(ctx, "div", a, b, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _binary_backward(ctx, "div", grads[0])


_BINARY = {"add": _QAdd, "sub": _QSub, "mul": _QMul, "div": _QDiv}


def _checked_operands(op: str, a: Any, b: Any) -> torch.Tensor:
    """The tensor operand of ``a op b``: two tensors of one float dtype, or a
    tensor and a Python number (a ``bool`` is not one)."""
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    for v, is_t in ((a, ta), (b, tb)):
        if not is_t and (isinstance(v, bool) or not isinstance(v, int | float)):
            raise TypeError(
                f"q{op} takes tensors and Python numbers (int, float), got {type(v).__name__}"
            )
    if not (ta or tb):
        raise TypeError(f"q{op} needs a tensor operand, got two numbers")
    ref = a if ta else b
    if not ref.is_floating_point() or (ta and tb and a.dtype != b.dtype):
        got = f"{a.dtype} and {b.dtype}" if ta and tb else str(ref.dtype)
        raise TypeError(
            f"q{op} takes float tensors of one dtype, got {got}: convert the operands first, "
            "so that the carrier and the stored dtype are those of one dtype"
        )
    return ref


def _binary(op: str, a, b, quant, grad_quant, carrier, carrier_rounding) -> torch.Tensor:
    # the op, and not the calls it makes, is what a torch function mode sees,
    # as for the Functions
    with torch._C.DisableTorchFunction():
        ref = _checked_operands(op, a, b)
        fused = _fused_op(quant, a, b) is not None
        odd = _checked_call(op, quant, grad_quant, carrier, carrier_rounding, ref.dtype, fused)
        if quant is None and grad_quant is None:
            return _PLAIN[op](a, b)
        return _BINARY[op].apply(a, b, quant, grad_quant, carrier, odd)


_SLOTS_DOC = """
    Args:
        quant: rounds the result: a :class:`~mptorch.quant.Quant` (its own
            ``carrier`` left unset), another callable, or ``None``.
            Default: ``None``
        grad_quant: rounds the gradient of each tensor operand, computed in
            the carrier with torch's formula. Default: ``None``
        carrier (torch.dtype, optional): ``None`` computes in binary32
            (float32, float16 and bfloat16 tensors, widened first) or binary64
            (float64 tensors); ``torch.float64`` in binary64 whatever the
            dtype, the result narrowed with one rounding afterwards.
            Default: ``None``
        carrier_rounding (RoundMode): ``RoundMode.RNE``, or ``RoundMode.RO``
            for the exact result rounded to odd, which the format then rounds
            correctly within the bounds of
            ``docs/source/kernels/pointwise.rst`` (on the CPU and CUDA, any
            float operands; elsewhere float32, float16 and bfloat16 ones).
            Default: ``RoundMode.RNE``

    Raises:
        TypeError: for operands of two dtypes or of no float dtype, or a
            format in a slot (pass ``Quant(fmt)``).
        ValueError: for a slot ``Quant`` with another carrier, or a format
            outside the round-to-odd bounds.
"""


def qadd(
    x: torch.Tensor | float,
    y: torch.Tensor | float,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``x + y``, computed in the carrier and rounded once by ``quant``.

    The carrier rounds the sum to nearest first (unless ``carrier_rounding``
    is ``RoundMode.RO``), which is exact when the operands' exponents differ
    by at most the carrier's precision less theirs, less one (for the carry).
    A Python number operand is converted to the carrier, as torch converts
    it. With both slots ``None`` this is ``x + y``.
    """
    return _binary("add", x, y, quant, grad_quant, carrier, carrier_rounding)


def qsub(
    x: torch.Tensor | float,
    y: torch.Tensor | float,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``x - y``, computed in the carrier and rounded once by ``quant``; the
    carrier's step is :func:`qadd`'s. With both slots ``None`` this is
    ``x - y``."""
    return _binary("sub", x, y, quant, grad_quant, carrier, carrier_rounding)


def qmul(
    x: torch.Tensor | float,
    y: torch.Tensor | float,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``x * y``, computed in the carrier and rounded once by ``quant``.

    The carrier's product is exact when the operands' precisions sum to at
    most the carrier's: always in binary64 for float32, float16 and bfloat16
    operands. With both slots ``None`` this is ``x * y``.
    """
    return _binary("mul", x, y, quant, grad_quant, carrier, carrier_rounding)


def qdiv(
    x: torch.Tensor | float,
    y: torch.Tensor | float,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``x / y``, computed in the carrier and rounded once by ``quant``.

    The carrier rounds the quotient to nearest first. In binary64 that is on
    the exact quotient's side of every number of at most 53 - p bits, for
    operands of p bits, so the format's rounding is the exact value's when
    its rounding boundaries are no wider. A number divisor or dividend
    becomes a 0-dim tensor of the carrier, since torch's CUDA division by a
    number multiplies by its reciprocal. With both slots ``None`` this is
    ``x / y``.
    """
    return _binary("div", x, y, quant, grad_quant, carrier, carrier_rounding)


def _with_section(f: Callable, section: str) -> None:
    """Append ``section`` (an ``Args:`` block written at the module's
    indentation) to ``f``'s docstring. Both are normalized first, since
    Python 3.13 strips a docstring's indentation and earlier versions keep
    it."""
    f.__doc__ = inspect.cleandoc(f.__doc__ or "") + "\n\n" + textwrap.dedent(section).strip("\n")


for _f in (qadd, qsub, qmul, qdiv):
    _with_section(_f, _SLOTS_DOC)


_UNARY_PLAIN: dict[str, Callable[..., torch.Tensor]] = {
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "exp": torch.exp,
    "pow": torch.pow,
}


def _carrier_sqrt(x: torch.Tensor) -> torch.Tensor:
    """The carrier's square root, correctly rounded in binary32 on every
    device. torch's eager CPU root is not (it can be an ulp off, in either
    dtype; CUDA's and Inductor's are correctly rounded), so a float32 root on
    the CPU is computed in float64 and narrowed: a float64 root within an ulp
    lies on the exact root's side of every float32 midpoint, so its nearest
    float32 is the correctly rounded root. A float64 root stays torch's,
    within an ulp, which ``_odd_reason``'s bound allows for."""
    if x.dtype is torch.float32 and x.device.type == "cpu":
        return torch.sqrt(x.double()).float()
    return torch.sqrt(x)


def _unary_forward(ctx, op: str, x, exponent, quant, grad_quant, carrier, odd: bool):
    ctx.save_for_backward(x)
    ctx.grad_quant, ctx.carrier, ctx.exponent = grad_quant, carrier, exponent
    args = () if exponent is None else (exponent,)
    if quant is None:
        return _UNARY_PLAIN[op](x, *args)
    fused = _fused_op(quant, x) if op == "sqrt" else None
    if fused is not None:
        return _fused_forward(fused, op, x, None, quant, carrier, odd)
    c = _carrier_dtype(carrier, x.dtype)
    if odd:
        t = _odd_step(op, x, None, x, c is torch.float64)
    elif op == "sqrt":
        t = _carrier_sqrt(x.to(c))
    else:
        t = _UNARY_PLAIN[op](x.to(c), *args)
    return _round_result(quant, t, x.dtype)


def _unary_grad(op: str, g: torch.Tensor, x: torch.Tensor, exponent: Any) -> torch.Tensor:
    """torch's formula for the gradient of ``op`` (``derivatives.yaml``), with
    the result recomputed from ``x``."""
    if op == "sqrt":
        return g / (2 * torch.sqrt(x))
    if op == "rsqrt":
        return -0.5 * g * torch.rsqrt(x).pow(3)
    if op == "exp":
        return g * torch.exp(x)
    if exponent == 0:
        return torch.zeros_like(x)
    return g * (exponent * x.pow(exponent - 1))


def _unary_backward(ctx, op: str, g: torch.Tensor):
    (x,) = ctx.saved_tensors
    q = ctx.grad_quant
    if q is None:
        return _unary_grad(op, g, x, ctx.exponent), None, None, None, None, None
    c = _carrier_dtype(ctx.carrier, g.dtype)
    gx = _unary_grad(op, g.to(c), x.to(c), ctx.exponent)
    return _round_result(q, gx, g.dtype), None, None, None, None, None


class _QSqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _unary_forward(ctx, "sqrt", x, exponent, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _unary_backward(ctx, "sqrt", grads[0])


class _QRsqrt(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _unary_forward(ctx, "rsqrt", x, exponent, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _unary_backward(ctx, "rsqrt", grads[0])


class _QExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _unary_forward(ctx, "exp", x, exponent, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _unary_backward(ctx, "exp", grads[0])


class _QPow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, exponent, quant, grad_quant, carrier, odd):
        with torch._C.DisableTorchFunction():
            return _unary_forward(ctx, "pow", x, exponent, quant, grad_quant, carrier, odd)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _unary_backward(ctx, "pow", grads[0])


_UNARY = {"sqrt": _QSqrt, "rsqrt": _QRsqrt, "exp": _QExp, "pow": _QPow}


def _unary(op: str, x, exponent, quant, grad_quant, carrier, carrier_rounding) -> torch.Tensor:
    with torch._C.DisableTorchFunction():  # an op, not the calls it makes, as in _binary
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            got = getattr(x, "dtype", type(x).__name__)
            raise TypeError(f"q{op} takes a float tensor, got {got}")
        if exponent is not None and (
            isinstance(exponent, bool) or not isinstance(exponent, int | float)
        ):
            raise TypeError(f"qpow's exponent is a Python number, got {type(exponent).__name__}")
        fused = op == "sqrt" and _fused_op(quant, x) is not None
        odd = _checked_call(op, quant, grad_quant, carrier, carrier_rounding, x.dtype, fused)
        if quant is None and grad_quant is None:
            return _UNARY_PLAIN[op](x, *(() if exponent is None else (exponent,)))
        return _UNARY[op].apply(x, exponent, quant, grad_quant, carrier, odd)


def qsqrt(
    x: torch.Tensor,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``torch.sqrt(x)``, computed in the carrier and rounded once by ``quant``.

    The carrier rounds the root to nearest first, correctly rounded: with a
    ``Quant`` slot on the CPU and CUDA the fused op computes it in either
    carrier, and otherwise a binary32 root is computed through float64 on
    the CPU, whose eager root is not correctly rounded. In binary64 the root
    is on the exact root's side of every number of at most 25 bits, so the
    format's rounding is the exact value's when its rounding boundaries are
    no wider. With both slots ``None`` this is ``torch.sqrt(x)``.
    """
    return _unary("sqrt", x, None, quant, grad_quant, carrier, carrier_rounding)


def qrsqrt(
    x: torch.Tensor,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``torch.rsqrt(x)``, computed in the carrier and rounded once by ``quant``.

    The carrier's result is torch's, which is not correctly rounded (and
    differs between the CPU, CUDA and MPS), so the format rounds a value
    that may already be off by an ulp of the carrier. ``RoundMode.RO`` is
    not offered. With both slots ``None`` this is ``torch.rsqrt(x)``.
    """
    return _unary("rsqrt", x, None, quant, grad_quant, carrier, carrier_rounding)


def qexp(
    x: torch.Tensor,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``torch.exp(x)``, computed in the carrier and rounded once by ``quant``.

    The carrier's result is torch's, as for :func:`qrsqrt`. With both slots
    ``None`` this is ``torch.exp(x)``.
    """
    return _unary("exp", x, None, quant, grad_quant, carrier, carrier_rounding)


def qpow(
    x: torch.Tensor,
    exponent: float,
    quant: Slot = None,
    grad_quant: Slot = None,
    *,
    carrier: torch.dtype | None = None,
    carrier_rounding: RoundMode = _RNE,
) -> torch.Tensor:
    """``torch.pow(x, exponent)`` for a number exponent, computed in the
    carrier and rounded once by ``quant``.

    The carrier's result is torch's, as for :func:`qrsqrt` (torch computes
    an exponent of 2 as a product, and 0.5 as a square root). With both slots
    ``None`` this is ``torch.pow(x, exponent)``.
    """
    return _unary("pow", x, exponent, quant, grad_quant, carrier, carrier_rounding)


for _f in (qsqrt, qrsqrt, qexp, qpow):
    _with_section(
        _f,
        _SLOTS_DOC.replace("the gradient of each tensor operand", "the input's gradient").replace(
            "operands of two dtypes or of no float dtype", "an operand that is not a float tensor"
        ),
    )


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
        from . import pwl  # imported here: pwl imports this module

        if f.table.name is not None and f.table.name != act:
            raise ValueError(
                f"q{act} was given a table of {f.table.name}: an activation evaluates a table of "
                "its own function, or of one of yours (a table with no name)"
            )
        return pwl._rounded(f.table, x, f.out, f.carrier, x.dtype)
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


# --- the embedding ---------------------------------------------------------------


def _padding(padding_idx: int | None, num_weights: int) -> int:
    """``padding_idx`` as ``F.embedding`` hands it to the op: from the end
    when negative, and -1 for none."""
    if padding_idx is None:
        return -1
    if padding_idx < 0:
        return num_weights + padding_idx
    return padding_idx


class _QEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, formats, padding_idx, max_norm, norm_type, by_freq, sparse):
        with torch._C.DisableTorchFunction():
            rows = F.embedding(input, weight, padding_idx, max_norm, norm_type, by_freq, sparse)
            ctx.save_for_backward(input)
            n = weight.shape[0]
            ctx.embedding = (formats, n, _padding(padding_idx, n), by_freq, sparse)
            q = formats.quant
            if q is None:
                return rows
            return _round_result(
                q, rows.to(_carrier_dtype(formats.carrier, rows.dtype)), rows.dtype
            )

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            g = grads[0]
            (input,) = ctx.saved_tensors
            formats, n, padding_idx, by_freq, sparse = ctx.embedding
            q = formats.grad_quant
            if q is None:
                gw = torch.ops.aten.embedding_backward(g, input, n, padding_idx, by_freq, sparse)
            else:
                gc = g.to(_carrier_dtype(formats.carrier, g.dtype))
                gw = torch.ops.aten.embedding_backward(gc, input, n, padding_idx, by_freq, sparse)
                gw = _round_result(q, gw, g.dtype)
            return None, gw, None, None, None, None, None, None


def qembedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    formats: PointwiseFormats | None = None,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    """``F.embedding``, its rows rounded by ``formats.quant`` and its weight's
    gradient by ``formats.grad_quant``.

    The gather is exact, so the rows are the format's rounding of the
    weight's. The weight's gradient is torch's (``embedding_backward``),
    summed over repeated indices in the carrier, then rounded once. The other
    arguments are ``F.embedding``'s.

    Raises:
        ValueError: for ``carrier_rounding=RoundMode.RO`` (a gather has no
            carrier step to round), or a sparse gradient with
            ``grad_quant`` set.
    """
    if formats is not None and not isinstance(formats, PointwiseFormats):
        raise TypeError(f"qembedding's formats is a PointwiseFormats or None, got {formats!r}")
    if formats is None or (formats.quant is None and formats.grad_quant is None):
        with torch._C.DisableTorchFunction():
            return F.embedding(
                input, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
            )
    if formats.carrier_rounding is _RO:
        raise ValueError(
            "carrier_rounding=RoundMode.RO: an embedding gathers its rows, which leaves no "
            "carrier step to round to odd -- use RoundMode.RNE"
        )
    if sparse and formats.grad_quant is not None:
        raise ValueError(
            "qembedding: a sparse gradient cannot be rounded by grad_quant -- pass sparse=False, "
            "or leave grad_quant unset"
        )
    return _QEmbedding.apply(
        input, weight, formats, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse
    )
