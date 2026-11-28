"""Softmax and the norms, each step rounded as their formats say.

``qsoftmax``, ``qlayernorm``, ``qrmsnorm``, ``qgroupnorm`` and ``qbatchnorm``
compute in a carrier, as ``functional/pointwise.py``'s ops do: binary32 for
float32, float16 and bfloat16 tensors, which are widened first, and binary64
for float64 ones or under ``carrier=torch.float64``. Each step whose slot is
set is rounded once there; the others are the carrier's own. Their sums are
:func:`~mptorch.quant.qsum`'s, on the GEMM kernels, with the ``acc`` slot's
mac. When every slot before the result's is ``None``, the op is torch's own
function in the carrier, and its result is rounded once. With every slot
``None`` it is torch's function.

The backward is the derivative of the unrounded function at the input, the
forward's roundings passing the gradient through. With a backward slot set it
is the closed form, in the carrier, each of its steps rounded as the
``bwd_*`` slots say. With none, it is torch's own gradient of its function,
which autograd computes from a forward of torch's function, so it equals
autograd's word for word.

``docs/source/kernels/reductions.rst`` is the user's account.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .._labels import _mac_label, _quant_label
from .._opaque import _opaque_backward
from ..mac import FusedMac, SplitMac, spec_for_mac
from ..ops._checks import _checked_carrier
from ..ops._spec import _GemmSpec, _round_result, _stored
from .pointwise import _carrier_dtype
from .pwl import FISR, PWLTable, _fisr, _rounded
from .reduce import _checked_mac, _checked_quant_slot, _reduced, _sum

__all__ = [
    "SoftmaxFormats",
    "NormFormats",
    "qsoftmax",
    "qlayernorm",
    "qrmsnorm",
    "qgroupnorm",
    "qbatchnorm",
]

# The default of the `uniform` constructors' gradient arguments: the forward's.
_SAME: Any = object()


# --- the formats -----------------------------------------------------------------------


def _mac_slot(mac: Any, carrier: torch.dtype | None, name: str) -> FusedMac | SplitMac | None:
    """A mac slot: its own carrier, when set, must be the op's."""
    mac = _checked_mac(mac, name)
    if mac is not None and mac.carrier is not None and mac.carrier is not carrier:
        raise ValueError(
            f"{name} is a mac with carrier={mac.carrier}, and the op's carrier is {carrier}: "
            "an op computes and rounds in the one carrier its formats name -- leave the mac's "
            "carrier unset, and set the formats' carrier"
        )
    return mac


def _resolve(
    formats: Any,
    macs: Sequence[str],
    quants: Sequence[str],
    inner: Sequence[str],
    last: str,
    backward: Sequence[str],
) -> None:
    """Check and convert a formats value's slots in place, resolve its macs,
    and note which of its steps round: those before the result (``inner``),
    the forward's, the backward's."""
    carrier = _checked_carrier(formats.carrier)
    specs = []
    for name in macs:
        mac = _mac_slot(getattr(formats, name), carrier, name)
        object.__setattr__(formats, name, mac)
        # resolved once, so that a call reads no cache (Dynamo warns about those)
        specs.append(None if mac is None else spec_for_mac(mac))
    for name in quants:
        object.__setattr__(
            formats, name, _checked_quant_slot(getattr(formats, name), carrier, name)
        )
    object.__setattr__(formats, "_specs", tuple(specs))
    rounds_inner = any([getattr(formats, n) is not None for n in inner])
    object.__setattr__(
        formats,
        "_flags",
        (
            rounds_inner,
            rounds_inner or getattr(formats, last) is not None,
            any([getattr(formats, n) is not None for n in backward]),
        ),
    )


def _label(formats: Any, macs: Sequence[str], quants: Sequence[str]) -> str:
    values = {n: getattr(formats, n) for n in (*macs, *quants)}
    set_macs = {values[n] for n in macs if values[n] is not None}
    set_quants = {values[n] for n in quants if values[n] is not None}
    if (
        len(set_macs) == 1
        and len(set_quants) == 1
        and all(values[n] is not None for n in (*macs, *quants))
    ):
        parts = [f"mac {_mac_label(next(iter(set_macs)))}", f"quant {_quant_label(*set_quants)}"]
    else:
        parts = [
            f"{n} {_mac_label(v) if n in macs else _quant_label(v)}"
            for n, v in values.items()
            if v is not None
        ]
    if formats.carrier is not None:
        parts.append(f"carrier {str(formats.carrier).removeprefix('torch.')}")
    return ", ".join(parts) or "torch"


_SOFTMAX_MACS = ("acc", "bwd_acc")
_SOFTMAX_QUANTS = ("sub", "exp", "div", "bwd_mul", "bwd_sub", "bwd_out")


@dataclass(frozen=True, kw_only=True)
class SoftmaxFormats:
    """The formats of a softmax: each step's rounding, forward and backward.

    The forward computes ``y = e / sum(e)`` with ``e = exp(x - max(x))``, the
    maximum exact; the backward ``y * (g - sum(g * y))``, with ``y`` the
    unrounded function's. It is the value of a recipe's ``softmax`` kind, and
    of :class:`~mptorch.quant.QSoftmax`'s ``formats``. A slot left ``None``
    does not round; with every slot ``None`` this is torch's softmax.

    Args:
        sub: rounds ``x - max(x)``. A format means ``Quant(fmt)``.
            Default: ``None``
        exp: rounds ``e``. Default: ``None``
        acc: each step of the sum of ``e``: a :class:`~mptorch.quant.FusedMac`,
            a :class:`~mptorch.quant.SplitMac`, or a format, which means
            ``FusedMac(fmt)`` (:func:`~mptorch.quant.qsum`). Default: ``None``,
            torch's sum in the carrier
        div: rounds the quotient, the result. Default: ``None``
        bwd_mul: rounds ``g * y``. Default: ``None``
        bwd_acc: each step of the sum of ``g * y``, as ``acc``.
            Default: ``None``
        bwd_sub: rounds ``g - sum(g * y)``. Default: ``None``
        bwd_out: rounds the input's gradient. Default: ``None``
        carrier (torch.dtype, optional): ``None`` for the tensor's own carrier
            (binary32, or binary64 for float64), ``torch.float64`` for binary64
            whatever its dtype. Default: ``None``

    Raises:
        TypeError: for a module in a slot, a slot that is not callable, or a
            block format in a mac slot.
        ValueError: for a ``Quant`` or a mac whose own carrier is set and
            differs, a palette, or an unknown carrier.
    """

    sub: Any = None
    exp: Any = None
    acc: Any = None
    div: Any = None
    bwd_mul: Any = None
    bwd_acc: Any = None
    bwd_sub: Any = None
    bwd_out: Any = None
    carrier: torch.dtype | None = None
    _specs: tuple[_GemmSpec | None, ...] = field(
        init=False, repr=False, compare=False, default=(None, None)
    )

    _flags: tuple[bool, bool, bool] = field(
        init=False, repr=False, compare=False, default=(False, False, False)
    )

    def __post_init__(self) -> None:
        _resolve(
            self,
            _SOFTMAX_MACS,
            _SOFTMAX_QUANTS,
            ("sub", "exp", "acc"),
            "div",
            ("bwd_mul", "bwd_acc", "bwd_sub", "bwd_out"),
        )

    @classmethod
    def uniform(
        cls, mac: Any, quant: Any, *, grad: Any = _SAME, carrier: torch.dtype | None = None
    ) -> SoftmaxFormats:
        """``mac`` for both sums and ``quant`` for every other step; ``grad``
        replaces the backward's roundings (``None`` leaves the backward
        unrounded, its sum included)."""
        g = quant if grad is _SAME else grad
        return cls(
            sub=quant,
            exp=quant,
            acc=mac,
            div=quant,
            bwd_mul=g,
            bwd_acc=None if g is None else mac,
            bwd_sub=g,
            bwd_out=g,
            carrier=carrier,
        )

    @property
    def _inner(self) -> bool:
        return self._flags[0]

    @property
    def _forward(self) -> bool:
        return self._flags[1]

    @property
    def _backward(self) -> bool:
        return self._flags[2]

    def label(self) -> str:
        """A one-line summary, for reports."""
        return _label(self, _SOFTMAX_MACS, _SOFTMAX_QUANTS)


_NORM_MACS = ("acc", "bwd_acc")
_NORM_QUANTS = ("div", "sub", "sq", "rsqrt", "norm", "mul", "add", "out", "bwd_mul", "bwd_out")
_NORM_INNER = ("acc", "div", "sub", "sq", "rsqrt", "norm", "mul", "add")
_DEVIATIONS = ("std", "l1", "linf")
# The l1 deviation's constant: E|X - mu| = sigma * sqrt(2/pi) for Gaussian X.
_L1_KAPPA = math.sqrt(2.0 / math.pi)


@dataclass(frozen=True, kw_only=True)
class NormFormats:
    """The formats of a norm: each step's rounding, forward and backward.

    A norm computes ``x_hat = (x - mean) / sqrt(var + eps)`` over its
    statistics' elements (a layer norm's last dimensions, a group's channels
    and positions, a batch norm's batch and positions for each channel; an
    RMS norm has no mean), then ``x_hat * weight + bias``. It is the value of
    a recipe's ``norm`` kind, and of the ``formats`` of
    :class:`~mptorch.quant.QLayerNorm` and its siblings. A slot left ``None``
    does not round; with every slot ``None`` a norm is torch's.

    Args:
        acc: each step of the statistics' sums (of ``x``, and of the squares):
            a :class:`~mptorch.quant.FusedMac`, a
            :class:`~mptorch.quant.SplitMac`, or a format, which means
            ``FusedMac(fmt)`` (:func:`~mptorch.quant.qsum`). Default: ``None``,
            torch's sum in the carrier
        div: rounds each statistic, a sum divided by the count: the mean and
            the variance. A format means ``Quant(fmt)``. Default: ``None``
        sub: rounds ``x - mean``. Default: ``None``
        sq: rounds the squares the variance sums (``(x - mean) ** 2``, or
            ``x ** 2`` for an RMS norm), each the carrier's product.
            Default: ``None``, the carrier's product
        rsqrt: rounds ``1 / sqrt(var + eps)``, the sum ``var + eps`` the
            carrier's. Default: ``None``
        norm: rounds ``x_hat``. Default: ``None``
        mul: rounds ``x_hat * weight``. Default: ``None``
        add: rounds ``... + bias``. Default: ``None``
        out: rounds the result. Default: ``None``
        bwd_acc: each step of the backward's sums (the means of
            ``g * weight`` and of ``g * weight * x_hat``, and the weight's and
            the bias's gradients over the batch), as ``acc``. Default: ``None``
        bwd_mul: rounds the backward's products (``g * weight``, its product
            with ``x_hat``, and ``g * x_hat``). Default: ``None``
        bwd_out: rounds each gradient: the input's, the weight's and the
            bias's. Default: ``None``
        carrier (torch.dtype, optional): as for :class:`SoftmaxFormats`.
            Default: ``None``
        deviation (str): the statistic ``x`` is divided by: ``"std"``, the
            standard deviation; ``"l1"``, the mean absolute deviation ``d``,
            ``x_hat = c * sqrt(2/pi) / (d + eps)`` with ``c = x - mean`` (``x``
            for an RMS norm), which matches the standard deviation on Gaussian
            data; ``"linf"``, the largest absolute deviation, ``x_hat = c /
            (d + eps)``. The ``rsqrt`` slot rounds the inverse either way, and
            ``sq`` does not apply. Layer, RMS and group norms only: a batch
            norm's running variance keeps torch's meaning. Default: ``"std"``
        inverse: how ``1 / sqrt(var + eps)`` is computed under ``"std"``:
            ``None``, the carrier's; a :class:`~mptorch.quant.FISR`, the fast
            inverse square root, each Newton step rounded by ``rsqrt``; or a
            :class:`~mptorch.quant.PWLTable` from
            :meth:`~mptorch.quant.PWLTable.rsqrt`, rounded by ``rsqrt``. The
            gradient is the exact one's, straight through. Default: ``None``

    Raises:
        TypeError, ValueError: as for :class:`SoftmaxFormats`, and for a
            ``deviation`` or an ``inverse`` that is not one of the above, or an
            ``inverse`` with a deviation other than ``"std"``.
    """

    acc: Any = None
    div: Any = None
    sub: Any = None
    sq: Any = None
    rsqrt: Any = None
    norm: Any = None
    mul: Any = None
    add: Any = None
    out: Any = None
    bwd_acc: Any = None
    bwd_mul: Any = None
    bwd_out: Any = None
    carrier: torch.dtype | None = None
    deviation: str = "std"
    inverse: Any = None
    _specs: tuple[_GemmSpec | None, ...] = field(
        init=False, repr=False, compare=False, default=(None, None)
    )

    _flags: tuple[bool, bool, bool] = field(
        init=False, repr=False, compare=False, default=(False, False, False)
    )

    def __post_init__(self) -> None:
        _resolve(
            self, _NORM_MACS, _NORM_QUANTS, _NORM_INNER, "out", ("bwd_acc", "bwd_mul", "bwd_out")
        )
        if self.deviation not in _DEVIATIONS:
            raise ValueError(
                f"NormFormats' deviation is one of {_DEVIATIONS}, got {self.deviation!r}"
            )
        inv = self.inverse
        if inv is not None:
            if isinstance(inv, PWLTable):
                if inv.reduction != "rsqrt":
                    raise ValueError(
                        "NormFormats' inverse is a table of 1/sqrt from PWLTable.rsqrt, which "
                        "reduces its input onto [1, 4); this table has no reduction"
                    )
            elif not isinstance(inv, FISR):
                raise TypeError(
                    "NormFormats' inverse is None, a FISR or a PWLTable.rsqrt table, "
                    f"got {type(inv).__name__}"
                )
            if self.deviation != "std":
                raise ValueError(
                    f"NormFormats' inverse approximates 1/sqrt(var + eps), and deviation "
                    f"{self.deviation!r} divides by its own statistic: set one of them"
                )
        variant = self.deviation != "std" or inv is not None
        inner, forward, backward = self._flags
        object.__setattr__(self, "_flags", (inner or variant, forward or variant, backward))

    @classmethod
    def uniform(
        cls, mac: Any, quant: Any, *, grad: Any = _SAME, carrier: torch.dtype | None = None
    ) -> NormFormats:
        """``mac`` for every sum and ``quant`` for every other step; ``grad``
        replaces the backward's roundings (``None`` leaves the backward
        unrounded, its sums included)."""
        g = quant if grad is _SAME else grad
        return cls(
            acc=mac,
            div=quant,
            sub=quant,
            sq=quant,
            rsqrt=quant,
            norm=quant,
            mul=quant,
            add=quant,
            out=quant,
            bwd_acc=None if g is None else mac,
            bwd_mul=g,
            bwd_out=g,
            carrier=carrier,
        )

    @property
    def _inner(self) -> bool:
        return self._flags[0]

    @property
    def _forward(self) -> bool:
        return self._flags[1]

    @property
    def _backward(self) -> bool:
        return self._flags[2]

    def label(self) -> str:
        """A one-line summary, for reports."""
        label = _label(self, _NORM_MACS, _NORM_QUANTS)
        extra = []
        if self.deviation != "std":
            extra.append(f"deviation {self.deviation}")
        if self.inverse is not None:
            extra.append(f"inverse {self.inverse.label()}")
        if not extra:
            return label
        return ", ".join(extra) if label == "torch" else f"{label}, {', '.join(extra)}"


# --- the steps -------------------------------------------------------------------------


def _r(slot: Any, t: torch.Tensor) -> torch.Tensor:
    """A step's result ``t``, fresh and in the carrier, rounded by ``slot`` there."""
    return t if slot is None else _round_result(slot, t, t.dtype)


def _finish(slot: Any, t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """The last step's result, rounded by ``slot`` and stored in ``dtype``."""
    return _stored(t, dtype) if slot is None else _round_result(slot, t, dtype)


def _total(spec: _GemmSpec | None, t: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    """``t`` summed over ``dims``, keeping them: on the GEMM ``spec`` names, or
    torch's sum in ``t``'s dtype."""
    return torch.sum(t, dims, keepdim=True) if spec is None else _sum(spec, t, dims, True)


def _over(t: torch.Tensor, n: int) -> torch.Tensor:
    """``t / n``, ``n`` as a 0-dim tensor of ``t``'s dtype: torch's CUDA
    division by a number multiplies by its reciprocal."""
    return t / torch.full((), n, dtype=t.dtype, device=t.device)


def _count(shape: Sequence[int], dims: Sequence[int]) -> int:
    return math.prod([shape[d] for d in dims])


def _normalized(
    xc: torch.Tensor,
    dims: tuple[int, ...],
    eps: float,
    f: NormFormats,
    rms: bool,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """``x_hat`` over ``dims`` of the carrier tensor ``xc``, each step rounded
    as ``f`` says, with the mean and the variance it used. ``stats`` (a batch
    norm's running statistics, in eval) replaces the batch's."""
    acc = f._specs[0]
    n = _count(xc.shape, dims)
    if stats is not None:
        mean, var = stats
        centred = _r(f.sub, xc - mean)
    elif rms:
        mean, centred = None, xc
        if f.deviation != "std":
            return _deviated(xc, None, centred, dims, eps, f)
        var = _r(f.div, _over(_total(acc, _r(f.sq, xc * xc), dims), n))
    else:
        mean = _r(f.div, _over(_total(acc, xc, dims), n))
        centred = _r(f.sub, xc - mean)
        if f.deviation != "std":
            return _deviated(xc, mean, centred, dims, eps, f)
        var = _r(f.div, _over(_total(acc, _r(f.sq, centred * centred), dims), n))
    rstd = _inverse(f, var + eps)
    return _r(f.norm, centred * rstd), mean, var


def _inverse(f: NormFormats, v: torch.Tensor) -> torch.Tensor:
    """``1 / sqrt(v)``, ``v = var + eps``: the carrier's, FISR's or a
    table's, as ``f.inverse`` says, rounded by ``rsqrt``."""
    inv = f.inverse
    if inv is None:
        return _r(f.rsqrt, torch.rsqrt(v))
    if isinstance(inv, FISR):
        return _fisr(v, inv.steps, f.rsqrt)
    return _rounded(inv, v, f.rsqrt, None, v.dtype)


def _kappa(deviation: str) -> float:
    return _L1_KAPPA if deviation == "l1" else 1.0


def _deviated(xc, mean, centred, dims, eps, f: NormFormats):
    """``x_hat`` under an ``l1`` or ``linf`` deviation ``d`` of ``centred``:
    ``kappa / (d + eps)`` rounded by ``rsqrt``, times ``centred`` rounded by
    ``norm``. Returns ``x_hat``, the mean and ``d``."""
    a = centred.abs()
    if f.deviation == "l1":
        d = _r(f.div, _over(_total(f._specs[0], a, dims), _count(xc.shape, dims)))
    else:
        d = a.amax(dims, keepdim=True)
    kappa = torch.full((), _kappa(f.deviation), dtype=xc.dtype, device=xc.device)
    inv = _r(f.rsqrt, kappa / (d + eps))
    return _r(f.norm, centred * inv), mean, d


def _affine(t, weight, bias, f: NormFormats, view: Sequence[int] | None) -> torch.Tensor:
    """``t * weight + bias`` in ``t``'s carrier, ``view`` the shape that
    broadcasts a per-channel parameter (``None``: as it is)."""
    c = t.dtype
    if weight is not None:
        w = weight.to(c)
        t = _r(f.mul, t * (w if view is None else w.view(view)))
    if bias is not None:
        b = bias.to(c)
        t = _r(f.add, t + (b if view is None else b.view(view)))
    return t


def _channel_view(x: torch.Tensor) -> tuple[int, ...]:
    """The shape that broadcasts a per-channel parameter over ``x`` ([N, C, ...])."""
    return (1, x.shape[1], *([1] * (x.dim() - 2)))


def _unrounded_stats(
    xc: torch.Tensor, dims: tuple[int, ...], eps: float, rms: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unrounded function's ``x_hat`` and ``1 / sqrt(var + eps)`` over
    ``dims``, in the carrier, for the closed-form backward."""
    if rms:
        var = (xc * xc).mean(dims, keepdim=True)
        centred = xc
    else:
        mean = xc.mean(dims, keepdim=True)
        centred = xc - mean
        var = (centred * centred).mean(dims, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    return centred * rstd, rstd


def _variant_stats(xc, dims, eps, rms: bool, deviation: str):
    """The unrounded ``c``, ``r = kappa / (d + eps)`` and ``x_hat = c * r``
    of an ``l1`` or ``linf`` deviation, for the closed-form backward."""
    c = xc if rms else xc - xc.mean(dims, keepdim=True)
    a = c.abs()
    d = a.mean(dims, keepdim=True) if deviation == "l1" else a.amax(dims, keepdim=True)
    r = _kappa(deviation) / (d + eps)
    return c, r, c * r


def _variant_input_grad(dyhat, c, r, dims, f: NormFormats, rms: bool) -> torch.Tensor:
    """The input's gradient under an ``l1`` or ``linf`` deviation ``d``: with
    ``y = c * r`` and ``r = kappa / (d + eps)``, ``dc = r * g - (r**2 /
    kappa) * sum(g * c) * dd/dc``, where ``dd/dc`` is ``sign(c) / n`` for
    ``l1`` and ``sign(c) / t`` on the ``t`` tied maxima for ``linf`` (as
    ``torch.amax`` splits a tie), and then ``dx = dc - mean(dc)`` for a
    centred norm. The sums are on ``bwd_acc``."""
    acc = f._specs[1]
    n = _count(dyhat.shape, dims)
    kappa = _kappa(f.deviation)
    s = _total(acc, _r(f.bwd_mul, dyhat * c), dims)
    if f.deviation == "l1":
        dd = torch.sign(c) / n
    else:
        a = c.abs()
        top = (a == a.amax(dims, keepdim=True)).to(c.dtype)
        dd = torch.sign(c) * top / top.sum(dims, keepdim=True)
    dc = r * dyhat - (r * r / kappa) * s * dd
    if not rms:
        dc = dc - _over(_total(acc, dc, dims), n)
    return dc


def _variant_plain(x, weight, bias, dims, eps, deviation: str, rms: bool, view=None):
    """The ``l1`` or ``linf`` norm written in torch's ops, unrounded, whose
    graph gives autograd's gradient when no backward slot is set."""
    c = x if rms else x - x.mean(dims, keepdim=True)
    a = c.abs()
    d = a.mean(dims, keepdim=True) if deviation == "l1" else a.amax(dims, keepdim=True)
    t = c * (_kappa(deviation) / (d + eps))
    if weight is not None:
        t = t * (weight if view is None else weight.view(view))
    if bias is not None:
        t = t + (bias if view is None else bias.view(view))
    return t


def _input_grad(
    dyhat: torch.Tensor,
    xhat: torch.Tensor,
    rstd: torch.Tensor,
    dims: tuple[int, ...],
    f: NormFormats,
    rms: bool,
) -> torch.Tensor:
    """``rstd * (dyhat - mean(dyhat) - xhat * mean(dyhat * xhat))`` over
    ``dims`` (an RMS norm drops ``mean(dyhat)``), its sums on ``bwd_acc``."""
    acc = f._specs[1]
    n = _count(dyhat.shape, dims)
    b = _over(_total(acc, _r(f.bwd_mul, dyhat * xhat), dims), n)
    centred = dyhat - xhat * b
    if not rms:
        centred = centred - _over(_total(acc, dyhat, dims), n)
    return rstd * centred


def _param_grad(
    t: torch.Tensor, dims: tuple[int, ...], f: NormFormats, param: torch.Tensor
) -> torch.Tensor:
    """A weight's or a bias's gradient: ``t`` summed over ``dims`` on
    ``bwd_acc``, shaped as ``param``, rounded by ``bwd_out``."""
    s = t if not dims else _total(f._specs[1], t, dims)
    return _finish(f.bwd_out, s.reshape(param.shape), param.dtype)


# --- softmax ---------------------------------------------------------------------------


def _softmax_forward(x: torch.Tensor, dim: int, f: SoftmaxFormats) -> torch.Tensor:
    dtype = x.dtype
    xc = x.to(_carrier_dtype(f.carrier, dtype))
    if not f._inner:
        return _finish(f.div, torch.softmax(xc, dim), dtype)
    d = _r(f.sub, xc - xc.amax(dim, keepdim=True))
    e = _r(f.exp, torch.exp(d))
    return _finish(f.div, e / _total(f._specs[0], e, (dim,)), dtype)


def _softmax_grad(x: torch.Tensor, dim: int, f: SoftmaxFormats, g: torch.Tensor) -> torch.Tensor:
    gc = g.to(_carrier_dtype(f.carrier, g.dtype))
    y = torch.softmax(x.to(gc.dtype), dim)  # the unrounded function's
    s = _total(f._specs[1], _r(f.bwd_mul, gc * y), (dim,))
    return _finish(f.bwd_out, y * _r(f.bwd_sub, gc - s), g.dtype)


# --- the norms -------------------------------------------------------------------------


def _layer_dims(x: torch.Tensor, shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(range(x.dim() - len(shape), x.dim()))


def _layer_norm_forward(x, weight, bias, shape, eps, f: NormFormats, rms: bool):
    dtype = x.dtype
    c = _carrier_dtype(f.carrier, dtype)
    xc = x.to(c)
    if not f._inner:
        w = None if weight is None else weight.to(c)
        if rms:
            return _finish(f.out, F.rms_norm(xc, shape, w, eps), dtype)
        b = None if bias is None else bias.to(c)
        return _finish(f.out, F.layer_norm(xc, shape, w, b, eps), dtype)
    xhat, _, _ = _normalized(xc, _layer_dims(x, shape), eps, f, rms)
    return _finish(f.out, _affine(xhat, weight, bias, f, None), dtype)


def _layer_norm_grads(x, weight, bias, g, needs, shape, eps, f: NormFormats, rms: bool):
    c = _carrier_dtype(f.carrier, g.dtype)
    xc, gc = x.to(c), g.to(c)
    dims = _layer_dims(x, shape)
    if f.deviation == "std":
        xhat, rstd = _unrounded_stats(xc, dims, eps, rms)
    else:
        centred, r, xhat = _variant_stats(xc, dims, eps, rms, f.deviation)
    dyhat = gc if weight is None else _r(f.bwd_mul, gc * weight.to(c))
    gx = None
    if needs[0]:
        if f.deviation == "std":
            t = _input_grad(dyhat, xhat, rstd, dims, f, rms)
        else:
            t = _variant_input_grad(dyhat, centred, r, dims, f, rms)
        gx = _finish(f.bwd_out, t, g.dtype)
    batch = tuple(range(x.dim() - len(shape)))
    gw = _param_grad(_r(f.bwd_mul, gc * xhat), batch, f, weight) if needs[1] else None
    gb = _param_grad(gc, batch, f, bias) if needs[2] else None
    return gx, gw, gb


def _group_view(x: torch.Tensor, groups: int) -> torch.Tensor:
    """``x`` ([N, C, ...]) as ``[N, groups, C / groups * positions]``."""
    return x.reshape(x.shape[0], groups, -1)


def _group_norm_forward(x, weight, bias, groups, eps, f: NormFormats):
    dtype = x.dtype
    c = _carrier_dtype(f.carrier, dtype)
    xc = x.to(c)
    if not f._inner:
        w = None if weight is None else weight.to(c)
        b = None if bias is None else bias.to(c)
        return _finish(f.out, F.group_norm(xc, groups, w, b, eps), dtype)
    xhat, _, _ = _normalized(_group_view(xc, groups), (2,), eps, f, False)
    t = _affine(xhat.reshape(x.shape), weight, bias, f, _channel_view(x))
    return _finish(f.out, t, dtype)


def _group_norm_grads(x, weight, bias, g, needs, groups, eps, f: NormFormats):
    c = _carrier_dtype(f.carrier, g.dtype)
    xc, gc = x.to(c), g.to(c)
    view = _channel_view(x)
    if f.deviation == "std":
        xhat_g, rstd = _unrounded_stats(_group_view(xc, groups), (2,), eps, False)
    else:
        centred, r, xhat_g = _variant_stats(_group_view(xc, groups), (2,), eps, False, f.deviation)
    xhat = xhat_g.reshape(x.shape)
    dyhat = gc if weight is None else _r(f.bwd_mul, gc * weight.to(c).view(view))
    gx = None
    if needs[0]:
        dy_g = _group_view(dyhat, groups)
        if f.deviation == "std":
            t = _input_grad(dy_g, xhat_g, rstd, (2,), f, False)
        else:
            t = _variant_input_grad(dy_g, centred, r, (2,), f, False)
        gx = _finish(f.bwd_out, t.reshape(x.shape), g.dtype)
    others = (0, *range(2, x.dim()))
    gw = _param_grad(_r(f.bwd_mul, gc * xhat), others, f, weight) if needs[1] else None
    gb = _param_grad(gc, others, f, bias) if needs[2] else None
    return gx, gw, gb


def _batch_dims(x: torch.Tensor) -> tuple[int, ...]:
    return (0, *range(2, x.dim()))


def _batch_norm_forward(x, weight, bias, rm, rv, training, momentum, eps, f: NormFormats):
    dtype = x.dtype
    c = _carrier_dtype(f.carrier, dtype)
    xc = x.to(c)
    view = _channel_view(x)
    if not f._inner:
        t = _plain_batch_norm(xc, weight, bias, rm, rv, training, momentum, eps)
        return _finish(f.out, t, dtype)
    dims = _batch_dims(x)
    if training:
        xhat, mean, var = _normalized(xc, dims, eps, f, False)
        if rm is not None and rv is not None:
            n = _count(x.shape, dims)
            _update_running(rm, rv, mean, var, n, momentum)
    else:
        assert rm is not None and rv is not None  # checked by qbatchnorm
        stats = (rm.to(c).view(view), rv.to(c).view(view))
        xhat, _, _ = _normalized(xc, dims, eps, f, False, stats)
    return _finish(f.out, _affine(xhat, weight, bias, f, view), dtype)


def _plain_batch_norm(xc, weight, bias, rm, rv, training, momentum, eps) -> torch.Tensor:
    """torch's batch norm on the carrier tensor ``xc``. The running statistics
    are given to it in ``xc``'s dtype, and copied back when that is not theirs."""
    c = xc.dtype
    w = None if weight is None else weight.to(c)
    b = None if bias is None else bias.to(c)
    rmc = None if rm is None else rm.to(c)
    rvc = None if rv is None else rv.to(c)
    out = F.batch_norm(xc, rmc, rvc, w, b, training, momentum, eps)
    with torch.no_grad():
        for buf, used in ((rm, rmc), (rv, rvc)):
            if buf is not None and used is not buf:
                buf.copy_(used)
    return out


def _update_running(rm, rv, mean, var, n: int, momentum: float) -> None:
    """torch's update of a batch norm's running statistics, from the batch's
    mean and its variance made unbiased, in the buffers' dtype."""
    with torch.no_grad():
        m = mean.reshape(-1).to(rm.dtype)
        unbiased = var.reshape(-1).to(rv.dtype) * (n / max(n - 1, 1))
        rm.copy_(momentum * m + (1 - momentum) * rm)
        rv.copy_(momentum * unbiased + (1 - momentum) * rv)


def _batch_norm_grads(x, weight, bias, g, needs, rm, rv, training, momentum, eps, f, stats):
    c = _carrier_dtype(f.carrier, g.dtype)
    xc, gc = x.to(c), g.to(c)
    view = _channel_view(x)
    dims = _batch_dims(x)
    if training:
        xhat, rstd = _unrounded_stats(xc, dims, eps, False)
    else:
        mean, var = stats[0].to(c).view(view), stats[1].to(c).view(view)
        rstd = torch.rsqrt(var + eps)
        xhat = (xc - mean) * rstd
    dyhat = gc if weight is None else _r(f.bwd_mul, gc * weight.to(c).view(view))
    gx = None
    if needs[0]:
        t = _input_grad(dyhat, xhat, rstd, dims, f, False) if training else dyhat * rstd
        gx = _finish(f.bwd_out, t, g.dtype)
    gw = _param_grad(_r(f.bwd_mul, gc * xhat), dims, f, weight) if needs[1] else None
    gb = _param_grad(gc, dims, f, bias) if needs[2] else None
    return gx, gw, gb


# --- the Functions ---------------------------------------------------------------------

_FORWARDS: dict[str, Callable[..., torch.Tensor]] = {
    "softmax": lambda x, w, b, dim, f: _softmax_forward(x, dim, f),
    "layer_norm": lambda x, w, b, shape, eps, f: _layer_norm_forward(x, w, b, shape, eps, f, False),
    "rms_norm": lambda x, w, b, shape, eps, f: _layer_norm_forward(x, w, b, shape, eps, f, True),
    "group_norm": _group_norm_forward,
    "batch_norm": _batch_norm_forward,
}


class _Rounded(torch.autograd.Function):
    """The rounded forward of an op whose backward is torch's: the gradient
    goes to ``plain``, torch's own result of the op, whose graph autograd
    differentiates."""

    @staticmethod
    def forward(ctx, plain, op, x, weight, bias, *rest):
        ctx.n_rest = len(rest)
        with torch._C.DisableTorchFunction():  # hidden from torch function modes
            return _FORWARDS[op](x, weight, bias, *rest)

    @staticmethod
    def backward(ctx, *grads):
        return (grads[0], None, None, None, None) + (None,) * ctx.n_rest


class _ClosedForm(torch.autograd.Function):
    """The rounded forward of an op with a backward slot set, and the closed
    form of its gradient, each step rounded as the slots say."""

    @staticmethod
    def forward(ctx, op, x, weight, bias, *rest):
        with torch._C.DisableTorchFunction():
            ctx.op, ctx.rest = op, rest
            stats: tuple[torch.Tensor, ...] = ()
            if op == "batch_norm" and not rest[2]:  # eval: the running statistics it used
                stats = (rest[0].clone(), rest[1].clone())
            ctx.save_for_backward(x, weight, bias, *stats)
            return _FORWARDS[op](x, weight, bias, *rest)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():  # as in forward, except under compile (_opaque.py)
            x, weight, bias, *stats = ctx.saved_tensors
            g, op, rest = grads[0], ctx.op, ctx.rest
            needs = ctx.needs_input_grad[1:4]
            if op == "softmax":
                gx = _softmax_grad(x, rest[0], rest[1], g) if needs[0] else None
                gw = gb = None
            elif op in ("layer_norm", "rms_norm"):
                shape, eps, f = rest
                rms = op == "rms_norm"
                gx, gw, gb = _layer_norm_grads(x, weight, bias, g, needs, shape, eps, f, rms)
            elif op == "group_norm":
                gx, gw, gb = _group_norm_grads(x, weight, bias, g, needs, *rest)
            else:
                gx, gw, gb = _batch_norm_grads(x, weight, bias, g, needs, *rest, stats)
            return (None, gx, gw, gb) + (None,) * len(rest)


def _run(op: str, plain: Callable[[], torch.Tensor], f: Any, x, weight, bias, *rest):
    """An op with formats ``f`` that rounds its forward: the closed form when
    a backward slot is set, otherwise the rounded forward over torch's
    gradient, built from ``plain()`` when a gradient is needed."""
    if f._backward:
        return _ClosedForm.apply(op, x, weight, bias, *rest)
    needs_grad = (
        x.requires_grad
        or (weight is not None and weight.requires_grad)
        or (bias is not None and bias.requires_grad)
    )
    if torch.is_grad_enabled() and needs_grad:
        return _Rounded.apply(plain(), op, x, weight, bias, *rest)
    return _FORWARDS[op](x, weight, bias, *rest)


def _checked_formats(formats: Any, cls: type, op: str) -> Any:
    if formats is not None and not isinstance(formats, cls):
        raise TypeError(f"{op}'s formats is a {cls.__name__} or None, got {type(formats).__name__}")
    return formats


def _rounding(f: Any) -> bool:
    return f is not None and (f._forward or f._backward)


# --- the ops ---------------------------------------------------------------------------


def qsoftmax(
    x: torch.Tensor,
    dim: int | None = None,
    formats: SoftmaxFormats | None = None,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """``F.softmax(x, dim, dtype=dtype)``, each step rounded as ``formats`` says.

    ``dtype`` converts ``x`` first, as ``F.softmax`` does. The maximum is
    exact; ``x - max(x)``, its exponential, their sum (on the GEMM kernels,
    with ``formats.acc``'s mac) and the quotient are the carrier's, each
    rounded by its slot. With ``sub``, ``exp`` and ``acc`` all ``None`` this is
    torch's softmax in the carrier, its result rounded by ``div``.

    Args:
        x (Tensor): a float tensor.
        dim (int): the dimension to normalize over. ``None`` chooses it as
            ``F.softmax`` does, with its warning. Default: ``None``
        formats (SoftmaxFormats, optional): the roundings.
            Default: ``None``, torch's softmax
        dtype (torch.dtype, optional): the dtype to convert ``x`` to first.
            Default: ``None``
    """
    with torch._C.DisableTorchFunction():  # the op, not the calls it makes
        f = _checked_formats(formats, SoftmaxFormats, "qsoftmax")
        if dim is None:
            dim = _implicit_dim(x.dim())
        if dtype is not None:
            x = x.to(dtype)
        if not _rounding(f):
            return F.softmax(x, dim)
        (d,) = _reduced(x, dim)
        return _run("softmax", lambda: F.softmax(x, d), f, x, None, None, d, f)


def qlayernorm(
    x: torch.Tensor,
    normalized_shape: int | Sequence[int],
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    formats: NormFormats | None = None,
) -> torch.Tensor:
    """``F.layer_norm``, each step rounded as ``formats`` says.

    The statistics are over the last ``len(normalized_shape)`` dimensions:
    the mean, ``x - mean``, the variance (the sum of the squares over the
    count), ``1 / sqrt(var + eps)``, ``x_hat`` and the affine part, each the
    carrier's and rounded by its slot (:class:`NormFormats`). The arguments
    before ``formats`` are ``F.layer_norm``'s.
    """
    with torch._C.DisableTorchFunction():
        f = _checked_formats(formats, NormFormats, "qlayernorm")
        shape = _shape(normalized_shape)
        if not _rounding(f):
            return F.layer_norm(x, shape, weight, bias, eps)
        return _run(
            "layer_norm",
            _layer_plain(x, shape, weight, bias, eps, f, rms=False),
            f,
            x,
            weight,
            bias,
            shape,
            eps,
            f,
        )


def qrmsnorm(
    x: torch.Tensor,
    normalized_shape: int | Sequence[int],
    weight: torch.Tensor | None = None,
    eps: float | None = None,
    formats: NormFormats | None = None,
) -> torch.Tensor:
    """``F.rms_norm``, each step rounded as ``formats`` says.

    As :func:`qlayernorm`, with no mean: the variance is the mean of the
    squares of ``x``. ``eps=None`` is ``torch.finfo(x.dtype).eps``, as for
    ``F.rms_norm``. The arguments before ``formats`` are ``F.rms_norm``'s.
    """
    with torch._C.DisableTorchFunction():
        f = _checked_formats(formats, NormFormats, "qrmsnorm")
        shape = _shape(normalized_shape)
        if not _rounding(f):
            return F.rms_norm(x, shape, weight, eps)
        e = torch.finfo(x.dtype).eps if eps is None else eps
        return _run(
            "rms_norm",
            _layer_plain(x, shape, weight, None, e, f, rms=True),
            f,
            x,
            weight,
            None,
            shape,
            e,
            f,
        )


def qgroupnorm(
    x: torch.Tensor,
    num_groups: int,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    formats: NormFormats | None = None,
) -> torch.Tensor:
    """``F.group_norm``, each step rounded as ``formats`` says.

    The statistics of each sample's group are over its channels and their
    positions, and the affine part is per channel. The arguments before
    ``formats`` are ``F.group_norm``'s.
    """
    with torch._C.DisableTorchFunction():
        f = _checked_formats(formats, NormFormats, "qgroupnorm")
        if not _rounding(f):
            return F.group_norm(x, num_groups, weight, bias, eps)
        if x.dim() < 2 or x.shape[1] % num_groups:
            raise ValueError(
                f"qgroupnorm: {x.shape[1] if x.dim() > 1 else 'no'} channels do not divide "
                f"into {num_groups} groups"
            )
        return _run(
            "group_norm",
            _group_plain(x, num_groups, weight, bias, eps, f),
            f,
            x,
            weight,
            bias,
            num_groups,
            eps,
            f,
        )


def qbatchnorm(
    x: torch.Tensor,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
    formats: NormFormats | None = None,
) -> torch.Tensor:
    """``F.batch_norm``, each step rounded as ``formats`` says.

    In training the statistics of each channel are over the batch and its
    positions, and the running statistics are updated from the rounded mean
    and variance (the variance made unbiased) by torch's formula, in their
    own dtype. In eval the running statistics are used. The arguments before
    ``formats`` are ``F.batch_norm``'s.
    """
    with torch._C.DisableTorchFunction():
        f = _checked_formats(formats, NormFormats, "qbatchnorm")
        if not _rounding(f):
            return F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)
        if training:
            size = x.size()
            if _count(size, _batch_dims(x)) <= 1:
                raise ValueError(
                    f"Expected more than 1 value per channel when training, got input size {size}"
                )
        elif running_mean is None or running_var is None:
            raise ValueError("qbatchnorm in eval needs the running mean and variance")
        if f.deviation != "std":
            raise ValueError(
                f"qbatchnorm takes no deviation {f.deviation!r}: a batch norm's running_var holds "
                "the variance torch's formula updates, which a checkpoint and eval read as such. "
                "A layer, RMS or group norm takes it, and a batch norm takes an inverse"
            )

        def plain():
            # torch's gradient in training reads the batch's statistics, not the running ones,
            # which the rounded forward updates (once)
            if training:
                return F.batch_norm(x, None, None, weight, bias, True, momentum, eps)
            return F.batch_norm(x, running_mean, running_var, weight, bias, False, momentum, eps)

        return _run(
            "batch_norm",
            plain,
            f,
            x,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            momentum,
            eps,
            f,
        )


def _layer_plain(x, shape, weight, bias, eps, f: NormFormats, rms: bool) -> Callable:
    """``plain`` for a layer or RMS norm: torch's own, or the variant's spelling."""
    if f.deviation == "std":
        if rms:
            return lambda: F.rms_norm(x, shape, weight, eps)
        return lambda: F.layer_norm(x, shape, weight, bias, eps)

    def plain():
        c = _carrier_dtype(f.carrier, x.dtype)
        dims = _layer_dims(x, shape)
        w = None if weight is None else weight.to(c)
        b = None if bias is None else bias.to(c)
        return _variant_plain(x.to(c), w, b, dims, eps, f.deviation, rms).to(x.dtype)

    return plain


def _group_plain(x, groups, weight, bias, eps, f: NormFormats) -> Callable:
    """``plain`` for a group norm: torch's own, or the variant's spelling."""
    if f.deviation == "std":
        return lambda: F.group_norm(x, groups, weight, bias, eps)

    def plain():
        c = _carrier_dtype(f.carrier, x.dtype)
        t = _variant_plain(_group_view(x.to(c), groups), None, None, (2,), eps, f.deviation, False)
        w = None if weight is None else weight.to(c)
        b = None if bias is None else bias.to(c)
        view = _channel_view(x)
        return _variant_plain_affine(t.reshape(x.shape), w, b, view).to(x.dtype)

    return plain


def _variant_plain_affine(t, weight, bias, view):
    if weight is not None:
        t = t * weight.view(view)
    if bias is not None:
        t = t + bias.view(view)
    return t


def _implicit_dim(ndim: int) -> int:
    """``F.softmax``'s choice of a dimension when given none, with its warning."""
    warnings.warn(
        "Implicit dimension choice for softmax has been deprecated. "
        "Change the call to include dim=X as an argument.",
        stacklevel=4,
    )
    return 0 if ndim in (0, 1, 3) else 1


def _shape(normalized_shape: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(normalized_shape)
