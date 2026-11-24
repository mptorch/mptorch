"""Sums and means whose every step rounds to a simulated format: ``qsum`` and ``qmean``.

A rounded sum runs on the GEMM kernels, as a product with a ones vector
(``docs/source/kernels/reductions.rst`` is the user's account).
``FusedMac(F)`` over ``ones · x`` takes ``fma(1, x_k, s)`` at each step, which
rounds ``s + x_k`` once in the carrier, and ``F`` then rounds that, in the
order of the ``x_k``; a ``SplitMac(M, F)`` first rounds each ``x_k`` to
``M``. So a sum has what a GEMM has: every accumulate algorithm (TREE for a
``SplitMac`` only), every rounding mode, round to odd (``carrier_rounding``),
both carriers, every backend, and ``torch.compile``.

The sum reads ``x`` where it lies whenever the reduced dimensions are
adjacent: the last ones as rows (``ones[1, K] @ x[R, K].mT``, which the CPU's
vector tile runs across 32 rows at a time), any others through a batch
(``ones[1, K] @ x[A, K, B]``, the ones matrix shared across the batch).
Dimensions that are not adjacent are moved next to each other first, which
copies, and summed in row-major order over them.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from mptorch.number import BlockFormat, Number

# A module reference rather than names: `pointwise` imports `modules`, whose
# QSoftmax imports `norm` and so this module, while `pointwise` is still loading.
from . import pointwise
from .mac import BlockMac, FusedMac, SplitMac, _as_quant, spec_for_mac
from .modules._opaque import _opaque_backward
from .ops._checks import _call_carrier, _checked_carrier
from .ops._spec import _GemmSpec, _round_result, _run_gemm

__all__ = ["ReductionFormats", "qsum", "qmean"]

Dims = int | Sequence[int] | None

# The default of `ReductionFormats.uniform`'s gradient argument: the result's.
_SAME: Any = object()


def _checked_mac(mac: Any, name: str) -> FusedMac | SplitMac | None:
    """A sum's mac: a ``FusedMac`` or a ``SplitMac`` of one format per slot, or
    a format, which means ``FusedMac(fmt)``."""
    if mac is None:
        return None
    if isinstance(mac, BlockFormat | BlockMac):
        raise TypeError(
            f"{name} is a block format: a sum's steps round to one elementwise format each, so "
            "pass a FusedMac, a SplitMac or a BinaryK, SuperFP or FixedPoint format"
        )
    if isinstance(mac, Number):
        mac = FusedMac(mac)
    if not isinstance(mac, FusedMac | SplitMac):
        raise TypeError(
            f"{name} takes a FusedMac, a SplitMac or a format, got {type(mac).__name__}"
        )
    if spec_for_mac(mac).mixed:
        raise ValueError(
            f"{name} holds a palette, which selects a format per output element from a prec_idx "
            "map: a sum rounds every step to one format, so give each slot a single format"
        )
    return mac


def _checked_quant_slot(q: Any, carrier: torch.dtype | None, name: str) -> Any:
    """A rounding slot of a formats value, a format turned into ``Quant(fmt)``,
    then checked as ``mptorch.quant.pointwise``'s slots are."""
    q = _as_quant(q, name)
    pointwise._checked_slot(q, carrier, name)
    return q


def _checked_keyword_slot(q: Any, carrier: torch.dtype | None, name: str) -> Any:
    """A rounding slot given to :func:`qsum` or :func:`qmean`: a ``Quant`` or a
    callable, as ``mptorch.quant.pointwise``'s keyword slots, so that nothing is
    built per call."""
    if isinstance(q, Number):
        raise TypeError(
            f"{name} is a format, {q!r}: pass Quant(fmt), which rounds to it, or put the format "
            "in a ReductionFormats, which turns it into one when it is built"
        )
    pointwise._checked_slot(q, carrier, name)
    return q


# --- the sum ---------------------------------------------------------------------------


def _reduced(x: torch.Tensor, dim: Dims) -> tuple[int, ...]:
    """The dimensions a reduction over ``dim`` sums, sorted: every one for
    ``None`` or an empty sequence, as ``torch.sum`` reads them."""
    n = x.dim()
    if dim is None:
        return tuple(range(n))
    dims = (dim,) if isinstance(dim, int) else tuple(dim)
    if not dims:
        return tuple(range(n))
    rank = max(n, 1)  # a 0-dim tensor takes dim 0 or -1, as torch's reductions do
    out: list[int] = []
    for d in dims:
        if isinstance(d, bool) or not isinstance(d, int):
            raise TypeError(f"dim takes ints, got {type(d).__name__}")
        if not -rank <= d < rank:
            raise IndexError(
                f"dim {d} is out of range for a tensor of {n} dimensions "
                f"(expected to be in [{-rank}, {rank - 1}])"
            )
        d %= rank
        if d in out:
            raise RuntimeError(f"dim {d} appears more than once in the dimensions to sum")
        out.append(d)
    return tuple(sorted(out))


def _as_torch_dim(dim: Dims) -> int | tuple[int, ...] | None:
    """``dim`` as torch's reductions take it: a sequence becomes a tuple."""
    return dim if dim is None or isinstance(dim, int) else tuple(dim)


def _prod(sizes: Sequence[int]) -> int:
    p = 1
    for s in sizes:
        p *= s
    return p


def _kept_shape(shape: Sequence[int], dims: tuple[int, ...], keepdim: bool) -> tuple[int, ...]:
    if keepdim:
        return tuple([1 if i in dims else s for i, s in enumerate(shape)])
    return tuple([s for i, s in enumerate(shape) if i not in dims])


def _sum(spec: _GemmSpec, x: torch.Tensor, dims: tuple[int, ...], keepdim: bool) -> torch.Tensor:
    """``x`` summed over ``dims`` by the GEMM ``spec`` names, with a ones vector."""
    shape = tuple(x.shape)
    if x.dim() == 0:
        x, dims = x.reshape(1), (0,)
        out_shape: tuple[int, ...] = ()
    else:
        out_shape = _kept_shape(shape, dims, keepdim)
    sizes = tuple(x.shape)
    first, last = dims[0], dims[-1]
    if dims == tuple(range(first, last + 1)):
        a, k, b = _prod(sizes[:first]), _prod(sizes[first : last + 1]), _prod(sizes[last + 1 :])
        src = x
    else:
        # the reduced dimensions moved last, in their order: row-major over them
        kept = [i for i in range(x.dim()) if i not in dims]
        src = x.permute(*kept, *dims)
        a, k, b = _prod([sizes[i] for i in kept]), _prod([sizes[i] for i in dims]), 1
    ones = torch.ones(1, k, dtype=x.dtype, device=x.device)
    if b == 1:
        s = _run_gemm(spec, ones, src.reshape(a, k), False, True)  # [1, a]
    else:
        s = _run_gemm(spec, ones, src.reshape(a, k, b), False, False)  # [a, 1, b]
    return s.reshape(out_shape)


def _plain_sum(x: torch.Tensor, dims: tuple[int, ...], keepdim: bool) -> torch.Tensor:
    """torch's sum over ``dims`` (empty only for a 0-dim tensor)."""
    return torch.sum(x, dims, keepdim)


def _divided(s: torch.Tensor, n: int, quant: Any, carrier: torch.dtype | None) -> torch.Tensor:
    """``s / n``: torch's division in ``s``'s dtype with no slot, and the
    carrier's, rounded once by ``quant``, with one. ``n`` is a 0-dim tensor of
    the carrier, since torch's CUDA division by a number multiplies by its
    reciprocal."""
    if quant is None:
        return s / n
    wide, _ = _call_carrier(carrier, s.dtype)
    c = torch.float64 if wide else torch.float32
    t = s.to(c) / torch.full((), n, dtype=c, device=s.device)
    return _round_result(quant, t, s.dtype)


def _reduce_forward(ctx, x, spec, dims, keepdim, quant, grad_quant, carrier, mean):
    n = _prod([x.shape[i] for i in dims]) if x.dim() else 1
    ctx.reduction = (tuple(x.shape), dims, keepdim, grad_quant, carrier, n if mean else None)
    s = _plain_sum(x, dims, keepdim) if spec is None else _sum(spec, x, dims, keepdim)
    return _divided(s, n, quant, carrier) if mean else s


def _reduce_backward(ctx, g: torch.Tensor) -> torch.Tensor:
    """The gradient broadcast back over the summed dimensions, divided by
    their size for a mean, as torch's ``sum_backward`` and ``mean_backward``
    compute it; with a gradient slot, in the carrier and rounded once."""
    shape, dims, keepdim, q, carrier, n = ctx.reduction
    if not keepdim and len(shape):
        g = g.reshape(_kept_shape(shape, dims, True))
    if q is None:
        gx = g.expand(shape)
        return gx if n is None else gx / n
    wide, _ = _call_carrier(carrier, g.dtype)
    c = torch.float64 if wide else torch.float32
    gx = g.to(c).expand(shape)
    if n is not None:
        gx = gx / torch.full((), n, dtype=c, device=g.device)
    return _round_result(q, gx, g.dtype)


class _QSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spec, dims, keepdim, grad_quant, carrier):
        with torch._C.DisableTorchFunction():  # hidden from torch function modes
            return _reduce_forward(ctx, x, spec, dims, keepdim, None, grad_quant, carrier, False)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():  # as in forward, except under compile (modules/_opaque.py)
            return _reduce_backward(ctx, grads[0]), None, None, None, None, None


class _QMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spec, dims, keepdim, quant, grad_quant, carrier):
        with torch._C.DisableTorchFunction():
            return _reduce_forward(ctx, x, spec, dims, keepdim, quant, grad_quant, carrier, True)

    @staticmethod
    def backward(ctx, *grads):
        with _opaque_backward():
            return _reduce_backward(ctx, grads[0]), None, None, None, None, None, None


def _checked_input(x: Any, op: str) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{op} takes a tensor, got {type(x).__name__}")
    if not x.is_floating_point():
        raise TypeError(
            f"{op} rounds a float tensor's sum, got {x.dtype}: convert it first, or leave "
            "every slot None for torch's own"
        )


def _qsum(
    x: torch.Tensor,
    dims: tuple[int, ...],
    keepdim: bool,
    spec: _GemmSpec | None,
    grad_quant: Any,
    carrier: torch.dtype | None,
) -> torch.Tensor:
    """:func:`qsum` with its arguments resolved, for the ops built on it."""
    if spec is None and grad_quant is None:
        return _plain_sum(x, dims, keepdim)
    return _QSum.apply(x, spec, dims, keepdim, grad_quant, carrier)


def _qmean(
    x: torch.Tensor,
    dims: tuple[int, ...],
    keepdim: bool,
    spec: _GemmSpec | None,
    quant: Any,
    grad_quant: Any,
    carrier: torch.dtype | None,
) -> torch.Tensor:
    """:func:`qmean` with its arguments resolved, for the ops built on it."""
    if spec is None and quant is None and grad_quant is None:
        return torch.mean(x, dims, keepdim)
    return _QMean.apply(x, spec, dims, keepdim, quant, grad_quant, carrier)


def qsum(
    x: torch.Tensor,
    dim: Dims = None,
    mac: Any = None,
    *,
    keepdim: bool = False,
    grad_quant: Any = None,
) -> torch.Tensor:
    """``torch.sum(x, dim, keepdim)``, each step of the sum rounded as ``mac`` says.

    The sum runs on the GEMM kernels as ``ones · x``, in the order of the
    summed elements (row-major over the summed dimensions): under a
    ``FusedMac(F)`` each step rounds ``s + x_k`` once in the carrier and then
    to ``F``; a ``SplitMac(M, A)`` rounds each ``x_k`` to ``M`` first and each
    sum to ``A``. The mac's accumulate algorithm (KAHAN, BLOCK, TREE), its
    rounding mode, its carrier and its ``carrier_rounding`` apply as in a
    GEMM. The gradient is the incoming gradient broadcast back over the summed
    dimensions, rounded by ``grad_quant``. With both slots ``None`` this is
    ``torch.sum``.

    Args:
        x (Tensor): a float tensor.
        dim (int or Sequence[int], optional): the dimensions to sum; ``None``
            (or an empty sequence) sums every element. Default: ``None``
        mac: a :class:`~mptorch.quant.FusedMac` or a
            :class:`~mptorch.quant.SplitMac` of one format per slot, or a
            format, which means ``FusedMac(fmt)``. Default: ``None``, torch's
            sum
        keepdim (bool): keep the summed dimensions, of size 1.
            Default: ``False``
        grad_quant: rounds the gradient: a :class:`~mptorch.quant.Quant`
            (its own ``carrier`` unset, or the mac's) or another callable.
            Default: ``None``

    Raises:
        TypeError: for a tensor that is not a float one under a mac, a block
            format, or a slot that is neither a ``Quant`` nor a callable (a
            bare format included: pass ``Quant(fmt)``).
        ValueError: for a mac that holds a palette.

    Example::

        >>> import torch
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import qsum
        >>> qsum(torch.tensor([1.0, 0.0625, 0.0625]), mac=BinaryK(8, 4))
        tensor(1.)
    """
    with torch._C.DisableTorchFunction():  # the op, not the calls it makes
        mac = _checked_mac(mac, "mac")
        carrier = None if mac is None else mac.carrier
        grad_quant = _checked_keyword_slot(grad_quant, carrier, "grad_quant")
        if mac is None and grad_quant is None:
            return torch.sum(x, dim=_as_torch_dim(dim), keepdim=keepdim)
        _checked_input(x, "qsum")
        spec = None if mac is None else spec_for_mac(mac)
        return _qsum(x, _reduced(x, dim), keepdim, spec, grad_quant, carrier)


def qmean(
    x: torch.Tensor,
    dim: Dims = None,
    mac: Any = None,
    *,
    keepdim: bool = False,
    quant: Any = None,
    grad_quant: Any = None,
) -> torch.Tensor:
    """``torch.mean(x, dim, keepdim)``: :func:`qsum`'s sum, then one division by the count.

    The division is computed in the carrier and rounded by ``quant``: a
    single rounding, where multiplying by ``1 / n`` would round ``1 / n``
    first. The gradient is the incoming gradient divided by the count and
    broadcast back, rounded once by ``grad_quant``. With every slot ``None``
    this is ``torch.mean``.

    Args:
        x (Tensor): a float tensor.
        dim (int or Sequence[int], optional): as for :func:`qsum`.
            Default: ``None``
        mac: the sum's, as for :func:`qsum`. Default: ``None``
        keepdim (bool): as for :func:`qsum`. Default: ``False``
        quant: rounds the quotient. Default: ``None``, torch's division in
            ``x``'s dtype
        grad_quant: rounds the gradient. Default: ``None``

    Raises:
        TypeError, ValueError: as for :func:`qsum`.
    """
    with torch._C.DisableTorchFunction():
        mac = _checked_mac(mac, "mac")
        carrier = None if mac is None else mac.carrier
        quant = _checked_keyword_slot(quant, carrier, "quant")
        grad_quant = _checked_keyword_slot(grad_quant, carrier, "grad_quant")
        if mac is None and quant is None and grad_quant is None:
            return torch.mean(x, dim=_as_torch_dim(dim), keepdim=keepdim)
        _checked_input(x, "qmean")
        spec = None if mac is None else spec_for_mac(mac)
        return _qmean(x, _reduced(x, dim), keepdim, spec, quant, grad_quant, carrier)


# --- the formats -----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class ReductionFormats:
    """The formats of a sum and a mean: the sum's mac, the mean's division, the gradient.

    It is the value of a recipe's ``reduction`` kind, which
    :class:`~mptorch.quant.simulate` gives a model's ``torch.sum`` and
    ``torch.mean`` calls (:func:`qsum`, :func:`qmean`). A slot left ``None``
    does not round; with every slot ``None`` they are torch's.

    Args:
        mac: each step of the sum: a :class:`~mptorch.quant.FusedMac`, a
            :class:`~mptorch.quant.SplitMac`, or a format, which means
            ``FusedMac(fmt)``. Its carrier is the reduction's. Default: ``None``
        div: rounds a mean's quotient. A format means ``Quant(fmt)``.
            Default: ``None``
        grad: rounds the gradient. A format means ``Quant(fmt)``.
            Default: ``None``

    Raises:
        TypeError: for a block format, a module in a slot, or a slot that is
            not callable.
        ValueError: for a palette, or a ``Quant`` whose own carrier is set
            and differs from the mac's.
    """

    mac: Any = None
    div: Any = None
    grad: Any = None
    _spec: _GemmSpec | None = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        mac = _checked_mac(self.mac, "mac")
        object.__setattr__(self, "mac", mac)
        carrier = None if mac is None else _checked_carrier(mac.carrier)
        for name in ("div", "grad"):
            object.__setattr__(self, name, _checked_quant_slot(getattr(self, name), carrier, name))
        # resolved once, so that a call reads no cache (Dynamo warns about those)
        object.__setattr__(self, "_spec", None if mac is None else spec_for_mac(mac))

    @classmethod
    def uniform(cls, mac: Any, quant: Any, *, grad: Any = _SAME) -> "ReductionFormats":
        """One mac for the sum, and one rounding for the quotient and the
        gradient; ``grad`` replaces the gradient's (``None`` leaves it
        unrounded)."""
        return cls(mac=mac, div=quant, grad=quant if grad is _SAME else grad)

    @property
    def carrier(self) -> torch.dtype | None:
        """The mac's carrier, where every step computes and rounds."""
        return None if self.mac is None else self.mac.carrier

    @property
    def _rounds(self) -> bool:
        return self.mac is not None or self.div is not None or self.grad is not None

    def label(self) -> str:
        """A one-line summary, for reports."""
        from .recipe import _mac_label, _quant_label  # imported here: recipe imports this module

        parts = [] if self.mac is None else [f"mac {_mac_label(self.mac)}"]
        parts += [
            f"{n} {_quant_label(q)}"
            for n, q in (("div", self.div), ("grad", self.grad))
            if q is not None
        ]
        return ", ".join(parts) or "torch"
