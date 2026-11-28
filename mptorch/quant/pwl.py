"""Piecewise-linear tables: lookup-table activations, and approximate inverse square roots.

A :class:`PWLTable` holds the entries of a continuous piecewise-linear
function on uniform knots over ``[lo, hi]``: ``segments + 1`` values, one at
each knot. Evaluating it in the carrier takes the knot below ``x`` and
interpolates to the next. :meth:`PWLTable.fit` computes the entries from a
function in one of three ways:

* ``"chord"``: the function's value at each knot.
* ``"balanced"``: LoFloat's correction (its ``init_lut``, which it calls
  minimax). Each entry is moved by the mean of the errors at the midpoints
  of the segments on either side of it, halved, so that the chord error is
  shared between the segments.
* ``"minimax"``: the entries whose largest error over a grid of points in
  each segment is the smallest possible. The fit is exact on the grid: for
  a bound ``E``, each point constrains its segment's two entries to a strip,
  the feasible values of each entry form an interval carried from knot to
  knot, and bisection on ``E`` finds the smallest bound that leaves every
  interval nonempty (``_minimax_entries``).

:func:`qpwl` evaluates a table as an op, and :class:`~mptorch.quant.ActivationFormats`'
``table`` makes an activation evaluate one. Their backward is the analytic
derivative of the function the table approximates, since a table's own
slope is piecewise constant and a poor gradient. :meth:`PWLTable.rsqrt` and
:class:`FISR` are the two approximate inverse square roots a
:class:`~mptorch.quant.NormFormats` can take as its ``inverse``.

``docs/source/kernels/pointwise.rst`` and ``kernels/reductions.rst`` are
the user's account.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from . import pointwise
from .ops._checks import _checked_carrier
from .ops._spec import _round_result, _stored

__all__ = ["FISR", "PWLTable", "gelu_table", "qpwl", "silu_table"]

_FITS = ("minimax", "balanced", "chord")
_ERRORS = ("absolute", "relative")
_TAILS = ("extend", "clamp")

# The bisection's stopping point, relative to the bound it brackets.
_BISECTION_TOLERANCE = 1e-13


# --- the fits ------------------------------------------------------------------------------


def _knots(lo: float, hi: float, segments: int) -> torch.Tensor:
    """The knots ``lo + k (hi - lo) / segments``, in float64, as evaluation reads them."""
    k = torch.arange(segments + 1, dtype=torch.float64)
    return lo + k * ((hi - lo) / segments)


def _fit_grid(fn, lo: float, hi: float, segments: int, grid: int):
    """The fit's points: ``grid`` in each segment at ``t = j / grid`` (the
    left knot included), and the last knot. Returns ``(t, y, y_end)``, ``t``
    and ``y`` of shape ``[segments, grid]``."""
    t = (torch.arange(grid, dtype=torch.float64) / grid).expand(segments, grid).contiguous()
    k = torch.arange(segments, dtype=torch.float64)[:, None]
    x = lo + (k + t) * ((hi - lo) / segments)
    y = fn(x)
    y_end = float(fn(torch.tensor([hi], dtype=torch.float64))[0])
    return t, y, y_end


def _grid_error(t, y, y_end, entries: torch.Tensor, relative: bool) -> float:
    """The largest error of ``entries`` on the fit's points."""
    p = (1 - t) * entries[:-1, None] + t * entries[1:, None]
    r, r_end = y - p, y_end - float(entries[-1])
    if relative:
        r, r_end = r / y.abs(), r_end / abs(y_end)
    return max(float(r.abs().max()), abs(r_end))


def _chord(fn, lo, hi, segments) -> torch.Tensor:
    return fn(_knots(lo, hi, segments))


def _balanced(fn, lo, hi, segments) -> torch.Tensor:
    """LoFloat's correction: each entry moved by its segments' midpoint errors."""
    knots = _knots(lo, hi, segments)
    at = fn(knots)
    delta = fn((knots[:-1] + knots[1:]) / 2) - 0.5 * (at[:-1] + at[1:])
    shift = torch.empty_like(at)
    shift[0], shift[-1] = delta[0] / 2, delta[-1] / 2
    shift[1:-1] = (delta[:-1] + delta[1:]) / 4
    return at + shift


def _feasible(t, y, y_end, bound: float, relative: bool, entries: bool = False):
    """Whether some entries keep every point's error at most ``bound``, and,
    with ``entries``, one such set of entries (or ``None`` if there is none).

    A point at ``t`` in segment ``k`` asks ``|y - ((1 - t) a + t b)| <= eps``
    of its segment's entries ``a`` and ``b``. The point at ``t = 0`` bounds
    ``a`` alone; each other one bounds ``b`` between two lines that fall as
    ``a`` grows. So the values of ``b`` reachable from an interval of ``a``
    form an interval, from the largest lower line at the interval's right
    end to the smallest upper line at its left end, over the values of ``a``
    where those two lines do not cross, which the pairs of points give.
    Carrying the interval from knot to knot decides the bound exactly, and
    walking back from the last knot picks entries inside it."""
    segments, grid = y.shape
    eps = bound * y.abs() if relative else torch.full_like(y, bound)
    eps_end = bound * abs(y_end) if relative else bound
    low, high = y - eps, y + eps
    tt, lo_b, hi_b = t[:, 1:], low[:, 1:], high[:, 1:]
    inv = 1 / tt
    # every pair (i, j) of a segment's points asks L_i(a) <= U_j(a):
    # (1/t_j - 1/t_i) a <= high_j/t_j - low_i/t_i
    c = inv[:, None, :] - inv[:, :, None]
    d = (hi_b * inv)[:, None, :] - (lo_b * inv)[:, :, None]
    if bool((d[c == 0] < 0).any()):
        return None if entries else False
    inf = torch.full_like(c, math.inf)
    a_hi = torch.minimum(torch.where(c > 0, d / c, inf).amin((1, 2)), high[:, 0])
    a_lo = torch.maximum(torch.where(c < 0, d / c, -inf).amax((1, 2)), low[:, 0])
    slope = ((1 - tt) * inv).tolist()
    lower, upper = (lo_b * inv).tolist(), (hi_b * inv).tolist()
    a_lo_l, a_hi_l = a_lo.tolist(), a_hi.tolist()
    span = (-math.inf, math.inf)
    spans = []
    for k in range(segments):
        left, right = max(span[0], a_lo_l[k]), min(span[1], a_hi_l[k])
        if left > right:
            return None if entries else False
        spans.append((left, right))
        s_k, l_k, u_k = slope[k], lower[k], upper[k]
        span = (
            max([l_k[j] - s_k[j] * right for j in range(grid - 1)]),
            min([u_k[j] - s_k[j] * left for j in range(grid - 1)]),
        )
    last = (max(span[0], y_end - eps_end), min(span[1], y_end + eps_end))
    if last[0] > last[1]:
        return None if entries else False
    if not entries:
        return True
    e = [0.0] * (segments + 1)
    e[segments] = 0.5 * (last[0] + last[1])
    lo_l, hi_l, t_l = lo_b.tolist(), hi_b.tolist(), tt.tolist()
    for k in range(segments - 1, -1, -1):
        b = e[k + 1]
        left = max(
            [spans[k][0]]
            + [(lo_l[k][j] - t_l[k][j] * b) / (1 - t_l[k][j]) for j in range(grid - 1)]
        )
        right = min(
            [spans[k][1]]
            + [(hi_l[k][j] - t_l[k][j] * b) / (1 - t_l[k][j]) for j in range(grid - 1)]
        )
        e[k] = 0.5 * (left + right) if left <= right else left
    return torch.tensor(e, dtype=torch.float64)


def _fitted(fn, lo, hi, segments, fit, error, grid, entry_quant) -> tuple[torch.Tensor, float]:
    """A fit's entries, rounded by ``entry_quant``, and their largest error
    on the fit's grid."""
    if type(segments) is not int or segments < 1:
        raise ValueError(f"segments is an int of at least 1, got {segments!r}")
    if type(grid) is not int or grid < 2:
        raise ValueError(f"grid is an int of at least 2, got {grid!r}")
    if fit not in _FITS:
        raise ValueError(f"fit is one of {_FITS}, got {fit!r}")
    if error not in _ERRORS:
        raise ValueError(f"error is one of {_ERRORS}, got {error!r}")
    relative = error == "relative"
    if fit == "chord":
        entries = _chord(fn, lo, hi, segments)
    elif fit == "balanced":
        entries = _balanced(fn, lo, hi, segments)
    else:
        entries, _ = _minimax_entries(fn, lo, hi, segments, grid, relative)
    if entry_quant is not None:
        entries = entry_quant(entries)
    t, y, y_end = _fit_grid(fn, lo, hi, segments, grid)
    return entries, _grid_error(t, y, y_end, entries, relative)


def _minimax_entries(fn, lo, hi, segments, grid, relative) -> tuple[torch.Tensor, float]:
    """The minimax entries on the fit's grid, and the bound bisection proved
    no entries can beat (the largest infeasible one)."""
    t, y, y_end = _fit_grid(fn, lo, hi, segments, grid)
    if relative and (bool((y == 0).any()) or y_end == 0):
        raise ValueError(
            "a relative-error fit divides by the function's values, and this function is 0 at "
            "a fit point: fit its absolute error, or move [lo, hi] off its zeros"
        )
    below, above = 0.0, _grid_error(t, y, y_end, _chord(fn, lo, hi, segments), relative)
    above *= 1 + 1e-12  # the chord itself is feasible, a hair above its error
    while above - below > _BISECTION_TOLERANCE * above:
        middle = 0.5 * (below + above)
        if _feasible(t, y, y_end, middle, relative):
            above = middle
        else:
            below = middle
    e = _feasible(t, y, y_end, above, relative, entries=True)
    assert isinstance(e, torch.Tensor)
    return e, below


# --- the table -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PWLTable:
    """A continuous piecewise-linear function on uniform knots over ``[lo, hi]``.

    Build one with :meth:`fit`, :meth:`rsqrt`, :func:`silu_table` or
    :func:`gelu_table`, or from entries you already have. ``entries`` holds
    its value at each of the ``segments + 1`` knots ``lo + k (hi - lo) /
    segments``. In the carrier, a value ``x`` is evaluated as

    * ``u = (x - lo) * (segments / (hi - lo))``,
    * ``i = clamp(floor(u), 0, segments - 1)`` and ``f = u - i``,
    * ``e[i] + f * (e[i + 1] - e[i])``,

    each operation rounded by the carrier. Outside ``[lo, hi]``, ``tails``
    says what it returns: ``"extend"`` the end segments' lines, ``"clamp"``
    the end entries, or a callable, applied to ``x`` there (the activation
    presets use ``torch.relu``, as LoFloat does). A table of 1/sqrt from
    :meth:`rsqrt` first reduces ``v`` to ``m * 4**k`` with ``m`` in
    ``[1, 4)``, both exactly, and returns ``T(m) * 2**-k``; ``v`` must be a
    positive normal number.

    Args:
        lo, hi (float): the range of the knots, ``lo < hi``.
        entries (tuple of float): the values at the knots, at least two.
        tails: ``"extend"``, ``"clamp"`` or a callable. Default: ``"extend"``
        name (str, optional): the activation the table approximates
            (``"silu"``, ``"gelu"``), which an activation given another
            activation's table refuses. Default: ``None``
        reduction (str, optional): ``"rsqrt"`` for a table of 1/sqrt over
            ``[1, 4)``. Default: ``None``
        fn (callable, optional): the function the entries were fitted to,
            on float64 tensors, whose autograd derivative :func:`qpwl`
            takes by default. Default: ``None``
        derivative (callable, optional): ``derivative(g, x)``, the input's
            gradient given the output's. Default: ``None``
        fit_error (float, optional): the largest error on the fit's grid, as
            :meth:`fit` measured it. Default: ``None``

    Raises:
        TypeError, ValueError: for a range, entries, tails or reduction that
            is not one of the above.
    """

    lo: float
    hi: float
    entries: tuple[float, ...]
    tails: Any = "extend"
    name: str | None = None
    reduction: str | None = None
    fn: Callable | None = field(default=None, compare=False, repr=False)
    derivative: Callable | None = field(default=None, compare=False, repr=False)
    fit_error: float | None = field(default=None, compare=False)
    _cpu: dict = field(init=False, compare=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        lo, hi = float(self.lo), float(self.hi)
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            raise ValueError(f"a PWLTable's range is finite with lo < hi, got [{lo}, {hi}]")
        entries = tuple(float(e) for e in self.entries)
        if len(entries) < 2 or not all(math.isfinite(e) for e in entries):
            raise ValueError("a PWLTable has at least two entries, each finite")
        if not (self.tails in _TAILS or callable(self.tails)):
            raise TypeError(f"tails is 'extend', 'clamp' or a callable, got {self.tails!r}")
        if self.reduction not in (None, "rsqrt"):
            raise ValueError(f"reduction is None or 'rsqrt', got {self.reduction!r}")
        if self.reduction == "rsqrt" and (lo, hi) != (1.0, 4.0):
            raise ValueError(
                "a table of 1/sqrt covers [1, 4), the range its reduction maps v onto, "
                f"got [{lo}, {hi}]"
            )
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "entries", entries)
        pin = torch.cuda.is_available()
        for dtype in (torch.float32, torch.float64):
            t = torch.tensor(entries, dtype=dtype)
            self._cpu[dtype] = t.pin_memory() if pin else t

    @property
    def segments(self) -> int:
        """The number of segments, one fewer than the entries."""
        return len(self.entries) - 1

    @classmethod
    def fit(
        cls,
        fn: Callable[[torch.Tensor], torch.Tensor],
        lo: float,
        hi: float,
        segments: int,
        *,
        fit: str = "minimax",
        error: str = "absolute",
        grid: int = 64,
        entry_quant: Callable | None = None,
        tails: Any = "extend",
        name: str | None = None,
        derivative: Callable | None = None,
    ) -> PWLTable:
        """A table of ``fn`` on ``segments`` uniform segments of ``[lo, hi]``.

        Args:
            fn: the function, on float64 tensors.
            lo, hi (float): the range.
            segments (int): the number of segments, at least 1.
            fit (str): ``"minimax"``, ``"balanced"`` or ``"chord"`` (the
                module's docstring). Default: ``"minimax"``
            error (str): ``"absolute"`` or ``"relative"``, the error the
                minimax fit holds and :attr:`fit_error` measures.
                Default: ``"absolute"``
            grid (int): the fit's points in each segment, the left knot
                included. Default: ``64``
            entry_quant (callable, optional): rounds the fitted entries, as a
                float64 tensor. A minimax table so rounded is no longer
                minimax. Default: ``None``
            tails, name, derivative: as for :class:`PWLTable`.

        Returns:
            PWLTable: the table, its ``fit_error`` the largest error of its
            final entries on the fit's grid.
        """
        lo, hi = float(lo), float(hi)
        entries, measured = _fitted(fn, lo, hi, segments, fit, error, grid, entry_quant)
        return cls(
            lo,
            hi,
            tuple(entries.tolist()),
            tails,
            name,
            fn=fn,
            derivative=derivative,
            fit_error=measured,
        )

    @classmethod
    def rsqrt(
        cls,
        segments: int = 32,
        *,
        fit: str = "minimax",
        error: str = "relative",
        grid: int = 64,
        entry_quant: Callable | None = None,
    ) -> PWLTable:
        """A table of ``1 / sqrt(m)`` over ``[1, 4)``, which a
        :class:`~mptorch.quant.NormFormats`' ``inverse`` evaluates for any
        positive normal ``v`` by ``v = m * 4**k``, both factors exact."""
        entries, measured = _fitted(_inv_sqrt, 1.0, 4.0, segments, fit, error, grid, entry_quant)
        return cls(
            1.0,
            4.0,
            tuple(entries.tolist()),
            "extend",
            "rsqrt",
            "rsqrt",
            fn=_inv_sqrt,
            fit_error=measured,
        )

    def max_error(
        self, fn: Callable | None = None, *, points: int = 1024, relative: bool = False
    ) -> float:
        """The table's largest error against ``fn`` (its own ``fn`` by
        default) on ``points`` points in each segment, in float64."""
        fn = self.fn if fn is None else fn
        if fn is None:
            raise ValueError("this table has no fn: pass the function to measure it against")
        t, y, y_end = _fit_grid(fn, self.lo, self.hi, self.segments, points)
        return _grid_error(t, y, y_end, torch.tensor(self.entries, dtype=torch.float64), relative)

    def label(self) -> str:
        """A one-line summary, for reports."""
        what = self.name or "table"
        return f"{what} PWL[{self.segments} on [{self.lo:g}, {self.hi:g}]]"

    def _entries(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        t = self._cpu[dtype]
        return t if device.type == "cpu" else t.to(device, non_blocking=True)


def _inv_sqrt(v: torch.Tensor) -> torch.Tensor:
    return 1 / torch.sqrt(v)


# --- evaluation ----------------------------------------------------------------------------


def _interpolate(table: PWLTable, e: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """The table at ``x`` (in its carrier), the tails applied."""
    n, lo, hi = table.segments, table.lo, table.hi
    xi = x.clamp(lo, hi) if table.tails == "clamp" else x
    u = (xi - lo) * (n / (hi - lo))
    i = torch.floor(u.masked_fill(u.isnan(), 0.0)).clamp(0, n - 1)
    f = u - i
    i = i.long()
    e0, e1 = e[i], e[i + 1]
    y = e0 + f * (e1 - e0)
    if callable(table.tails):
        y = torch.where((x < lo) | (x > hi), table.tails(x), y)
    return y


def _power_of_two(p: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``2 ** p`` for integer ``p`` in the normal range, built on the word."""
    if dtype is torch.float32:
        return ((p + 127).to(torch.int32) << 23).view(torch.float32)
    return ((p + 1023).to(torch.int64) << 52).view(torch.float64)


def _exponent(v: torch.Tensor) -> torch.Tensor:
    """The unbiased exponent of each positive normal ``v``, from its word."""
    if v.dtype is torch.float32:
        return ((v.view(torch.int32) >> 23) & 0xFF) - 127
    return ((v.view(torch.int64) >> 52) & 0x7FF) - 1023


def _rsqrt_reduced(table: PWLTable, e: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``1 / sqrt(v)`` by the table over ``[1, 4)``: ``v = m * 4**k`` with
    ``k = floor(exponent / 2)``, ``T(m) * 2**-k``; both scalings exact."""
    k = _exponent(v) >> 1
    m = v * _power_of_two(-2 * k, v.dtype)
    return _interpolate(table, e, m) * _power_of_two(-k, v.dtype)


def _evaluate(table: PWLTable, x: torch.Tensor) -> torch.Tensor:
    """The table at ``x``, a float32 or float64 tensor in its carrier."""
    e = table._entries(x.device, x.dtype)
    if table.reduction == "rsqrt":
        return _rsqrt_reduced(table, e, x)
    return _interpolate(table, e, x)


# Each family's fused table op (``csrc/common/pointwise.h``'s ``PwlStep``), by
# the name of its flat quantizer: the evaluation above and the slot's rounding
# in one pass, word for word the two passes, stochastic rounding included.
_FUSED_PWL: dict[str, Callable[..., torch.Tensor]] = {
    "binaryK_quantize": torch.ops.mptorch.binaryK_pwl.default,
    "superfp_quantize": torch.ops.mptorch.superfp_pwl.default,
    "fixedpoint_quantize": torch.ops.mptorch.fixedpoint_pwl.default,
}


def _tails_code(tails: Any) -> int | None:
    """The fused op's code for a table's tails, or ``None`` for a callable
    other than ``torch.relu`` (the activation presets' tails)."""
    if tails is torch.relu:
        return 2
    if isinstance(tails, str):
        return _TAILS.index(tails)
    return None


def _rounded(
    table: PWLTable, x: torch.Tensor, slot: Any, carrier: torch.dtype | None, dtype: torch.dtype
) -> torch.Tensor:
    """The table at ``x``, computed in the carrier, rounded by ``slot`` and
    stored in ``dtype``: one fused op where the slot is a ``Quant`` of a
    binaryK, superfp or fixed-point format on a device that has the op's
    kernels (as for ``pointwise._fused_op``) and the tails are ones it
    evaluates; otherwise ``_evaluate`` and the slot."""
    c = pointwise._carrier_dtype(carrier, x.dtype)
    code = _tails_code(table.tails)
    spec = slot._spec if isinstance(slot, pointwise.Quant) else None
    fused = None
    if code is not None and spec is not None and x.device.type in pointwise._FUSED_DEVICES:
        fused = _FUSED_PWL.get(spec.name)
    if fused is None:
        y = _evaluate(table, x.to(c))
        return _stored(y, dtype) if slot is None else _round_result(slot, y, dtype)
    assert spec is not None  # a fused op was found for it
    wide = c is torch.float64
    pointwise._report_fused(spec, wide, dtype)
    xc = x.double() if wide and x.dtype is not torch.float64 else x
    entries = table._entries(x.device, c)
    rsqrt = table.reduction == "rsqrt"
    return _stored(fused(xc, entries, table.lo, table.hi, code, rsqrt, *spec.args), dtype)


# --- FISR ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class FISR:
    """The fast inverse square root: ``0x5F3759DF`` and Newton steps.

    The first guess is the bit trick on the carrier's word: ``0x5F3759DF -
    (i >> 1)`` in binary32, and ``0x5FE6EB50C7B537A9 - (i >> 1)`` in binary64,
    whose errors are the same (3.4% before any step, 1.75e-3 after one and
    4.7e-6 after two, measured over ``[1, 4)``). Each step is ``y * (1.5 -
    (0.5 * v) * y * y)``, computed in the carrier and rounded by the norm's
    ``rsqrt`` slot, as an iterative unit of that precision would; with no
    steps the slot rounds the guess.

    Args:
        steps (int): the Newton steps. Default: ``1``, the classic, so that
            its error shows

    Raises:
        TypeError: for ``steps`` that is not an ``int``.
        ValueError: for ``steps`` below 0.
    """

    steps: int = 1

    def __post_init__(self) -> None:
        if type(self.steps) is not int:
            raise TypeError(f"FISR's steps is an int, got {self.steps!r}")
        if self.steps < 0:
            raise ValueError(f"FISR's steps is at least 0, got {self.steps}")

    def label(self) -> str:
        return f"FISR({self.steps})"


_FISR_MAGIC = {
    torch.float32: (0x5F3759DF, torch.int32),
    torch.float64: (0x5FE6EB50C7B537A9, torch.int64),
}


def _round(slot: Any, t: torch.Tensor) -> torch.Tensor:
    return t if slot is None else _round_result(slot, t, t.dtype)


def _fisr(v: torch.Tensor, steps: int, slot: Any) -> torch.Tensor:
    """``1 / sqrt(v)`` by FISR in ``v``'s carrier, each step rounded by ``slot``."""
    magic, itype = _FISR_MAGIC[v.dtype]
    y = (magic - (v.view(itype) >> 1)).view(v.dtype)
    if steps == 0:
        return _round(slot, y)
    half = v * 0.5
    for _ in range(steps):
        y = _round(slot, y * (1.5 - half * y * y))
    return y


# --- the op --------------------------------------------------------------------------------


def _derivative(table: PWLTable, derivative: Callable | None, g, x) -> torch.Tensor:
    """The input's gradient: ``derivative(g, x)``, the table's own, or
    autograd's of the table's ``fn``."""
    d = derivative if derivative is not None else table.derivative
    if d is not None:
        return d(g, x)
    if table.fn is None:
        raise ValueError(
            "qpwl's backward takes the derivative of the function the table approximates, and "
            "this table has neither fn nor derivative: pass derivative=, or build the table "
            "with PWLTable.fit"
        )
    with torch.enable_grad():
        xr = x.detach().requires_grad_()
        (gx,) = torch.autograd.grad(table.fn(xr), xr, g)
    return gx


class _QPWL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, table, quant, grad_quant, derivative, carrier):
        with torch._C.DisableTorchFunction():  # hidden from torch function modes
            ctx.save_for_backward(x)
            ctx.table, ctx.grad_quant, ctx.derivative, ctx.carrier = (
                table,
                grad_quant,
                derivative,
                carrier,
            )
            return _rounded(table, x, quant, carrier, x.dtype)

    @staticmethod
    def backward(ctx, *grads):
        from .modules._opaque import _opaque_backward  # imported here: modules imports this

        with _opaque_backward():
            (x,) = ctx.saved_tensors
            g = grads[0]
            c = pointwise._carrier_dtype(ctx.carrier, g.dtype)
            gx = _derivative(ctx.table, ctx.derivative, g.to(c), x.to(c))
            q = ctx.grad_quant
            return (
                _stored(gx, g.dtype) if q is None else _round_result(q, gx, g.dtype),
                None,
                None,
                None,
                None,
                None,
            )


def qpwl(
    x: torch.Tensor,
    table: PWLTable,
    *,
    quant: Any = None,
    grad_quant: Any = None,
    derivative: Callable | None = None,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """``table`` evaluated at ``x`` in the carrier, rounded by ``quant``.

    The gradient is ``derivative(g, x)`` (by default the table's own, or the
    derivative autograd gives of its ``fn``), computed in the carrier and
    rounded by ``grad_quant``: the function's, not the table's.

    Args:
        x (Tensor): a float tensor.
        table (PWLTable): the table.
        quant, grad_quant: a :class:`~mptorch.quant.Quant`, a callable or
            ``None``, as for :func:`~mptorch.quant.qmul`. Default: ``None``
        derivative (callable, optional): ``derivative(g, x)``.
            Default: ``None``
        carrier (torch.dtype, optional): as for :func:`~mptorch.quant.qmul`.
            Default: ``None``

    Returns:
        Tensor: of ``x``'s dtype and shape.

    Raises:
        TypeError: for a ``table`` that is not a ``PWLTable``, or a format in a
            slot.
        ValueError: for a table of 1/sqrt, which is a norm's ``inverse``.

    Example::

        >>> import torch
        >>> from mptorch.quant import qpwl, silu_table
        >>> y = qpwl(torch.linspace(-4, 4, 9), silu_table())
    """
    with torch._C.DisableTorchFunction():
        if not isinstance(table, PWLTable):
            raise TypeError(f"qpwl's table is a PWLTable, got {type(table).__name__}")
        if table.reduction is not None:
            raise ValueError(
                "this is a table of 1/sqrt, which a NormFormats takes as its inverse: an "
                "activation's table has no reduction"
            )
        _checked_carrier(carrier)
        for name, slot in (("quant", quant), ("grad_quant", grad_quant)):
            pointwise._checked_slot(slot, carrier, name)
        return _QPWL.apply(x, table, quant, grad_quant, derivative, carrier)


# --- the presets ---------------------------------------------------------------------------


def _silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (1 + torch.erf(x * math.sqrt(0.5)))


def _silu_backward(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.silu_backward(g, x)


def _gelu_backward(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.gelu_backward(g, x)


def silu_table(
    segments: int = 16,
    lo: float = -8.0,
    hi: float = 8.0,
    *,
    fit: str = "minimax",
    entry_quant: Callable | None = None,
) -> PWLTable:
    """A table of SiLU, ``x * sigmoid(x)``, ReLU outside ``[lo, hi]``
    (LoFloat's ``PWLSiLU`` is ``silu_table(fit="balanced")``)."""
    return PWLTable.fit(
        _silu,
        lo,
        hi,
        segments,
        fit=fit,
        entry_quant=entry_quant,
        tails=torch.relu,
        name="silu",
        derivative=_silu_backward,
    )


def gelu_table(
    segments: int = 16,
    lo: float = -8.0,
    hi: float = 8.0,
    *,
    fit: str = "minimax",
    entry_quant: Callable | None = None,
) -> PWLTable:
    """A table of GELU, ``0.5 * x * (1 + erf(x / sqrt(2)))``, ReLU outside
    ``[lo, hi]``."""
    return PWLTable.fit(
        _gelu,
        lo,
        hi,
        segments,
        fit=fit,
        entry_quant=entry_quant,
        tails=torch.relu,
        name="gelu",
        derivative=_gelu_backward,
    )
