"""Numeric event counts: how a format's casts went.

Each event is a property of one cast, read from its input ``x`` (a value of
the carrier) and its result ``q`` (a value of the format):

=============  ===============================================================
event          when
=============  ===============================================================
``casts``      every cast: the denominator
``overflow``   ``x`` finite, and its magnitude past the format's largest
               finite value on ``x``'s side of zero, whatever it rounds to (an
               infinity, or the largest value under a saturating mode)
``underflow``  ``x`` finite and nonzero, and ``q`` zero
``subnormal``  ``q`` nonzero and below the format's normal range: a binaryK
               subnormal under ``SUBNORMALS``, a superfp supernormal; never
               under ``NORMALS`` or ``EXTENDED_NORMALS``, nor for fixed point
``inexact``    ``x`` not a NaN, and ``q`` a different value from ``x``
``nan``        ``q`` a NaN, made by this cast or carried into it
=============  ===============================================================

The largest finite magnitude is the same on both sides of zero, except for a
signed fixed-point format that is not symmetric, whose bottom is one step
further out than its top. An unsigned format sends a negative input to zero,
an underflow (and an overflow too, if its magnitude is past the largest
value). The GEMM kernels classify their casts with the same table, on the
words (``csrc/common/event_counts.h``); a quantizer's casts are recounted
here from its input and its rounded output, which is the same thing, since
an elementwise cast sees nothing else.

:class:`~mptorch.quant.simulate` with ``events=True`` counts a model's events
by site, pass and role: each simulated product runs the counted twin of its
GEMM op (``csrc/common/gemm_counted.h``), whose roles are ``mul`` (a split
mac's products), ``acc`` (its sums, or a fused mac's one rounding) and
``outer`` (BLOCK's and TREE's outer format), and each quantizer slot that
holds a :class:`~mptorch.quant.Quant` is recounted, under its slot's name.
The counts stay on the device until a report is made.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NamedTuple

import torch

from mptorch.number import BinaryK, FixedPoint, SubnormalsMode, SuperFP

__all__ = ["EVENTS", "ROLES", "EventCounts", "EventReport", "EventRow"]

EVENTS = ("casts", "overflow", "underflow", "subnormal", "inexact", "nan")
"""The events, in the order of a counts row (``event_counts.h``'s ``Event``)."""

ROLES = ("mul", "acc", "outer")
"""A counted GEMM's roles, the rows of its counts (``event_counts.h``'s ``Role``)."""


class EventCounts(NamedTuple):
    """The six counts of one role's casts (see :mod:`mptorch.quant.events`)."""

    casts: int = 0
    overflow: int = 0
    underflow: int = 0
    subnormal: int = 0
    inexact: int = 0
    nan: int = 0


class _Thresholds(NamedTuple):
    """A format's thresholds as values of the carrier: the largest finite
    magnitude for a positive and for a negative input, and the bottom of the
    normal range (0 when the format has no subnormal results)."""

    pos_max: float
    neg_max: float
    normal_min: float


def _in_carrier(v: float, carrier: torch.dtype) -> float:
    """``v``, a format's value or a power of two, as the carrier holds it:
    the infinity past the carrier's finite range and zero below its
    subnormals, as ``event_counts.h``'s ``largest_finite_word`` and
    ``pow2_word`` give them. Every other such value is exact in the carrier,
    so comparing a tensor with it compares with the format's value."""
    info = torch.finfo(carrier)
    if v > info.max:
        return math.inf
    if 0 < v < info.smallest_normal * info.eps:
        return 0.0
    return v


def _pow2(e: int) -> float:
    """``2**e`` as a Python float: infinite past binary64's range, zero below it."""
    return math.inf if e > 1023 else math.ldexp(1.0, max(e, -1100))


@lru_cache(maxsize=256)
def _thresholds(fmt: BinaryK | SuperFP | FixedPoint, carrier: torch.dtype) -> _Thresholds:
    """The thresholds of ``fmt``'s casts in ``carrier``, from the format's
    own fields, as the kernels build them (``event_counts.h``)."""
    if isinstance(fmt, FixedPoint):
        top = fmt.max_value
        bottom = -fmt.min_value if fmt.is_signed else top
        return _Thresholds(_in_carrier(top, carrier), _in_carrier(bottom, carrier), 0.0)
    try:
        top = fmt.max_finite
    except OverflowError:  # a largest value past binary64's
        top = math.inf
    if isinstance(fmt, BinaryK):
        assert fmt.bias is not None  # set by __post_init__
        floor = _pow2(1 - fmt.bias) if fmt.subnormals is SubnormalsMode.SUBNORMALS else 0.0
    else:
        floor = _pow2((1 << fmt.exp_bits) - fmt.bias - fmt.normal_binades)
    top = _in_carrier(top, carrier)
    return _Thresholds(top, top, _in_carrier(floor, carrier))


def _carrier_of(x: torch.Tensor, carrier: torch.dtype | None) -> torch.dtype:
    """The carrier a cast of ``x`` rounds in: binary64 for a float64 tensor or
    a call naming ``carrier=torch.float64``, binary32 otherwise."""
    if carrier is not None:
        return carrier
    return torch.float64 if x.dtype == torch.float64 else torch.float32


def _recount(
    x: torch.Tensor,
    q: torch.Tensor,
    fmt: BinaryK | SuperFP | FixedPoint,
    carrier: torch.dtype | None = None,
) -> torch.Tensor:
    """The events of the casts ``x -> q`` of a format, as an int64 tensor of
    six counts in :data:`EVENTS`' order, on ``x``'s device, with no wait for
    the device.

    ``x`` is what the quantizer was given and ``q`` what it returned; both
    widen exactly to the carrier, where the table is read. A result rounded
    again when stored, for a format that does not fit its dtype (which
    warns), is counted as stored.
    """
    c = _carrier_of(x, carrier)
    t = _thresholds(fmt, c)
    x = x.detach().to(c).reshape(-1)
    q = q.detach().to(c).reshape(-1)
    ax, aq = x.abs(), q.abs()
    finite = torch.isfinite(x)
    past = torch.where(torch.signbit(x), ax > t.neg_max, ax > t.pos_max)
    masks = torch.stack(
        [
            finite & past,
            finite & (x != 0) & (q == 0),
            (q != 0) & (aq < t.normal_min),
            ~torch.isnan(x) & (q != x),
            torch.isnan(q),
        ]
    )
    casts = torch.full((1,), x.numel(), dtype=torch.int64, device=x.device)
    return torch.cat([casts, masks.sum(dim=1)])


# --- counting a model's sites -----------------------------------------------------

# Above zero while a saved operand's quantizer runs again in backward
# (`modules._saved.restore`): the replay repeats the forward's casts, which
# were counted then.
_replaying: int = 0


class _SiteTally:
    """One simulated site's counts, by (pass, role, format label), each an
    int64 tensor of six on the device, added to without waiting for it; and
    what the site does not count, by (pass, role), with the reason."""

    __slots__ = ("counts", "gaps")

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str, str], torch.Tensor] = {}
        self.gaps: dict[tuple[str, str], str] = {}

    def add(self, pass_: str, role: str, label: str, n: torch.Tensor) -> None:
        key = (pass_, role, label)
        before = self.counts.get(key)
        self.counts[key] = n if before is None else before + n

    def gap(self, pass_: str, role: str, reason: str) -> None:
        self.gaps[(pass_, role)] = reason


class _CountedCall:
    """A resolved GEMM's op, run as its counted twin: ``spec.op`` of a
    counting spec, which `ops._spec._run_gemm` calls with the spec's
    arguments. The counts go to ``tally`` under ``pass_``, a role's row only
    where ``labels`` names its format (a role the mac does not round has
    none). ``conv`` is the counted conv op and its tail, which a counting
    spec's `ops._conv._ConvOp` runs instead."""

    __slots__ = ("op", "tail", "conv", "tally", "pass_", "labels")

    def __init__(
        self,
        op: Callable[..., Any],
        tail: tuple[Any, ...],
        conv: tuple[Callable[..., Any], tuple[Any, ...]] | None,
        tally: _SiteTally,
        pass_: str,
        labels: tuple[str, str, str],
    ):
        self.op, self.tail, self.tally, self.pass_, self.labels = op, tail, tally, pass_, labels
        self.conv = None if conv is None else (_CountedConv(self, conv[0]), conv[1])

    def add(self, n: torch.Tensor) -> None:
        for role, label, row in zip(ROLES, self.labels, n, strict=True):
            if label:
                self.tally.add(self.pass_, role, label, row)

    def __call__(self, a: torch.Tensor, b: torch.Tensor, *args: Any) -> torch.Tensor:
        c, n = self.op(a, b, *args, *self.tail)
        self.add(n)
        return c


class _CountedConv:
    """A counting spec's conv op: the counted conv twin, its counts added as
    its GEMM's are."""

    __slots__ = ("call", "op")

    def __init__(self, call: _CountedCall, op: Callable[..., Any]):
        self.call, self.op = call, op

    def __call__(self, *args: Any) -> torch.Tensor:
        c, n = self.op(*args)
        self.call.add(n)
        return c


class _CountingQuant:
    """A quantizer slot's :class:`~mptorch.quant.Quant`, its casts recounted
    into a site's tally, except while it replays in backward. It is the
    slot's value, and `modules._saved._known` sees through it to the Quant,
    so the layer saves and recomputes as it would without counting."""

    __slots__ = ("quant", "tally", "pass_", "role", "label")

    def __init__(self, quant: Any, tally: _SiteTally, pass_: str, role: str, label: str):
        self.quant, self.tally, self.pass_, self.role, self.label = quant, tally, pass_, role, label

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        q = self.quant(x)
        if not _replaying:
            n = _recount(x, q, self.quant.fmt, self.quant.carrier)
            self.tally.add(self.pass_, self.role, self.label, n)
        return q


class EventRow(NamedTuple):
    """One row of an :class:`EventReport`: a site's casts of one format in
    one pass and role."""

    module: str
    kind: str
    site: str | None
    pass_name: str
    role: str
    format: str
    counts: EventCounts


class EventReport:
    """A model's numeric events, by site, pass and role (``simulate(...,
    events=True)``'s :meth:`~mptorch.quant.simulate.events`).

    ``rows`` holds the counts, in the order the sites were made; ``not_counted``
    the (module, kind, site, pass, role, reason) of each part of a simulated
    site whose casts are not counted, a palette mac or a quantizer that is
    not a ``Quant`` for instance. ``str()`` is a table.
    """

    def __init__(self, rows: tuple[EventRow, ...], not_counted: tuple[tuple, ...]):
        self.rows = rows
        self.not_counted = not_counted

    def by_format(self) -> dict[str, EventCounts]:
        """The counts summed over every site, pass and role, by format label."""
        out: dict[str, list[int]] = {}
        for row in self.rows:
            acc = out.setdefault(row.format, [0] * len(EVENTS))
            for i, v in enumerate(row.counts):
                acc[i] += v
        return {k: EventCounts(*v) for k, v in out.items()}

    def __str__(self) -> str:
        head = ("module", "kind", "pass", "role", "format", *EVENTS)
        body = [
            (
                (r.module or "(model)") + ("" if r.site is None else f"[{r.site}]"),
                r.kind,
                r.pass_name,
                r.role,
                r.format,
                *(str(v) for v in r.counts),
            )
            for r in self.rows
        ]
        widths = [max(len(x) for x in col) for col in zip(head, *body, strict=True)]
        lines = [
            "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in [head, *body]
        ]
        for module, kind, site, pass_, role, reason in self.not_counted:
            where = (module or "(model)") + ("" if site is None else f"[{site}]")
            lines.append(f"not counted: {where} {kind} {pass_} {role}: {reason}")
        return "\n".join(lines)


def _report(sites: dict[tuple[str, str, str | None], _SiteTally]) -> EventReport:
    """The report of a simulate's tallies, read from the device once."""
    keys, tensors = [], []
    for (module, kind, site), tally in sites.items():
        for (pass_, role, label), n in tally.counts.items():
            keys.append((module, kind, site, pass_, role, label))
            tensors.append(n.detach().to("cpu", torch.int64))
    values = torch.stack(tensors).tolist() if tensors else []
    rows = tuple(
        EventRow(*key, counts=EventCounts(*v)) for key, v in zip(keys, values, strict=True)
    )
    gaps = tuple(
        (module, kind, site, pass_, role, reason)
        for (module, kind, site), tally in sites.items()
        for (pass_, role), reason in tally.gaps.items()
    )
    return EventReport(rows, gaps)
