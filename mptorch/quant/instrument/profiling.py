"""Counting a model's arithmetic by format: :func:`profile`.

A mixed-precision study weighs accuracy against cost, and a model's cost is
mostly its products: how many multiply-accumulates (MACs) each product
does, and in which arithmetic. :func:`profile` runs one forward pass and
records every product and every sum the model computes, each with the
formats it computes in, and adds them up per format and against a cost per
MAC.

It sees what :class:`~mptorch.quant.simulate` sees, under the recipe it is
given, or under none: the model as it is, its converted layers counted from
their own formats. That is:

* the products of ``F.linear``, ``torch.matmul`` and its relatives
  (``mm``, ``bmm``, ``@``), ``torch.addmm`` (GPT-2's ``Conv1D``),
  ``torch.baddbmm``, ``F.conv1d/2d/3d``,
  ``F.scaled_dot_product_attention`` and ``nn.MultiheadAttention``, whether
  a module or a function call computes them;
* the sums of softmaxes, norms and ``torch.sum``/``torch.mean``, which
  mptorch runs as products with a vector of ones;
* as element counts, in a table of their own: activations, a model's
  elementwise arithmetic, its embeddings, and the steps of its softmaxes
  and norms.

``backward=True`` adds the products and sums of the backward pass, from the
forward's shapes, for each operand that requires a gradient. ``run=False``
counts without computing anything, under a fake tensor mode.
"""

import itertools
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch._subclasses.fake_tensor import (
    DataDependentOutputException,
    DynamicOutputShapeException,
    FakeTensorMode,
    unset_fake_temporarily,
)
from torch.nn.modules.utils import _ntuple

from .._labels import _mac_label
from ..events import EventReport, _SiteTally
from ..factories.conv import _leading_padding
from ..functional._flash import _keeps_scores, _skipped, _tiles
from ..functional.reduce import ReductionFormats
from ..mac import BlockMac, FusedMac, SplitMac, cast_quant
from ..modules.attention import QMultiheadAttention
from ..modules.matmul import QMatmul
from ..modules.norm import QGroupNorm, QLayerNorm, QRMSNorm, QSoftmax, _QBatchNorm
from ._counting import _counting_formats
from ._handlers import (
    _MHA_DEFAULTS,
    _MHA_NAMES,
    _NAMES,
    _OPT_IN_HANDLERS,
    _OPT_IN_KINDS,
    _OTHER_PRODUCTS,
    _SDPA_NAMES,
    OWNED,
    SIMULATED,
    WHOLE,
    _bind,
    _owner,
    _projection,
)
from .conversion import _REGISTRY
from .palettes import _PaletteMap, set_palette_maps_
from .recipe import GemmFormats, Recipe
from .simulation import _Interceptor, _State, simulate

__all__ = ["profile", "Profile", "ProfileRecord", "ElementRecord", "UncountedRecord"]

# The key of a product whose formats were not built by a GemmFormats, such as
# a layer configured with a *_gemm_formats factory: its arithmetic has no value
# to report it by.
_CUSTOM = "custom"


class ProfileRecord(NamedTuple):
    """One product or sum of the profiled pass, as the GEMM the kernels run:
    ``batch`` products of an ``[M, K]`` and a ``[K, N]`` matrix,
    ``macs = batch * M * N * K``. A sum of ``K`` terms into each of ``N``
    outputs is ``(1, 1, N, K)``.

    ``name`` is the module it ran under (``""`` for the model itself),
    ``kind`` its kind (``linear``, ``conv``, ``matmul``, ``attention``,
    ``softmax``, ``norm`` or ``reduction``), ``site`` what it is within its
    call (``"matmul#0"``, attention's ``"qk"``, ``"softmax"`` and ``"pv"``, a
    norm's ``"mean"``), ``call`` the function or the layer that computed it,
    and ``step`` the pass: ``"fwd"``, ``"igrad"`` (the gradient of the
    input, or of a product's left operand) or ``"wgrad"`` (of the weight,
    the right operand, or a norm's parameters).

    ``formats`` is what computed it (a
    :class:`~mptorch.quant.GemmFormats`, or a softmax's, a norm's or a
    reduction's formats; ``None`` for PyTorch's arithmetic), and ``key``
    what the costs and :meth:`Profile.by_format` file it under: the step's
    mac (``fwd_mac``, ``igrad_mac`` or ``wgrad_mac``; a sum's ``acc`` or
    ``bwd_acc``), ``None`` for PyTorch's arithmetic, or ``"custom"`` for a
    layer whose formats were built otherwise."""

    name: str
    kind: str
    batch: int
    M: int
    N: int
    K: int
    macs: int
    formats: Any
    key: Any
    site: str | None = None
    call: str = ""
    step: str = "fwd"


class ElementRecord(NamedTuple):
    """One elementwise call of the profiled pass: an activation, an
    arithmetic op, an embedding, or a softmax's or a norm's steps, with the
    number of elements of its result and its formats (``None`` for
    PyTorch's)."""

    name: str
    kind: str
    call: str
    elements: int
    formats: Any


class UncountedRecord(NamedTuple):
    """A call :func:`profile` saw and could not count, and why."""

    name: str
    kind: str
    call: str
    reason: str


@dataclass(frozen=True)
class Profile:
    """The records of a profiled pass, and the cost table they are weighed by.

    ``records`` holds the products and sums, ``elements`` the elementwise
    calls and ``uncounted`` what could not be counted, each in the order the
    calls ran. With ``profile(..., events=True)``, ``events`` holds the pass's
    numeric events (:class:`~mptorch.quant.EventReport`).
    """

    records: tuple[ProfileRecord, ...]
    cost: Mapping[Any, float] | None = None
    elements: tuple[ElementRecord, ...] = ()
    uncounted: tuple[UncountedRecord, ...] = ()
    # the pass's numeric events, with profile(events=True)
    events: EventReport | None = None

    @property
    def macs(self) -> int:
        """The MACs of every product and sum recorded."""
        return sum(r.macs for r in self.records)

    def by_format(self) -> dict[Any, int]:
        """The MACs per key, in the order the keys were first met."""
        out: dict[Any, int] = {}
        for r in self.records:
            out[r.key] = out.get(r.key, 0) + r.macs
        return out

    def cost_of(self, key: Any) -> float:
        """The cost per MAC of ``key`` in the cost table: the key itself, and
        for a mac, then its product format (a ``SplitMac``'s ``mul``, a
        ``FusedMac``'s ``fma``, a ``BlockMac``'s ``a``), so that a table keyed
        by formats (``{BinaryK(8, 4): 1.0, None: 4.0}``) prices macs. A
        ``FusedMac`` with no ``fma`` computes in the carrier, and is priced
        as PyTorch's arithmetic, the table's ``None``. A sum on a
        ``SplitMac`` is priced by its ``mul`` too, like any product.

        Raises:
            ValueError: without a cost table.
            KeyError: for a key the table does not price.
        """
        if self.cost is None:
            raise ValueError("this profile has no cost table: pass cost= to profile")
        for k in (key, _product_format(key)):
            if k is not _NOTHING and k in self.cost:
                return float(self.cost[k])
        raise KeyError(f"the cost table has no entry for {key!r} or for its product format")

    @property
    def total_cost(self) -> float:
        """The sum over the records of ``macs * cost_of(key)``, which weighs
        each product's and each sum's MACs by the cost table. The element
        counts are not priced."""
        return math.fsum(r.macs * self.cost_of(r.key) for r in self.records)

    def summary(self) -> str:
        """The MACs per key with their share, and their cost with a cost
        table, one line each, then the total cost."""
        total = self.macs or 1
        lines = []
        for key, macs in self.by_format().items():
            line = f"{_key_label(key)}: {macs:,} MACs ({macs / total:.1%})"
            if self.cost is not None:
                line += f", cost {macs * self.cost_of(key):,.6g}"
            lines.append(line)
        if self.cost is not None:
            lines.append(f"total cost {self.total_cost:,.6g}")
        return "\n".join(lines)

    def table(self) -> str:
        """The records as tables: one line per product or sum, then the
        :meth:`summary`, then the element counts per module, call and
        formats, then what was not counted."""
        sites = any(r.site for r in self.records)
        steps = any(r.step != "fwd" for r in self.records)
        head = ["module", "kind"] + ["site"] * sites + ["step"] * steps
        head += ["batch", "M", "N", "K", "MACs", "arithmetic"]
        body = [
            [r.name or "(model)", r.kind]
            + [r.site or ""] * sites
            + [r.step] * steps
            + [str(r.batch), str(r.M), str(r.N), str(r.K), f"{r.macs:,}", _key_label(r.key)]
            for r in self.records
        ]
        numbers = range(len(head) - 6, len(head) - 1)
        lines = _columns(head, body, numbers)
        lines += ["", self.summary()]
        if self.elements:
            groups: dict[tuple[str, str, str, str], list[int]] = {}
            for e in self.elements:
                g = groups.setdefault(
                    (e.name or "(model)", e.kind, e.call, _label(e.formats)), [0, 0]
                )
                g[0] += 1
                g[1] += e.elements
            lines.append("")
            lines += _columns(
                ["module", "kind", "call", "calls", "elements", "formats"],
                [[*k[:3], str(n), f"{m:,}", k[3]] for k, (n, m) in groups.items()],
                range(3, 5),
            )
        for u in self.uncounted:
            lines.append(f"not counted: {u.call} in {u.name or '(model)'} ({u.kind}): {u.reason}")
        return "\n".join(lines)


def _columns(head: Sequence[str], body: Sequence[Sequence[str]], numbers: range) -> list[str]:
    """Rows as aligned columns, the ``numbers`` columns right-justified."""
    widths = [max(len(row[i]) for row in (head, *body)) for i in range(len(head))]
    return [
        "  ".join(
            c.rjust(w) if i in numbers else c.ljust(w)
            for i, (c, w) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in (head, *body)
    ]


_NOTHING = object()


def _product_format(key: Any) -> Any:
    """The format a mac rounds its products to (``None`` for a fused mac
    that rounds none, which computes in the carrier), or ``_NOTHING``."""
    if isinstance(key, SplitMac):
        return key.mul[0]
    if isinstance(key, FusedMac):
        return None if key.fma is None else key.fma[0]
    if isinstance(key, BlockMac):
        return cast_quant(key.a).fmt
    return _NOTHING


def _key_label(key: Any) -> str:
    if key is None:
        return "torch"
    if key == _CUSTOM:
        return "custom formats"
    return _mac_label(key)


def _label(formats: Any) -> str:
    return "torch" if formats is None else formats.label()


# --- what each product and sum is filed under -------------------------------------------

_STEP_MACS = {"fwd": "fwd_mac", "igrad": "igrad_mac", "wgrad": "wgrad_mac"}
# a container's hooks for each step: QAffineFormats' names, then QMatmulFormats'
_STEP_HOOKS = {
    "fwd": ("fwd_math",),
    "igrad": ("bwd_igrad_math", "bwd_agrad_math"),
    "wgrad": ("bwd_wgrad_math", "bwd_bgrad_math"),
}


def _gemm_key(formats: Any, step: str) -> tuple[GemmFormats | None, Any]:
    """The formats and the key of a product computed by ``formats``: a
    GemmFormats, a layer's container, or ``None`` for PyTorch's."""
    if formats is None:
        return None, None
    source = formats if isinstance(formats, GemmFormats) else getattr(formats, "_source", None)
    if source is not None:
        return source, getattr(source, _STEP_MACS[step])
    hooked = any(getattr(formats, h, None) is not None for h in _STEP_HOOKS[step])
    return None, (_CUSTOM if hooked else None)


def _sum_key(formats: Any, step: str) -> Any:
    """The mac a sum of a softmax, a norm or a reduction runs on."""
    if formats is None:
        return None
    if isinstance(formats, ReductionFormats):
        return formats.mac
    return formats.acc if step == "fwd" else formats.bwd_acc


def _palette_parts(formats: Any) -> list[tuple[Any, int | None]] | None:
    """A palette layer's entries and each one's count of outputs (``None``
    for all of them, which an unset map gives entry 0), or ``None`` for a
    layer without a palette. The counts are the map's, read outside any fake
    tensor mode; a map on the meta device, which has no values, gives the
    counts kept when it was set."""
    if not isinstance(formats, _PaletteMap):
        return None
    macs, m = formats._palette.macs, formats.prec_idx
    if m is None:
        return [(macs[0], None)]
    if m.device.type == "meta":
        counts = formats._palette_counts
        if counts is None:
            raise RuntimeError(
                "a palette layer's map is on the meta device, and has no values to count: set "
                "it with set_palette_maps_, which keeps its counts"
            )
    else:
        with unset_fake_temporarily():
            counts = torch.bincount(m.reshape(-1).long(), minlength=len(macs)).tolist()
    return list(zip(macs, counts, strict=True))


def _needs(t: Any) -> bool:
    """Whether ``t``'s gradient is computed: it requires one, in grad mode."""
    return isinstance(t, torch.Tensor) and t.requires_grad and torch.is_grad_enabled()


class _Counter:
    """What a profiled pass records."""

    def __init__(self, backward: bool):
        self.backward = backward
        self.records: list[ProfileRecord] = []
        self.elements: list[ElementRecord] = []
        self.uncounted: dict[UncountedRecord, None] = {}

    def product(self, name, kind, site, call, step, shape, formats, key=_NOTHING) -> None:
        batch, M, N, K = shape
        gf, found = _gemm_key(formats, step)
        key = found if key is _NOTHING else key
        record = ProfileRecord(name, kind, batch, M, N, K, batch * M * N * K, gf, key)
        self.records.append(record._replace(site=site, call=call, step=step))

    def forward(self, name, kind, site, call, shape, split: int, formats) -> None:
        """A forward product's record, or, for a palette layer, one per entry:
        ``shape``'s dimension ``split`` (its output neurons, channels or
        columns) divided as the layer's map divides them, each part filed
        under its entry's mac."""
        parts = _palette_parts(formats)
        if parts is None:
            self.product(name, kind, site, call, "fwd", shape, formats)
            return
        for mac, count in parts:
            part = list(shape)
            part[split] = shape[split] if count is None else count
            if part[split]:
                self.product(name, kind, site, call, "fwd", tuple(part), formats, key=mac)

    def sum(self, name, kind, site, call, step, outputs, terms, formats) -> None:
        key = _sum_key(formats, step)
        record = ProfileRecord(name, kind, 1, 1, outputs, terms, outputs * terms, formats, key)
        self.records.append(record._replace(site=site, call=call, step=step))

    def element(self, name, kind, call, elements, formats) -> None:
        self.elements.append(ElementRecord(name, kind, call, elements, formats))

    def not_counted(self, name, kind, call, reason) -> None:
        self.uncounted[UncountedRecord(name, kind, call, reason)] = None


# --- the formulas, shared by the calls and the converted layers ---------------------------


def _rotations(c: _Counter, name, call, formats, k, counts) -> None:
    """A rotated product's rotations (``GemmFormats.rotation``), one element
    record each: its two operands' in the forward pass, and each gradient's
    rotation back that the backward pass computes. ``counts`` gives each
    one's number of elements, ``None`` for one not computed. The record
    carries the rotation at the block size a ``k``-channel axis takes."""
    source = formats if isinstance(formats, GemmFormats) else getattr(formats, "_source", None)
    rotation = getattr(source, "rotation", None)
    if rotation is None or rotation.block(k) == 1:
        return
    rotation = replace(rotation, size=rotation.block(k))
    for n in counts:
        if n is not None:
            c.element(name, "rotation", call, n, rotation)


def _linear(c: _Counter, name, call, rows, n_out, k_in, formats, x_grad, w_grad) -> None:
    """``rows`` rows times a weight ``[n_out, k_in]``, and its gradients."""
    c.forward(name, "linear", None, call, (1, rows, n_out, k_in), 2, formats)
    if c.backward:
        if x_grad:
            c.product(name, "linear", None, call, "igrad", (1, rows, k_in, n_out), formats)
        if w_grad:
            c.product(name, "linear", None, call, "wgrad", (1, n_out, k_in, rows), formats)
    grads = (rows * k_in if x_grad else None, n_out * k_in if w_grad else None)
    _rotations(
        c, name, call, formats, k_in, (rows * k_in, n_out * k_in, *(grads if c.backward else ()))
    )


def _matmul(c: _Counter, name, site, call, a, b, out, formats) -> None:
    """``a @ b`` as ``torch.matmul`` broadcasts it, and its gradients; a
    broadcast operand's gradient is a product per batch, then a sum, which
    is not counted."""
    K = a.shape[-1]
    M = a.shape[-2] if a.dim() > 1 else 1
    N = b.shape[-1] if b.dim() > 1 else 1
    batch = out.numel() // (M * N) if M * N else 0
    c.forward(name, "matmul", site, call, (batch, M, N, K), 2, formats)
    if c.backward:
        if _needs(a):
            c.product(name, "matmul", site, call, "igrad", (batch, M, K, N), formats)
        if _needs(b):
            c.product(name, "matmul", site, call, "wgrad", (batch, K, N, M), formats)
    # a gradient is rotated back in the broadcast shape, before it is summed
    grads = (batch * M * K if _needs(a) else None, batch * K * N if _needs(b) else None)
    _rotations(c, name, call, formats, K, (a.numel(), b.numel(), *(grads if c.backward else ())))


def _igrad_classes(size, kernel, stride, lead, dilation) -> Iterator[tuple[int, int]]:
    """A convolution's input gradient as mptorch's kernels run it: one GEMM
    per residue class of the stride, each over the input positions of its
    class and the kernel taps that reach them. Yields each class's numbers
    of positions and of taps, leaving out the classes with none."""
    per = []
    for n_in, k, s, p, d in zip(size, kernel, stride, lead, dilation, strict=True):
        e = s // math.gcd(s, d)
        classes = []
        for r in range(s):
            h0 = (r - p) % s
            n = (n_in - 1 - h0) // s + 1 if h0 < n_in else 0
            first = next((j for j in range(min(k, s)) if (j * d) % s == r), None)
            classes.append((n, 0 if first is None else (k - 1 - first) // e + 1))
        per.append(classes)
    for combo in itertools.product(*per):
        n, taps = math.prod(x for x, _ in combo), math.prod(t for _, t in combo)
        if n and taps:
            yield n, taps


def _conv(c: _Counter, name, call, x, weight, out, stride, padding, dilation, groups, formats):
    """A convolution: one GEMM per sample and group, of the group's output
    channels by the output positions, summing over the group's input
    channels times the kernel taps; and its gradients."""
    nd = weight.dim() - 2
    kernel = tuple(weight.shape[2:])
    B = x.shape[0] if x.dim() == nd + 2 else 1
    M = weight.shape[0] // groups
    N = math.prod(out.shape[-nd:])
    K = weight.shape[1] * math.prod(kernel)
    if _palette_parts(formats) is None:
        c.product(name, "conv", None, call, "fwd", (B * groups, M, N, K), formats)
    else:  # a map of output channels cuts across the groups
        c.forward(name, "conv", None, call, (B, weight.shape[0], N, K), 1, formats)
    if not c.backward:
        return
    if _needs(weight):
        c.product(name, "conv", None, call, "wgrad", (groups, M, K, B * N), formats)
    if _needs(x):
        expand = _ntuple(nd)
        s, d = expand(stride), expand(dilation)
        lead = _leading_padding(padding, kernel, s, d)
        for n, taps in _igrad_classes(tuple(x.shape[-nd:]), kernel, s, lead, d):
            shape = (B * groups, weight.shape[1], n, M * taps)
            c.product(name, "conv", None, call, "igrad", shape, formats)


def _core(
    c: _Counter, name, call, BH, L, S, E, Ev, af, q_grad, k_grad, v_grad, causal=False
) -> None:
    """An attention's core over ``BH`` heads (query heads: grouped keys and
    values are repeated), ``L`` queries, ``S`` keys, head dimensions ``E``
    and ``Ev``: the scores, the softmax's sum, the output; and their
    gradients. Under a flash part, the products flash runs (``_flash_core``)."""
    if af is not None and af.flash is not None:
        _flash_core(c, name, call, BH, L, S, E, Ev, af, (q_grad, k_grad, v_grad), causal)
        return
    qk = None if af is None else af.qk
    sm = None if af is None else af.softmax
    pv = None if af is None else af.pv
    c.product(name, "attention", "qk", call, "fwd", (BH, L, S, E), qk)
    c.sum(name, "attention", "softmax", call, "fwd", BH * L, S, sm)
    c.product(name, "attention", "pv", call, "fwd", (BH, L, Ev, S), pv)
    if not c.backward:
        return
    if q_grad or k_grad:
        c.product(name, "attention", "pv", call, "igrad", (BH, L, S, Ev), pv)
        c.sum(name, "attention", "softmax", call, "igrad", BH * L, S, sm)
        if q_grad:
            c.product(name, "attention", "qk", call, "igrad", (BH, L, E, S), qk)
        if k_grad:
            c.product(name, "attention", "qk", call, "wgrad", (BH, E, S, L), qk)
    if v_grad:
        c.product(name, "attention", "pv", call, "wgrad", (BH, S, Ev, L), pv)


def _flash_core(c: _Counter, name, call, BH, L, S, E, Ev, af, grads, causal) -> None:
    """Flash attention's products as it runs them, tile by tile, the causal
    tiles it skips left out. Tiles of one shape are one record, their count
    folded into the batch. The backward is FA2's: per tile, the scores again
    (unless flash keeps them), dV, dP, dQ and dK, and per query tile the sum
    of ``dO * O``. The running updates (``rescale``, ``combine``, ``lse``)
    are one element record per pass."""
    fl = af.flash
    shapes: dict[tuple[int, int], int] = {}
    for rows in _tiles(L, fl.block_q):
        for cols in _tiles(S, fl.block_kv):
            if not _skipped(causal, rows, cols):
                key = (rows[1] - rows[0], cols[1] - cols[0])
                shapes[key] = shapes.get(key, 0) + 1
    q_grad, k_grad, v_grad = grads
    backward = c.backward and (q_grad or k_grad or v_grad)
    recompute = backward and not _keeps_scores(af)
    fwd_elements = BH * L  # the logsumexp
    bwd_elements = 0
    for (br, bc), n in shapes.items():
        b = BH * n
        c.product(name, "attention", "qk", call, "fwd", (b, br, bc, E), af.qk)
        c.sum(name, "attention", "softmax", call, "fwd", b * br, bc, af.softmax)
        c.product(name, "attention", "pv", call, "fwd", (b, br, Ev, bc), af.pv)
        fwd_elements += b * br * (2 + Ev)  # alpha, and the updates of l and O
        if not backward:
            continue
        if recompute:
            c.product(name, "attention", "qk", call, "fwd", (b, br, bc, E), af.qk)
        if v_grad:
            c.product(name, "attention", "pv", call, "wgrad", (b, bc, Ev, br), af.pv)
            bwd_elements += b * bc * Ev
        if q_grad or k_grad:
            c.product(name, "attention", "pv", call, "igrad", (b, br, bc, Ev), af.pv)
        if q_grad:
            c.product(name, "attention", "qk", call, "igrad", (b, br, E, bc), af.qk)
            bwd_elements += b * br * E
        if k_grad:
            c.product(name, "attention", "qk", call, "wgrad", (b, E, bc, br), af.qk)
            bwd_elements += b * bc * E
    if backward and (q_grad or k_grad):
        c.sum(name, "attention", "softmax", call, "igrad", BH * L, Ev, af.softmax)
    c.element(name, "attention", call, fwd_elements, fl)
    if bwd_elements:
        c.element(name, "attention", call, bwd_elements, fl)


def _mha(c: _Counter, name, out_name, call, shape, same, grads, weights, bias, extra, out_w, parts):
    """``nn.MultiheadAttention``: its in-projections (one product when the
    query, key and value are one tensor, two when the key and value are,
    three otherwise), its core and its output projection.

    ``shape`` is (L, N, S, E, H), ``same`` (query is key, key is value),
    ``grads`` whether the query, key and value need gradients, ``weights``
    the packed in-projection weight or the three separate ones, ``bias`` the
    in-projections' bias, ``extra`` the keys added to the core's (bias_k,
    add_zero_attn, static_k), ``parts`` the in-projections', the output
    projection's and the core's formats."""
    L, N, S, E, H = shape
    in_proj, out_proj, af = parts
    w_grad = any(_needs(w) for w in weights) or _needs(bias)
    if len(weights) == 1:
        f = in_proj[0] if isinstance(in_proj, tuple) else in_proj
        w = weights[0]
        if same[0] and same[1]:
            _linear(c, name, call, L * N, 3 * E, E, f, grads[0], _needs(w))
        elif same[1]:
            _linear(c, name, call, L * N, E, E, f, grads[0], _needs(w))
            _linear(c, name, call, S * N, 2 * E, E, f, grads[1], _needs(w))
        else:
            for rows, g in ((L * N, grads[0]), (S * N, grads[1]), (S * N, grads[2])):
                _linear(c, name, call, rows, E, E, f, g, _needs(w))
    else:
        fs = in_proj if isinstance(in_proj, tuple) else (in_proj,) * 3
        for rows, w, f, g in zip((L * N, S * N, S * N), weights, fs, grads, strict=True):
            _linear(c, name, call, rows, w.shape[0], w.shape[1], f, g, _needs(w))
    bias_k, zero_attn, static_k = extra
    keys = static_k.shape[1] if static_k is not None else S + (bias_k is not None)
    keys += bool(zero_attn)
    q_grad = grads[0] or w_grad
    k_grad = (grads[1] or w_grad) if static_k is None else _needs(static_k)
    v_grad = grads[2] or w_grad
    D = E // H
    _core(c, name, call, N * H, L, keys, D, D, af, q_grad, k_grad, v_grad)
    o_grad = q_grad or k_grad or v_grad
    _linear(
        c, out_name, call, L * N, out_w.shape[0], out_w.shape[1], out_proj, o_grad, _needs(out_w)
    )


def _softmax(c: _Counter, name, call, x, dim, formats) -> None:
    if dim is None:  # F.softmax's choice of a dimension
        dim = 0 if x.dim() in (0, 1, 3) else 1
    K = x.shape[dim] if x.dim() else 1
    rows = x.numel() // K if K else 0
    c.sum(name, "softmax", "sum", call, "fwd", rows, K, formats)
    if c.backward and _needs(x):
        c.sum(name, "softmax", "sum", call, "igrad", rows, K, formats)
    c.element(name, "softmax", call, x.numel(), formats)


def _norm(
    c: _Counter, name, call, x, out, formats, *, rows, sums, param_rows, weight, bias, bwd=None
):
    """A norm's statistics, ``sums`` of them (none for a batch norm in eval),
    each a sum of ``numel / rows`` terms into each of ``rows`` outputs; in
    backward the input's ``bwd`` (as many as ``sums`` by default), and one
    each for a weight and a bias, over ``param_rows`` outputs."""
    terms = x.numel() // rows if rows else 0
    for site in sums:
        c.sum(name, "norm", site, call, "fwd", rows, terms, formats)
    if c.backward:
        if _needs(x):
            for _ in range(len(sums) if bwd is None else bwd):
                c.sum(name, "norm", "input", call, "igrad", rows, terms, formats)
        per_param = x.numel() // param_rows if param_rows else 0
        for site, p in (("weight", weight), ("bias", bias)):
            if _needs(p):
                c.sum(name, "norm", site, call, "wgrad", param_rows, per_param, formats)
    c.element(name, "norm", call, out.numel(), formats)


def _norm_sums(formats, rms: bool) -> tuple[tuple[str, ...], int]:
    """A norm's forward sums, which its deviation decides (an L-infinity
    deviation is a maximum, not a sum), and its input gradient's: two for a
    centred norm, one for an RMS norm, whatever the deviation."""
    deviation = getattr(formats, "deviation", "std")
    if deviation == "l1":
        second = ("mean absolute deviation",)
    elif deviation == "linf":
        second = ()
    else:
        second = ("mean of squares",) if rms else ("variance",)
    return (() if rms else ("mean",)) + second, 1 if rms else 2


def _layer_norm(c, name, call, x, out, shape, weight, bias, formats, rms=False) -> None:
    k = math.prod((shape,) if isinstance(shape, int) else tuple(shape))
    rows = x.numel() // k if k else 0
    sums, bwd = _norm_sums(formats, rms)
    _norm(
        c,
        name,
        call,
        x,
        out,
        formats,
        rows=rows,
        sums=sums,
        param_rows=k,
        weight=weight,
        bias=bias,
        bwd=bwd,
    )


def _group_norm(c, name, call, x, out, groups, weight, bias, formats) -> None:
    rows = x.shape[0] * groups if x.dim() > 1 else groups
    channels = x.shape[1] if x.dim() > 1 else x.shape[0]
    sums, bwd = _norm_sums(formats, False)
    _norm(
        c,
        name,
        call,
        x,
        out,
        formats,
        rows=rows,
        sums=sums,
        param_rows=channels,
        weight=weight,
        bias=bias,
        bwd=bwd,
    )


def _batch_norm(c, name, call, x, out, training, weight, bias, formats) -> None:
    channels = x.shape[1]
    sums = ("mean", "variance") if training else ()
    _norm(
        c,
        name,
        call,
        x,
        out,
        formats,
        rows=channels,
        sums=sums,
        param_rows=channels,
        weight=weight,
        bias=bias,
    )


# --- the calls a simulate sees ---------------------------------------------------------


def _route_of(state: _CountingState, decision: tuple) -> Any:
    """The formats a call was simulated with, or ``None`` for PyTorch's."""
    scope, kind, site, action = decision
    if action != SIMULATED:
        return None
    route = state.routes.get((scope, kind, site if site in state.sites else None))
    return None if route is None else route.container


def _count_linear(c, state, d, func, args, kwargs, out):
    x, weight, _ = _bind(args, kwargs, ("input", "weight", "bias"), (None,))
    n_out, k_in = (weight.shape[0], weight.shape[1]) if weight.dim() == 2 else (1, weight.shape[0])
    rows = x.numel() // k_in if k_in else 0
    f = _route_of(state, d)
    _linear(c, d[0], _NAMES[func], rows, n_out, k_in, f, _needs(x), _needs(weight))


def _count_product(c, state, d, func, a, b, out):
    """A product of the matmul family: a linear when its right operand is one
    of the model's weights (as simulate runs it), and otherwise a matmul."""
    f = _route_of(state, d)
    if d[1] == "linear" and _owner(state, b) is not None and b.dim() == 2:
        k_in, n_out = b.shape
        rows = a.numel() // k_in if k_in else 0
        _linear(c, d[0], _NAMES[func], rows, n_out, k_in, f, _needs(a), _needs(b))
    else:
        _matmul(c, d[0], d[2], _NAMES[func], a, b, out, f)


def _binary_product(names: tuple[str, str], swap: bool = False):
    def count(c, state, d, func, args, kwargs, out):
        a, b = _bind(args, kwargs, names, ())
        if swap:
            a, b = b, a
        _count_product(c, state, d, func, a, b, out)

    return count


def _count_addmm(c, state, d, func, args, kwargs, out):
    _, a, b = _bind(args, kwargs, ("input", "mat1", "mat2"), ())
    _count_product(c, state, d, func, a, b, out)


def _count_baddbmm(c, state, d, func, args, kwargs, out):
    _, a, b = _bind(args, kwargs, ("input", "batch1", "batch2"), ())
    _count_product(c, state, d, func, a, b, out)


def _count_conv(c, state, d, func, args, kwargs, out):
    x, weight, _, stride, padding, dilation, groups = _bind(
        args,
        kwargs,
        ("input", "weight", "bias", "stride", "padding", "dilation", "groups"),
        (None, 1, 0, 1, 1),
    )
    f = _route_of(state, d)
    _conv(c, d[0], _NAMES[func], x, weight, out, stride, padding, dilation, groups, f)


def _count_mha(c, state, d, func, args, kwargs, out):
    values = _bind(args, kwargs, _MHA_NAMES, _MHA_DEFAULTS)
    query, key, value = values[0:3]
    separate = bool(values[17])
    weights = tuple(values[18:21]) if separate else (values[5],)
    scope = d[0]
    in_routes = [_projection(state, w) for w in weights]
    containers = tuple(None if r is None else r.container for r in in_routes)
    out_route = _projection(state, values[11])
    core = state.routes.get((scope, "attention", None))
    parts = (
        containers if separate else containers[0],
        None if out_route is None else out_route.container,
        None if core is None else core.container,
    )
    batched = query.dim() == 3
    shape = (
        query.shape[0],
        query.shape[1] if batched else 1,
        key.shape[0],
        query.shape[-1],
        values[4],
    )
    _mha(
        c,
        scope,
        _owner(state, values[11]) or scope,
        _NAMES[func],
        shape,
        (query is key, key is value),
        (_needs(query), _needs(key), _needs(value)),
        weights,
        values[6],
        (values[7], values[9], values[21]),
        values[11],
        parts,
    )


def _count_sdpa(c, state, d, func, args, kwargs, out):
    values = _bind(args, kwargs, _SDPA_NAMES, (None, 0.0, False, None, False))
    q, k, v, causal = values[0], values[1], values[2], bool(values[5])
    BH = math.prod(q.shape[:-2])
    af = _route_of(state, d)
    L, E, S, Ev = q.shape[-2], q.shape[-1], k.shape[-2], v.shape[-1]
    needs = (_needs(q), _needs(k), _needs(v))
    _core(c, d[0], _NAMES[func], BH, L, S, E, Ev, af, *needs, causal=causal)


def _count_elements(c, state, d, func, args, kwargs, out):
    n = out.numel() if isinstance(out, torch.Tensor) else 0
    c.element(d[0], d[1], _NAMES[func], n, _route_of(state, d))


def _count_softmax(names: tuple[str, ...], defaults: tuple):
    def count(c, state, d, func, args, kwargs, out):
        values = _bind(args, kwargs, names, defaults)
        _softmax(c, d[0], _NAMES[func], values[0], values[1], _route_of(state, d))

    return count


def _count_norm(op: str, names: tuple[str, ...], defaults: tuple):
    def count(c, state, d, func, args, kwargs, out):
        v = dict(zip(names, _bind(args, kwargs, names, defaults), strict=True))
        x, f, call = v["input"], _route_of(state, d), _NAMES[func]
        if op in ("layer_norm", "rms_norm"):
            shape = v["normalized_shape"]
            _layer_norm(
                c, d[0], call, x, out, shape, v["weight"], v.get("bias"), f, op == "rms_norm"
            )
        elif op == "group_norm":
            _group_norm(c, d[0], call, x, out, v["num_groups"], v["weight"], v["bias"], f)
        else:
            _batch_norm(c, d[0], call, x, out, bool(v["training"]), v["weight"], v["bias"], f)

    return count


def _count_reduction(c, state, d, func, args, kwargs, out):
    x = args[0] if args else kwargs.get("input")
    outputs = out.numel() if isinstance(out, torch.Tensor) else 1
    terms = x.numel() // outputs if outputs else 0
    c.sum(d[0], "reduction", None, _NAMES[func], "fwd", outputs, terms, _route_of(state, d))


def _uncounted(c, state, d, func, args, kwargs, out):
    c.not_counted(d[0], d[1], _NAMES[func], "profile does not count this product")


def _norm_counter(op: str) -> Callable:
    names = {
        "layer_norm": ("input", "normalized_shape", "weight", "bias", "eps"),
        "rms_norm": ("input", "normalized_shape", "weight", "eps"),
        "group_norm": ("input", "num_groups", "weight", "bias", "eps"),
        "batch_norm": (
            "input",
            "running_mean",
            "running_var",
            "weight",
            "bias",
            "training",
            "momentum",
            "eps",
        ),
    }[op]
    defaults = {
        "layer_norm": (None, None, 1e-5),
        "rms_norm": (None, None),
        "group_norm": (None, None, 1e-5),
        "batch_norm": (None, None, False, 0.1, 1e-5),
    }[op]
    return _count_norm(op, names, defaults)


# Each intercepted function's counter, called after its handler decided.
_COUNTERS: dict[Any, Callable] = {
    F.linear: _count_linear,
    torch.matmul: _binary_product(("input", "other")),
    torch.Tensor.matmul: _binary_product(("input", "other")),
    torch.Tensor.__matmul__: _binary_product(("input", "other")),
    torch.Tensor.__rmatmul__: _binary_product(("self", "other"), swap=True),
    torch.mm: _binary_product(("input", "mat2")),
    torch.Tensor.mm: _binary_product(("input", "mat2")),
    torch.bmm: _binary_product(("input", "mat2")),
    torch.Tensor.bmm: _binary_product(("input", "mat2")),
    torch.addmm: _count_addmm,
    torch.Tensor.addmm: _count_addmm,
    torch.baddbmm: _count_baddbmm,
    torch.Tensor.baddbmm: _count_baddbmm,
    F.conv1d: _count_conv,
    F.conv2d: _count_conv,
    F.conv3d: _count_conv,
    F.multi_head_attention_forward: _count_mha,
    F.scaled_dot_product_attention: _count_sdpa,
    F.embedding: _count_elements,
    F.softmax: _count_softmax(("input", "dim", "_stacklevel", "dtype"), (None, 3, None)),
    torch.softmax: _count_softmax(("input", "dim", "dtype"), (None,)),
    torch.Tensor.softmax: _count_softmax(("input", "dim", "dtype"), (None,)),
    F.layer_norm: _norm_counter("layer_norm"),
    F.rms_norm: _norm_counter("rms_norm"),
    F.group_norm: _norm_counter("group_norm"),
    F.batch_norm: _norm_counter("batch_norm"),
    torch.sum: _count_reduction,
    torch.Tensor.sum: _count_reduction,
    torch.mean: _count_reduction,
    torch.Tensor.mean: _count_reduction,
    **{f: _uncounted for funcs in _OTHER_PRODUCTS.values() for f in funcs},
}
# the activations' and the elementwise ops' handlers, in place ones included
for _kind in ("activation", "elementwise"):
    for _f in _OPT_IN_HANDLERS[_kind]:
        _COUNTERS.setdefault(_f, _count_elements)
# SDPA has an entry in _OTHER_PRODUCTS (for a recipe without the attention kind)
_COUNTERS[F.scaled_dot_product_attention] = _count_sdpa


class _CountingState(_State):
    """A simulate's state that sees every kind, and keeps each call's decision."""

    def __init__(self, recipe: Recipe, model: nn.Module, strict: bool, events: bool = False):
        super().__init__(recipe, model, strict, events)
        # every kind's handler, so that the calls of a kind the recipe does not
        # name are seen too: they find no route, and run as PyTorch's
        for kind in _OPT_IN_KINDS:
            self.handlers.update(_OPT_IN_HANDLERS[kind])
        self.decision: tuple | None = None
        self.counter = _Counter(False)

    def record(self, scope, kind, site, call, action, label=""):
        super().record(scope, kind, site, call, action, label)
        self.decision = (scope, kind, site, action)


class _CountingInterceptor(_Interceptor):
    """A simulate's mode that counts each call after its handler ran it."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        state = self.state
        handler = state.handlers.get(func)
        if handler is None or not state.stack:
            return func(*args, **kwargs)
        assert isinstance(state, _CountingState)
        state.decision = None
        out = handler(state, func, args, kwargs)
        d = state.decision
        # an mptorch layer's own calls are counted by its hook, and the calls
        # of a module simulated whole are not the arithmetic that runs
        if d is not None and d[3] != OWNED and d[3] != WHOLE:
            count = _COUNTERS.get(func)
            if count is not None:
                count(state.counter, state, d, func, args, kwargs, out)
        return out


class _Counting(simulate):
    _state_class = _CountingState
    _mode_class = _CountingInterceptor


# --- converted layers --------------------------------------------------------------------

_CONVS = (nn.Conv1d, nn.Conv2d, nn.Conv3d)
_ACTIVATIONS = {
    entry.qcls: entry.kind
    for entry in _REGISTRY.values()
    if entry.kind in ("activation", "embedding") and entry.forward is None
}


def _first(args: tuple, kwargs: dict, name: str) -> Any:
    return args[0] if args else kwargs[name]


def _count_layer(c: _Counter, name: str, module: nn.Module, args, kwargs, out) -> None:
    """Count a converted (mptorch) layer's call, from its class and its formats."""
    call = type(module).__name__
    f = getattr(module, "formats", None)
    if isinstance(module, QMultiheadAttention):
        _count_mha_layer(c, name, module, args, kwargs, f)
    elif isinstance(module, nn.Linear):
        x = _first(args, kwargs, "input")
        k_in, n_out = module.in_features, module.out_features
        rows = x.numel() // k_in if k_in else 0
        w_grad = module.weight is not None and _needs(module.weight)
        _linear(c, name, call, rows, n_out, k_in, f, _needs(x), w_grad)
    elif isinstance(module, _CONVS):
        x = _first(args, kwargs, "input")
        padding = module.padding
        _conv(
            c,
            name,
            call,
            x,
            module.weight,
            out,
            module.stride,
            padding,
            module.dilation,
            module.groups,
            f,
        )
    elif isinstance(module, QMatmul):
        a, b = (list(args) + [kwargs.get(k) for k in ("a", "b")[len(args) :]])[:2]
        _matmul(c, name, None, call, a, b, out, f)
    elif isinstance(module, QSoftmax):
        _softmax(c, name, call, _first(args, kwargs, "input"), module.dim, f)
    elif isinstance(module, QLayerNorm):
        x = _first(args, kwargs, "input")
        _layer_norm(c, name, call, x, out, module.normalized_shape, module.weight, module.bias, f)
    elif isinstance(module, QRMSNorm):
        x = _first(args, kwargs, "x")
        _layer_norm(c, name, call, x, out, module.normalized_shape, module.weight, None, f, True)
    elif isinstance(module, QGroupNorm):
        x = _first(args, kwargs, "input")
        _group_norm(c, name, call, x, out, module.num_groups, module.weight, module.bias, f)
    elif isinstance(module, _QBatchNorm):
        assert isinstance(module, nn.modules.batchnorm._BatchNorm)
        x = _first(args, kwargs, "input")
        training = module.training or (module.running_mean is None and module.running_var is None)
        _batch_norm(c, name, call, x, out, training, module.weight, module.bias, f)
    elif type(module) in _ACTIVATIONS:
        n = out.numel() if isinstance(out, torch.Tensor) else 0
        c.element(name, _ACTIVATIONS[type(module)], call, n, f)


def _count_mha_layer(c: _Counter, name: str, module: QMultiheadAttention, args, kwargs, f) -> None:
    query, key, value = (
        list(args) + [kwargs.get(k) for k in ("query", "key", "value")[len(args) :]]
    )[:3]
    batched = query.dim() == 3
    if module.batch_first and batched:
        L, N, S = query.shape[1], query.shape[0], key.shape[1]
    else:
        L, N, S = query.shape[0], query.shape[1] if batched else 1, key.shape[0]
    if module._qkv_same_embed_dim:
        weights = (module.in_proj_weight,)
    else:
        weights = (module.q_proj_weight, module.k_proj_weight, module.v_proj_weight)
    plain = f is None or f._plain
    parts = (None, None, None) if plain else (f.in_proj, f.out_proj, f.attention)
    out_proj = module.out_proj
    _mha(
        c,
        name,
        f"{name}.out_proj" if name else "out_proj",
        type(module).__name__,
        (L, N, S, module.embed_dim, module.num_heads),
        (query is key, key is value),
        (_needs(query), _needs(key), _needs(value)),
        weights,
        module.in_proj_bias,
        (module.bias_k, module.add_zero_attn, None),
        out_proj.weight,
        parts,
    )


def _whole_entry(module: nn.Module) -> Any:
    """The registry entry of a module converted to a class registered with a forward."""
    for entry in _REGISTRY.values():
        if entry.forward is not None and entry.qcls is type(module):
            return entry
    return None


_WHOLE = "it is simulated by the forward registered for it, whose sums profile does not count"


def _hooks(model: nn.Module, state: _CountingState) -> list:
    """The forward hooks that count the converted layers and the modules
    simulated whole. They are registered before the simulate's own, so that
    the module's frame is still on its stack when they run."""
    c = state.counter
    handles = []
    for module in model.modules():
        name = state.names[id(module)]
        owned = getattr(type(module), "_mptorch_owned", False)
        whole = state.whole.get(name)
        if not owned and whole is None:
            continue

        def hook(module, args, kwargs, out, name=name, owned=owned, whole=whole):
            stack = state.stack
            if whole is not None:
                # inside another whole module's own forward: not what runs
                if len(stack) > 1 and stack[-2][2]:
                    return None
                n = out.numel() if isinstance(out, torch.Tensor) else 0
                c.element(name, whole[2], f"{whole[4]}.forward", n, whole[1])
                c.not_counted(name, whole[2], f"{whole[4]}.forward", _WHOLE)
                return None
            if stack and stack[-1][2]:
                return None
            entry = _whole_entry(module)
            if entry is not None:
                call = f"{type(module).__name__}.forward"
                n = out.numel() if isinstance(out, torch.Tensor) else 0
                c.element(name, entry.kind, call, n, getattr(module, "formats", None))
                c.not_counted(name, entry.kind, call, _WHOLE)
                return None
            _count_layer(c, name, module, args, kwargs, out)
            return None

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    return handles


# --- profile -------------------------------------------------------------------------------


def _cuda_devices(model: nn.Module, inputs: Sequence[Any], kwargs: Mapping[str, Any]) -> list[int]:
    """The CUDA devices the model's tensors and its inputs are on."""
    tensors = itertools.chain(
        model.parameters(),
        model.buffers(),
        (t for t in (*inputs, *kwargs.values()) if isinstance(t, torch.Tensor)),
    )
    return sorted({t.device.index or 0 for t in tensors if t.is_cuda})


def profile(
    model: nn.Module,
    *inputs: Any,
    recipe: Recipe | None = None,
    cost: Mapping[Any, float] | None = None,
    backward: bool = False,
    run: bool = True,
    palette_maps: Mapping[str, torch.Tensor | None] | None = None,
    events: bool = False,
    **kwargs: Any,
) -> Profile:
    """Run one forward pass of ``model`` and count its products and sums by their formats.

    The pass is ``model(*inputs, **kwargs)``, under a
    :class:`~mptorch.quant.simulate` of ``recipe`` that counts what it sees,
    and a converted layer is counted from its own formats. Without a recipe
    the model runs as it is. Each product is recorded as the GEMM the
    kernels run (:class:`ProfileRecord`): ``F.linear``, ``torch.matmul`` and
    its relatives, ``torch.addmm`` (GPT-2's ``Conv1D``), ``torch.baddbmm``,
    ``F.conv1d/2d/3d`` (one GEMM per sample and group),
    ``F.scaled_dot_product_attention`` and ``nn.MultiheadAttention``. Each
    sum of a softmax, a norm or a ``torch.sum``/``torch.mean`` is recorded
    as a GEMM with a vector of ones, and each elementwise call (an
    activation, an arithmetic op, an embedding, a softmax's or a norm's
    steps) as its number of elements (:class:`ElementRecord`).

    Not counted, and listed in :attr:`Profile.uncounted`: the products
    simulate cannot run (``einsum``, ``tensordot``, ...), and the arithmetic
    of a module simulated whole, by a forward registered for it (its
    elements are counted). Not seen: mptorch's own functions
    (:func:`~mptorch.quant.qmatmul`, :func:`~mptorch.quant.qattention`, ...)
    called in a model's code, whose bodies hide their calls.

    The pass leaves the model's buffers (a batch norm's running statistics)
    and the CPU's and CUDA's random generators as it found them.

    Args:
        model (nn.Module): the model.
        *inputs: its positional inputs.
        recipe (Recipe, optional): the formats to count it in, as
            ``simulate(recipe, model)`` would run it. Default: ``None``,
            the model's own
        cost (Mapping, optional): a cost per MAC for each key (see
            :meth:`Profile.cost_of`), such as ``{BinaryK(8, 4): 1.0, None: 4.0}``
            for fp8 products against float32 ones. Default: ``None``
        backward (bool): count the backward pass's products and sums too,
            each for an operand that requires a gradient, keyed by its own
            step's mac (a GEMM's ``igrad_mac`` and ``wgrad_mac``, a sum's
            ``bwd_acc``). The forward then runs in grad mode. A product's
            gradients, an attention's four and a convolution's input
            gradient (one GEMM per residue class of the stride) are counted
            as mptorch's kernels run them. Default: ``False``
        run (bool): compute the pass. ``False`` runs it under a fake tensor
            mode, which computes nothing and allocates nothing, so that a
            model too large to run, or built on the meta device, is counted
            from its shapes. Default: ``True``
        palette_maps (Mapping, optional): with a ``recipe`` that has
            palettes, the maps of its palette layers, as
            :func:`~mptorch.quant.set_palette_maps_` takes them for a
            ``simulate``. A palette layer's forward is counted per entry, as
            its map divides its outputs, an unset map's all in entry 0; a
            converted model's layers are counted by their own maps.
            Default: ``None``
        events (bool): also count the pass's numeric events, into
            :attr:`Profile.events`, as ``simulate(recipe, model,
            events=True)`` counts them; a converted layer's through a
            counting container rebuilt from the formats it was converted
            with, swapped in for the pass (a GEMM layer whose formats were
            built otherwise, or hold a palette, is listed as not counted). The
            pass is a forward one: ``backward=True`` counts the backward's
            MACs from the forward's shapes, and runs no backward. It needs
            ``run=True``, since a fake tensor has no values. Default:
            ``False``
        **kwargs: its keyword inputs.

    Returns:
        Profile: the records, in the order the calls ran.

    Raises:
        RuntimeError: with ``backward=True`` under ``torch.inference_mode``,
            which no gradient is computed in; with ``run=False``, for a
            model whose pass reads a tensor's values, which a fake tensor
            does not have.

    Example::

        >>> import torch
        >>> from torch import nn
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import convert_, presets, profile
        >>> model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 2))
        >>> _ = convert_(model, presets.uniform(BinaryK(8, 4)).override("2", linear=None))
        >>> p = profile(model, torch.randn(4, 16), cost={BinaryK(8, 4): 1.0, None: 4.0})
        >>> p.macs, p.total_cost
        (576, 768.0)
    """
    if backward and torch.is_inference_mode_enabled():
        raise RuntimeError(
            "profile(backward=True) was called under torch.inference_mode, where no gradient "
            "is computed: call it outside inference_mode"
        )
    if events and not run:
        raise RuntimeError(
            "profile(events=True) counts the values the pass computes, and run=False runs it on "
            "fake tensors, which have none: pass run=True"
        )
    model = getattr(model, "_orig_mod", model)
    sim = _Counting(Recipe() if recipe is None else recipe, model, events=events)
    if palette_maps is not None:
        if recipe is None:
            raise TypeError(
                "palette_maps= gives the maps of a recipe's palette layers, and profile was given "
                "no recipe: a converted model's layers are counted by their own maps"
            )
        set_palette_maps_(sim, palette_maps)
    state = sim._state
    assert isinstance(state, _CountingState)
    counter = _Counter(backward)
    state.counter = counter
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    handles = _hooks(model, state)
    swapped: list[tuple[nn.Module, nn.Module]] = []
    try:
        with ExitStack() as stack:
            stack.enter_context(torch.random.fork_rng(devices=_cuda_devices(model, inputs, kwargs)))
            if not run:
                stack.enter_context(FakeTensorMode(allow_non_fake_inputs=True))
            stack.enter_context(torch.enable_grad() if backward else torch.no_grad())
            stack.enter_context(sim)
            if events:
                # after simulate installed its hooks, which know the model's modules
                swapped = _counting_layers(model, state)
            try:
                model(*inputs, **kwargs)
            except (DataDependentOutputException, DynamicOutputShapeException) as e:
                raise RuntimeError(
                    f"profile(run=False) cannot count this model: its pass reads a tensor's "
                    f"values ({e}), which a fake tensor does not have (an NVFP4 input without a "
                    "tensor_scale reads its largest magnitude). Pass run=True, or give the "
                    "format what it reads"
                ) from e
    finally:
        for module, formats in swapped:
            module.formats = formats
        for h in handles:
            h.remove()
        with torch.no_grad():
            for b, saved in buffers:
                b.copy_(saved)
    return Profile(
        tuple(counter.records),
        cost,
        tuple(counter.elements),
        tuple(counter.uncounted),
        sim.events() if events else None,
    )


# The kinds of the converted layers whose formats are an opt-in kind's
# frozen value (QLayerNorm, QSoftmax, QGELU, ...).
_OPT_IN_LAYERS = {
    entry.qcls: entry.kind
    for entry in _REGISTRY.values()
    if entry.qcls is not None and entry.kind not in ("linear", "conv", "matmul")
}


def _counting_layers(model: nn.Module, state: _State) -> list[tuple[nn.Module, Any]]:
    """Give each converted layer counting formats for the pass, its site's
    counts in ``state.tallies``: a GEMM layer a container rebuilt from the
    GemmFormats it was converted with, an opt-in kind's layer a counting copy
    of its frozen formats; return the (layer, formats) pairs to put back."""
    swapped: list[tuple[nn.Module, Any]] = []
    for name, module in model.named_modules():
        f = getattr(module, "formats", None)
        opt_in = _OPT_IN_LAYERS.get(type(module))
        if opt_in is not None and f is not None:
            tally = state.tallies.setdefault((name, opt_in, None), _SiteTally())
            swapped.append((module, f))
            module.formats = _counting_formats(f, tally)
            continue
        if isinstance(module, _CONVS):
            kind, build = "conv", "conv"
        elif isinstance(module, QMatmul):
            kind, build = "matmul", "matmul"
        elif isinstance(module, nn.Linear):
            kind, build = "linear", "affine"
        else:
            continue
        if f is None or not getattr(type(module), "_mptorch_owned", False):
            continue
        tally = state.tallies.setdefault((name, kind, None), _SiteTally())
        source = getattr(f, "_source", None)
        if not isinstance(source, GemmFormats) or source.palette:
            why = (
                "a palette mac"
                if isinstance(source, GemmFormats)
                else "formats not built from a GemmFormats"
            )
            tally.gap("fwd", "mac", why)
            continue
        swapped.append((module, f))
        module.formats = getattr(source, build)(_tally=tally)
    return swapped
