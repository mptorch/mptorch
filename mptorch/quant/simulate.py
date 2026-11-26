"""Simulating a model's arithmetic while it runs, without converting it: ``simulate``.

:func:`~mptorch.quant.convert_` gives mptorch's arithmetic to the layers of a
registered class, by switching their class. Many of a model's products are
not such layers: eager attention multiplies two activations with
``torch.matmul``, GPT-2's ``Conv1D`` calls ``torch.addmm``, and swin calls
``F.linear`` on its ``qkv`` layer's weight without calling the layer.
:class:`simulate` reaches those too. While it is active, a torch function
mode intercepts the model's calls, and each product the recipe gives formats
to runs through the same autograd Function a converted layer runs, with the
same formats, so both give the same words on what both cover.

Which formats a call gets comes from the recipe, resolved under the
innermost module running (forward hooks on the model's modules keep a stack
of their names), and from the kind of product:

* ``linear``: ``F.linear``, and a ``torch.addmm`` or ``torch.matmul`` whose
  right operand is one of the model's Parameters. It is resolved under the
  innermost module if that module owns the Parameter (a tied ``lm_head``
  does), and otherwise under the module that does (swin's ``qkv``).
* ``conv``: ``F.conv1d/2d/3d``, resolved as ``linear`` is.
* ``nn.MultiheadAttention``'s ``F.multi_head_attention_forward``: its
  projections take the ``linear`` kind's formats of their weights' owners,
  and its core the ``attention`` kind's, as a converted
  :class:`~mptorch.quant.QMultiheadAttention` takes them. Without an
  ``attention`` kind in the recipe the core is PyTorch's, and reported. (The
  mode keeps the module off its fast path, compiled code included.)
* ``matmul``: a product of two activations (anything but a Parameter itself:
  a view of one, ``W.T``, counts as an activation). It is resolved under the
  innermost module, at a *site*: ``"matmul#0"`` for that module's first
  such product in a forward pass, ``"matmul#1"`` for its second, and so on,
  which a :class:`~mptorch.quant.Rule`'s ``site`` names.

The kinds past the GEMMs are reached only when a rule of the recipe names
them, so a recipe for the GEMMs pays nothing for them and reports nothing
about them:

* ``activation``: ``F.gelu``, ``F.silu``, ``F.hardswish``, ``F.hardsigmoid``,
  ``F.relu``, ``torch.sigmoid`` and ``torch.tanh`` (and their ``torch.`` and
  ``Tensor.`` spellings), under the innermost module. A call with
  ``inplace=True`` computes out of place, as the mptorch activation modules
  do, since a rounded result is a new tensor; the ``Tensor.relu_`` kind of
  in-place method passes through.
* ``elementwise``: additions, subtractions, products and quotients (``+``,
  ``-``, ``*``, ``/``, ``torch.add`` and the others), square roots,
  reciprocal square roots, exponentials and powers with a number exponent,
  under the innermost module. Their in-place forms pass through, and so does
  a call with no float tensor operand (index arithmetic), unrecorded.
* ``embedding``: ``F.embedding``, resolved under its weight's owner, as
  ``linear`` is.
* ``softmax``: ``F.softmax``, ``torch.softmax`` and ``Tensor.softmax``, with
  their ``dtype``, under the innermost module.
* ``norm``: ``F.layer_norm``, ``F.rms_norm``, ``F.group_norm`` and
  ``F.batch_norm``, resolved under the owner of their weight, as ``linear``
  is (the innermost module, without one). A norm a model writes out of
  elementwise ops is reached as a whole when its class is registered with
  the forward it simulates (``register(cls, kind="norm", forward=...)``):
  the module's own calls pass through, and that forward's result replaces
  its output.
* ``reduction``: ``torch.sum`` and ``torch.mean`` (and their ``Tensor.``
  spellings) over a float tensor, under the innermost module; a sum of
  integers passes through, unrecorded.
* ``attention``: ``F.scaled_dot_product_attention``, under the innermost
  module, as :func:`~mptorch.quant.qattention`. Without the kind it is
  reported as not covered, since its two products run inside that one call.

Everything is resolved, and every format container built, when the
``simulate`` is made, so that a call only looks its route up: Dynamo can
trace that, and a container built while it traced gave wrong gradients.
"""

import weakref
from dataclasses import dataclass
from typing import Any, NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair, _single, _triple
from torch.overrides import TorchFunctionMode, resolve_name

from . import attention, norm, pointwise, reduce
from ._counting import _counting_formats
from .convert import _REGISTRY
from .events import EventReport, _SiteTally
from .events import _report as _event_report
from .modules.conv import CustomArithConvNd
from .modules.linear import CustomArithLinear
from .modules.matmul import CustomArithMatmul
from .palettes import _PaletteMap
from .recipe import _GEMM_KINDS, Recipe

__all__ = ["simulate", "SimulateReport", "SimulateRow"]

SIMULATED = "simulated"
PLAIN = "plain: the recipe gives None"
OWNED = "an mptorch layer's own call"
WHOLE = "inside a module simulated whole"


class SimulateRow(NamedTuple):
    """One kind of call :class:`simulate` saw: where (the innermost module's
    name, ``""`` for the model itself), the kind and site it was resolved
    as, the function, what was done with it, and the formats' label."""

    scope: str
    kind: str
    site: str | None
    call: str
    action: str
    formats: str


@dataclass(frozen=True)
class SimulateReport:
    """What a :class:`simulate` saw: one row per distinct call, in the order
    first seen. ``str(report)`` is a table.

    Only calls run eagerly are recorded: a compiled model's are not, since
    recording them while Dynamo traced made every later call recompile. Run
    the model once without ``torch.compile`` to see its report.
    """

    rows: tuple[SimulateRow, ...]
    #: The palette layers' maps, each as its name and whether it is set (an
    #: unset map computes every output in the palette's entry 0).
    palette_maps: tuple[tuple[str, bool], ...] = ()

    @property
    def simulated(self) -> tuple[SimulateRow, ...]:
        """The calls that ran in mptorch's arithmetic."""
        return tuple(r for r in self.rows if r.action == SIMULATED)

    @property
    def not_covered(self) -> tuple[SimulateRow, ...]:
        """The products ``simulate`` cannot run in mptorch's arithmetic, each
        with its reason; they ran in PyTorch's."""
        return tuple(r for r in self.rows if r.action.startswith("not covered"))

    def __str__(self) -> str:
        header = ("module", "kind", "site", "call", "formats", "action")
        body = [
            (r.scope or "(model)", r.kind, r.site or "", r.call, r.formats, r.action)
            for r in self.rows
        ]
        widths = [max(len(row[i]) for row in (header, *body)) for i in range(5)]
        lines = [
            "  ".join(c.ljust(w) for c, w in zip(row[:5], widths, strict=True)) + "  " + row[5]
            for row in (header, *body)
        ]
        lines.append(
            f"{len(self.simulated)} simulated, {len(self.not_covered)} not covered, "
            f"{len(self.rows) - len(self.simulated) - len(self.not_covered)} left plain"
        )
        if self.palette_maps:
            maps = ", ".join(
                f"{n} ({'set' if is_set else 'unset: entry 0'})" for n, is_set in self.palette_maps
            )
            lines.append(f"palette maps: {maps}")
        return "\n".join(lines)


class _Route(NamedTuple):
    container: Any  # a GEMM kind's container, or another kind's frozen formats
    label: str


# The kinds past the GEMMs, which a simulate reaches only when the recipe names them.
_OPT_IN_KINDS = (
    "activation",
    "elementwise",
    "embedding",
    "softmax",
    "norm",
    "reduction",
    "attention",
)


class _State:
    """Everything a call reads, built once from the model and the recipe."""

    def __init__(self, recipe: Recipe, model: nn.Module, strict: bool, events: bool = False):
        self.strict = strict
        # simulate(events=True)'s counts, by (module, kind, site): one counting
        # container per site, which counts into its tally
        self.tallies: dict[tuple[str, str, str | None], _SiteTally] = {}
        self.names: dict[int, str] = {}
        self.owners: dict[int, tuple[str, ...]] = {}
        self.owned: set[str] = set()
        # the modules whose registered forward replaces their own: name -> (forward,
        # its formats, kind, label, the module's class name)
        self.whole: dict[str, tuple[Any, Any, str, str, str]] = {}
        self.routes: dict[tuple[str, str, str | None], _Route | None] = {}
        self.covered: dict[str, set[str]] = {}  # each module's kinds with formats
        self.sites = {r.site for r in recipe.rules if r.site is not None}
        named = {kind for rule in recipe.rules for kind, _ in rule.kinds}
        opt_in = tuple(kind for kind in _OPT_IN_KINDS if kind in named)
        # whether a missing attention route means "plain" (the recipe names the
        # kind) or "not covered" (it does not), for nn.MultiheadAttention's core
        self.attention_named = "attention" in named
        self.handlers = dict(_HANDLERS)
        for kind in opt_in:
            self.handlers.update(_OPT_IN_HANDLERS[kind])
        # one [name, products of two activations so far, whether the frame is
        # inside the own forward of a module simulated whole] per module
        # running: kept on the stack, which is empty between calls, so that
        # Dynamo has no counter to guard on (a dict of counts made every call
        # recompile)
        self.stack: list[list] = []
        self.rows: dict[SimulateRow, None] = {}  # a dict keeps the order rows were seen in

        containers: dict[tuple, nn.Module] = {}
        first: dict[int, tuple[str, tuple]] = {}
        # a module reached under several names: each later name -> its first
        self.aliases: dict[str, str] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            resolved = tuple(
                (kind, site, recipe.resolve(name, module, kind, site))
                for kind in _GEMM_KINDS
                for site in ((None, *self.sites) if kind == "matmul" else (None,))
            ) + tuple((kind, None, recipe.resolve(name, module, kind)) for kind in opt_in)
            seen = first.get(id(module))
            if seen is not None:
                if seen[1] != resolved:
                    raise ValueError(
                        f"the module at {seen[0]!r} is also at {name!r}, and the recipe gives the "
                        "two names different formats: give both names the same formats"
                    )
                self.aliases[name] = seen[0]
                continue
            first[id(module)] = (name, resolved)
            self.names[id(module)] = name
            if getattr(type(module), "_mptorch_owned", False):
                self.owned.add(name)
            for p in module.parameters(recurse=False):
                self.owners[id(p)] = (*self.owners.get(id(p), ()), name)
            for kind, site, gf in resolved:
                if gf is None:
                    continue
                self.covered.setdefault(name, set()).add(kind)
                if kind in _OPT_IN_KINDS:  # frozen formats, read as they are
                    value = gf
                    if events:  # or a counting copy (mptorch.quant._counting)
                        tally = self.tallies.setdefault((name, kind, site), _SiteTally())
                        value = _counting_formats(gf, tally)
                    self.routes[(name, kind, site)] = _Route(value, gf.label())
                    continue
                # a palette's container holds its map, and a counting one its
                # site's counts, so each such route has its own
                key = (name, kind, site) if gf.palette or events else (id(gf), kind)
                if key not in containers:
                    build = {"linear": gf.affine, "conv": gf.conv, "matmul": gf.matmul}[kind]
                    if events:
                        tally = self.tallies.setdefault((name, kind, site), _SiteTally())
                        containers[key] = build(_tally=tally)
                    else:
                        containers[key] = build()
                self.routes[(name, kind, site)] = _Route(containers[key], gf.label())
            entry = _REGISTRY.get(type(module))
            if entry is not None and entry.forward is not None:
                route = self.routes.get((name, entry.kind, None))
                if route is not None:
                    cls = type(module).__name__
                    self.whole[name] = (
                        entry.forward,
                        route.container,
                        entry.kind,
                        route.label,
                        cls,
                    )

        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                for scope in (name, f"{name}.out_proj" if name else "out_proj"):
                    route = self.routes.get((scope, "linear", None))
                    if route is not None and isinstance(route.container, _PaletteMap):
                        raise ValueError(
                            f"the recipe gives {scope!r}, a projection of the "
                            "nn.MultiheadAttention at "
                            f"{name or '(the model)'!r}, a palette fwd_mac: its in-projection "
                            "packs the query's, the key's and the value's into one product, "
                            "whose width a map of output neurons does not follow. Give it a "
                            "single format"
                        )

    def record(self, scope: str, kind: str, site: str | None, call: Any, action: str, label=""):
        # Not while Dynamo traces: it guards on what it reads, and a report
        # that grew during the first compiled call made the second recompile.
        if not torch.compiler.is_compiling():
            name = call if isinstance(call, str) else _NAMES[call]
            self.rows[SimulateRow(scope, kind, site, name, action, label)] = None


def _bind(args: tuple, kwargs: dict, names: tuple[str, ...], defaults: tuple) -> list:
    """The values of a call's parameters ``names``, positional or keyword;
    the trailing ones default to ``defaults``."""
    values = list(args) + [kwargs.get(n, _MISSING) for n in names[len(args) :]]
    first_default = len(names) - len(defaults)
    for i in range(first_default, len(names)):
        if values[i] is _MISSING:
            values[i] = defaults[i - first_default]
    return values


_MISSING: Any = object()


def _floats(*ts: Any) -> bool:
    """Whether every operand is a real floating tensor, all of one dtype."""
    first = ts[0]
    return all(
        isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype == first.dtype for t in ts
    )


class _Interceptor(TorchFunctionMode):
    """The mode a :class:`simulate` enters. Its ``__enter__`` and
    ``__exit__`` are the default ones, which Dynamo traces."""

    def __init__(self, state: _State):
        super().__init__()
        self.state = state

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        handler = self.state.handlers.get(func)
        if handler is None or not self.state.stack:
            return func(*args, **kwargs)
        return handler(self.state, func, args, kwargs)


# --- handlers --------------------------------------------------------------------------

_NOT_FLOAT = "the operands are not of one float dtype"
_OUT = "out= writes into a given tensor"
_NOT_MATRICES = "the operands are not float matrices"
_NOT_BATCHES = "the operands are not float batches"
_LEFT_WEIGHT = "the weight is the left operand"


def _owner(state: _State, weight: Any) -> str | None:
    """The module a product with ``weight`` is resolved under, if ``weight``
    is one of the model's Parameters: the innermost module, if it owns it,
    and otherwise the first module that does."""
    owners = state.owners.get(id(weight))
    if owners is None:
        return None
    top = state.stack[-1][0]
    return top if top in owners else owners[0]


def _route(state: _State, func: Any, kind: str, scope: str, site: str | None = None):
    """The route of a call, or ``None`` (and the reason recorded) to pass it through."""
    top, _, whole = state.stack[-1]
    if top in state.owned:
        state.record(top, kind, site, func, OWNED)
        return None
    if whole:
        state.record(top, kind, site, func, WHOLE)
        return None
    key = (scope, kind, site if site in state.sites else None)
    route = state.routes.get(key)
    if route is None:
        state.record(scope, kind, site, func, PLAIN)
        return None
    state.record(scope, kind, site, func, SIMULATED, route.label)
    return route


def _site(state: _State) -> tuple[str, str]:
    """The innermost module and the site of its next product of two activations."""
    frame = state.stack[-1]
    n = frame[1]
    frame[1] = n + 1
    return frame[0], f"matmul#{n}"


def _not_covered(state: _State, func: Any, kind: str, reason: str, args, kwargs):
    """Pass a call through, recording why; under ``strict``, raise instead
    when the recipe gives the module formats of the call's kind, or, for a
    product, of some GEMM kind."""
    scope, _, whole = state.stack[-1]
    if scope in state.owned or whole:
        state.record(scope, kind, None, func, OWNED if scope in state.owned else WHOLE)
        return func(*args, **kwargs)
    _uncovered_call(state, func, kind, reason, scope)
    return func(*args, **kwargs)


def _uncovered_call(state: _State, func: Any, kind: str, reason: str, scope: str) -> None:
    """Record that ``func`` in ``scope`` is not simulated, or, under
    ``strict``, raise when the recipe covers it: when it gives the module
    formats of the call's kind, or, for a product (attention's included), of
    some GEMM kind."""
    kinds = state.covered.get(scope, set())
    products = not kinds.isdisjoint(_GEMM_KINDS)
    if kind == "attention":
        covered = kind in kinds or products
    else:
        covered = kind in kinds if kind in _OPT_IN_KINDS else products
    if state.strict and covered:
        raise RuntimeError(
            f"{_NAMES[func]} in {scope or '(the model)'} is not simulated: {reason}. With "
            "strict=True a product the recipe covers raises when simulate cannot run it; "
            "pass strict=False to let it run in PyTorch's arithmetic, as the report then says"
        )
    state.record(scope, kind, None, func, f"not covered: {reason}")


def _linear(state: _State, func, args, kwargs):
    x, weight, bias = _bind(args, kwargs, ("input", "weight", "bias"), (None,))
    if not _floats(x, weight) or (bias is not None and not _floats(x, bias)):
        return _not_covered(state, func, "linear", _NOT_FLOAT, args, kwargs)
    route = _route(state, func, "linear", _owner(state, weight) or state.stack[-1][0])
    if route is None:
        return func(*args, **kwargs)
    return CustomArithLinear.apply(x, weight, bias, route.container)


def _matmul_route(state: _State, func, a, b, args, kwargs, *, rank: int | None = None):
    """``a @ b``, simulated: a linear when ``b`` is one of the model's 2D
    Parameters, and otherwise a product of two activations at its site."""
    if kwargs.get("out") is not None:
        return _not_covered(state, func, "matmul", _OUT, args, kwargs)
    if not _floats(a, b):
        return _not_covered(state, func, "matmul", _NOT_FLOAT, args, kwargs)
    if rank is not None and (a.dim() != rank or b.dim() != rank):
        return func(*args, **kwargs)  # let PyTorch raise its own error
    if _owner(state, a) is not None:
        return _not_covered(state, func, "linear", _LEFT_WEIGHT, args, kwargs)
    owner = _owner(state, b)
    if owner is not None:
        if b.dim() != 2:
            return _not_covered(state, func, "linear", "the weight is not a matrix", args, kwargs)
        route = _route(state, func, "linear", owner)
        if route is None:
            return func(*args, **kwargs)
        return CustomArithLinear.apply(a, b.mT, None, route.container)
    scope, site = _site(state)
    route = _route(state, func, "matmul", scope, site)
    if route is None:
        return func(*args, **kwargs)
    return CustomArithMatmul.apply(a, b, route.container)


def _matmul(state: _State, func, args, kwargs):
    a, b = _bind(args, kwargs, ("input", "other"), ())
    return _matmul_route(state, func, a, b, args, kwargs)


def _rmatmul(state: _State, func, args, kwargs):
    # Tensor.__rmatmul__(self, other) is other @ self
    b, a = _bind(args, kwargs, ("self", "other"), ())
    return _matmul_route(state, func, a, b, args, kwargs)


def _mm(state: _State, func, args, kwargs):
    a, b = _bind(args, kwargs, ("input", "mat2"), ())
    return _matmul_route(state, func, a, b, args, kwargs, rank=2)


def _bmm(state: _State, func, args, kwargs):
    a, b = _bind(args, kwargs, ("input", "mat2"), ())
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dim() == b.dim() == 3:
        if a.shape[0] != b.shape[0]:
            return func(*args, **kwargs)  # let PyTorch raise its own error
    return _matmul_route(state, func, a, b, args, kwargs, rank=3)


def _addmm(state: _State, func, args, kwargs):
    """``beta * input + alpha * (mat1 @ mat2)``: a linear with ``input`` as
    its bias when ``mat2`` is a weight (GPT-2's ``Conv1D``), and otherwise a
    product of two activations plus ``input``; ``alpha = beta = 1`` only."""
    c, a, b = _bind(args, kwargs, ("input", "mat1", "mat2"), ())
    if kwargs.get("alpha", 1) != 1 or kwargs.get("beta", 1) != 1:
        return _not_covered(state, func, "matmul", "alpha or beta is not 1", args, kwargs)
    if kwargs.get("out") is not None:
        return _not_covered(state, func, "matmul", _OUT, args, kwargs)
    if not _floats(a, b, c) or a.dim() != 2 or b.dim() != 2:
        return _not_covered(state, func, "matmul", _NOT_MATRICES, args, kwargs)
    if _owner(state, a) is not None:
        return _not_covered(state, func, "linear", _LEFT_WEIGHT, args, kwargs)
    owner = _owner(state, b)
    if owner is not None:
        route = _route(state, func, "linear", owner)
        if route is None:
            return func(*args, **kwargs)
        if c.dim() == 1 and c.shape[0] == b.shape[1]:
            return CustomArithLinear.apply(a, b.mT, c, route.container)
        return CustomArithLinear.apply(a, b.mT, None, route.container) + c
    scope, site = _site(state)
    route = _route(state, func, "matmul", scope, site)
    if route is None:
        return func(*args, **kwargs)
    return CustomArithMatmul.apply(a, b, route.container) + c


def _baddbmm(state: _State, func, args, kwargs):
    c, a, b = _bind(args, kwargs, ("input", "batch1", "batch2"), ())
    if kwargs.get("alpha", 1) != 1 or kwargs.get("beta", 1) != 1:
        return _not_covered(state, func, "matmul", "alpha or beta is not 1", args, kwargs)
    if kwargs.get("out") is not None:
        return _not_covered(state, func, "matmul", _OUT, args, kwargs)
    if not _floats(a, b, c) or a.dim() != 3 or b.dim() != 3 or a.shape[0] != b.shape[0]:
        return _not_covered(state, func, "matmul", _NOT_BATCHES, args, kwargs)
    if _owner(state, a) is not None or _owner(state, b) is not None:
        return _not_covered(state, func, "matmul", "an operand is a weight", args, kwargs)
    scope, site = _site(state)
    route = _route(state, func, "matmul", scope, site)
    if route is None:
        return func(*args, **kwargs)
    return CustomArithMatmul.apply(a, b, route.container) + c


def _conv(nd: int):
    expand = {1: _single, 2: _pair, 3: _triple}[nd]

    def handler(state: _State, func, args, kwargs):
        x, weight, bias, stride, padding, dilation, groups = _bind(
            args,
            kwargs,
            ("input", "weight", "bias", "stride", "padding", "dilation", "groups"),
            (None, 1, 0, 1, 1),
        )
        if not _floats(x, weight) or (bias is not None and not _floats(x, bias)):
            return _not_covered(state, func, "conv", _NOT_FLOAT, args, kwargs)
        route = _route(state, func, "conv", _owner(state, weight) or state.stack[-1][0])
        if route is None:
            return func(*args, **kwargs)
        # as nn.Conv*d stores them, which QConv*d passes on; a string padding stays one
        padding = padding if isinstance(padding, str) else expand(padding)
        return CustomArithConvNd.apply(
            x,
            weight,
            bias,
            route.container,
            expand(stride),
            padding,
            expand(dilation),
            groups,
            nd,
            torch.is_grad_enabled(),
        )

    return handler


def _uncovered(kind: str, reason: str):
    def handler(state: _State, func, args, kwargs):
        return _not_covered(state, func, kind, reason, args, kwargs)

    return handler


# --- nn.MultiheadAttention -----------------------------------------------------------

_NO_ATTENTION = "the recipe names no attention kind, and its products run inside this one call"

_MHA_NAMES = (
    "query",
    "key",
    "value",
    "embed_dim_to_check",
    "num_heads",
    "in_proj_weight",
    "in_proj_bias",
    "bias_k",
    "bias_v",
    "add_zero_attn",
    "dropout_p",
    "out_proj_weight",
    "out_proj_bias",
    "training",
    "key_padding_mask",
    "need_weights",
    "attn_mask",
    "use_separate_proj_weight",
    "q_proj_weight",
    "k_proj_weight",
    "v_proj_weight",
    "static_k",
    "static_v",
    "average_attn_weights",
    "is_causal",
)
_MHA_DEFAULTS = (True, None, True, None, False, None, None, None, None, None, True, False)


def _projection(state: _State, weight: Any) -> _Route | None:
    """The ``linear`` route of a projection with ``weight``, under its owner."""
    owner = _owner(state, weight)
    return None if owner is None else state.routes.get((owner, "linear", None))


def _mha(state: _State, func, args, kwargs):
    """``F.multi_head_attention_forward``, its projections as their weights'
    owners' ``linear`` routes say and its core as the ``attention`` route of
    the in-projection's owner says, as a converted ``QMultiheadAttention``
    computes it."""
    values = _bind(args, kwargs, _MHA_NAMES, _MHA_DEFAULTS)
    top, _, whole = state.stack[-1]
    if top in state.owned or whole:
        state.record(top, "attention", None, func, OWNED if top in state.owned else WHOLE)
        return func(*args, **kwargs)
    separate = bool(values[17])
    weights = values[18:21] if separate else values[5:6]
    scope = _owner(state, weights[0]) or top
    in_routes = [_projection(state, w) for w in weights]
    out_route = _projection(state, values[11])
    core = state.routes.get((scope, "attention", None))
    if not state.attention_named:
        _uncovered_call(state, func, "attention", _NO_ATTENTION, scope)
    containers = [None if r is None else r.container for r in in_routes]
    af = None if core is None else core.container
    if all(c is None for c in containers) and out_route is None and (af is None or af._plain):
        if state.attention_named:
            state.record(scope, "attention", None, func, PLAIN)
        return func(*args, **kwargs)
    label = "; ".join(
        f"{name} [{'torch' if r is None else r.label}]"
        for name, r in (("in_proj", in_routes[0]), ("out_proj", out_route), ("attention", core))
    )
    state.record(scope, "attention", None, func, SIMULATED, label)
    return attention._qmha_forward(
        *values,
        in_proj=tuple(containers) if separate else containers[0],
        out_proj=None if out_route is None else out_route.container,
        attention=af,
        qk=None if af is None else af._qk,
        pv=None if af is None else af._pv,
    )


# --- the handlers ----------------------------------------------------------------------

_HANDLERS: dict[Any, Any] = {
    F.linear: _linear,
    torch.matmul: _matmul,
    torch.Tensor.matmul: _matmul,
    torch.Tensor.__matmul__: _matmul,
    torch.Tensor.__rmatmul__: _rmatmul,
    torch.mm: _mm,
    torch.Tensor.mm: _mm,
    torch.bmm: _bmm,
    torch.Tensor.bmm: _bmm,
    torch.addmm: _addmm,
    torch.Tensor.addmm: _addmm,
    torch.baddbmm: _baddbmm,
    torch.Tensor.baddbmm: _baddbmm,
    F.conv1d: _conv(1),
    F.conv2d: _conv(2),
    F.conv3d: _conv(3),
    F.multi_head_attention_forward: _mha,
}
_OTHER_PRODUCTS = {
    "gemm": [
        torch.einsum,
        torch.tensordot,
        torch.chain_matmul,
        torch.linalg.multi_dot,
        torch.linalg.vecdot,
        F.bilinear,
        *(
            getattr(owner, name)
            for owner in (torch, torch.Tensor)
            for name in ("addbmm", "addmv", "mv", "dot", "vdot", "inner", "outer", "ger")
        ),
        torch.Tensor.addmm_,
        torch.Tensor.baddbmm_,
    ],
    "conv": [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d],
    "attention": [F.scaled_dot_product_attention],
}
_REASONS = {
    "gemm": "simulate runs F.linear, matmul, mm, bmm, addmm and baddbmm, out of place, only",
    "conv": "simulate runs conv1d, conv2d and conv3d only",
    "attention": _NO_ATTENTION,
}
for _kind, _funcs in _OTHER_PRODUCTS.items():
    for _f in _funcs:
        _HANDLERS.setdefault(_f, _uncovered(_kind, _REASONS[_kind]))
# --- the kinds a recipe opts into --------------------------------------------------------

_IN_PLACE = "it writes its operand in place"
_NOT_FLOAT_TENSOR = "the operand is not a float tensor"


def _activation(op: str, names: tuple[str, ...], defaults: tuple):
    """The handler of an activation call whose parameters are ``names``."""

    def handler(state: _State, func, args, kwargs):
        if kwargs.get("out") is not None:
            return _not_covered(state, func, "activation", _OUT, args, kwargs)
        values = _bind(args, kwargs, names, defaults)
        x = values[0]
        if not _floats(x):
            return _not_covered(state, func, "activation", _NOT_FLOAT_TENSOR, args, kwargs)
        route = _route(state, func, "activation", state.stack[-1][0])
        if route is None:
            return func(*args, **kwargs)
        if op == "gelu":
            return pointwise.qgelu(x, route.container, values[1])
        return _ACTIVATION_OPS[op](x, route.container)

    return handler


_BINARY_OPS = {
    "add": pointwise.qadd,
    "sub": pointwise.qsub,
    "mul": pointwise.qmul,
    "div": pointwise.qdiv,
}
_UNARY_OPS = {"sqrt": pointwise.qsqrt, "rsqrt": pointwise.qrsqrt, "exp": pointwise.qexp}

_ACTIVATION_OPS = {
    "silu": pointwise.qsilu,
    "sigmoid": pointwise.qsigmoid,
    "tanh": pointwise.qtanh,
    "hardswish": pointwise.qhardswish,
    "hardsigmoid": pointwise.qhardsigmoid,
    "relu": pointwise.qrelu,
}


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _elementwise_operands(a: Any, b: Any) -> str | None:
    """Why ``a op b`` cannot be simulated, ``""`` when it can, or ``None``
    when it is not float arithmetic at all (index arithmetic passes through,
    unrecorded)."""
    tensors = [t for t in (a, b) if isinstance(t, torch.Tensor)]
    if not any(t.is_floating_point() for t in tensors):
        return None
    if not all(isinstance(v, torch.Tensor) or _is_number(v) for v in (a, b)):
        return "an operand is neither a tensor nor an int or float"
    if not _floats(*tensors):
        return _NOT_FLOAT
    return ""


def _binary(op: str, swap: bool = False):
    """The handler of ``a op b``; ``swap`` for ``Tensor.__rsub__`` and
    ``Tensor.__rtruediv__``, whose ``self`` is the right operand."""

    def handler(state: _State, func, args, kwargs):
        if kwargs.get("out") is not None:
            return _not_covered(state, func, "elementwise", _OUT, args, kwargs)
        if kwargs.get("alpha", 1) != 1:
            return _not_covered(state, func, "elementwise", "alpha is not 1", args, kwargs)
        if kwargs.get("rounding_mode") is not None:
            reason = "it rounds the quotient to an integer"
            return _not_covered(state, func, "elementwise", reason, args, kwargs)
        a, b = _bind(
            args,
            {k: v for k, v in kwargs.items() if k in ("input", "other")},
            ("input", "other"),
            (),
        )
        if swap:
            a, b = b, a
        why = _elementwise_operands(a, b)
        if why is None:
            return func(*args, **kwargs)
        if why:
            return _not_covered(state, func, "elementwise", why, args, kwargs)
        route = _route(state, func, "elementwise", state.stack[-1][0])
        if route is None:
            return func(*args, **kwargs)
        f = route.container
        return _BINARY_OPS[op](a, b, f.quant, f.grad_quant, **f._keywords)

    return handler


def _unary(op: str):
    def handler(state: _State, func, args, kwargs):
        if kwargs.get("out") is not None:
            return _not_covered(state, func, "elementwise", _OUT, args, kwargs)
        if op == "pow":
            x, exponent = _bind(args, kwargs, ("input", "exponent"), ())
        else:
            (x,), exponent = _bind(args, kwargs, ("input",), ()), None
        if not (isinstance(x, torch.Tensor) and x.is_floating_point()):
            if isinstance(exponent, torch.Tensor) and exponent.is_floating_point():
                reason = "a number is raised to a tensor"
                return _not_covered(state, func, "elementwise", reason, args, kwargs)
            return func(*args, **kwargs)  # index arithmetic
        if op == "pow" and not _is_number(exponent):
            reason = "the exponent is not an int or float"
            return _not_covered(state, func, "elementwise", reason, args, kwargs)
        route = _route(state, func, "elementwise", state.stack[-1][0])
        if route is None:
            return func(*args, **kwargs)
        f = route.container
        if op == "pow":
            return pointwise.qpow(x, cast(float, exponent), f.quant, f.grad_quant, **f._keywords)
        return _UNARY_OPS[op](x, f.quant, f.grad_quant, **f._keywords)

    return handler


def _in_place(kind: str):
    """The handler of an in-place method: passed through, and reported when
    its tensor is a float one (an index counter's is not arithmetic)."""

    def handler(state: _State, func, args, kwargs):
        x = args[0] if args else None
        if isinstance(x, torch.Tensor) and x.is_floating_point():
            return _not_covered(state, func, kind, _IN_PLACE, args, kwargs)
        return func(*args, **kwargs)

    return handler


def _embedding(state: _State, func, args, kwargs):
    names = (
        "input",
        "weight",
        "padding_idx",
        "max_norm",
        "norm_type",
        "scale_grad_by_freq",
        "sparse",
    )
    values = _bind(args, kwargs, names, (None, None, 2.0, False, False))
    input, weight, padding_idx, max_norm, norm_type, by_freq, sparse = values
    if not _floats(weight):
        return _not_covered(state, func, "embedding", _NOT_FLOAT_TENSOR, args, kwargs)
    route = _route(state, func, "embedding", _owner(state, weight) or state.stack[-1][0])
    if route is None:
        return func(*args, **kwargs)
    f = route.container
    if sparse and f.grad_quant is not None:
        reason = "its gradient is sparse, which grad_quant cannot round"
        return _not_covered(state, func, "embedding", reason, args, kwargs)
    return pointwise.qembedding(input, weight, f, padding_idx, max_norm, norm_type, by_freq, sparse)


def _softmax(names: tuple[str, ...], defaults: tuple):
    """The handler of a softmax call whose parameters are ``names``."""

    def handler(state: _State, func, args, kwargs):
        values = _bind(args, kwargs, names, defaults)
        x, dim, dtype = values[0], values[1], values[-1]
        if not _floats(x):
            return _not_covered(state, func, "softmax", _NOT_FLOAT_TENSOR, args, kwargs)
        route = _route(state, func, "softmax", state.stack[-1][0])
        if route is None:
            return func(*args, **kwargs)
        return norm.qsoftmax(x, dim, route.container, dtype=dtype)

    return handler


def _is_float_param(p: Any) -> bool:
    return p is None or (isinstance(p, torch.Tensor) and p.is_floating_point())


def _norm(op: str, names: tuple[str, ...], defaults: tuple):
    """The handler of a norm functional whose parameters are ``names``, the
    ones of its mptorch op before ``formats``."""
    weight_at = names.index("weight")
    params = [i for i, n in enumerate(names) if n in ("weight", "bias")]

    def handler(state: _State, func, args, kwargs):
        values = _bind(args, kwargs, names, defaults)
        if not _floats(values[0]) or not all(_is_float_param(values[i]) for i in params):
            return _not_covered(state, func, "norm", _NOT_FLOAT, args, kwargs)
        scope = _owner(state, values[weight_at]) or state.stack[-1][0]
        route = _route(state, func, "norm", scope)
        if route is None:
            return func(*args, **kwargs)
        return _NORM_OPS[op](*values, route.container)

    return handler


_NORM_OPS = {
    "layer_norm": norm.qlayernorm,
    "rms_norm": norm.qrmsnorm,
    "group_norm": norm.qgroupnorm,
    "batch_norm": norm.qbatchnorm,
}

_REDUCTION_KEYWORDS = {"input", "dim", "keepdim", "axis", "dtype", "out"}


def _reduction(op: str):
    """The handler of ``torch.sum`` or ``torch.mean``, or their ``Tensor.``
    spellings. NumPy's ``axis`` is read as ``dim``, as torch reads it."""

    def handler(state: _State, func, args, kwargs):
        x = args[0] if args else kwargs.get("input")
        if not (isinstance(x, torch.Tensor) and x.is_floating_point()):
            return func(*args, **kwargs)  # counting, not arithmetic
        if kwargs.get("out") is not None:
            return _not_covered(state, func, "reduction", _OUT, args, kwargs)
        if kwargs.get("dtype") is not None:
            reason = "dtype= converts the operand first"
            return _not_covered(state, func, "reduction", reason, args, kwargs)
        unknown = set(kwargs) - _REDUCTION_KEYWORDS
        if unknown or ("axis" in kwargs and "dim" in kwargs):
            reason = f"its arguments ({', '.join(sorted(kwargs))}) are not sum's or mean's"
            return _not_covered(state, func, "reduction", reason, args, kwargs)
        named = {
            ("dim" if k == "axis" else k): v for k, v in kwargs.items() if k not in ("dtype", "out")
        }
        _, dim, keepdim = _bind(args, named, ("input", "dim", "keepdim"), (None, False))
        route = _route(state, func, "reduction", state.stack[-1][0])
        if route is None:
            return func(*args, **kwargs)
        f = route.container
        dims = reduce._reduced(x, dim)
        if op == "sum":
            return reduce._qsum(x, dims, keepdim, f._spec, f.grad, f.carrier)
        return reduce._qmean(x, dims, keepdim, f._spec, f.div, f.grad, f.carrier)

    return handler


_SDPA_NAMES = (
    "query",
    "key",
    "value",
    "attn_mask",
    "dropout_p",
    "is_causal",
    "scale",
    "enable_gqa",
)


def _sdpa(state: _State, func, args, kwargs):
    q, k, v, mask, p, causal, scale, gqa = _bind(
        args, kwargs, _SDPA_NAMES, (None, 0.0, False, None, False)
    )
    if not _floats(q, k, v):
        return _not_covered(state, func, "attention", _NOT_FLOAT, args, kwargs)
    route = _route(state, func, "attention", state.stack[-1][0])
    if route is None:
        return func(*args, **kwargs)
    return attention.qattention(
        q,
        k,
        v,
        route.container,
        attn_mask=mask,
        dropout_p=p,
        is_causal=causal,
        scale=scale,
        enable_gqa=gqa,
    )


_OPT_IN_HANDLERS: dict[str, dict[Any, Any]] = {
    "activation": {
        F.gelu: _activation("gelu", ("input", "approximate"), ("none",)),
        F.silu: _activation("silu", ("input", "inplace"), (False,)),
        F.hardswish: _activation("hardswish", ("input", "inplace"), (False,)),
        F.hardsigmoid: _activation("hardsigmoid", ("input", "inplace"), (False,)),
        F.relu: _activation("relu", ("input", "inplace"), (False,)),
        torch.relu: _activation("relu", ("input",), ()),
        torch.Tensor.relu: _activation("relu", ("input",), ()),
        torch.sigmoid: _activation("sigmoid", ("input",), ()),
        torch.Tensor.sigmoid: _activation("sigmoid", ("input",), ()),
        torch.tanh: _activation("tanh", ("input",), ()),
        torch.Tensor.tanh: _activation("tanh", ("input",), ()),
        **{
            f: _in_place("activation")
            for f in (
                torch.relu_,
                torch.Tensor.relu_,
                torch.sigmoid_,
                torch.Tensor.sigmoid_,
                torch.tanh_,
                torch.Tensor.tanh_,
            )
        },
    },
    "elementwise": {
        **{
            getattr(owner, name): _binary(op)
            for owner in (torch, torch.Tensor)
            for name, op in (
                ("add", "add"),
                ("sub", "sub"),
                ("mul", "mul"),
                ("div", "div"),
                ("true_divide", "div"),
            )
        },
        torch.Tensor.__rsub__: _binary("sub", swap=True),
        torch.Tensor.__rtruediv__: _binary("div", swap=True),
        **{
            getattr(owner, op): _unary(op)
            for owner in (torch, torch.Tensor)
            for op in ("sqrt", "rsqrt", "exp", "pow")
        },
        torch.Tensor.__pow__: _unary("pow"),
        torch.Tensor.__rpow__: lambda state, func, args, kwargs: (
            _not_covered(state, func, "elementwise", "a number is raised to a tensor", args, kwargs)
            if isinstance(args[0], torch.Tensor) and args[0].is_floating_point()
            else func(*args, **kwargs)
        ),
        **{
            f: _in_place("elementwise")
            for f in (
                *(
                    getattr(torch.Tensor, n)
                    for n in ("add_", "sub_", "mul_", "div_", "true_divide_")
                ),
                *(getattr(torch.Tensor, n) for n in ("sqrt_", "rsqrt_", "exp_", "pow_")),
                torch.sqrt_,
                torch.rsqrt_,
                torch.exp_,
            )
        },
    },
    "embedding": {F.embedding: _embedding},
    "softmax": {
        F.softmax: _softmax(("input", "dim", "_stacklevel", "dtype"), (None, 3, None)),
        torch.softmax: _softmax(("input", "dim", "dtype"), (None,)),
        torch.Tensor.softmax: _softmax(("input", "dim", "dtype"), (None,)),
    },
    "norm": {
        F.layer_norm: _norm(
            "layer_norm", ("input", "normalized_shape", "weight", "bias", "eps"), (None, None, 1e-5)
        ),
        F.rms_norm: _norm("rms_norm", ("input", "normalized_shape", "weight", "eps"), (None, None)),
        F.group_norm: _norm(
            "group_norm", ("input", "num_groups", "weight", "bias", "eps"), (None, None, 1e-5)
        ),
        F.batch_norm: _norm(
            "batch_norm",
            (
                "input",
                "running_mean",
                "running_var",
                "weight",
                "bias",
                "training",
                "momentum",
                "eps",
            ),
            (None, None, False, 0.1, 1e-5),
        ),
    },
    "reduction": {
        torch.sum: _reduction("sum"),
        torch.Tensor.sum: _reduction("sum"),
        torch.mean: _reduction("mean"),
        torch.Tensor.mean: _reduction("mean"),
    },
    "attention": {F.scaled_dot_product_attention: _sdpa},
}

# each call's name in a report: F.linear, Tensor.matmul, torch.einsum
_NAMES = {
    f: (resolve_name(f) or repr(f))
    .replace("torch.nn.functional.", "F.")
    .replace("torch.functional.", "torch.")
    .replace("torch.Tensor.", "Tensor.")
    for f in (*_HANDLERS, *(f for table in _OPT_IN_HANDLERS.values() for f in table))
}
_NAMES[torch.mm] = "torch.mm"  # torch's resolve_name gives "torch.spmm" (torch 2.15)


# --- the context ---------------------------------------------------------------------


class simulate:
    """Run ``model`` in the arithmetic ``recipe`` gives it, while active.

    ``with simulate(recipe, model) as sim:`` leaves the model as it is and
    intercepts its calls: each product the recipe gives formats to runs
    through the autograd Function a converted layer runs (``F.linear`` as
    :class:`~mptorch.quant.QLinear` does, ``torch.matmul`` as
    :class:`~mptorch.quant.QMatmul` does), and the others run as before. So
    on the layers :func:`~mptorch.quant.convert` would convert, the results
    are the converted model's, word for word, gradients and stochastic
    rounding included; and the products only a function call computes
    (attention's, GPT-2's ``Conv1D``, swin's) are reached as well. A recipe
    that names the ``activation``, ``elementwise`` or ``embedding`` kinds
    reaches a model's activations, its pointwise arithmetic (residual sums,
    scalings, a norm's ``rsqrt``) and its embeddings too, through
    ``mptorch.quant.pointwise``'s ops, and one that names ``softmax``,
    ``norm`` or ``reduction`` its softmaxes, norms, sums and means, through
    ``mptorch.quant.norm``'s and ``mptorch.quant.reduce``'s. The module
    docstring says which
    formats a call gets: by kind, under the innermost module running, and
    at a site for a product of two activations.

    The model's structure and the recipe are read when the ``simulate`` is
    made: every module's formats are resolved then, and the containers
    built. A module added afterwards counts as part of the module that calls
    it. Calls made outside the model's modules are not intercepted, and
    neither are those inside an mptorch layer (a converted model's layers
    run as converted). ``report()`` lists what was seen.

    Under ``torch.compile``, compile :meth:`wrap`'s module, or enter the
    ``simulate`` outside the compiled call; entering it inside compiled code
    raises. One model is simulated by one thread at a time.

    Args:
        recipe (Recipe): the formats.
        model (nn.Module): the model, or a module compiled with
            ``torch.compile`` (its names are the original module's).
        strict (bool): raise, instead of reporting, when a call cannot be
            simulated in a module the recipe gives formats to: a product (an
            ``einsum``, ``scaled_dot_product_attention``) where some GEMM
            kind has formats, an in-place activation or arithmetic where its
            own kind has. Default: ``False``
        events (bool): count the numeric events of the simulated products'
            casts (:mod:`mptorch.quant.events`), by site, pass and role, which
            :meth:`events` reports: each product runs its GEMM op's counted
            twin, and each quantizer slot holding a ``Quant`` is recounted,
            the opt-in kinds' too (their sums through the counted GEMMs).
            It needs a build with ``MPTORCH_COUNTING=1`` (the report says so
            otherwise), runs in eager calls only, and gives each site a
            container of its own. Default: ``False``

    Raises:
        ValueError: for a module reachable under two names that the recipe
            gives different formats.

    Example::

        >>> import torch
        >>> from torch import nn
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import GemmFormats, Recipe, SplitMac, simulate
        >>> model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        >>> recipe = Recipe(linear=GemmFormats.uniform(SplitMac(BinaryK(8, 4), BinaryK(16, 11))))
        >>> with simulate(recipe, model) as sim:
        ...     out = model(torch.randn(4, 8))
        >>> len(sim.report().simulated)
        2
    """

    # what a simulate builds its state and its mode from; mptorch.quant.profile
    # counts the calls through subclasses of both
    _state_class: type[_State] = _State
    _mode_class: type[_Interceptor] = _Interceptor

    def __init__(
        self, recipe: Recipe, model: nn.Module, *, strict: bool = False, events: bool = False
    ):
        _outside_compile("made")
        if not isinstance(recipe, Recipe):
            raise TypeError(f"recipe must be a Recipe, got {type(recipe).__name__}")
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        model = getattr(model, "_orig_mod", model)  # a torch.compile'd module's original
        self.model = model
        self._events = events
        self._state = (
            self._state_class(recipe, model, strict, events)
            if events
            else self._state_class(recipe, model, strict)
        )
        self._mode = self._mode_class(self._state)
        self._hooks: list[Any] = []
        self._users = 0

    def _install(self) -> None:
        if self._users == 0:
            state = self._state
            state.stack.clear()

            def post(module, args, output):
                state.stack.pop()  # returns None: a value would replace the output

            for module in self.model.modules():
                name = state.names[id(module)]
                self._hooks.append(module.register_forward_pre_hook(_push(state, name)))
                whole = state.whole.get(name)
                if whole is not None:  # before the pop, so that the module is still running
                    hook = _replace(state, name, whole)
                    self._hooks.append(module.register_forward_hook(hook, with_kwargs=True))
                self._hooks.append(module.register_forward_hook(post, always_call=True))
        self._users += 1

    def _release(self) -> None:
        self._users -= 1
        if self._users == 0:
            for h in self._hooks:
                h.remove()
            self._hooks.clear()

    def __enter__(self) -> "simulate":
        _outside_compile("entered")
        self._install()
        try:
            self._mode.__enter__()
        except BaseException:
            self._release()
            raise
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            self._mode.__exit__(*exc)
        finally:
            self._release()

    def report(self) -> SimulateReport:
        """What was seen so far, as a :class:`SimulateReport`."""
        from .palettes import _simulate_slots

        # the maps that are set, and the unset ones of the modules that ran a
        # simulated product of their kind
        used = {(r.scope, r.kind) for r in self._state.rows if r.action == SIMULATED}
        maps = tuple(
            (name, slot[0].prec_idx is not None)
            for name, slot in _simulate_slots(self).items()
            if slot[0].prec_idx is not None or not used.isdisjoint(_route_keys(name))
        )
        return SimulateReport(tuple(self._state.rows), maps)

    def events(self) -> EventReport:
        """The numeric events counted so far, as an :class:`EventReport`: by
        site, pass and role, with the parts of the simulated sites that are
        not counted and why. It reads the counts from the device.

        Raises:
            RuntimeError: for a ``simulate`` made without ``events=True``.
        """
        if not self._events:
            raise RuntimeError("this simulate counts no events: make it with events=True")
        return _event_report(self._state.tallies)

    def wrap(self) -> "SimulatedModule":
        """The model as a module that simulates every call to it: the one to
        compile, as ``torch.compile(sim.wrap(), fullgraph=True)``.

        Its forward enters the mode inside the compiled code, and the hooks
        stay on the model's modules while the wrapper lives (until it is
        collected or its ``close()`` is called). Call it outside
        ``with sim:``, which would enter the mode a second time.

        Raises:
            RuntimeError: for a ``simulate`` made with ``events=True``: events
                are counted in eager calls only.
        """
        if self._events:
            raise RuntimeError(
                "a simulate made with events=True counts in eager calls only, and cannot be "
                "wrapped for torch.compile: run it eagerly, or make one without events"
            )
        return SimulatedModule(self)


def _route_keys(name: str) -> tuple[tuple[str, str], ...]:
    """The (scope, kind) pairs a palette map's name can stand for: a module's
    linear or conv layer, or, for ``"module#matmul..."``, its matmuls."""
    scope, sep, _ = name.partition("#")
    return ((scope, "matmul"),) if sep else ((scope, "linear"), (scope, "conv"))


def _outside_compile(what: str) -> None:
    """Refuse to make or enter a ``simulate`` inside compiled code: the
    containers it builds and the hooks it installs there come out wrong
    (wrong gradients, an empty stack) rather than failing."""
    if torch.compiler.is_compiling():
        raise RuntimeError(
            f"simulate was {what} inside compiled code, where its formats and hooks cannot be "
            "set up: make and enter it outside torch.compile, or compile sim.wrap() instead"
        )


def _push(state: _State, name: str):
    whole = name in state.whole

    def pre(module, args):
        # a module's frame is inside a whole module's own forward if it is that
        # module's, or its parent frame is
        inside = whole or (bool(state.stack) and state.stack[-1][2])
        state.stack.append([name, 0, inside])

    return pre


def _replace(state: _State, name: str, whole: tuple):
    """The hook that runs a module's registered forward and returns its result
    in place of the module's own (which ran with its calls passed through)."""
    forward, formats, kind, label, cls = whole
    call = f"{cls}.forward"

    def hook(module, args, kwargs, output):
        stack = state.stack
        if len(stack) > 1 and stack[-2][2]:
            # inside another whole module's own forward, whose result is
            # discarded: keep this one's own output too, rather than run its
            # registered forward for nothing
            state.record(name, kind, None, call, WHOLE)
            return None
        # its own forward is over: the modules the registered forward calls
        # are simulated, as a converted module's are
        stack[-1][2] = False
        state.record(name, kind, None, call, SIMULATED, label)
        return forward(module, formats, *args, **kwargs)

    return hook


class SimulatedModule(nn.Module):
    """A model whose every call runs under its :class:`simulate`
    (:meth:`simulate.wrap`). It holds the model as ``module``."""

    def __init__(self, sim: simulate):
        super().__init__()
        self.module = sim.model
        self._interceptor = sim._mode
        sim._install()
        self._finalizer = weakref.finalize(self, sim._release)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Call the model, simulated."""
        with self._interceptor:
            return self.module(*args, **kwargs)

    def close(self) -> None:
        """Remove the hooks this wrapper keeps on the model's modules."""
        self._finalizer()
