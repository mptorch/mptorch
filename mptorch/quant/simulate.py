"""Simulating a model's GEMMs while it runs, without converting it: ``simulate``.

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
* ``matmul``: a product of two activations (anything but a Parameter itself:
  a view of one, ``W.T``, counts as an activation). It is resolved under the
  innermost module, at a *site*: ``"matmul#0"`` for that module's first
  such product in a forward pass, ``"matmul#1"`` for its second, and so on,
  which a :class:`~mptorch.quant.Rule`'s ``site`` names.

Everything is resolved, and every format container built, when the
``simulate`` is made, so that a call only looks its route up: Dynamo can
trace that, and a container built while it traced gave wrong gradients.
"""

import weakref
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.utils import _pair, _single, _triple
from torch.overrides import TorchFunctionMode, resolve_name

from .modules.conv import CustomArithConvNd
from .modules.linear import CustomArithLinear
from .modules.matmul import CustomArithMatmul
from .recipe import _GEMM_KINDS, Recipe

__all__ = ["simulate", "SimulateReport", "SimulateRow"]

SIMULATED = "simulated"
PLAIN = "plain: the recipe gives None"
OWNED = "an mptorch layer's own call"


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
        return "\n".join(lines)


class _Route(NamedTuple):
    container: nn.Module
    label: str


class _State:
    """Everything a call reads, built once from the model and the recipe."""

    def __init__(self, recipe: Recipe, model: nn.Module, strict: bool):
        self.strict = strict
        self.names: dict[int, str] = {}
        self.owners: dict[int, tuple[str, ...]] = {}
        self.owned: set[str] = set()
        self.routes: dict[tuple[str, str, str | None], _Route | None] = {}
        self.covered: set[str] = set()  # the modules some GEMM kind has formats in
        self.sites = {r.site for r in recipe.rules if r.site is not None}
        # one [name, products of two activations so far] per module running:
        # kept on the stack, which is empty between calls, so that Dynamo has no
        # counter to guard on (a dict of counts made every call recompile)
        self.stack: list[list] = []
        self.rows: dict[SimulateRow, None] = {}  # a dict keeps the order rows were seen in

        containers: dict[tuple[int, str], nn.Module] = {}
        first: dict[int, tuple[str, tuple]] = {}
        for name, module in model.named_modules(remove_duplicate=False):
            resolved = tuple(
                (kind, site, recipe.resolve(name, module, kind, site))
                for kind in _GEMM_KINDS
                for site in ((None, *self.sites) if kind == "matmul" else (None,))
            )
            seen = first.get(id(module))
            if seen is not None:
                if seen[1] != resolved:
                    raise ValueError(
                        f"the module at {seen[0]!r} is also at {name!r}, and the recipe gives the "
                        "two names different formats: give both names the same formats"
                    )
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
                self.covered.add(name)
                key = (id(gf), kind)
                if key not in containers:
                    build = {"linear": gf.affine, "conv": gf.conv, "matmul": gf.matmul}[kind]
                    containers[key] = build()
                self.routes[(name, kind, site)] = _Route(containers[key], gf.label())

    def record(self, scope: str, kind: str, site: str | None, call: Any, action: str, label=""):
        # Not while Dynamo traces: it guards on what it reads, and a report
        # that grew during the first compiled call made the second recompile.
        if not torch.compiler.is_compiling():
            self.rows[SimulateRow(scope, kind, site, _NAMES[call], action, label)] = None


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
        handler = _HANDLERS.get(func)
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
    top = state.stack[-1][0]
    if top in state.owned:
        state.record(top, kind, site, func, OWNED)
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
    when the recipe gives the module formats of some GEMM kind."""
    scope = state.stack[-1][0]
    if scope in state.owned:
        state.record(scope, kind, None, func, OWNED)
        return func(*args, **kwargs)
    if state.strict and scope in state.covered:
        raise RuntimeError(
            f"{_NAMES[func]} in {scope or '(the model)'} is not simulated: {reason}. With "
            "strict=True a product the recipe covers raises when simulate cannot run it; "
            "pass strict=False to let it run in PyTorch's arithmetic, as the report then says"
        )
    state.record(scope, kind, None, func, f"not covered: {reason}")
    return func(*args, **kwargs)


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
    "attention": [F.scaled_dot_product_attention, F.multi_head_attention_forward],
}
_REASONS = {
    "gemm": "simulate runs F.linear, matmul, mm, bmm, addmm and baddbmm, out of place, only",
    "conv": "simulate runs conv1d, conv2d and conv3d only",
    "attention": "attention is not simulated yet: its products run inside one call",
}
for _kind, _funcs in _OTHER_PRODUCTS.items():
    for _f in _funcs:
        _HANDLERS.setdefault(_f, _uncovered(_kind, _REASONS[_kind]))
# each call's name in a report: F.linear, Tensor.matmul, torch.einsum
_NAMES = {
    f: (resolve_name(f) or repr(f))
    .replace("torch.nn.functional.", "F.")
    .replace("torch.functional.", "torch.")
    .replace("torch.Tensor.", "Tensor.")
    for f in _HANDLERS
}


# --- the context ---------------------------------------------------------------------


class simulate:
    """Run ``model`` in the arithmetic ``recipe`` gives its GEMMs, while active.

    ``with simulate(recipe, model) as sim:`` leaves the model as it is and
    intercepts its calls: each product the recipe gives formats to runs
    through the autograd Function a converted layer runs (``F.linear`` as
    :class:`~mptorch.quant.QLinear` does, ``torch.matmul`` as
    :class:`~mptorch.quant.QMatmul` does), and the others run as before. So
    on the layers :func:`~mptorch.quant.convert` would convert, the results
    are the converted model's, word for word, gradients and stochastic
    rounding included; and the products only a function call computes
    (attention's, GPT-2's ``Conv1D``, swin's) are reached as well. The module
    docstring says which formats a call gets: by kind, under the innermost
    module running, and at a site for a product of two activations.

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
        strict (bool): raise, instead of reporting, when a product in a
            module the recipe gives formats to cannot be simulated (an
            ``einsum``, ``scaled_dot_product_attention``). Default: ``False``

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

    def __init__(self, recipe: Recipe, model: nn.Module, *, strict: bool = False):
        _outside_compile("made")
        if not isinstance(recipe, Recipe):
            raise TypeError(f"recipe must be a Recipe, got {type(recipe).__name__}")
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        model = getattr(model, "_orig_mod", model)  # a torch.compile'd module's original
        self.model = model
        self._state = _State(recipe, model, strict)
        self._mode = _Interceptor(self._state)
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
        return SimulateReport(tuple(self._state.rows))

    def wrap(self) -> "SimulatedModule":
        """The model as a module that simulates every call to it: the one to
        compile, as ``torch.compile(sim.wrap(), fullgraph=True)``.

        Its forward enters the mode inside the compiled code, and the hooks
        stay on the model's modules while the wrapper lives (until it is
        collected or its ``close()`` is called). Call it outside
        ``with sim:``, which would enter the mode a second time.
        """
        return SimulatedModule(self)


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
    def pre(module, args):
        state.stack.append([name, 0])

    return pre


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
