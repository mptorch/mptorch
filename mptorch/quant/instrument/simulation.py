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
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from ..events import EventReport, _SiteTally
from ..events import _report as _event_report
from ._counting import _counting_formats
from ._handlers import _HANDLERS, _NAMES, _OPT_IN_HANDLERS, _OPT_IN_KINDS, SIMULATED, WHOLE
from .conversion import _REGISTRY
from .palettes import _PaletteMap
from .recipe import _GEMM_KINDS, Recipe

__all__ = ["simulate", "SimulateReport", "SimulateRow"]


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
                    if events:  # or a counting copy (instrument/_counting.py)
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
    scalings, a norm's ``rsqrt``) and its embeddings too, through the
    pointwise ops, activations and embedding of ``mptorch.quant.functional``,
    and one that names ``softmax``, ``norm`` or ``reduction`` its softmaxes,
    norms, sums and means, through that package's ``norm`` and ``reduce``.
    The module docstring says which
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

    def __enter__(self) -> simulate:
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

    def wrap(self) -> SimulatedModule:
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
