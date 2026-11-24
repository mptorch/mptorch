"""Converting a model's layers to mptorch's, from a recipe.

:func:`convert_` walks a model, asks the recipe for each layer's formats, and
turns each layer it gives formats to into its mptorch counterpart *in place*:
an ``nn.Linear`` becomes a :class:`~mptorch.quant.QLinear` and an
``nn.Conv2d`` a :class:`~mptorch.quant.QConv2d`, the same object with its
class changed (as ``torch.nn.utils.parametrize`` changes the class of the
modules it wraps) and a new ``formats`` container. Its parameters, buffers,
hooks and ``state_dict`` keys are unchanged, so an optimizer built before the
conversion still updates it. :func:`convert` converts a deep copy instead,
and :func:`unconvert_` turns the layers back.

Which classes convert, and to what, is a registry of exact classes:
:func:`register` adds one. The built-in entries are ``nn.Linear`` and
``nn.Conv1d/2d/3d``, torch's activation modules (``nn.GELU``, ``nn.SiLU``,
``nn.Sigmoid``, ``nn.Tanh``, ``nn.Hardswish``, ``nn.Hardsigmoid``,
``nn.ReLU``) and ``nn.Embedding``. A subclass of a registered class, such as a module
that ``torch.nn.utils.parametrize`` has wrapped, is not converted, since the
class switch would drop what the subclass adds; the report lists it, and
``strict=True`` raises instead.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from torch import nn
from torch.nn.utils import parametrize

from .modules.activation import QGELU, QHardsigmoid, QHardswish, QReLU, QSigmoid, QSiLU, QTanh
from .modules.conv import QConv1d, QConv2d, QConv3d
from .modules.embedding import QEmbedding
from .modules.linear import QLinear, QPackedLinear, _switch_class
from .pointwise import ActivationFormats, PointwiseFormats
from .recipe import _GEMM_KINDS, _KINDS, GemmFormats, Recipe, Rule

__all__ = ["register", "convert_", "convert", "unconvert_", "ConvertReport", "ConvertRow"]


class _Entry(NamedTuple):
    cls: type
    kind: str
    qcls: type


_REGISTRY: dict[type, _Entry] = {}

# How each kind's formats build the container its layer reads. An activation's
# and an embedding's formats are frozen values, which the layer reads as they are.
_CONTAINER: dict[str, Callable[[Any], Any]] = {
    "linear": GemmFormats.affine,
    "conv": GemmFormats.conv,
    "matmul": GemmFormats.matmul,
    "activation": lambda formats: formats,
    "embedding": lambda formats: formats,
}


def register(cls: type, *, kind: str, qcls: type) -> None:
    """Make :func:`convert_` turn modules of exactly ``cls`` into ``qcls``.

    ``qcls`` must subclass ``cls`` and compute through ``self.formats``: the
    container that the kind's :class:`GemmFormats` builds
    (:meth:`GemmFormats.affine` for ``linear``, :meth:`GemmFormats.conv` for
    ``conv``, :meth:`GemmFormats.matmul` for ``matmul``), or the kind's
    formats themselves for ``activation`` (an
    :class:`~mptorch.quant.ActivationFormats`) and ``embedding`` (a
    :class:`~mptorch.quant.PointwiseFormats`). A conversion switches the
    module's class and sets that attribute, and runs no ``__init__``.
    Registering a class again replaces its entry. The built-in entries are
    ``nn.Linear`` (``linear``), ``nn.Conv1d``, ``nn.Conv2d`` and
    ``nn.Conv3d`` (``conv``), torch's activation modules (``activation``)
    and ``nn.Embedding`` (``embedding``).

    Raises:
        ValueError: for an unknown kind, or ``elementwise``, which has no
            module to convert (only :class:`~mptorch.quant.simulate`
            reaches it).
        TypeError: if ``qcls`` is not a subclass of ``cls``.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}: the kinds are {', '.join(_KINDS)}")
    if kind not in _CONTAINER:
        raise ValueError(
            f"no module converts to the {kind!r} kind, which only simulate reaches (a model's "
            f"arithmetic is not a module): the kinds a module converts to are "
            f"{', '.join(_CONTAINER)}"
        )
    if not (isinstance(cls, type) and isinstance(qcls, type) and issubclass(qcls, cls)):
        raise TypeError(
            f"qcls must be a subclass of cls, which a conversion switches the module's class "
            f"to: got {qcls!r} for {cls!r}"
        )
    _REGISTRY[cls] = _Entry(cls, kind, qcls)


register(nn.Linear, kind="linear", qcls=QLinear)
register(nn.Conv1d, kind="conv", qcls=QConv1d)
register(nn.Conv2d, kind="conv", qcls=QConv2d)
register(nn.Conv3d, kind="conv", qcls=QConv3d)
register(nn.GELU, kind="activation", qcls=QGELU)
register(nn.SiLU, kind="activation", qcls=QSiLU)
register(nn.Sigmoid, kind="activation", qcls=QSigmoid)
register(nn.Tanh, kind="activation", qcls=QTanh)
register(nn.Hardswish, kind="activation", qcls=QHardswish)
register(nn.Hardsigmoid, kind="activation", qcls=QHardsigmoid)
register(nn.ReLU, kind="activation", qcls=QReLU)
register(nn.Embedding, kind="embedding", qcls=QEmbedding)


class ConvertRow(NamedTuple):
    """One module the conversion considered: where it is, what it is, the
    formats the recipe gives it, and what was done (``"converted"``, or why
    it was kept)."""

    name: str
    type: str
    kind: str
    formats: GemmFormats | ActivationFormats | PointwiseFormats | None
    action: str


@dataclass(frozen=True)
class ConvertReport:
    """What :func:`convert_` did, or, from :meth:`Recipe.plan`, would do.

    It has one row per module of a registered class or of a subclass of one,
    in the model's order, except the modules of a kind past the GEMMs
    (activations, embeddings) that no rule of the recipe names, so that a
    recipe for the GEMMs reports its layers alone. ``str(report)`` is a
    table.
    """

    rows: tuple[ConvertRow, ...]
    dry_run: bool = False

    @property
    def converted(self) -> tuple[ConvertRow, ...]:
        """The rows of the modules converted (or, in a dry run, to convert)."""
        return tuple(r for r in self.rows if r.action in (_CONVERTED, _TO_CONVERT))

    @property
    def kept(self) -> tuple[ConvertRow, ...]:
        """The rows of the modules left as they were, each with its reason."""
        return tuple(r for r in self.rows if r.action not in (_CONVERTED, _TO_CONVERT))

    def __str__(self) -> str:
        header = ("module", "type", "kind", "formats", "action")
        body = [
            (
                r.name or "(model)",
                r.type,
                r.kind,
                "plain torch" if r.formats is None else r.formats.label(),
                r.action,
            )
            for r in self.rows
        ]
        # every column padded to its widest cell, except the last
        widths = [max(len(row[i]) for row in (header, *body)) for i in range(4)]
        lines = [
            "  ".join(c.ljust(w) for c, w in zip(row[:4], widths, strict=True)) + "  " + row[4]
            for row in (header, *body)
        ]
        verb = "to convert" if self.dry_run else "converted"
        lines.append(f"{len(self.converted)} {verb}, {len(self.kept)} kept")
        return "\n".join(lines)


_CONVERTED, _TO_CONVERT = "converted", "to convert"


class _Decision(NamedTuple):
    row: ConvertRow
    module: nn.Module
    entry: _Entry | None


def _registered_base(module: nn.Module) -> _Entry | None:
    """The entry of the nearest registered class ``module`` derives from."""
    for c in type(module).__mro__[1:]:
        entry = _REGISTRY.get(c)
        if entry is not None:
            return entry
    return None


def _skipped(skip: Any, name: str, module: nn.Module) -> bool:
    """Whether ``skip`` names the module: one ``where`` of a :class:`Rule`, or
    a list or tuple of them (a tuple of types is one ``where``)."""
    wheres = skip if isinstance(skip, list | tuple) and not _is_type_tuple(skip) else (skip,)
    return any(Rule(w).matches(name, module) for w in wheres)


def _is_type_tuple(value: Any) -> bool:
    """Whether ``value`` is a non-empty tuple of types, one ``where``."""
    return isinstance(value, tuple) and bool(value) and all(isinstance(v, type) for v in value)


def _decide(
    model: nn.Module, recipe: Recipe, kinds: Any, skip: Any
) -> tuple[list[_Decision], list[str]]:
    """Decide each module's fate, and list what ``strict=True`` refuses."""
    if not isinstance(recipe, Recipe):
        raise TypeError(f"recipe must be a Recipe, got {type(recipe).__name__}")
    if kinds is not None:
        unknown = [k for k in kinds if k not in _KINDS]
        if unknown:
            raise ValueError(f"unknown kinds {unknown}: the kinds are {', '.join(_KINDS)}")
    qclasses = {e.qcls: e for e in _REGISTRY.values()}
    named = {kind for rule in recipe.rules for kind, _ in rule.kinds}
    seen: dict[int, tuple[str, Any]] = {}
    decisions: list[_Decision] = []
    refused: list[str] = []
    for name, module in model.named_modules(remove_duplicate=False):
        entry = _REGISTRY.get(type(module))
        base = None if entry is not None else _registered_base(module)
        kind_entry = entry or base
        if kind_entry is None or (kinds is not None and kind_entry.kind not in kinds):
            continue
        if kind_entry.kind not in _GEMM_KINDS and kind_entry.kind not in named:
            continue
        gf = recipe.resolve(name, module, kind_entry.kind)
        first = seen.get(id(module))
        if first is not None:
            if first[1] != gf:
                raise ValueError(
                    f"the module at {first[0]!r} is also at {name!r}, and the recipe gives the two "
                    "names different formats: a shared module converts once, so give both "
                    "names the same formats"
                )
            continue
        seen[id(module)] = (name, gf)
        type_name = type(module).__name__
        if entry is not None:
            if gf is None:
                action = "kept: the recipe gives it None"
            elif _skipped(skip, name, module):
                action = "kept: skipped"
            else:
                action = _CONVERTED
        elif type(module) in qclasses or isinstance(module, QPackedLinear):
            action = f"kept: already a {type_name}"
        else:
            why = (
                "parametrized"
                if parametrize.is_parametrized(module)
                else f"a subclass of {kind_entry.cls.__name__}"
            )
            action = f"kept: {why}, and only the exact class converts"
            if gf is not None and not _skipped(skip, name, module):
                refused.append(f"{name or '(model)'} ({type_name}, {why})")
        row = ConvertRow(name, type_name, kind_entry.kind, gf, action)
        decisions.append(_Decision(row, module, entry if action == _CONVERTED else None))
    return decisions, refused


def _plan(
    model: nn.Module, recipe: Recipe, *, kinds: Any = None, skip: Any = (), strict: bool = False
) -> ConvertReport:
    """:meth:`Recipe.plan`: the report of a conversion, with nothing converted."""
    decisions, _ = _decide(model, recipe, kinds, skip)
    rows = tuple(
        d.row._replace(action=_TO_CONVERT) if d.row.action == _CONVERTED else d.row
        for d in decisions
    )
    return ConvertReport(rows, dry_run=True)


def convert_(
    model: nn.Module,
    recipe: Recipe,
    *,
    kinds: Any = None,
    skip: Any = (),
    strict: bool = False,
) -> ConvertReport:
    """Convert ``model``'s layers in place, as ``recipe`` says.

    Each module of a registered class (``nn.Linear``, ``nn.Conv1d/2d/3d``,
    torch's activations, ``nn.Embedding``, and any :func:`register` adds)
    that the recipe gives formats to becomes its mptorch counterpart: the
    same object, with its class switched and a new ``formats`` container
    built from the formats (:meth:`GemmFormats.affine` or
    :meth:`GemmFormats.conv`; an activation's or an embedding's formats are
    the recipe's value itself). Its parameters,
    buffers, hooks and ``state_dict`` keys are unchanged, so an optimizer
    built before the conversion still updates it. A module reachable under
    two names converts once. A model is saved through its ``state_dict``: the
    containers hold closures, which ``pickle`` cannot save, and a recipe
    rebuilds them.

    Args:
        model (nn.Module): the model, or one layer.
        recipe (Recipe): the formats.
        kinds (Sequence[str], optional): convert only modules of these kinds.
            Default: ``None``, every kind.
        skip: modules to leave as they are, in any form a :class:`Rule`'s
            ``where`` takes (a name glob, a type, a predicate), or a list or
            tuple of them. Default: ``()``
        strict (bool): raise, instead of reporting, when the recipe gives
            formats to a module that cannot be converted: a subclass of a
            registered class, a parametrized module among them. A module
            that is already an mptorch layer is kept either way.
            Default: ``False``

    Returns:
        ConvertReport: what was converted, and what was kept and why.

    Raises:
        ValueError: under ``strict``, for a module that cannot be converted;
            for a module reachable under two names that the recipe gives
            different formats; or for an unknown kind.

    Example::

        >>> from torch import nn
        >>> from mptorch import BinaryK
        >>> from mptorch.quant import convert_, presets
        >>> model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        >>> report = convert_(model, presets.uniform(BinaryK(8, 4)).override("2", linear=None))
        >>> [type(m).__name__ for m in model]
        ['QLinear', 'ReLU', 'Linear']
    """
    decisions, refused = _decide(model, recipe, kinds, skip)
    if strict and refused:
        raise ValueError(
            "the recipe gives formats to modules that cannot be converted, since only a "
            "registered class converts and these are subclasses: " + "; ".join(refused)
        )
    for d in decisions:
        if d.entry is not None:
            assert d.row.formats is not None  # only modules with formats convert
            formats = _CONTAINER[d.entry.kind](d.row.formats)
            _switch_class(d.module, d.entry.qcls)
            d.module.formats = formats
    return ConvertReport(tuple(d.row for d in decisions))


def convert(model: nn.Module, recipe: Recipe, **kwargs: Any) -> tuple[nn.Module, ConvertReport]:
    """:func:`convert_` on a deep copy of ``model``, which is left as it was.

    The copy keeps tied weights tied, since ``copy.deepcopy`` copies a tensor
    shared by two modules once. It takes :func:`convert_`'s keyword arguments.

    Returns:
        tuple[nn.Module, ConvertReport]: the converted copy and the report.
    """
    copied = copy.deepcopy(model)
    return copied, convert_(copied, recipe, **kwargs)


def unconvert_(model: nn.Module) -> int:
    """Turn ``model``'s mptorch layers back into their PyTorch classes, in place.

    Every module whose class is exactly a registered mptorch class
    (:class:`~mptorch.quant.QLinear`, ``QConv1d/2d/3d``, the activations,
    :class:`~mptorch.quant.QEmbedding`, and any :func:`register` added) gets
    back its PyTorch class, and its ``formats`` are removed. Its parameters
    and their values are unchanged. A layer built as an mptorch layer, not
    converted, is turned back too.

    Returns:
        int: how many modules were turned back.

    Raises:
        TypeError: for a :class:`~mptorch.quant.QPackedLinear`, which holds its
            weight as codes: call its ``unfreeze_packed_()`` first. Nothing is
            changed then.
    """
    back = {e.qcls: e.cls for e in _REGISTRY.values()}
    modules = list(model.modules())
    for m in modules:
        if isinstance(m, QPackedLinear):
            raise TypeError(
                "this model holds a frozen QPackedLinear, whose weight is block codes and not a "
                "float tensor: unfreeze_packed_() it before turning the model back"
            )
    count = 0
    for m in modules:
        cls = back.get(type(m))
        if cls is None:
            continue
        del m.formats
        _switch_class(m, cls)
        count += 1
    return count
