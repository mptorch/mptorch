"""The calls a :class:`~mptorch.quant.simulate` intercepts, and what it does with each.

``_HANDLERS`` maps each torch function of the GEMM kinds (``F.linear``,
``torch.matmul`` and its relatives, ``addmm`` and ``baddbmm``, the
convolutions, ``F.multi_head_attention_forward``) to its handler, and every
other product torch has to a handler that passes it through and says why.
``_OPT_IN_HANDLERS`` holds, by kind, the functions of the kinds a recipe opts
into: the activations, the elementwise ops, the embedding, softmax, the
norms, the reductions and SDPA. A handler finds the formats the recipe
resolved for the innermost module running (``state.routes``) and runs the
Function a converted layer runs, or records why the call ran as torch's.
``simulation.py`` builds each simulation's state from these tables, and
``profiling.py`` counts the same calls from them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _single, _triple
from torch.overrides import resolve_name

from ..functional.activation import qgelu, qhardsigmoid, qhardswish, qrelu, qsigmoid, qsilu, qtanh
from ..functional.attention import _qmha_forward, qattention
from ..functional.embedding import qembedding
from ..functional.norm import qbatchnorm, qgroupnorm, qlayernorm, qrmsnorm, qsoftmax
from ..functional.pointwise import qadd, qdiv, qexp, qmul, qpow, qrsqrt, qsqrt, qsub
from ..functional.reduce import _qmean, _qsum, _reduced
from ..modules.conv import CustomArithConvNd
from ..modules.linear import CustomArithLinear
from ..modules.matmul import CustomArithMatmul
from .recipe import _GEMM_KINDS

if TYPE_CHECKING:
    from .simulation import _Route, _State


SIMULATED = "simulated"
PLAIN = "plain: the recipe gives None"
OWNED = "an mptorch layer's own call"
WHOLE = "inside a module simulated whole"


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
    return _qmha_forward(
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
            return qgelu(x, route.container, values[1])
        return _ACTIVATION_OPS[op](x, route.container)

    return handler


_BINARY_OPS = {
    "add": qadd,
    "sub": qsub,
    "mul": qmul,
    "div": qdiv,
}
_UNARY_OPS = {"sqrt": qsqrt, "rsqrt": qrsqrt, "exp": qexp}

_ACTIVATION_OPS = {
    "silu": qsilu,
    "sigmoid": qsigmoid,
    "tanh": qtanh,
    "hardswish": qhardswish,
    "hardsigmoid": qhardsigmoid,
    "relu": qrelu,
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
            return qpow(x, cast(float, exponent), f.quant, f.grad_quant, **f._keywords)
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
    return qembedding(input, weight, f, padding_idx, max_norm, norm_type, by_freq, sparse)


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
        return qsoftmax(x, dim, route.container, dtype=dtype)

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
    "layer_norm": qlayernorm,
    "rms_norm": qrmsnorm,
    "group_norm": qgroupnorm,
    "batch_norm": qbatchnorm,
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
        dims = _reduced(x, dim)
        if op == "sum":
            return _qsum(x, dims, keepdim, f._spec, f.grad, f.carrier)
        return _qmean(x, dims, keepdim, f._spec, f.div, f.grad, f.carrier)

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
    return qattention(
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
