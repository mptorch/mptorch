"""The fused pointwise and table ops: one carrier step and one cast, in one pass.

``binaryK_pointwise``, ``superfp_pointwise`` and ``fixedpoint_pointwise``
compute what ``mptorch.quant.pointwise`` states for ``qadd``, ``qsub``,
``qmul``, ``qdiv`` and ``qsqrt`` with a ``Quant`` slot: torch's operation in
the carrier, the elementwise quantizer, the store in the operands' dtype.
They are held here to that two-pass spelling, written with torch's
operations and the quantizer ops rather than through
``mptorch.quant.pointwise``, word for word (NaN where NaN): every op, number
operands on either side, every round mode, the four dtypes, broadcast,
strided and empty layouts, and stochastic rounding under one seed, with the
generator left where the quantizer leaves it.

Under ``carrier_rounding=RO`` the step is the exact result rounded to odd: it
is held to ``_carrier_odd``'s primitives (which ``test_carrier_rounding``
holds to a ``Fraction`` reference) followed by the quantizer, and in binary32
also to ``mptorch.quant.pointwise``'s own odd step, computed in float64.

``binaryK_pwl``, ``superfp_pwl`` and ``fixedpoint_pwl`` evaluate a
``PWLTable`` as ``mptorch.quant.pwl._evaluate`` does with torch's operations,
then cast: they are held to that evaluation followed by the quantizer, for
the three tails it takes, the 1/sqrt table's reduction, inputs outside the
range, at its knots and ends, and the specials.
"""

import math
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mptorch import BinaryK, FixedPoint, RoundMode, SuperFP
from mptorch.quant import (
    ActivationFormats,
    NormFormats,
    PWLTable,
    Quant,
    pwl,
    qlayernorm,
    qpwl,
    qsilu,
    silu_table,
)
from mptorch.quant.pointwise import _odd_step, qadd, qdiv, qmul, qsqrt, qsub

from .markers import cuda_devices, requires_mps, requires_odd

R = RoundMode
ADD, SUB, MUL, DIV, SQRT, RSUB, RDIV = range(7)
OPS = {"add": ADD, "sub": SUB, "mul": MUL, "div": DIV, "sqrt": SQRT, "rsub": RSUB, "rdiv": RDIV}
MODES = [R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO, R.SR, R.SR_B, R.SR_C]
DTYPES = [torch.float32, torch.float64, torch.float16, torch.bfloat16]
# Two formats per family: a narrow one, inside every dtype's range and
# precision, and a wide one, whose last bits are near binary32's, so that a
# carrier step that differs in its last bits shows.
FORMATS = {
    "binaryK": BinaryK(8, 4, prng_bits=6),
    "superfp": SuperFP(3, 4, 2, 7, prng_bits=6),
    "fixedpoint": FixedPoint(8, 4, prng_bits=6),
    "binaryK_wide": BinaryK(28, 20, prng_bits=4),
    "superfp_wide": SuperFP(19, 6, 40, 31, prng_bits=4),
    "fixedpoint_wide": FixedPoint(20, 10, prng_bits=3),
}
DEVICES = ["cpu", *cuda_devices, pytest.param("mps", marks=requires_mps)]


def _reference_device(device: str) -> str:
    """Where a test computes its reference: the device itself, except MPS,
    whose own arithmetic flushes binary32 subnormals (2^-140 is among the
    operands). The MPS ops return the CPU's words, so there the reference is
    the CPU's two passes."""
    return "cpu" if device == "mps" else device


def _to(t, device):
    return None if t is None else t.to(device)


# (a's shape, b's shape, or None for a number operand)
LAYOUTS = {
    "same": ((6, 5), (6, 5)),
    "row": ((6, 5), (5,)),
    "outer": ((4, 1), (1, 5)),
    "middle": ((3, 4, 5), (3, 1, 5)),
    "scalar_tensor": ((), ()),
    "empty": ((0, 5), (5,)),
    "number": ((7, 3), None),
    "wide": ((2, 1, 2, 1, 2, 1, 2, 1), (1, 2, 1, 2, 1, 2, 1, 2)),
}


def _spec(fmt, mode):
    """A Quant's resolved spec: its quantizer op and that op's arguments."""
    spec = Quant(fmt, mode)._spec
    assert spec is not None  # an elementwise format
    return spec


def _op(spec):
    """The fused op of a resolved Quant's family."""
    family = spec.plain_op.name().split("::")[1].removesuffix("_quant")
    return getattr(torch.ops.mptorch, f"{family}_pointwise").default


def _values(shape, dtype, device, seed, positive=False) -> torch.Tensor:
    """Normal values with a few zeros, both signs, tiny and huge values, an
    infinity and a NaN among them."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64) * 3
    flat = x.view(-1)
    specials = [0.0, -0.0, 1e-30, -1e30, math.inf, math.nan, 2.0**-140, 0.5]
    for i, v in enumerate(specials[: flat.numel() // 3]):
        flat[(7 * i + seed) % flat.numel()] = v
    if positive:
        x = x.abs()
    return x.to(device=device, dtype=dtype)


def _same(got: torch.Tensor, want: torch.Tensor) -> bool:
    """Equal words, NaN where NaN, and the same dtype, shape and layout."""
    if got.dtype != want.dtype or got.shape != want.shape or not got.is_contiguous():
        return False
    nan = want.isnan()
    return torch.equal(nan, got.isnan()) and torch.equal(
        got[~nan].view(torch.int16 if got.element_size() == 2 else got.dtype),
        want[~nan].view(torch.int16 if want.element_size() == 2 else want.dtype),
    )


def _carrier(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype is torch.float64 else torch.float32


def _two_pass(op: int, a, b, number, spec, device) -> torch.Tensor:
    """mptorch.quant.pointwise's statement: torch's operation in the carrier
    (a number divisor or dividend as a 0-dim tensor of the carrier, as it
    does; the binary32 root through float64 on the CPU, where torch's own is
    not correctly rounded; the binary64 root correctly rounded, element by
    element), then the quantizer on the contiguous result, then the store."""
    dtype = a.dtype
    c = _carrier(dtype)
    x = a.to(c)
    if op == SQRT:
        if c is torch.float64:
            t = torch.tensor([math.sqrt(v) if v >= 0 else math.nan for v in x.flatten().tolist()],
                             dtype=c).view(x.shape).to(device)  # fmt: skip
        else:
            t = torch.sqrt(x.double()).float() if x.device.type == "cpu" else torch.sqrt(x)
    else:
        if b is None:
            y: Any = torch.full((), number, dtype=c, device=device) if op in (DIV, RDIV) else number
        else:
            y = b.to(c)
        t = {
            ADD: lambda: x + y,
            SUB: lambda: x - y,
            MUL: lambda: x * y,
            DIV: lambda: x / y,
            RSUB: lambda: y - x,
            RDIV: lambda: y / x,
        }[op]()
    t = t.contiguous()
    return spec.plain_op(t, *spec.args).to(dtype)


def _fused(op: int, a, b, number, spec, carrier_rounding=R.RNE) -> torch.Tensor:
    return _op(spec)(a, b, number, op, carrier_rounding.value, *spec.args)


def _operands(layout, op, dtype, device):
    a_shape, b_shape = LAYOUTS[layout]
    positive = op == SQRT
    a = _values(a_shape, dtype, device, 1, positive)
    if op in (SQRT,):
        return a, None, None
    if b_shape is None or op in (RSUB, RDIV):
        return a, None, 0.37
    return a, _values(b_shape, dtype, device, 2), None


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("op", list(OPS.values()), ids=list(OPS))
@pytest.mark.parametrize("family", list(FORMATS))
def test_a_fused_op_is_its_operation_then_its_quantizer(device, dtype, mode, op, family):
    spec = _spec(FORMATS[family], mode)
    r = _reference_device(device)
    for layout in LAYOUTS:
        a, b, number = _operands(layout, op, dtype, device)
        torch.manual_seed(11)
        want = _two_pass(op, a.to(r), _to(b, r), number, spec, r)
        after_want = torch.rand(3, device=r)
        torch.manual_seed(11)
        got = _fused(op, a, b, number, spec)
        after_got = torch.rand(3, device=r)
        assert _same(got.to(r), want), (layout, got, want)
        assert torch.equal(after_got, after_want), layout  # the generator is where it was


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["float32", "float16"])
@pytest.mark.parametrize("op", [ADD, DIV, RSUB], ids=["add", "div", "rsub"])
def test_strided_operands_are_read_where_they_are(device, dtype, op):
    """A transposed operand, a slice with a step, and one expanded with
    stride 0: each read through its strides gives the copy's result."""
    spec = _spec(FORMATS["binaryK"], R.RNE)
    base = _values((10, 12), dtype, device, 3)
    other = _values((12, 5), dtype, device, 4)
    cases = [
        (base.t(), None),
        (base[:, ::2], None),
        (base[::3, 1:6], other[:4, :]),
        (base[:5, :5].t(), other[:5, :]),
        (other[:1, :].expand(7, 5), None),
    ]
    for a, b in cases:
        if op == RSUB:
            b = None
        elif b is None:  # a transposed b of a's shape
            b = _values(a.shape[::-1], dtype, device, 5).t()
        number = 0.37 if b is None else None
        got = _fused(op, a, b, number, spec)
        want = _fused(op, a.contiguous(), None if b is None else b.contiguous(), number, spec)
        assert _same(got, want)
        r = _reference_device(device)
        assert _same(got.to(r), _two_pass(op, a.to(r), _to(b, r), number, spec, r))


def _odd_reference(op: int, a, b, number, spec, device) -> torch.Tensor:
    """The exact result rounded to odd in the carrier by _carrier_odd (a
    number first rounded to the operands' dtype), then the quantizer."""
    dtype = a.dtype
    c = _carrier(dtype)
    shape = a.shape if b is None else torch.broadcast_shapes(a.shape, b.shape)
    x = a.to(c).expand(shape).contiguous()
    if b is None and number is not None:
        y = torch.full(shape, torch.full((), number, dtype=dtype).item(), dtype=c, device=device)
    elif b is None:
        y = x
    else:
        y = b.to(c).expand(shape).contiguous()
    odd = torch.ops.mptorch._carrier_odd
    t = {
        ADD: lambda: odd(x, y, x, 1),
        SUB: lambda: odd(x, -y, x, 1),
        MUL: lambda: odd(x, y, x, 2),
        DIV: lambda: odd(x, y, x, 3),
        SQRT: lambda: odd(x, x, x, 4),
        RSUB: lambda: odd(y, -x, x, 1),
        RDIV: lambda: odd(y, x, x, 3),
    }[op]()
    return spec.plain_op(t, *spec.args).to(dtype)


@requires_odd
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("op", list(OPS.values()), ids=list(OPS))
@pytest.mark.parametrize("family", list(FORMATS))
def test_round_to_odd_rounds_the_exact_result(device, dtype, mode, op, family):
    spec = _spec(FORMATS[family], mode)
    r = _reference_device(device)
    for layout in LAYOUTS:
        a, b, number = _operands(layout, op, dtype, device)
        torch.manual_seed(12)
        want = _odd_reference(op, a.to(r), _to(b, r), number, spec, r)
        torch.manual_seed(12)
        got = _fused(op, a, b, number, spec, R.RO)
        assert _same(got.to(r), want), (layout, got, want)


_PHASE_OPS = {ADD: "add", SUB: "sub", MUL: "mul", DIV: "div", SQRT: "sqrt"}


@requires_odd
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("op", list(_PHASE_OPS), ids=list(_PHASE_OPS.values()))
def test_binary32_round_to_odd_is_the_float64_odd_step(device, dtype, op):
    """In binary32 the odd step mptorch.quant.pointwise computes in float64
    (an exact sum and product, a quotient and root that binary64 rounds on
    the exact side of every binary32 value) is the exact result rounded to
    odd: the C++ primitives give it, before the cast."""
    a = _values((40, 30), dtype, device, 6, positive=op == SQRT)
    b = None if op == SQRT else _values((40, 30), dtype, device, 7)
    r = _reference_device(device)  # the odd step's float64, which MPS does not have
    want = _odd_step(_PHASE_OPS[op], a.to(r), _to(b, r), a.to(r), False)
    x = a.float()
    y = x if b is None else b.float()
    code = {ADD: 1, SUB: 1, MUL: 2, DIV: 3, SQRT: 4}[op]
    got = torch.ops.mptorch._carrier_odd(x, -y if op == SUB else y, x, code)
    assert _same(got.to(r), want)


# --- what the ops refuse -----------------------------------------------------------


def _call(device="cpu", **kw):
    spec = _spec(FORMATS["binaryK"], R.RNE)
    args: dict[str, Any] = {"a": torch.ones(3, device=device), "b": torch.ones(3, device=device)}
    args |= {"scalar": None, "op": ADD, "carrier_rounding": R.RNE.value}
    args |= kw
    return torch.ops.mptorch.binaryK_pointwise(
        args["a"], args["b"], args["scalar"], args["op"], args["carrier_rounding"], *spec.args
    )


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"op": 7}, "op is 0 .a . b., .* got 7"),
        ({"carrier_rounding": R.RZ.value}, "carrier_rounding is RoundMode.RNE"),
        ({"a": torch.ones(3, dtype=torch.int32)}, "float32, float64, float16 or bfloat16"),
        ({"op": SQRT}, "sqrt .op 4. takes one operand"),
        ({"op": RSUB, "scalar": 1.0, "b": None, "a": torch.ones(2)}, None),
        ({"op": RSUB, "scalar": 1.0}, "takes the tensor a and a number"),
        ({"b": None}, "got neither"),
        ({"scalar": 2.0}, "got both"),
        ({"b": torch.ones(3, dtype=torch.float64)}, "share one dtype"),
        ({"b": torch.ones(4)}, "must match the size"),
    ],
)
def test_what_the_ops_refuse(kw, message):
    if message is None:
        _call(**kw)
        return
    with pytest.raises(RuntimeError, match=message):
        _call(**kw)


def test_a_format_is_checked_first():
    """The quantizer's own checks, in its words, before the operands'."""
    with pytest.raises(RuntimeError, match="binaryK_pointwise: .*prng_bits"):
        torch.ops.mptorch.binaryK_pointwise(
            torch.ones(3, dtype=torch.int32), None, None, 99, 0, 8, 4, 7, 30, True, R.SR.value, 1, 0
        )


def test_a_gradient_is_refused_under_grad_mode():
    a = torch.ones(3, requires_grad=True)
    with pytest.raises(RuntimeError, match="not differentiable: .*qadd"):
        _call(a=a)
    with torch.no_grad():
        _call(a=a)


# --- mptorch.quant.pointwise runs them ---------------------------------------------


class _Ops(TorchDispatchMode):
    """The names of the mptorch ops the code under it calls."""

    def __init__(self):
        super().__init__()
        self.names: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func.namespace == "mptorch":
            self.names.append(func._schema.name.split("::")[1])
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["binaryK", "superfp", "fixedpoint"])
def test_the_pointwise_ops_run_one_fused_op(device, family):
    """qadd, qsub, qmul, qdiv and qsqrt with a Quant slot: one fused op per
    call, numbers on either side, either carrier, and RO; the gradient slot
    stays a quantizer."""
    q = Quant(FORMATS[family], R.SR)
    x = _values((4, 5), torch.float32, device, 8, positive=True)
    y = _values((5,), torch.float32, device, 9)
    wide: dict[str, Any] = (
        {} if device == "mps" else {"carrier": torch.float64}
    )  # MPS has no float64
    calls = [
        lambda: qadd(x, y, q),
        lambda: qsub(0.5, x, q),
        lambda: qmul(x, 3, q, **wide),
        lambda: qdiv(2.0, x, q, carrier_rounding=R.RO),
        lambda: qsqrt(x, q, **wide, carrier_rounding=R.RO),
    ]
    for call in calls:
        with _Ops() as ops:
            call()
        assert ops.names == [f"{family}_pointwise"], ops.names


def test_a_compiled_call_with_a_number_is_one_graph():
    """Under RO the number becomes a 0-dim tensor of the dtype, so a compiled
    call does not break its graph on reading it back."""
    q = Quant(FORMATS["binaryK_wide"], R.RD)
    f = torch.compile(
        lambda x: qsub(0.1, x, q, carrier_rounding=R.RO), fullgraph=True, backend="eager"
    )
    x = _values((6, 7), torch.float32, "cpu", 10)
    assert _same(f(x), qsub(0.1, x, q, carrier_rounding=R.RO))


# --- the table ops ---------------------------------------------------------------------

# Ranges whose ends and scale (segments / (hi - lo)) binary32 does not hold,
# so that where each is rounded shows.
TABLES = {
    "extend": PWLTable(-1.3, 2.45, (0.25, -1.0, 3.0, 0.5, 0.125, 2.0, -0.75, 1.0, 0.3)),
    "clamp": PWLTable(-0.7, 3.1, (1.0, 0.5, 0.0, 2.0, 1.5), tails="clamp"),
    # 7 / (1.43 + 0.91) rounds to another binary32 value than the quotient
    # of the bounds' binary32 values does
    "scale": PWLTable(-0.91, 1.43, (0.5, -0.25, 1.75, 1.0, -1.5, 0.375, 2.25, 0.0625)),
    "relu": silu_table(),
    "rsqrt": PWLTable.rsqrt(16),
}
_TAILS = {"extend": 0, "clamp": 1, "scale": 0, "relu": 2, "rsqrt": 0}


def _table_input(name, shape, dtype, device, seed):
    """Values across the range, outside it, its knots and ends, and the
    specials; positive normal values for the 1/sqrt table, with a zero, a
    negative, a subnormal, an infinity and a NaN among them."""
    table = TABLES[name]
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(shape, generator=g, dtype=torch.float64)
    flat = x.view(-1)
    if name == "rsqrt":
        flat.copy_(torch.exp2(flat * 80 - 40))
        specials = [0.0, -1.0, 1e-40, math.inf, math.nan, 1.0, 4.0, 2.0**-126]
    else:
        width = table.hi - table.lo
        flat.copy_(table.lo - width / 2 + flat * width * 2)
        knots = [table.lo + k * width / table.segments for k in range(table.segments + 1)]
        specials = [*knots, math.inf, -math.inf, math.nan, 0.0, -0.0, 1e30]
    for i, v in enumerate(specials[: flat.numel()]):
        flat[(5 * i + seed) % flat.numel()] = v
    return x.to(device=device, dtype=dtype)


def _table_op(spec):
    family = spec.plain_op.name().split("::")[1].removesuffix("_quant")
    return getattr(torch.ops.mptorch, f"{family}_pwl").default


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: str(d)[6:])
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("table", list(TABLES))
@pytest.mark.parametrize("family", list(FORMATS))
def test_a_fused_table_is_its_evaluation_then_its_quantizer(device, dtype, mode, table, family):
    spec = _spec(FORMATS[family], mode)
    t = TABLES[table]
    c = _carrier(dtype)
    r = _reference_device(device)
    for shape, view in [((31, 33), None), ((33, 31), "t"), ((0, 4), None), ((), None)]:
        x = _table_input(table, shape, dtype, device, len(shape))
        if view == "t":
            x = x.t()
        torch.manual_seed(13)
        y = pwl._evaluate(t, x.to(r).to(c)).contiguous()
        want = spec.plain_op(y, *spec.args).to(dtype)
        after_want = torch.rand(3, device=r)
        torch.manual_seed(13)
        got = _table_op(spec)(
            x, t._entries(x.device, c), t.lo, t.hi, _TAILS[table], table == "rsqrt", *spec.args
        )
        after_got = torch.rand(3, device=r)
        assert _same(got.to(r), want), (shape, got, want)
        assert torch.equal(after_got, after_want)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("family", ["binaryK", "superfp", "fixedpoint"])
def test_the_table_call_sites_run_one_fused_op(device, family):
    """qpwl, an activation given a table, and a norm whose inverse is a
    table: the table and its rounding are one fused op."""
    q = Quant(FORMATS[family], R.RNE)
    x = _values((4, 6), torch.float32, device, 14)
    with _Ops() as ops:
        qpwl(x, TABLES["extend"], quant=q)
    assert ops.names == [f"{family}_pwl"]
    with _Ops() as ops:
        qsilu(x, ActivationFormats(table=silu_table(), out=q))
    assert ops.names == [f"{family}_pwl"]
    with _Ops() as ops:
        qlayernorm(x, (6,), formats=NormFormats(rsqrt=q, inverse=PWLTable.rsqrt()))
    assert f"{family}_pwl" in ops.names


def test_a_callable_tail_keeps_the_two_passes():
    """A tail other than torch.relu is evaluated by torch, then rounded."""
    table = PWLTable(-1.0, 1.0, (0.0, 1.0, 0.0), tails=torch.tanh)
    with _Ops() as ops:
        qpwl(torch.linspace(-3, 3, 7), table, quant=Quant(FORMATS["binaryK"]))
    assert ops.names == ["binaryK_quant_"]


def _table_call(**kw):
    spec = _spec(FORMATS["binaryK"], R.RNE)
    args: dict[str, Any] = {"x": torch.ones(3), "entries": torch.ones(3), "lo": 0.0, "hi": 1.0}
    args |= {"tails": 0, "rsqrt": False} | kw
    return torch.ops.mptorch.binaryK_pwl(
        args["x"], args["entries"], args["lo"], args["hi"], args["tails"], args["rsqrt"], *spec.args
    )


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"tails": 3}, "tails is 0 .extend., 1 .clamp. or 2 .relu., got 3"),
        ({"lo": 1.0}, "finite with lo < hi"),
        ({"hi": math.inf}, "finite with lo < hi"),
        ({"entries": torch.ones(3, dtype=torch.float64)}, "values of the carrier"),
        ({"entries": torch.ones(3, 1)}, "a 1D tensor"),
        ({"entries": torch.ones(1)}, "2 to 2\\^31 - 1 entries"),
        ({"x": torch.ones(3, dtype=torch.int32)}, "float32, float64, float16 or bfloat16"),
    ],
)
def test_what_the_table_ops_refuse(kw, message):
    with pytest.raises(RuntimeError, match=message):
        _table_call(**kw)
