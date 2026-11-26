"""The pointwise ops: torch's arithmetic in a carrier, then one rounding.

Tier 1: with every slot ``None`` an op is torch's function, and with only the
result rounded its gradients are torch's. Tier 2: a real format comes out
close to float. Tier 3: an op is its carrier computation followed by the
slot's quantizer, word for word, gradients and stochastic rounding included.

Then what the carrier rounds (``docs/source/kernels/pointwise.rst``'s table):
every ordered pair of ``BinaryK(8, 4)`` values through the four arithmetic
ops, and every value through the square root, in every rounding mode, in
both carriers and rounded to odd, against the exact value's rounding (the
exact value rounded to odd at 53 bits, then the binary64 quantizer, as
``test_carrier_rounding`` builds its references). The table's bounds on a
float64 quotient and root are checked on worst cases built for them, and its
claims that the default carrier can round first, on witnesses.
"""

import math
import random
from fractions import Fraction

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from mptorch import BinaryK, FixedPoint, RoundMode, SuperFP
from mptorch.quant import Quant
from mptorch.quant.pointwise import (
    PointwiseFormats,
    _odd_binary32,
    _odd_sum,
    qadd,
    qdiv,
    qexp,
    qmul,
    qpow,
    qrsqrt,
    qsqrt,
    qsub,
)
from tests.markers import available_devices, float64_devices, requires_cuda, requires_mps
from tests.test_carrier_rounding import _ro as _ro_carrier

R = RoundMode
P8 = BinaryK(8, 4)
P16 = BinaryK(16, 11)
BINARY = {"add": qadd, "sub": qsub, "mul": qmul, "div": qdiv}
TORCH_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}
UNARY = {"sqrt": qsqrt, "rsqrt": qrsqrt, "exp": qexp}
TORCH_UNARY = {"sqrt": torch.sqrt, "rsqrt": torch.rsqrt, "exp": torch.exp}
DETERMINISTIC = [R.RNE, R.RNA, R.RU, R.RD, R.RZ, R.RO]


def _operands(device, dtype=torch.float32, shape=(6, 5), positive=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    x = x.abs() + 0.25 if positive else x
    return x.to(device=device, dtype=dtype)


def _grads(out, *inputs, seed=1):
    g = torch.randn(out.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.autograd.grad(out, inputs, g.to(out), allow_unused=True, retain_graph=True)


def _same(a, b) -> bool:
    """Equal words, NaN where NaN: same dtype, shape and values."""
    if a is None or b is None:
        return a is b
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    nan = a.isnan()
    return torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])


# --- tier 1: torch's function -------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("op", sorted(BINARY))
def test_without_slots_a_binary_op_is_torchs(device, dtype, op):
    if device == "mps" and dtype is torch.float64:
        pytest.skip("MPS has no float64")
    x = _operands(device, dtype, positive=op == "div").requires_grad_()
    y = _operands(device, dtype, (5,), positive=op == "div", seed=2).requires_grad_()
    for a, b in ((x, y), (x, 3.0), (2.5, y)):
        got, want = BINARY[op](a, b), TORCH_BINARY[op](a, b)
        assert _same(got, want)
        inputs = [t for t in (a, b) if isinstance(t, torch.Tensor)]
        assert all(
            _same(g, w) for g, w in zip(_grads(got, *inputs), _grads(want, *inputs), strict=True)
        )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("op", sorted(BINARY))
def test_rounding_the_result_only_keeps_torchs_gradients(device, dtype, op):
    """With ``grad_quant=None`` the backward is torch's formula in the
    gradient's dtype, broadcast operands and numbers on the right included."""
    x = _operands(device, dtype, positive=op == "div").requires_grad_()
    y = _operands(device, dtype, (5,), positive=op == "div", seed=2).requires_grad_()
    for a, b in ((x, y), (x, 3.0)):
        got, want = BINARY[op](a, b, Quant(P8)), TORCH_BINARY[op](a, b)
        inputs = [t for t in (a, b) if isinstance(t, torch.Tensor)]
        assert all(
            _same(g, w) for g, w in zip(_grads(got, *inputs), _grads(want, *inputs), strict=True)
        )


def _unary(op, x, *slots):
    return qpow(x, 1.5, *slots) if op == "pow" else UNARY[op](x, *slots)


def _torch_unary(op, x):
    return torch.pow(x, 1.5) if op == "pow" else TORCH_UNARY[op](x)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", sorted(UNARY) + ["pow"])
def test_without_slots_a_unary_op_is_torchs(device, op):
    """With no slot an op is torch's, and with only its result rounded its
    gradient is torch's formula."""
    x = _operands(device, positive=True).requires_grad_()
    want = _torch_unary(op, x)
    assert _same(_unary(op, x), want)
    for got in (_unary(op, x), _unary(op, x, Quant(P8))):
        assert _same(_grads(got, x)[0], _grads(want, x)[0])


def test_a_zero_exponent_has_a_zero_gradient():
    x = torch.rand(5).requires_grad_()
    out = qpow(x, 0, Quant(P8), Quant(P8))
    assert torch.equal(out, torch.ones(5)) and torch.equal(_grads(out, x)[0], torch.zeros(5))


# --- tier 2: close to float ----------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("op", sorted(BINARY) + sorted(UNARY))
def test_a_real_format_comes_out_close(device, op):
    x = _operands(device, positive=True)
    y = _operands(device, positive=True, seed=2)
    q = Quant(P16)
    got = BINARY[op](x, y, q) if op in BINARY else UNARY[op](x, q)
    want = TORCH_BINARY[op](x, y) if op in BINARY else TORCH_UNARY[op](x)
    torch.testing.assert_close(got, want, rtol=2**-10, atol=0)


# --- tier 3: the carrier, then the slot -----------------------------------------------------


def _slots(rounding):
    fmt = BinaryK(8, 4, prng_bits=12 if rounding.is_stochastic else 0)
    return Quant(fmt, rounding), Quant(
        BinaryK(12, 8, prng_bits=8 if rounding.is_stochastic else 0), rounding
    )


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("rounding", [R.RNE, R.RU, R.SR])
@pytest.mark.parametrize("op", sorted(BINARY))
def test_a_binary_op_is_its_carrier_step_then_the_slots(device, dtype, rounding, op):
    """The result is the slot's rounding of the binary32 result, and each
    gradient the gradient slot's rounding of torch's formula in binary32,
    summed over a broadcast operand's dimensions first; under SR the
    roundings draw in that order (result, then each input's gradient)."""
    q, gq = _slots(rounding)
    x = _operands(device, dtype, positive=op == "div").requires_grad_()
    y = _operands(device, dtype, (5,), positive=op == "div", seed=2).requires_grad_()
    g = _operands(device, dtype, seed=3)
    torch.manual_seed(7)
    out = BINARY[op](x, y, q, gq)
    gx, gy = torch.autograd.grad(out, (x, y), g)
    xc, yc, gc = x.detach().float(), y.detach().float(), g.float()
    torch.manual_seed(7)
    want = q(TORCH_BINARY[op](xc, yc)).to(dtype)
    wx = {"add": gc, "sub": gc, "mul": gc * yc, "div": gc / yc}[op]
    wy = {"add": gc, "sub": -gc, "mul": gc * xc, "div": -gc * ((xc / yc) / yc)}[op]
    want_x, want_y = gq(wx).to(dtype), gq(wy.sum_to_size(yc.shape)).to(dtype)
    assert _same(out, want) and _same(gx, want_x) and _same(gy, want_y)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
@pytest.mark.parametrize("op", sorted(UNARY) + ["pow"])
def test_a_unary_op_is_its_carrier_step_then_the_slots(device, rounding, op):
    q, gq = _slots(rounding)
    x = _operands(device, positive=True).requires_grad_()
    g = _operands(device, seed=3)
    torch.manual_seed(7)
    out = qpow(x, 1.5, q, gq) if op == "pow" else UNARY[op](x, q, gq)
    (gx,) = torch.autograd.grad(out, (x,), g)
    xd = x.detach()
    torch.manual_seed(7)
    if op == "pow":
        want, wg = q(torch.pow(xd, 1.5)), g * (1.5 * xd.pow(0.5))
    else:
        # correctly rounded, unlike torch's CPU root; float64 on the CPU, which MPS lacks
        carrier_root = torch.sqrt(xd.cpu().double()).float().to(xd.device)
        want = q(carrier_root if op == "sqrt" else TORCH_UNARY[op](xd))
        r = TORCH_UNARY[op](xd)
        wg = {"sqrt": g / (2 * r), "rsqrt": -0.5 * g * r.pow(3), "exp": g * r}[op]
    assert _same(out, want) and _same(gx, gq(wg))


@pytest.mark.parametrize("device", available_devices)
def test_the_binary32_root_is_correctly_rounded(device):
    """torch's eager CPU square root can be an ulp off, in float32 and
    float64 (CUDA's is correctly rounded); the op's binary32 step is the
    correctly rounded root on every device, the float64 one's nearest."""
    g = torch.Generator().manual_seed(9)
    x = torch.rand(2**16, generator=g) * torch.exp2(
        torch.randint(-60, 60, (2**16,), generator=g).float()
    )
    want = torch.tensor([math.sqrt(v) for v in x.double().tolist()]).float()
    got = qsqrt(x.to(device), lambda t: t)
    assert torch.equal(got.cpu(), want)


@pytest.mark.parametrize("device", available_devices)
def test_a_number_divisor_is_a_tensor_of_the_carrier(device):
    """torch's CUDA division by a CPU number multiplies by its reciprocal,
    which is not the correctly rounded quotient; the op divides by a 0-dim
    tensor instead, on every device."""
    x = _operands(device, positive=True, shape=(4096,))
    q = Quant(BinaryK(24, 20))
    got = qdiv(x, 3.0, q)
    want = q(x / torch.full((), 3.0, device=device))
    assert _same(got, want)
    assert _same(qdiv(x.cpu(), 3.0, q), q(x.cpu() / 3.0))


# --- what the carrier rounds: every pair of an 8-bit format ---------------------------------


def _p8_values() -> torch.Tensor:
    """Every value of ``BinaryK(8, 4)``: the distinct roundings of every
    bfloat16, whose values hold all of the format's, then the infinities and
    NaN."""
    words = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16)
    every = Quant(P8)(words.view(torch.bfloat16).float())
    finite = torch.unique(every[torch.isfinite(every)])
    return torch.cat([finite, torch.tensor([math.inf, -math.inf, math.nan])])


def _ro(x: Fraction, p: int) -> Fraction:
    """``x`` rounded to odd at ``p`` significant bits, any exponent."""
    if x == 0:
        return x
    sign, n, d = (1 if x > 0 else -1), abs(x.numerator), x.denominator
    e = n.bit_length() - d.bit_length()
    if (n << max(0, -e)) < (d << max(0, e)):
        e -= 1
    s = p - 1 - e
    q, r = divmod(n << s, d) if s >= 0 else divmod(n, d << -s)
    return sign * Fraction(q | (1 if r else 0)) / Fraction(2) ** s


def _ro_sqrt(x: Fraction, p: int) -> Fraction:
    """``sqrt(x)`` of a dyadic ``x > 0``, rounded to odd at ``p`` bits."""
    n, t = x.numerator, x.denominator.bit_length() - 1  # x = n / 2^t
    r = math.isqrt(n << (2 * 200 - t))  # sqrt(x) to 200 fractional bits
    e = r.bit_length() - 1 - 200
    s = p - 1 - e  # q = floor(sqrt(x) * 2^s) = floor(sqrt(n * 2^(2s - t)))
    j = 2 * s - t
    y = n << j if j >= 0 else n >> -j
    q = math.isqrt(y)
    exact = q * q == y and (j >= 0 or (y << -j) == n)
    return Fraction(q | (0 if exact else 1)) / Fraction(2) ** s


def _exact53(op: str, a: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """Each pair's exact result rounded to odd at 53 bits, as float64: a
    value whose binary64 rounding is the exact value's for every format of
    at most 51 bits. Infinities, NaN and division by zero are IEEE's."""
    out = []
    av = a.double().tolist()
    bv = av if b is None else b.double().tolist()
    for u, v in zip(av, bv, strict=True):
        if not (math.isfinite(u) and math.isfinite(v)) or (op == "div" and v == 0):
            t = torch.tensor(u, dtype=torch.float64)
            w = torch.tensor(v, dtype=torch.float64)
            out.append(float(torch.sqrt(t) if op == "sqrt" else TORCH_BINARY[op](t, w)))
            continue
        fu, fv = Fraction(u), Fraction(v)
        if op == "sqrt":
            out.append(float(_ro_sqrt(fu, 53)) if u > 0 else (0.0 if u == 0 else math.nan))
            continue
        exact = {"add": fu + fv, "sub": fu - fv, "mul": fu * fv}[op] if op != "div" else fu / fv
        out.append(float(_ro(exact, 53)))
    return torch.tensor(out, dtype=torch.float64)


@pytest.fixture(scope="module")
def p8_pairs():
    vals = _p8_values()
    n = vals.numel()
    a, b = vals.repeat_interleave(n), vals.repeat(n)
    exact = {op: _exact53(op, a, b) for op in BINARY}
    exact["sqrt"] = _exact53("sqrt", vals, None)
    return vals, a, b, exact


def _run_p8(op, a, b, vals, q, **kw):
    return BINARY[op](a, b, q, **kw) if op in BINARY else qsqrt(vals, q, **kw)


def _reference(exact: torch.Tensor, fmt, rounding, seed=None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return Quant(fmt, rounding, carrier=torch.float64)(exact)


@pytest.mark.parametrize("op", sorted(BINARY) + ["sqrt"])
@pytest.mark.parametrize("fmt", [P8, P16], ids=["P4", "P11"])
@pytest.mark.parametrize("rounding", DETERMINISTIC)
def test_every_pair_rounds_to_the_exact_values_rounding(p8_pairs, op, fmt, rounding):
    """The operands have 4 bits, so every carrier result the table calls
    exact is: binary32's sums and products always, its quotients up to 20-bit
    boundaries and its roots up to 11-bit ones, and binary64's everything.
    The one exception is binary32's root into P = 11 (its nearest-mode
    boundaries have 12 bits), which the table does not claim."""
    vals, a, b, exact = p8_pairs
    want = _reference(exact[op], fmt, rounding)
    q = Quant(fmt, rounding)
    for carrier in (None, torch.float64):
        for cr in (R.RNE, R.RO):
            got = _run_p8(op, a, b, vals, q, carrier=carrier, carrier_rounding=cr).double()
            claimed = not (op == "sqrt" and fmt is P16 and carrier is None and cr is R.RNE)
            if claimed:
                assert _same(got, want), (carrier, cr)


def _faithful(carrier_value: torch.Tensor, exact53: torch.Tensor, k: int, op: str) -> torch.Tensor:
    """Whether each carrier value is on the exact value's side of every
    number of ``k`` significant bits (and equal to one only when the exact
    value is): their roundings to odd at ``k + 1`` bits agree. ``exact53``
    already holds that information for every ``k`` up to 51."""
    out = []
    for c, x in zip(carrier_value.double().tolist(), exact53.tolist(), strict=True):
        if not (math.isfinite(c) and math.isfinite(x)) or x == 0:
            out.append(c == x or (math.isnan(c) and math.isnan(x)))
            continue
        out.append(_ro(Fraction(c), k + 1) == _ro(Fraction(x), k + 1))
    return torch.tensor(out)


@pytest.mark.parametrize("op", ["div", "sqrt"])
def test_binary32_holds_the_boundaries_the_table_says(p8_pairs, op):
    """Of 4-bit operands, a binary32 quotient is faithful at every boundary
    of at most 24 - 4 = 20 bits, and a root at every one of at most 11; past
    those bounds some are not, which is why stochastic rounding with many
    random bits rounds first in binary32 (rule 12)."""
    vals, a, b, exact = p8_pairs
    if op == "div":
        carrier = (a / b)[::7]
        e = exact["div"][::7]
    else:
        carrier, e = torch.sqrt(vals), exact["sqrt"]
    bound = 20 if op == "div" else 11
    assert bool(_faithful(carrier, e, bound, op).all())
    assert not bool(_faithful(carrier, e, 24, op).all())


@pytest.mark.parametrize("op", sorted(BINARY) + ["sqrt"])
@pytest.mark.parametrize("rounding", [R.SR, R.SR_B, R.SR_C])
def test_every_pair_rounds_stochastically_as_the_exact_value_would(p8_pairs, op, rounding):
    """binary64 draws as the binary64 quantizer does, so each pair's result
    equals that quantizer's on the exact value under the same seed: with 20
    random bits (P + N = 24 boundaries, inside every bound), in binary64 and
    rounded to odd there. Rounded to odd in binary32, the cast draws as the
    binary32 quantizer, on the exact value rounded to odd at 24 bits, which
    16 random bits keep inside ``_odd_findings``' binary32 bound."""
    vals, a, b, exact = p8_pairs
    fmt = BinaryK(8, 4, prng_bits=20)
    q = Quant(fmt, rounding)
    want = _reference(exact[op], fmt, rounding, seed=11)
    for cr in (R.RNE, R.RO):
        torch.manual_seed(11)
        got = _run_p8(op, a, b, vals, q, carrier=torch.float64, carrier_rounding=cr)
        assert _same(got.double(), want), cr
    fmt16 = BinaryK(8, 4, prng_bits=16)
    q16 = Quant(fmt16, rounding)
    torch.manual_seed(12)
    want32 = q16(_odd_binary32(exact[op]))
    torch.manual_seed(12)
    got32 = _run_p8(op, a, b, vals, q16, carrier_rounding=R.RO)
    assert _same(got32, want32)


# --- float32 operands: the bounds' worst cases, and the default carrier's double rounding -----


def _division_worst_cases(k: int, n: int = 200, seed: int = 0):
    """24-bit A and B whose quotient is 1/(M B) from M / 2^k, a k-bit number,
    in relative terms: the closest a quotient of float32 values gets to one
    without equalling it (the extended-Euclid construction)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        B = rng.randrange(2**23, 2**24) | 1
        M = (-pow(B, -1, 2**k)) % 2**k
        if M < 2 ** (k - 1):
            continue
        A, r = divmod(M * B + 1, 2**k)
        if r == 0 and 2**23 <= A < 2**24:
            out.append((A, B))
    a = torch.tensor([A for A, _ in out], dtype=torch.float64).float()
    b = torch.tensor([B for _, B in out], dtype=torch.float64).float()
    return a, b


@pytest.mark.parametrize("k", [29, 30, 31])
def test_a_binary64_quotient_is_faithful_up_to_its_bound(k):
    """For float32 operands the bound is 53 - 24 = 29 bits; the worst cases
    still hold at 30 and fail from 31."""
    a, b = _division_worst_cases(k)
    exact = _exact53("div", a, b)
    faithful = _faithful(a.double() / b.double(), exact, k, "div")
    assert bool(faithful.all()) == (k <= 30)


def test_the_default_carrier_rounds_first_where_the_table_says():
    """The witnesses of rule 12: an 8-bit format's midpoint plus 2^-30 is a
    tie in binary32, which RNE sends to even; 1 + 2^-30 is 1 in binary32,
    which RU leaves at 1. binary64 holds both sums, and fails past 53 bits
    (2^-60), where rounding to odd still gives the exact value's rounding."""
    q_ne, q_up = Quant(P8, R.RNE), Quant(P8, R.RU)
    mid, one = torch.tensor([1.0625]), torch.tensor([1.0])
    for tiny, b64_holds in ((2.0**-30, True), (2.0**-60, False)):
        t = torch.tensor([tiny])
        assert qadd(mid, t, q_ne).item() == 1.0 and qadd(one, t, q_up).item() == 1.0
        b64 = qadd(mid, t, q_ne, carrier=torch.float64).item()
        assert (b64 == 1.125) == b64_holds
        for carrier in (None, torch.float64):
            assert qadd(mid, t, q_ne, carrier=carrier, carrier_rounding=R.RO).item() == 1.125
            assert qadd(one, t, q_up, carrier=carrier, carrier_rounding=R.RO).item() == 1.125


def test_float32_operands_round_to_odd_exactly():
    """2^14 random float32 pairs, their exponents up to 2^±20 apart, and the
    sums of format values with addends 2^-60 below them: every rounding of
    every op rounded to odd, in either carrier, is the exact value's. The
    sums in binary64 alone round onto the format values instead."""
    g = torch.Generator().manual_seed(5)
    n = 2**14
    a = torch.randn(n, generator=g) * torch.exp2(torch.randint(-20, 20, (n,), generator=g).float())
    b = torch.randn(n, generator=g)
    fmt = BinaryK(16, 11)
    for op in sorted(BINARY):
        exact = _exact53(op, a, b)
        for rounding in DETERMINISTIC:
            want = _reference(exact, fmt, rounding)
            for carrier in (None, torch.float64):
                got = BINARY[op](a, b, Quant(fmt, rounding), carrier=carrier, carrier_rounding=R.RO)
                assert _same(got.double(), want), (op, rounding, carrier)
    on_grid = Quant(fmt)(a.clamp(-1000, 1000))
    tiny = b.abs() * on_grid.abs() * 2.0**-60
    exact = _exact53("add", on_grid, tiny)
    want = _reference(exact, fmt, R.RU)
    got = qadd(on_grid, tiny, Quant(fmt, R.RU), carrier=torch.float64, carrier_rounding=R.RO)
    assert _same(got.double(), want)
    nearest = qadd(on_grid, tiny, Quant(fmt, R.RU), carrier=torch.float64)
    assert not _same(nearest.double(), want)


def test_figueroa_holds_for_values_of_the_format():
    """Operands that are values of a P = 11 format round correctly to it to
    nearest in binary32 (2P + 1 <= 24 for + - * /, 2P + 2 <= 24 for the root),
    on 2^14 sampled pairs."""
    g = torch.Generator().manual_seed(6)
    n = 2**14
    q = Quant(P16)
    a = q(torch.randn(n, generator=g) * 4)
    b = q(torch.randn(n, generator=g) * 4)
    for op in sorted(BINARY):
        want = _reference(_exact53(op, a, b), P16, R.RNE)
        assert _same(BINARY[op](a, b, q).double(), want), op
    ab = a.abs()
    assert _same(qsqrt(ab, q).double(), _reference(_exact53("sqrt", ab, None), P16, R.RNE))


def test_the_odd_primitives():
    """``_odd_sum`` is the exact sum rounded to odd at 53 bits, and
    ``_odd_binary32`` a 53-bit odd value rounded to odd at 24, the largest
    finite float32 past the range."""
    g = torch.Generator().manual_seed(8)
    a = torch.randn(4096, generator=g, dtype=torch.float64)
    b = torch.randn(4096, generator=g, dtype=torch.float64) * 2.0**-70
    want = [
        float(_ro(Fraction(u) + Fraction(v), 53))
        for u, v in zip(a.tolist(), b.tolist(), strict=True)
    ]
    assert _odd_sum(a, b).tolist() == want
    v = [1 + 2.0**-30, 1 + 2.0**-23 + 2.0**-40, 3.5e38, -1e-46, 2.0**-140 * 3.3, 0.0, math.inf]
    got = _odd_binary32(torch.tensor(v, dtype=torch.float64))
    want32 = [float(_ro_carrier(Fraction(x), torch.float32)) if math.isfinite(x) else x for x in v]
    assert got.dtype is torch.float32 and got.tolist() == want32
    assert _odd_binary32(torch.tensor([math.nan], dtype=torch.float64)).isnan().all()


# --- the carrier and round-to-odd rules ---------------------------------------------------


def test_a_slot_rounds_in_the_ops_carrier():
    x = torch.ones(3)
    with pytest.raises(ValueError, match="carrier argument names"):
        qadd(x, x, Quant(P8, carrier=torch.float64))
    with pytest.raises(ValueError, match="carrier argument names"):
        PointwiseFormats(quant=Quant(P8, carrier=torch.float64))
    assert torch.equal(qadd(x, x, Quant(P8, carrier=torch.float64), carrier=torch.float64), x * 2)
    with pytest.raises(TypeError, match=r"pass Quant\(fmt\)"):
        qadd(x, x, P8)  # ty: ignore[invalid-argument-type]
    assert PointwiseFormats(quant=P8).quant == Quant(P8)
    with pytest.raises(TypeError, match="a module"):
        PointwiseFormats(quant=torch.nn.Identity())


@pytest.mark.parametrize("device", float64_devices)
def test_a_carrier_is_never_narrower_than_the_operands(device):
    x = torch.ones(3, dtype=torch.float64, device=device)
    with pytest.raises(ValueError, match="narrower than float64"):
        qmul(x, x, Quant(P8), carrier=torch.float32)


def test_a_binary64_result_is_stored_with_one_rounding():
    """A binary64 result goes to bfloat16 with one rounding (``_narrowed``):
    just above a bfloat16 midpoint it goes up, where a conversion through
    float32 makes it the midpoint and then rounds the tie to even. A slot
    whose format bfloat16 cannot hold is refused before that can happen, so
    it shows with a slot that is not a ``Quant``."""
    x = torch.tensor([1.0], dtype=torch.bfloat16)
    y = torch.tensor([2.0**-8 + 2.0**-30])
    want = torch.tensor([1 + 2.0**-7], dtype=torch.bfloat16)
    assert torch.equal(qadd(x, y.to(torch.bfloat16)), x)  # bfloat16 cannot hold y
    got = qmul(x, 1.0, lambda t: t + y.double(), carrier=torch.float64)
    assert torch.equal(got, want)
    assert torch.equal((x.double() + y.double()).float().to(torch.bfloat16), x)


def test_a_result_narrower_than_its_format_is_refused():
    """A computed result reaches the format's whole value set, so a format
    more precise than float16 raises on a float16 result (the GEMM's rule),
    where the quantizer on a float16 input does not."""
    x = torch.ones(3, dtype=torch.float16)
    q = Quant(BinaryK(16, 12))
    q(x)  # a quantizer's input is already a float16 value
    with pytest.raises(ValueError, match="float16"):
        qadd(x, x, q)


def test_round_to_odd_refuses_what_it_cannot_round_exactly():
    x = torch.ones(3)
    with pytest.raises(ValueError, match="no exact carrier step"):
        qexp(x, Quant(P8), carrier_rounding=R.RO)
    with pytest.raises(ValueError, match="P is 23"):
        qadd(x, x, Quant(BinaryK(30, 23)), carrier_rounding=R.RO)  # binary32: P <= 22
    qadd(x, x, Quant(BinaryK(30, 23)), carrier=torch.float64, carrier_rounding=R.RO)
    # The fused ops round a binary64 quotient or root to odd exactly, and
    # float64 operands too, so only the format's own bound applies.
    sr = Quant(BinaryK(8, 4, prng_bits=24), R.SR)
    qsqrt(x, sr, carrier=torch.float64, carrier_rounding=R.RO)
    qdiv(x, x, sr, carrier=torch.float64, carrier_rounding=R.RO)
    qadd(x.double(), x.double(), Quant(P8), carrier_rounding=R.RO)
    with pytest.raises(ValueError, match="P \\+ prng_bits is 53"):
        qdiv(
            x.double(), x.double(), Quant(BinaryK(8, 4, prng_bits=49), R.SR), carrier_rounding=R.RO
        )
    with pytest.raises(ValueError, match="a Quant of a BinaryK"):
        qadd(x, x, lambda t: t, carrier_rounding=R.RO)
    for fmt in (SuperFP(3, 4, 8, 7), FixedPoint(8, 4)):
        qmul(x, x, Quant(fmt), carrier_rounding=R.RO)
    with pytest.raises(ValueError, match="RoundMode.RNE .* or RoundMode.RO"):
        qadd(x, x, Quant(P8), carrier_rounding=R.RU)


@requires_cuda
def test_the_two_passes_keep_their_round_to_odd_bounds():
    """A CUDA tensor beside a CPU 0-dim tensor runs the two passes, whose odd
    step is computed in float64: float64 operands, and a binary64 quotient
    or root past its bound, are refused there."""
    x, one = torch.ones(3, device="cuda"), torch.tensor(1.0)
    with pytest.raises(ValueError, match="float64 operands"):
        qadd(x.double(), one.double(), Quant(P8), carrier_rounding=R.RO)
    with pytest.raises(ValueError, match="P \\+ prng_bits is 30 .* at most 29"):
        qdiv(x, one, Quant(BinaryK(8, 4, prng_bits=26), R.SR), carrier=torch.float64,
             carrier_rounding=R.RO)  # fmt: skip


@requires_mps
def test_round_to_odd_runs_and_binary64_fails_on_mps():
    """Round to odd runs the fused op on MPS, which gives the CPU's words; a
    binary64 carrier needs float64 tensors, which MPS does not have."""
    x = torch.linspace(-2.0, 3.0, 11)
    y = torch.linspace(0.1, 5.0, 11)
    q = Quant(BinaryK(8, 4, prng_bits=6), R.SR)
    for f in (qadd, qsub, qmul, qdiv):
        torch.manual_seed(3)
        want = f(x, y, q, carrier_rounding=R.RO)
        torch.manual_seed(3)
        got = f(x.to("mps"), y.to("mps"), q, carrier_rounding=R.RO)
        assert torch.equal(got.cpu(), want), f.__name__
    assert torch.equal(
        qsqrt(y.to("mps"), Quant(P8), carrier_rounding=R.RO).cpu(),
        qsqrt(y, Quant(P8), carrier_rounding=R.RO),
    )
    m = torch.ones(3, device="mps")
    with pytest.raises((TypeError, RuntimeError)):
        qadd(m, m, Quant(P8), carrier=torch.float64)


# --- the result is rounded in place, the inputs never ------------------------------------


@pytest.mark.parametrize("device", available_devices)
def test_the_inputs_are_never_written(device):
    x = _operands(device).requires_grad_()
    y = _operands(device, seed=2)
    xs, ys, version = x.detach().clone(), y.clone(), y._version
    out = qadd(x, y, Quant(P8), Quant(P8))
    out.backward(torch.ones_like(out))
    assert torch.equal(x.detach(), xs) and torch.equal(y, ys) and y._version == version
    # a gradient that is the incoming gradient itself is copied before rounding
    g = torch.full_like(out, 0.3)
    gs = g.clone()
    torch.autograd.grad(qadd(x, y, Quant(P8), Quant(P8)), x, g)
    assert torch.equal(g, gs)


@pytest.mark.parametrize("device", available_devices)
def test_a_result_that_is_not_contiguous_is_rounded_out_of_place(device):
    x = _operands(device, shape=(5, 6)).mT
    y = _operands(device, shape=(5, 6), seed=2).mT
    out = qmul(x, y, Quant(P8))
    assert not (x * y).is_contiguous()
    assert torch.equal(out, Quant(P8)(x * y))
    assert torch.equal(qmul(x, y, Quant(P8, inplace=True)), out)


# --- compile ------------------------------------------------------------------------------


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("rounding", [R.RNE, R.SR])
def test_each_op_compiles_in_one_graph(device, rounding):
    q, gq = _slots(rounding)

    def f(x, y):
        z = qmul(qadd(x, y, q, gq), 0.5, q, gq)
        z = qdiv(qsub(z, y, q, gq), y, q, gq)
        return qpow(qexp(qrsqrt(qsqrt(z.abs() + 1, q, gq), q, gq), q, gq), 2, q, gq)

    torch._dynamo.reset()
    x = _operands(device, positive=True).requires_grad_()
    y = _operands(device, positive=True, seed=2).requires_grad_()
    torch.manual_seed(3)
    want = f(x, y)
    want_g = torch.autograd.grad(want.sum(), (x, y))
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(f, backend=counter, fullgraph=True)
    torch.manual_seed(3)
    got = compiled(x, y)
    got_g = torch.autograd.grad(got.sum(), (x, y))
    assert counter.frame_count == 1
    assert _same(got, want) and all(_same(a, b) for a, b in zip(got_g, want_g, strict=True))


def test_inductor_keeps_the_roundings():
    """Under Inductor torch's arithmetic is Inductor's, and mptorch's
    roundings are the same ops: a result on the format's grid stays on it."""
    torch._dynamo.reset()
    q = Quant(P8)
    x = _operands("cpu", positive=True)
    out = torch.compile(lambda x: qmul(qadd(x, 1.0, q), x, q))(x)
    assert torch.equal(q(out), out)


# --- MPS: the CPU's words, away from the subnormals torch's arithmetic flushes there -----------


@requires_mps
@pytest.mark.parametrize("op", sorted(BINARY) + ["sqrt"])
@pytest.mark.parametrize("rounding", [R.RNE, R.RD, R.SR])
def test_mps_is_the_cpu(op, rounding):
    q = Quant(BinaryK(8, 4, prng_bits=8 if rounding.is_stochastic else 0), rounding)
    x = _operands("cpu", positive=True, shape=(64, 64))
    y = _operands("cpu", positive=True, shape=(64, 64), seed=2)
    torch.manual_seed(4)
    want = qsqrt(x, q) if op == "sqrt" else BINARY[op](x, y, q)
    torch.manual_seed(4)
    got = qsqrt(x.to("mps"), q) if op == "sqrt" else BINARY[op](x.to("mps"), y.to("mps"), q)
    assert torch.equal(got.cpu(), want)
