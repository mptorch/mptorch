"""An independent reference for the fixed-point GEMM: its definition, step by step.

For operands ``a`` and ``b`` in the carrier ``c`` (binary32 for float32,
float16 and bfloat16 operands, binary64 for float64 ones or under
``carrier=torch.float64``), and ``Q_F`` the rounding to fixed-point format
``F`` with saturation or, for the accumulate format, wrap-around, each
output element is

    split:  p = Q_M(RN_c(a * b)),  s = Q_A(s + p)
    fused:                         s = Q_F(RN_c(a * b) + s)

summed over ``k`` in order from ``s = 0``, every sum exact. Written from that
definition and not from the kernels: the carrier's product is torch's own
multiplication in the carrier (IEEE, correctly rounded), and every rounding
is integer arithmetic on the exact rational value. Stochastic rounding takes
each step's random word from the caller and reads it as
``cast_fixedpoint_stochastic`` does, and decides as P3109's Fig. 2 does
(``tests/test_stochastic_rounding.py`` transcribes it), so the reference
rounds a step as the kernels' casts do, word for word.

Every value is a ``Fraction``, or ``None`` for a NaN. A result is returned as
float64, which holds every value of a format of at most 53 magnitude bits.
"""

import math
from collections.abc import Callable
from fractions import Fraction

import torch

from mptorch import FixedPoint, RoundMode

R = RoundMode
STOCHASTIC = (R.SR, R.SR_B, R.SR_C)

# A step's random word: draw(i, j, k, slot) for output element (i, j), step k,
# and slot 0 (the product's rounding) or 1 (the sum's). A fused mac draws one
# word a step, slot 0.
Draw = Callable[[int, int, int, int], int]


def codes(fmt: FixedPoint) -> tuple[int, int]:
    """The format's smallest and largest integer codes."""
    if not fmt.is_signed:
        return 0, (1 << fmt.wl) - 1
    top = (1 << (fmt.wl - 1)) - 1
    return (-top if fmt.symmetric else -top - 1), top


def _stochastic_up(frac: Fraction, r: int, n: int, mode: RoundMode) -> bool:
    """P3109's decision to round a magnitude away from zero, with ``frac`` its
    fraction between its two neighbours and ``r`` the draw's ``n`` bits."""
    X = frac * 2**n
    if mode is R.SR:
        return math.floor(X) + r >= 2**n
    if mode is R.SR_B:
        return math.floor(2 * X) + 2 * r + 1 >= 2 ** (n + 1)
    return round(X) + r >= 2**n  # SR_C: Fraction's round is RNITE


def _round(t: Fraction, mode: RoundMode, r: int, n: int) -> int:
    """``t``, a value in units of the format's step, rounded to an integer."""
    if t.denominator == 1:
        return int(t)
    lo = math.floor(t)
    hi = lo + 1
    if mode in (R.RNE, R.RNA):
        d = t - lo
        if d != Fraction(1, 2):
            return hi if d > Fraction(1, 2) else lo
        if mode is R.RNE:
            return lo if lo % 2 == 0 else hi
        return hi if t > 0 else lo  # RNA: away from zero
    if mode is R.RU:
        return hi
    if mode is R.RD:
        return lo
    if mode is R.RZ:
        return lo if t > 0 else hi
    if mode is R.RO:
        return lo if lo % 2 != 0 else hi
    # stochastic: on the magnitude, as the casts round, then the sign
    m = abs(t)
    m_lo = math.floor(m)
    q = m_lo + (1 if _stochastic_up(m - m_lo, r, n, mode) else 0)
    return q if t > 0 else -q


def draw_bits(word: int, n: int, carrier: torch.dtype) -> int:
    """The ``n`` random bits of a carrier word, as ``cast_fixedpoint_stochastic``
    reads them: the top ``n`` bits of its mantissa field."""
    man = 52 if carrier is torch.float64 else 23
    return ((word & ((1 << man) - 1)) >> (man - n)) if n else 0


def quantize(
    x: Fraction | float | None,
    fmt: FixedPoint,
    mode: RoundMode,
    word: int = 0,
    *,
    wrap: bool = False,
    carrier: torch.dtype = torch.float32,
) -> int | None:
    """``Q_F(x)`` as an integer code, or ``None`` for a NaN.

    ``x`` is an exact value, a float infinity, or ``None`` (a NaN). An
    infinity saturates to the end on its side, under wrap-around too: an
    infinity has no residue. Otherwise the value is rounded to an integer
    number of steps in ``mode``, then saturated to the codes (an unsigned
    format's negatives to 0) or, with ``wrap``, reduced modulo ``2**wl`` into
    them, as two's complement or unsigned hardware wraps."""
    if x is None:
        return None
    lo, hi = codes(fmt)
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return hi if x > 0 else lo
        x = Fraction(x)
    n = fmt.prng_bits if mode in STOCHASTIC else 0
    q = _round(x * Fraction(2) ** fmt.fl, mode, draw_bits(word, n, carrier), n)
    if wrap:
        q %= 1 << fmt.wl
        return q - (1 << fmt.wl) if fmt.is_signed and q > hi else q
    return lo if q < lo else (hi if q > hi else q)


def value(code: int | None, fmt: FixedPoint) -> Fraction | None:
    """A code's value, or ``None`` for a NaN."""
    return None if code is None else Fraction(code) * Fraction(2) ** -fmt.fl


def _exact(code: int, fmt: FixedPoint) -> Fraction:
    """A code's value (``fl`` may be negative: a step above 1)."""
    return Fraction(code) * Fraction(2) ** -fmt.fl


def carrier_of(dtype: torch.dtype, carrier: torch.dtype | None) -> torch.dtype:
    """The carrier a call computes in."""
    return torch.float64 if dtype is torch.float64 or carrier is torch.float64 else torch.float32


def gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    acc: FixedPoint,
    mul: FixedPoint | None = None,
    mode: RoundMode = R.RNE,
    wrap: bool = False,
    carrier: torch.dtype | None = None,
    draw: Draw | None = None,
) -> torch.Tensor:
    """``a @ b`` (``[M, K]`` by ``[K, N]``) under the fixed-point mac: split
    when ``mul`` is given, fused otherwise. Returns float64 values."""
    c = carrier_of(a.dtype, carrier)
    if mode in STOCHASTIC and draw is None:
        raise ValueError("a stochastic mode takes each step's random word through draw")
    products = a.detach().cpu().to(c)[:, :, None] * b.detach().cpu().to(c)[None, :, :]  # RN_c
    M, K, N = products.shape
    out = torch.empty(M, N, dtype=torch.float64)
    for i in range(M):
        for j in range(N):
            s = 0  # the running sum's code; a NaN ends the element
            nan = False
            for k in range(K):
                p = products[i, k, j].item()
                if math.isnan(p):
                    nan = True
                    break
                w0 = draw(i, j, k, 0) if draw is not None else 0
                if mul is not None:
                    pq = quantize(p, mul, mode, w0, carrier=c)
                    assert pq is not None  # a non-NaN product rounds to a code
                    w1 = draw(i, j, k, 1) if draw is not None else 0
                    x = _exact(s, acc) + _exact(pq, mul)
                    q = quantize(x, acc, mode, w1, wrap=wrap, carrier=c)
                elif math.isinf(p):
                    q = quantize(p, acc, mode, w0, wrap=wrap, carrier=c)
                else:
                    x = Fraction(p) + _exact(s, acc)
                    q = quantize(x, acc, mode, w0, wrap=wrap, carrier=c)
                assert q is not None
                s = q
            out[i, j] = float("nan") if nan else float(_exact(s, acc))
    return out
