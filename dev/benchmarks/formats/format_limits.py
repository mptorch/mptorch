#!/usr/bin/env python3
"""Safe parameter limits for simulating a format in its carrier.

The kernels do not build encodings: they round a value of their *carrier* onto
the format's grid and hand back a value of it (``docs/source/concepts.rst``) --
binary64 for a float64 tensor, binary32 for every other dtype. So a format is
only simulated faithfully while the carrier can hold every value it represents
*and* the casts can reach every one of them. Past that point quantization
silently stops being the format the caller named -- values at the top of the
range saturate early, values at the bottom land on the wrong grid -- which is
exactly what a caller cannot see from the output.

This script answers where that point is, for either carrier
(``--carrier binary64``; binary32 is the default). Binary32's numbers below,
binary64's in parentheses. Two questions, and they have different answers:

* **representable** -- is every value of the format a value of the carrier?
  That is arithmetic alone: at most 24 (53) significand bits, nothing above the
  carrier's largest finite value, and a finest spacing no finer than 2**-149
  (2**-1074).
* **faithful** -- does the cast actually land on those values, for every
  input? Strictly stronger, because a subnormal's exponent field is 0 whatever
  its magnitude, so any arm that reads that field is blind below 2**-126
  (2**-1022). Which arms do is the difference between the modes:
  ``SUBNORMALS`` and superfp's supernormals derive their grid from it and need
  their floor at 2**-125 (2**-1021), a binade clear; ``NORMALS`` and
  ``EXTENDED_NORMALS`` decide by comparing magnitudes (``bit_helper.h``'s
  normal clip tests the input against the floor and half of it,
  ``NormalRangeParams``' ``min_num`` and ``half_num``) and are exact to the
  smallest normal -- except at ``P == 1``, whose round reads the field after
  all, and ``EXTENDED_NORMALS`` at the carrier's full precision, ``P == 24``
  (53), whose half-floor the carrier cannot hold. Both need the floor a
  binade higher.

``--audit`` is what says so rather than this docstring: it brackets each input
between the two format values around it -- computed in binary64, from the
format's own definition, not from the cast -- and checks the kernel returns
one of them, the right one for the directed modes, and the nearer one for the
round-to-nearest modes away from ties. ``sweep`` walks each parameter until
that audit fails and prints the boundary next to the predicted one and next to
what :mod:`mptorch.number` enforces.

``--dtype float16`` or ``--dtype bfloat16`` asks the same question of a result
*stored* in that dtype (the library's rule for it is ``mptorch.number``'s
``check_binaryK_storage``, ``check_superfp_storage`` and
``check_fixedpoint_storage``). The cast rounds in binary32, their default
carrier, or in binary64 with ``--carrier binary64`` (the operands widened, the
result narrowed by ``mptorch.quant.ops._narrowed``); writing the result rounds
it again, so the format's values must also be the dtype's. That is a third
question, representability in the dtype, and the audit answers it twice,
because two ops reach different parts of the format: a K=1 fused GEMM over
operands of that dtype, whose products ``a * b`` land anywhere and so reach
every value the format has, and the elementwise quantizer over every value of
the dtype, whose inputs are already the dtype's values and so reach only some.
The library enforces the first, which is the stricter of the two.

Examples::

    format_limits.py binaryK -K 8 -P 4              # Binary8p4se: fine
    format_limits.py binaryK -K 16 -P 8             # Binary16p8se: not
    format_limits.py superfp --man 4 --exp 4 --normal-binades 2 --bias 7
    format_limits.py binaryK -K 8 -P 4 --bias 124 --audit
    format_limits.py sweep --audit                  # the boundaries, measured
    format_limits.py binaryK -K 8 -P 3 --bias 15 --dtype float16 --audit
    format_limits.py sweep --dtype float16 --audit  # the float16 boundaries
    format_limits.py binaryK -K 63 -P 53 --bias 970 --carrier binary64 --audit
    format_limits.py sweep --carrier binary64 --audit
    format_limits.py fixedpoint --wl 8 --fl 6 --symmetric --audit
    format_limits.py sweep --fixedpoint-only --audit   # its bounds, measured

Only ``--audit`` and ``sweep --audit`` need the built extension.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class Carrier:
    """The binary float a cast rounds in.

    Written out here rather than read from `mptorch.number`'s `_Carrier`
    records: the prediction is meant to be checked against the library, so it
    does not borrow the library's constants.
    """

    name: str
    man_bits: int  # stored significand bits
    top_exp: int  # exponent of the largest finite value
    min_normal_exp: int
    min_subnormal_exp: int  # exponent of the smallest positive value
    min_simulable_exp: int  # the lowest floor an exponent-field-reading arm can place
    dtype: str  # the torch dtype that holds it


CARRIERS = {
    "binary32": Carrier("binary32", 23, 127, -126, -149, -125, "float32"),
    "binary64": Carrier("binary64", 52, 1023, -1022, -1074, -1021, "float64"),
}
B32 = CARRIERS["binary32"]

SATURATIONS = ("SAT_NONE", "SAT_FINITE", "SAT_PROPAGATE", "SAT_FINITE_EXTENDED")
SUBNORMALS = ("SUBNORMALS", "NORMALS", "EXTENDED_NORMALS")
DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class Storage:
    """A dtype a rounded result is written in, read off ``torch.finfo``.

    Deliberately not `mptorch.number`'s table: the prediction below is meant to
    be checked against the library, so it takes the dtype's shape from torch.
    """

    name: str
    man_bits: int  # stored significand bits
    top_exp: int  # exponent of the largest finite value
    min_exp: int  # exponent of the smallest positive value, a subnormal


def storage(name: str) -> Storage:
    import torch

    fi = torch.finfo(getattr(torch, name))
    man_bits = -round(math.log2(fi.eps))
    min_normal = round(math.log2(fi.smallest_normal))
    return Storage(name, man_bits, math.floor(math.log2(fi.max)), min_normal - man_bits)


# --- the value set, as exponents ---------------------------------------------


@dataclass(frozen=True)
class Grid:
    """A format's value set, in the only terms the limits depend on.

    ``normal_floor`` is the lowest exponent whose binade carries a mantissa,
    and the two families differ only in what sits below it: binaryK's
    subnormals, a single grid of fixed spacing ``2**below_step``, or superfp's
    supernormals, one value per binade down to ``2**super_floor``. Both are
    spelled so that the bracket below reads ``step = 2**s`` for one ``s`` per
    input, which is what makes the oracle a few lines.
    """

    family: str
    label: str
    carrier: Carrier
    man_bits: int
    exp_bits: int
    normal_floor: int
    below_step: int  # binaryK: the subnormal spacing; superfp: unused
    super_floor: int | None  # superfp: the smallest supernormal's exponent
    bottom_exp: int  # exponent of the smallest positive value
    step_exp: int  # exponent of the finest spacing anywhere in the set
    hole_exp: int | None  # EXTENDED_NORMALS: the binade whose t=0 code is the zero
    min_bottom: int  # how low `bottom_exp` may go before the casts lose it
    top_exp: int  # exponent of the largest finite value
    top_code: int  # its significand code within that binade
    top_man: int  # the significand width that code is counted in
    # fixed point in two's complement: the negative end is one step past -max,
    # a power of two, 2**(top_exp + 1)
    neg_extra: bool = False

    @property
    def max_finite(self) -> float:
        try:
            return math.ldexp(
                float((1 << self.top_man) + self.top_code), self.top_exp - self.top_man
            )
        except OverflowError:
            return math.inf  # a format binary64 cannot hold either

    @property
    def max_negative(self) -> float:
        """The largest magnitude a negative result can have."""
        if not self.neg_extra:
            return self.max_finite
        try:
            return math.ldexp(1.0, self.top_exp + 1)
        except OverflowError:
            return math.inf

    @property
    def in_binary64(self) -> bool:
        """Whether the oracle below can hold this format's values at all (the
        two's complement bottom, a binade past the top, included)."""
        end_exp = self.top_exp + 1 if self.neg_extra else self.top_exp
        return end_exp <= 1023 and self.step_exp >= -1074 and self.man_bits <= 52


def _normal_range_top(exp_bits: int, man_bits: int, bias: int, reserved: int, drop: int = 0):
    """``make_normal_range_params``' top of the range, minus the carrier's clamp.

    The kernel counts the top code in ``min(man_bits, 23)`` (52) bits because
    that is the finest grid its carrier's word has; here the format's own
    ``man_bits`` is used, so that a format asking for more precision than the
    carrier holds is reported as such rather than quietly measured on the
    carrier's grid.
    """
    codes = 1 << man_bits
    top_code = codes - 1 - reserved
    top_field = (1 << exp_bits) - 1
    while top_code < 0:  # more reserved codes than the top binade has
        top_code += codes
        top_field -= 1
    return top_field, top_code, top_field - bias - drop


def _binaryK_min_bottom(man_bits: int, subnormals: str, c: Carrier) -> int:
    """How low a binaryK format's smallest value may go, by subnormals mode.

    In binary32's numbers, binary64's in parentheses. `SUBNORMALS` reads the
    input's exponent field to place it on its grid, and every subnormal of the
    carrier shares one, so it is blind below 2**-126 (2**-1022) and needs its
    floor a binade clear of that. The other two decide by comparing magnitudes
    as words and are exact to the smallest normal -- below which the floor
    leaves the carrier's normals and nothing flushes.

    Their compare needs a second point, though, half the floor, and two of
    their formats cannot have it a binade below the smallest normal. At
    ``man_bits == 0`` the round ahead of the compare works on the exponent
    field, and meets 2**-127 (2**-1023) -- a subnormal of the carrier -- as a
    tie between binades, which it carries up to the floor. And
    `EXTENDED_NORMALS`' floor is not a power of two: its half, 2**-127 * (1 +
    2**-man_bits), has its last bit at 2**(-127 - man_bits), which the carrier
    holds only down to 2**-149 (2**-1074).
    """
    if subnormals == "SUBNORMALS" or man_bits == 0:
        return c.min_simulable_exp
    if subnormals == "EXTENDED_NORMALS":
        # the half-floor's last bit, 2**(bottom - 1 - man_bits), at or above the finest step
        return max(c.min_normal_exp, c.min_subnormal_exp + 1 + man_bits)
    return c.min_normal_exp


def binaryK_grid(
    K: int,
    P: int,
    bias: int,
    is_signed: bool,
    saturation: str,
    subnormals: str,
    carrier: Carrier = B32,
) -> Grid:
    man_bits = P - 1
    exp_bits = K - P if is_signed else K - P + 1
    min_exp = 1 - bias  # the smallest normal's exponent
    reserved = (saturation != "SAT_FINITE") + (0 if is_signed else 1)
    _, top_code, top_exp = _normal_range_top(exp_bits, man_bits, bias, reserved)
    if subnormals == "SUBNORMALS":
        normal_floor, below_step = min_exp, min_exp - man_bits
    elif subnormals == "NORMALS":
        normal_floor, below_step = min_exp, min_exp
    elif man_bits:  # EXTENDED_NORMALS: code 0's binade becomes one more binade of normals
        normal_floor, below_step = min_exp - 1, min_exp - 1
    else:
        # -- which at man_bits == 0 is a single code, the zero, so the grid is
        # NORMALS'. Spelled as the lower binade, the bracket offered its foot as
        # a value, and every mode was measured wrong against it.
        normal_floor, below_step = min_exp, min_exp
    sign = "s" if is_signed else "u"
    domain = "f" if saturation == "SAT_FINITE" else "e"
    sub = "" if subnormals == "SUBNORMALS" else f" {subnormals}"
    return Grid(
        family="binaryK",
        label=f"Binary{K}p{P}{sign}{domain} bias={bias}{sub}",
        carrier=carrier,
        man_bits=man_bits,
        exp_bits=exp_bits,
        normal_floor=normal_floor,
        below_step=below_step,
        super_floor=None,
        # EXTENDED_NORMALS spends its floor binade's mantissa-zero code on the
        # zero, so the values there start one step up and that one power of two
        # is a hole in the grid; with man_bits == 0 the binade is that single
        # code and the floor is the one NORMALS has.
        hole_exp=(min_exp - 1 if subnormals == "EXTENDED_NORMALS" and man_bits else None),
        min_bottom=_binaryK_min_bottom(man_bits, subnormals, carrier),
        bottom_exp=(min_exp if subnormals == "EXTENDED_NORMALS" and not man_bits else below_step),
        step_exp=min(below_step, normal_floor - man_bits),
        top_exp=top_exp,
        top_code=top_code,
        top_man=man_bits,
    )


def superfp_grid(
    man_bits: int,
    exp_bits: int,
    normal_binades: int,
    bias: int,
    is_signed: bool,
    saturation: str,
    carrier: Carrier = B32,
) -> Grid:
    normal_cutoff = ((1 << exp_bits) - 1 - bias) - normal_binades + 1
    super_cutoff = normal_cutoff - ((1 << exp_bits) - normal_binades) * (1 << man_bits) + 1
    reserved = saturation != "SAT_FINITE"  # superfp spends no code on a NaN
    _, top_code, top_exp = _normal_range_top(exp_bits, man_bits, bias, reserved)
    sign = "s" if is_signed else "u"
    return Grid(
        family="superfp",
        label=f"m{man_bits}e{exp_bits}n{normal_binades}b{bias}{sign}",
        carrier=carrier,
        man_bits=man_bits,
        exp_bits=exp_bits,
        normal_floor=normal_cutoff,
        below_step=0,
        super_floor=super_cutoff,
        hole_exp=None,
        min_bottom=carrier.min_simulable_exp,  # the supernormal arm rounds the exponent field
        # normal_binades == 2**exp_bits leaves no supernormal codes, and then
        # the supernormal floor sits *above* the normal one: the value set is
        # the normals alone, and the bottom is where they start.
        bottom_exp=min(super_cutoff, normal_cutoff),
        step_exp=min(super_cutoff, normal_cutoff, normal_cutoff - man_bits),
        top_exp=top_exp,
        top_code=top_code,
        top_man=man_bits,
    )


def fixedpoint_grid(
    wl: int, fl: int, is_signed: bool, symmetric: bool, carrier: Carrier = B32
) -> Grid:
    """A fixed-point format (``cast_fixedpoint.h``): the multiples of 2**-fl a
    wl-bit word holds, spelled as a binaryK grid that is subnormal everywhere --
    one step, 2**-fl, from zero to the top, whose binade is ``mag_bits - 1 -
    fl`` -- so the bracket below needs no branch of its own. Two things it does
    not share with binaryK: it always saturates (no code for an infinity), and
    in two's complement the negative end is a step further out than the
    positive one, ``neg_extra``."""
    mag = wl - (1 if is_signed else 0)
    top_exp = mag - 1 - fl
    kind = "" if not is_signed else (", symmetric" if symmetric else ", two's complement")
    return Grid(
        family="fixedpoint",
        label=f"FixedPoint({wl}, {fl}{'' if is_signed else ', unsigned'}{kind})",
        carrier=carrier,
        man_bits=mag - 1,
        exp_bits=1,  # so that 2**(man_bits + exp_bits) counts its codes
        normal_floor=top_exp + 1,  # nothing below the top is a normal: one step throughout
        below_step=-fl,
        super_floor=None,
        hole_exp=None,
        # the casts place an input by its exponent field, as binaryK's subnormal arm
        min_bottom=carrier.min_simulable_exp,
        bottom_exp=-fl,
        step_exp=-fl,
        top_exp=top_exp,
        top_code=(1 << (mag - 1)) - 1,
        top_man=mag - 1,
        neg_extra=is_signed and not symmetric,
    )


# --- the limits ---------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    level: str  # "unrepresentable" | "unfaithful"
    what: str
    detail: str


def check(grid: Grid) -> list[Finding]:
    """Every way this format outruns its carrier, worst first."""
    out: list[Finding] = []
    c = grid.carrier

    if grid.man_bits > c.man_bits:
        out.append(
            Finding(
                "unrepresentable",
                "precision",
                f"{grid.man_bits + 1} bits of precision, {c.name} has {c.man_bits + 1}: every "
                f"binade's grid is finer than {c.name}'s, so only the top {c.man_bits + 1} bits "
                f"of each value survive",
            )
        )
    if grid.top_exp > c.top_exp:
        out.append(
            Finding(
                "unrepresentable",
                "top of range",
                f"largest finite value is 2^{grid.top_exp}-ish, {c.name} stops at "
                f"2^{c.top_exp + 1}; the cast saturates at {c.name}'s largest value on the "
                f"format's grid instead, and the codes above it are unreachable",
            )
        )
    if grid.neg_extra and grid.top_exp + 1 > c.top_exp:
        out.append(
            Finding(
                "unrepresentable",
                "bottom of the two's complement range",
                f"the negative end is -2^{grid.top_exp + 1}, past {c.name}'s largest value: "
                f"the cast saturates there to the largest negative value on the grid instead",
            )
        )
    if grid.step_exp < c.min_subnormal_exp:
        out.append(
            Finding(
                "unrepresentable",
                "bottom of range",
                f"finest spacing is 2^{grid.step_exp}, {c.name}'s is 2^{c.min_subnormal_exp}: "
                f"values that far down are not {c.name} values at all",
            )
        )
    if grid.top_code < 0 or grid.top_exp < c.min_normal_exp:
        # make_normal_range_params' max_num == 0: the reserved codes ate the
        # whole top binade, or the largest finite value is below the carrier's
        # normals. Either way every nonzero result overflows.
        out.append(
            Finding(
                "malformed",
                "top of range",
                "no finite normal value at all: max_num is zero, so every nonzero "
                "result of the normal arm overflows",
            )
        )
    if grid.super_floor is not None and grid.super_floor > grid.normal_floor:
        # normal_binades >= 2**exp_bits: the cast's region tests are ordered
        # supernormal-then-underflow, so an exponent equal to normal_cutoff
        # satisfies the underflow test and the format's lowest normal binade
        # flushes to zero (the RNE fast path's gate refuses this format; the
        # integer path does not).
        out.append(
            Finding(
                "malformed",
                "normal_binades",
                "normal_binades leaves no supernormal codes, which puts the "
                "supernormal floor above the normal one: the cast flushes the "
                "format's lowest normal binade to zero",
            )
        )
    if grid.bottom_exp < grid.min_bottom and grid.step_exp >= c.min_subnormal_exp:
        # The casts read the input's exponent field to place it in the
        # format's regions, and every subnormal of the carrier carries field 0
        # -- exponent -127 by that reading in binary32, -126 in fact. So a cast
        # can tell nothing apart below the smallest normal, and a format whose
        # own grid reaches into that range is rounded on a grid one binade
        # coarse (binaryK) or onto powers of two it cannot name (superfp). A
        # binade above it is the lowest value a format can have and still be
        # decided entirely by the carrier's normals: the round-to-nearest
        # boundary below it, the smallest normal, is the last one a cast can see.
        why = (
            f"the casts classify an input by its exponent field, which every {c.name} "
            f"subnormal shares, so inputs below 2^{c.min_normal_exp} cannot be placed on "
            "this grid"
            if grid.bottom_exp < c.min_normal_exp
            or grid.family != "binaryK"
            or grid.below_step < grid.normal_floor  # SUBNORMALS, which reads the field
            else "half of it, which the underflow compare decides at, is a point the "
            "cast cannot place: a tie between binades to the man_bits == 0 round, or "
            f"not a {c.name} value at all"
        )
        out.append(
            Finding(
                "unfaithful",
                "bottom of range",
                f"smallest value is 2^{grid.bottom_exp}, below 2^{grid.min_bottom}: {why}",
            )
        )
    return out


def check_storage(grid: Grid, st: Storage) -> list[Finding]:
    """Every way this format's values outrun a result stored in ``st``, worst first.

    Only the ways binary32, these dtypes' carrier, does not already account
    for: what `check` reports
    stays `check`'s. Stated on the value set, as `check`'s "representable"
    findings are -- the store rounds correctly and reads no exponent field, so
    there is no "faithful" twin here.
    """
    out: list[Finding] = []
    if grid.man_bits > st.man_bits:
        out.append(
            Finding(
                "unstorable",
                f"{st.name} precision",
                f"{grid.man_bits + 1} bits of precision, {st.name} has {st.man_bits + 1}: "
                f"every value the cast returns is rounded again when it is stored",
            )
        )
    elif grid.top_exp < st.min_exp:
        out.append(
            Finding(
                "unstorable",
                f"{st.name} top of range",
                f"largest finite value is below 2^{grid.top_exp + 1}, and {st.name}'s "
                f"smallest is 2^{st.min_exp}: nothing but zero and that value is stored",
            )
        )
    else:
        # With the precision inside the dtype's, the largest finite value is at
        # most the dtype's exactly when its binade is, and the grid is the
        # dtype's (or coarser) exactly when the finest step is.
        if grid.top_exp > st.top_exp:
            out.append(
                Finding(
                    "unstorable",
                    f"{st.name} top of range",
                    f"largest finite value is 2^{grid.top_exp}-ish, {st.name} stops below "
                    f"2^{st.top_exp + 1}: the values above are stored as infinity",
                )
            )
        if grid.step_exp < st.min_exp:
            out.append(
                Finding(
                    "unstorable",
                    f"{st.name} bottom of range",
                    f"finest spacing is 2^{grid.step_exp}, {st.name}'s is 2^{st.min_exp}: "
                    f"the values down there are rounded again when they are stored",
                )
            )
    return out


# --- the oracle ---------------------------------------------------------------


def _bracket(ax, grid: Grid):
    """The two format magnitudes around ``|x|``, in binary64, exactly, and the
    step between them.

    The grid above the format's largest finite value is continued rather than
    stopped, because that is what the casts do: P3109 rounds first and
    saturates second, so an input beyond the range is rounded on the grid it
    would have had and only then turned into the format's largest value or an
    infinity. Continued past binary64's own top, that is 2**1024, which
    binary64 spells as an infinity -- the one inexact quantity here, and why
    the step is handed back: :func:`expected` measures the distance up from
    it instead.
    """
    import torch

    _, e = torch.frexp(ax)
    ebin = e.to(torch.int64) - 1  # floor(log2 |x|)
    if grid.family in ("binaryK", "fixedpoint"):
        step_exp = torch.where(ebin < grid.normal_floor, grid.below_step, ebin - grid.man_bits)
    else:
        assert grid.super_floor is not None
        step_exp = torch.where(
            ebin >= grid.normal_floor,
            ebin - grid.man_bits,
            torch.where(ebin >= grid.bottom_exp, ebin, grid.bottom_exp),
        )
    # ldexp(1, s) is exact, and ax/step is exact whenever it matters: it is a
    # division by a power of two, which can only round when the quotient falls
    # below binary64's smallest subnormal -- an input far under a coarse step,
    # whose floor is 0 however that rounds. The floor, and its product with
    # the step, are exact because the step is never finer than the input's own
    # last bit (`in_binary64` holds man_bits to 52), so every quantity is a
    # multiple of that bit with at most 53 significant ones. hi = lo + step is
    # exact too, save the 2**1024 the docstring names.
    step = torch.ldexp(torch.ones_like(ax), step_exp.to(torch.int32))
    lo = torch.floor(ax / step) * step
    hi = lo + step
    if grid.hole_exp is not None:
        # that one power of two is not a value: reached from below it is the
        # zero, reached from above it is the first value of the binade
        hole = math.ldexp(1.0, grid.hole_exp)
        first = hole + math.ldexp(1.0, grid.hole_exp - grid.man_bits)
        lo = torch.where(lo == hole, torch.zeros_like(lo), lo)
        hi = torch.where(hi == hole, torch.full_like(hi, first), hi)
    lo = torch.where(ax == 0, torch.zeros_like(ax), lo)
    hi = torch.where(ax == 0, torch.zeros_like(ax), hi)
    return lo, hi, step


def _saturate(mag, grid: Grid, saturation: str):
    """P3109's second step, on a magnitude the first step has already rounded."""
    import torch

    maxf = torch.full_like(mag, grid.max_finite)
    over = mag > maxf
    if saturation == "SAT_NONE":
        return torch.where(over, torch.full_like(mag, math.inf), mag)
    return torch.where(over, maxf, mag)


def _ends(mag, neg, grid: Grid, saturation: str):
    """`_saturate`, or fixed point's ends: it always saturates, and onto the end
    on the input's side, which in two's complement is a step further out below
    zero than above it."""
    import torch

    if grid.family != "fixedpoint":
        return _saturate(mag, grid, saturation)
    top = torch.where(
        neg, torch.full_like(mag, grid.max_negative), torch.full_like(mag, grid.max_finite)
    )
    return torch.minimum(mag, top)


def expected(x, grid: Grid, saturation: str, is_signed: bool, mode: str):
    """What the format says the answer is, derived without the cast.

    Returns ``(want, mask)``: the value, and which inputs the answer is
    unambiguous for. The round-to-nearest modes are masked off exactly at
    ties, where the tie rule is a question about the encoding rather than
    about the value set, and round-to-odd is masked off entirely -- it is
    checked by membership instead, in :func:`audit`.

    Except the tie between zero and the smallest value, which the value set
    does decide: zero is code 0, which is even, and the other candidate is the
    one away from zero. That tie is half the floor, where the bottom-end limits
    live, and masking it hid one of them -- ``P == 1`` under ``NORMALS`` at
    bias 127, which rounds 2**-127 to the floor in round-to-nearest-even.
    """
    import torch

    ax = torch.abs(x)
    neg = x < 0
    lo, hi, step = _bracket(ax, grid)
    exact = lo == ax
    up = torch.where(exact, ax, hi)  # away from zero
    down = lo  # toward zero

    # ax - lo is exact (both are multiples of ax's last bit, and the result is
    # under a step); hi - ax is too, except where hi is binary64's 2**1024 --
    # and there the step minus that first distance is the exact one
    d_lo = ax - lo
    d_hi = torch.where(torch.isinf(hi), step - d_lo, hi - ax)
    tie = (d_lo == d_hi) & ~exact
    near = torch.where(d_lo <= d_hi, lo, hi)
    away = torch.where(d_lo < d_hi, lo, hi)

    if mode == "RNE" or mode == "RNA":
        # `near` takes zero at that tie and `away` the value, which are the answers
        mag, mask = (near if mode == "RNE" else away), ~(tie & (lo != 0))
    elif mode == "RZ":
        mag, mask = down, torch.ones_like(tie)
    elif mode == "RU":
        mag, mask = torch.where(neg, down, up), torch.ones_like(tie)
    elif mode == "RD":
        mag, mask = torch.where(neg, up, down), torch.ones_like(tie)
    else:
        raise ValueError(mode)

    want = _ends(mag, neg, grid, saturation)
    want = torch.where(neg, -want, want)
    if not is_signed:
        want = torch.where(neg, torch.zeros_like(want), want)
    # P3109 has one zero and it is unsigned (no cast returns -0.0), so a
    # magnitude that rounded away takes +0.0 whatever the sign that got there.
    return torch.where(want == 0, torch.zeros_like(want), want), mask


# --- the inputs ---------------------------------------------------------------


def _critical_words(grid: Grid, args) -> list[int]:
    """Every value of the format, and the midpoints between them.

    Taken from the two enumerators rather than from the grid above, so the
    probes come from the same code points ``binaryK_values.py`` prints -- or,
    for a binary64 format too wide to enumerate, from `_sampled_values`.
    """
    import torch

    values = _format_values(grid, args)
    if not values and (grid.carrier.name == "binary64" or grid.family == "fixedpoint"):
        values = _sampled_values(grid)
    # zero is in the value set of every format here, whether or not the
    # enumeration gives it a code point (EXTENDED_NORMALS spends code 0 on a
    # normal, and the cast still answers a zero input with zero), and the gap
    # between it and the smallest value is where the bottom-end limits live.
    vals = sorted({0.0} | {v for v in values if math.isfinite(v)})
    probes: list[float] = []
    for i, v in enumerate(vals):
        probes.append(v)
        if i + 1 < len(vals):
            probes.append((v + vals[i + 1]) / 2.0)
    t = torch.tensor(probes, dtype=torch.float64)
    words = set()
    if grid.carrier.name == "binary64":
        # the midpoint of two values on a 53-bit grid is not a binary64 value,
        # and rounds to one side of itself; the ulps either side cover both
        for f64 in (t, torch.nextafter(t, t * 0 + math.inf)):
            w = f64.view(torch.int64) & 0x7FFFFFFFFFFFFFFF
            words.update(int(i) for i in w.tolist())
        limit = 0x7FF0000000000000
    else:
        for f32 in (t.to(torch.float32), torch.nextafter(t, t * 0 + math.inf).to(torch.float32)):
            w = f32.view(torch.int32).to(torch.int64) & 0x7FFFFFFF
            words.update(int(i) for i in w.tolist())
        limit = 0x7F800000
    # one ulp of the carrier either side of each, where a grid that is one
    # binade out shows up first
    out = set()
    for w in words:
        out.update((w - 1, w, w + 1))
    return [w for w in out if 0 <= w < limit]


# How many codes `_sampled_values` takes from each end of a binade.
SAMPLED_CODES = 1 << 11


def _sampled_values(grid: Grid) -> list[float]:
    """Some of a format's finite magnitudes, where enumerating them all is out
    of the question: the first and last `SAMPLED_CODES` codes of the binades
    where the limits live -- the bottom two normal binades and the top two --
    and as many of the values below the normals from each end.

    Read off the grid rather than from code points, which for these formats
    means ``2**(exp_bits + man_bits)`` of them. Exact: every value is a code
    below ``2**53`` times a power of two binary64 holds (`in_binary64`).
    """
    m = grid.man_bits
    codes = 1 << m
    ends = sorted(
        set(range(min(codes, SAMPLED_CODES))) | set(range(max(0, codes - SAMPLED_CODES), codes))
    )
    out: set[float] = set()
    for e in {grid.normal_floor, grid.normal_floor + 1, grid.top_exp - 1, grid.top_exp}:
        if not grid.normal_floor <= e <= grid.top_exp:
            continue
        for t in ends:
            if e == grid.top_exp and t > grid.top_code:
                continue
            if e == grid.hole_exp and t == 0:
                continue  # EXTENDED_NORMALS' hole: that code is the zero
            out.add(math.ldexp(float(codes + t), e - m))
    if grid.family in ("binaryK", "fixedpoint"):
        # the subnormals, a single grid of 2**below_step, up to the normals
        below = 1 << max(0, grid.normal_floor - grid.below_step)
        for k in set(range(1, min(below, SAMPLED_CODES))) | set(
            range(max(1, below - SAMPLED_CODES), below)
        ):
            out.add(math.ldexp(float(k), grid.below_step))
    elif grid.super_floor is not None and grid.super_floor <= grid.normal_floor:
        # the supernormals, one power of two a binade
        for e in set(
            range(grid.super_floor, min(grid.normal_floor, grid.super_floor + SAMPLED_CODES))
        ) | set(range(max(grid.super_floor, grid.normal_floor - SAMPLED_CODES), grid.normal_floor)):
            out.add(math.ldexp(1.0, e))
    return sorted(out)


def audit_words(grid: Grid, args, light: bool):
    """The inputs of an audit, as the carrier's words with the sign bit clear."""
    if grid.carrier.name == "binary64":
        return _audit_words64(grid, args, light)
    import torch

    parts = [torch.arange(0, 1 << 23, dtype=torch.int64)]  # every binary32 subnormal
    man_n = 1 << (12 if light else 16)
    g = torch.Generator().manual_seed(20260912)
    edges = torch.tensor(
        [
            0,
            1,
            2,
            3,
            (1 << 21),
            (1 << 22) - 1,
            (1 << 22),
            (1 << 22) + 1,
            (1 << 23) - 2,
            (1 << 23) - 1,
        ],
        dtype=torch.int64,
    )
    # the binary32 binades that matter: the bottom (where a format that
    # reaches below 2**-126 is decided), the top (where saturation is), and
    # the ones around this format's own floor and ceiling, which move with the
    # bias and so cannot be a fixed list
    around = set()
    for e in (grid.bottom_exp, grid.normal_floor, grid.top_exp):
        around.update(range(e + 127 - 2, e + 127 + 3))
    fields = sorted({f for f in around if 1 <= f <= 254} | set(range(1, 9)) | set(range(246, 255)))
    for f in fields:
        man = torch.cat(
            [edges, torch.randint(0, 1 << 23, (man_n,), generator=g, dtype=torch.int64)]
        )
        parts.append((f << 23) | man)
    if not light:
        parts.append(torch.randint(0, 0x7F800000, (1 << 20,), generator=g, dtype=torch.int64))
    parts.append(torch.tensor(_critical_words(grid, args), dtype=torch.int64))
    return torch.unique(torch.cat(parts))


def _audit_words64(grid: Grid, args, light: bool):
    """`audit_words` for binary64, which has 2**52 subnormals and 2046 binades
    of normals, so no field is swept whole: the edges and random mantissas of
    field 0, of the fields around the format's floors and top, and of the
    carrier's own bottom and top fields; uniform random words; and the
    format's values with the midpoints between them."""
    import torch

    man_n = 1 << (12 if light else 16)
    g = torch.Generator().manual_seed(20260914)
    top = 1 << 52
    edges = torch.tensor(
        [0, 1, 2, 3, top >> 2, (top >> 1) - 1, top >> 1, (top >> 1) + 1, top - 2, top - 1],
        dtype=torch.int64,
    )
    around = set()
    for e in (grid.bottom_exp, grid.min_bottom, grid.normal_floor, grid.top_exp):
        around.update(range(e + 1023 - 2, e + 1023 + 3))
    fields = sorted(
        {0} | {f for f in around if 1 <= f <= 2046} | set(range(1, 9)) | set(range(2038, 2047))
    )
    parts = []
    for f in fields:
        man = torch.cat([edges, torch.randint(0, top, (man_n,), generator=g, dtype=torch.int64)])
        parts.append((f << 52) | man)
    if not light:
        parts.append(
            torch.randint(0, 0x7FF0000000000000, (1 << 20,), generator=g, dtype=torch.int64)
        )
    parts.append(torch.tensor(_critical_words(grid, args), dtype=torch.int64))
    return torch.unique(torch.cat(parts))


def audit(grid: Grid, args, device: str, light: bool = False, chunk: int = 1 << 22) -> dict:
    """Compare the kernel against :func:`expected` over a targeted input set."""
    import torch

    from mptorch.number import RoundMode

    words = audit_words(grid, args, light)
    modes = ("RNE", "RNA", "RU", "RD", "RZ")
    bad = {m: 0 for m in modes}
    bad["RO"] = 0
    first: dict[str, tuple[float, float, float]] = {}
    total = 0

    for start in range(0, words.numel(), chunk):
        w = words[start : start + chunk].to(device)
        if grid.carrier.name == "binary64":
            mag = w.view(torch.float64)
        else:
            mag = w.to(torch.int32).view(torch.float32)
        x = torch.cat([mag, -mag])
        total += x.numel()
        for name in modes + ("RO",):
            got = _quantize(x, grid, args, RoundMode[name])
            _tally(x.double(), got.double(), grid, args, name, bad, first)
    return {"total": total, "bad": bad, "first": first, **_reachable(grid, args, device)}


def _tally(xd, gd, grid: Grid, args, name: str, bad: dict, first: dict) -> None:
    """Count the kernel's answers ``gd`` to inputs ``xd`` that the format disagrees with."""
    import torch

    if name == "RO":
        # no value-level rule: round-to-odd picks between the two neighbours
        # by the encoding's last bit, so all that is checked here is that it
        # picks one of them (and that the sign and the saturation are right),
        # which is what a grid one binade out breaks.
        lo, hi, _ = _bracket(torch.abs(xd), grid)
        pos = _ends(hi, xd < 0, grid, args.saturation)
        neg = _ends(lo, xd < 0, grid, args.saturation)
        s = torch.where(xd < 0, -1.0, 1.0)
        ok = ((gd == s * pos) | (gd == s * neg) | (gd.abs() == 0)) & ~torch.isnan(xd)
        ok |= gd == xd  # an exactly representable input is its own answer
        if not args.signed:
            ok |= (xd < 0) & (gd == 0)
        wrong = ~ok
        want = None  # nothing to quote: RO is checked by membership
    else:
        want, mask = expected(xd, grid, args.saturation, args.signed, name)
        wrong = mask & ((gd != want) | (torch.signbit(gd) != torch.signbit(want)))
    n = int(wrong.sum())
    if n:
        bad[name] += n
        if name not in first:
            i = int(wrong.nonzero()[0])
            first[name] = (
                xd[i].item(),
                gd[i].item(),
                float("nan") if want is None else want[i].item(),
            )


def _every_value(st: Storage, device: str):
    """Every positive finite value of the dtype, from its bit patterns."""
    import torch

    top_word = ((1 << (15 - st.man_bits)) - 1) << st.man_bits  # the infinity
    words = torch.arange(1, top_word, dtype=torch.int32).to(torch.int16)
    return words.view(getattr(torch, st.name)).to(device)


def _gemm_multipliers(st: Storage, light: bool = False) -> list[float]:
    """The ``b`` of ``a * b``: powers of two across the dtype, and a few odd
    significands so the products carry more bits than one operand holds --
    which is what reaches a format finer than the dtype. ``a`` runs over every
    value of the dtype, so every fourth power still reaches every binade."""
    pows = [math.ldexp(1.0, e) for e in range(st.min_exp, st.top_exp + 1, 4 if light else 1)]
    odd = []
    for e in range(-8, 9, 2):
        for sig in (1.0 + math.ldexp(1.0, -st.man_bits), 1.5, 2.0 - math.ldexp(1.0, -st.man_bits)):
            odd.append(math.ldexp(sig, e))
    return pows + odd


def audit_storage(grid: Grid, args, device: str, st: Storage, light: bool = False) -> dict:
    """The format against its own value set, through a result stored in ``st``.

    ``gemm``: a K=1 fused GEMM, ``fma(a * b)``, over operands of the dtype. Its
    products are exact in binary32 and in binary64 alike (two significands of
    at most eleven bits), and they sit anywhere in the format's range -- so the
    format's answer to each is known, and every value it has is an answer to
    some. Products outside binary32's normals are left out, since what goes
    wrong there is `check`'s business and not the dtype's.

    ``quantize``: the elementwise quantizer over every value of the dtype. A
    format at least as fine as the dtype at some end of its range returns the
    dtype's own values there, so this can come out clean where ``gemm`` does
    not -- the difference is reported, not hidden.
    """
    import torch

    from mptorch.number import RoundMode

    dt = getattr(torch, st.name)
    a_mag = _every_value(st, device)
    modes = ("RNE", "RNA", "RU", "RD", "RZ", "RO")
    out: dict = {}
    if grid.carrier.name == "binary64":
        # `carrier=torch.float64`: the operands widened, the result narrowed
        # once, as the wrappers do it
        from mptorch.quant.ops import _narrowed

        def fma(a, b, mode):
            return _narrowed(_fma(a.double(), b.double(), grid, args, mode), dt)

        def quantize(x, mode):
            return _narrowed(_quantize(x.double(), grid, args, mode), dt)
    else:

        def fma(a, b, mode):
            return _fma(a, b, grid, args, mode)

        def quantize(x, mode):
            return _quantize(x, grid, args, mode)

    bad = {m: 0 for m in modes}
    first: dict = {}
    total = 0
    if grid.family == "fixedpoint":
        multipliers = []  # there is no fixed-point GEMM: the elementwise rule only
    else:
        multipliers = _gemm_multipliers(st, light)
    a = torch.cat([a_mag, -a_mag]).reshape(1, -1, 1)
    ad = a.double()
    for bv in multipliers:
        b = torch.full((1, 1, 1), bv, dtype=dt, device=device)
        assert b.double().item() == bv, f"{bv} is not a {st.name} value"
        xd = (ad * bv).reshape(-1)
        keep = (xd.abs() >= math.ldexp(1.0, B32.min_normal_exp)) & (
            xd.abs() < math.ldexp(1.0, B32.top_exp + 1)
        )
        if not bool(keep.any()):
            continue
        total += int(keep.sum())
        for name in modes:
            got = fma(a, b, RoundMode[name]).reshape(-1)
            _tally(xd[keep], got.double()[keep], grid, args, name, bad, first)
    out["gemm"] = {"total": total, "bad": bad, "first": first}

    bad = {m: 0 for m in modes}
    first = {}
    x = torch.cat([a_mag, -a_mag])
    for name in modes:
        got = quantize(x, RoundMode[name])
        _tally(x.double(), got.double(), grid, args, name, bad, first)
    out["quantize"] = {"total": x.numel(), "bad": bad, "first": first}

    exact = [v for v in _format_values(grid, args) if v != 0.0 and math.isfinite(v)]
    t64 = torch.tensor(exact, dtype=torch.float64)
    out["values"] = len(exact)
    out["not_in_dtype"] = int((t64.to(dt).double() != t64).sum()) if exact else 0
    return out


def _fma(a, b, grid: Grid, args, mode):
    """The fused GEMM, past `mptorch.quant`'s wrappers for the reason `_quantize` gives."""
    import torch

    from mptorch.number import SaturationMode, SubnormalsMode

    if grid.family == "binaryK":
        return torch.ops.mptorch.custom_matmul_binaryK_fma.default(
            a,
            b,
            False,
            False,
            True,
            args.K,
            args.P,
            args.bias,
            args.signed,
            0,
            mode.value,
            SaturationMode[args.saturation].value,
            SubnormalsMode[args.subnormals].value,
            0,
        )
    return torch.ops.mptorch.custom_matmul_superfp_fma.default(
        a,
        b,
        False,
        False,
        True,
        args.man,
        args.exp,
        args.normal_binades,
        args.bias,
        args.signed,
        0,
        mode.value,
        SaturationMode[args.saturation].value,
        0,
    )


def _quantize(x, grid: Grid, args, mode):
    """The kernel, reached past `mptorch.quant`'s wrappers.

    Those wrappers now hold a format against what binary32 can carry, and half
    the formats this script exists to measure are exactly the ones they refuse
    or warn about. The op underneath takes the same arguments and does the same
    work, so calling it directly keeps the measurement possible -- and keeps
    the tool independent of the rule it is checking, which is the point of the
    `enforced` column in the sweep.
    """
    import torch

    from mptorch.number import SaturationMode, SubnormalsMode

    if grid.family == "fixedpoint":
        return torch.ops.mptorch.fixedpoint_quant.default(
            x.contiguous(), args.wl, args.fl, 0, args.signed, args.symmetric, mode.value
        )
    if grid.family == "binaryK":
        return torch.ops.mptorch.binaryK_quant.default(
            x.contiguous(),
            args.K,
            args.P,
            args.bias,
            0,
            args.signed,
            mode.value,
            SaturationMode[args.saturation].value,
            SubnormalsMode[args.subnormals].value,
        )
    return torch.ops.mptorch.superfp_quant.default(
        x.contiguous(),
        args.man,
        args.exp,
        args.normal_binades,
        args.bias,
        0,
        args.signed,
        mode.value,
        SaturationMode[args.saturation].value,
    )


def _reachable(grid: Grid, args, device: str) -> dict:
    """How much of the format its carrier can hold, and the cast can reach.

    The audit above asks whether the kernel rounds correctly; this asks the
    prior question -- whether the values are there to round to. A value the
    cast does not give back unchanged is one no input can ever produce, which
    is the failure a caller cannot see: the tensor is full of numbers, they
    are simply not this format's. A binary64 format too wide to enumerate is
    asked about its `_sampled_values`, and says so.
    """
    import torch

    from mptorch.number import RoundMode

    exact = [v for v in _format_values(grid, args) if v != 0.0]
    sampled = False
    if not exact and (grid.carrier.name == "binary64" or grid.family == "fixedpoint"):
        if grid.in_binary64:
            exact, sampled = [v for v in _sampled_values(grid) if v != 0.0], True
    if not exact:
        return {"values": 0, "not_in_carrier": 0, "not_fixed": 0, "sampled": False}
    t64 = torch.tensor(exact, dtype=torch.float64)
    held_t = t64.to(getattr(torch, grid.carrier.dtype))
    held = (held_t.double() == t64) & torch.isfinite(t64)
    x = held_t[held].to(device)
    got = _quantize(x, grid, args, RoundMode.RNE)
    return {
        "values": len(exact),
        "not_in_carrier": int((~held).sum()),
        "not_fixed": int((got != x).sum()),
        "sampled": sampled,
    }


# Enumerating code points is 2**(exp_bits + man_bits) of them, so it answers
# for the formats anyone writes down and not for the ones that only exist to
# probe a limit. Past this the checks above are analytic anyway -- the finest
# spacing and the top exponent are what decide representability, and both are
# read off the parameters.
MAX_ENUMERATED = 1 << 22


# A fixed-point format has a value at every code, and the audit probes each
# value and the midpoints between them three ulps wide, so past this many codes
# it takes `_sampled_values` instead: both ends of the range, which is where its
# limits live, rather than tens of millions of probes.
MAX_FIXEDPOINT_ENUMERATED = 1 << 16


def _format_values(grid: Grid, args) -> list:
    """The format's own finite magnitudes, from the enumerators' code points."""
    if grid.family == "fixedpoint":
        if (1 << (grid.man_bits + 1)) > MAX_FIXEDPOINT_ENUMERATED:
            return []
        import fixedpoint_values

        # the positive codes: the two's complement end, a power of two one step
        # past the top, is a negative value only, and the audit's inputs reach it
        _, hi = fixedpoint_values.code_range(args.wl, args.signed, args.symmetric)
        return [math.ldexp(float(n), -args.fl) for n in range(hi + 1)]
    if (1 << (grid.man_bits + grid.exp_bits)) > MAX_ENUMERATED:
        return []
    if grid.family == "binaryK":
        import binaryK_values as enum_mod

        entries = enum_mod.enumerate_format(
            args.K, args.P, args.bias, args.signed, args.saturation, args.subnormals
        )
    else:
        import superfp_values as enum_mod

        entries = enum_mod.enumerate_format(
            args.man, args.exp, args.normal_binades, args.bias, args.signed, args.saturation
        )
    out = []
    for e in entries:
        if e.kind != "finite":
            continue
        v = e.value
        # a value binary64 cannot hold is out of binary32's reach a fortiori,
        # and math.ldexp on the exact (m, e) is how we know rather than guess
        out.append(abs(v) if v is not None and math.isfinite(v) else math.inf)
    return out


# --- reporting ----------------------------------------------------------------


def report(grid: Grid, verbose: bool = True, st: Storage | None = None) -> int:
    findings = check(grid) + (check_storage(grid, st) if st is not None else [])
    rank = {"ok": 0, "unfaithful": 1, "unstorable": 2, "unrepresentable": 2, "malformed": 3}
    worst = max((f.level for f in findings), key=lambda lv: rank[lv], default="ok")
    if verbose:
        print(f"{grid.label}, in {grid.carrier.name}")
        print(f"  precision      {grid.man_bits + 1} bits ({grid.man_bits} stored)")
        print(f"  exponent       {grid.exp_bits} bits")
        print(f"  smallest > 0   2^{grid.bottom_exp}")
        print(f"  finest step    2^{grid.step_exp}")
        print(f"  largest finite 2^{grid.top_exp} x (1 + {grid.top_code}/2^{grid.top_man})")
        print(f"  verdict        {worst}")
        for f in findings:
            print(f"    [{f.level}] {f.what}: {f.detail}")
    return rank[worst]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--unsigned", dest="signed", action="store_false", help="no sign bit")
        q.add_argument("--saturation", choices=SATURATIONS, default="SAT_NONE")
        q.add_argument(
            "--audit", action="store_true", help="check the kernel against the value set"
        )
        q.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
        q.add_argument("--light", action="store_true", help="a smaller audit input set")
        q.add_argument(
            "--dtype", choices=DTYPES, default="float32", help="the dtype the result is stored in"
        )
        q.add_argument(
            "--carrier", choices=tuple(CARRIERS), default="binary32", help="what the cast rounds in"
        )

    b = sub.add_parser("binaryK", help="an IEEE P3109 binaryK format")
    b.add_argument("-K", type=int, required=True)
    b.add_argument("-P", type=int, required=True)
    b.add_argument("--bias", type=int, default=None)
    b.add_argument("--subnormals", choices=SUBNORMALS, default="SUBNORMALS")
    common(b)

    s = sub.add_parser("superfp", help="a supernormal format")
    s.add_argument("--man", type=int, required=True)
    s.add_argument("--exp", type=int, required=True)
    s.add_argument("--normal-binades", dest="normal_binades", type=int, required=True)
    s.add_argument("--bias", type=int, required=True)
    common(s)

    f = sub.add_parser("fixedpoint", help="a fixed-point format (no saturation mode: it saturates)")
    f.add_argument("--wl", type=int, required=True, help="word length, the sign bit included")
    f.add_argument("--fl", type=int, required=True, help="fractional bits: the step is 2**-fl")
    f.add_argument("--symmetric", action="store_true", help="no two's complement bottom code")
    common(f)

    w = sub.add_parser("sweep", help="walk each parameter to its boundary")
    w.add_argument("--audit", action="store_true")
    w.add_argument("--device", default="cuda" if _has_cuda() else "cpu")
    w.add_argument("--light", action="store_true")
    w.add_argument("--dtype", choices=DTYPES, default="float32")
    w.add_argument("--carrier", choices=tuple(CARRIERS), default="binary32")
    w.add_argument(
        "--fixedpoint-only", action="store_true", help="only the fixed-point rows (both carriers)"
    )

    args = p.parse_args()
    # float32 is no storage question in binary32, and the storage audit
    # enumerates a 16-bit dtype, so a float32 result in binary64 is not asked
    # here: `tests/test_format_limits.py` pins float32's bounds
    if args.cmd == "sweep":
        return sweep(args)

    carrier = CARRIERS[args.carrier]
    if args.cmd == "fixedpoint":
        args.subnormals = "SUBNORMALS"
        grid = fixedpoint_grid(args.wl, args.fl, args.signed, args.symmetric, carrier)
    elif args.cmd == "binaryK":
        if args.bias is None:
            args.bias = 2 ** (args.K - args.P - 1) if args.signed else 2 ** (args.K - args.P)
        grid = binaryK_grid(
            args.K, args.P, args.bias, args.signed, args.saturation, args.subnormals, carrier
        )
    else:
        args.subnormals = "SUBNORMALS"
        grid = superfp_grid(
            args.man,
            args.exp,
            args.normal_binades,
            args.bias,
            args.signed,
            args.saturation,
            carrier,
        )
    st = storage(args.dtype) if args.dtype != "float32" else None
    code = report(grid, st=st)
    if args.audit and st is not None:
        r = audit_storage(grid, args, args.device, st, args.light)
        if r["values"]:
            print(
                f"  values         {r['values']} finite magnitudes, "
                f"{r['not_in_dtype']} not {st.name} values"
            )
        for op in ("gemm", "quantize"):
            if op == "gemm" and grid.family == "fixedpoint":
                print(f"  {op:<14} n/a: there is no fixed-point GEMM")
                continue
            q = r[op]
            n = sum(q["bad"].values())
            print(f"  {op:<14} {q['total']:,} inputs x 6 modes on {args.device}: {n} wrong")
            for m, c in q["bad"].items():
                if c:
                    x, got, want = q["first"][m]
                    extra = "" if math.isnan(want) else f", want {want!r}"
                    print(f"    {m}: {c} wrong, e.g. {x!r} -> {got!r}{extra}")
    elif args.audit:
        r = audit(grid, args, args.device, args.light)
        n = sum(r["bad"].values())
        if r["values"]:
            print(
                f"  values         {r['values']} finite magnitudes"
                f"{' (sampled)' if r['sampled'] else ''}, {r['not_in_carrier']} not "
                f"{grid.carrier.name} values, {r['not_fixed']} the cast cannot return"
            )
        else:
            print(f"  values         2^{grid.man_bits + grid.exp_bits} code points, not enumerated")
        print(f"  audit          {r['total']:,} inputs x 6 modes on {args.device}: {n} wrong")
        for m, c in r["bad"].items():
            if c:
                x, got, want = r["first"][m]
                extra = "" if math.isnan(want) else f", want {want!r}"
                print(f"    {m}: {c} wrong, e.g. {x!r} -> {got!r}{extra}")
    return code


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _enforced(grid: Grid, a) -> str:
    """What the shipped rule does with this format, as `mptorch.number` states it.

    Printed next to this script's own prediction and the audit, so the three
    can be read against each other: the script derives the boundary, the audit
    measures it, and this is what the library actually enforces. They agree on
    *whether* a format is clean; they differ on severity, since the library
    raises only for a format that cannot function and warns for one whose
    range merely outruns the carrier -- the call's carrier, which is the
    grid's here.
    """
    import warnings

    import torch

    from mptorch.number import (
        FormatRangeWarning,
        SaturationMode,
        SubnormalsMode,
        check_binaryK,
        check_fixedpoint,
        check_superfp,
    )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FormatRangeWarning)
            if grid.family == "fixedpoint":
                check_fixedpoint(
                    a.wl,
                    a.fl,
                    a.signed,
                    a.symmetric,
                    getattr(a, "prng_bits", 0),
                    carrier=getattr(torch, grid.carrier.dtype),
                )
            elif grid.family == "binaryK":
                check_binaryK(
                    a.K,
                    a.P,
                    a.bias,
                    a.signed,
                    SaturationMode[a.saturation],
                    SubnormalsMode[a.subnormals],
                    getattr(a, "prng_bits", 0),
                    carrier=getattr(torch, grid.carrier.dtype),
                )
            else:
                check_superfp(
                    a.man,
                    a.exp,
                    a.normal_binades,
                    a.bias,
                    SaturationMode[a.saturation],
                    getattr(a, "prng_bits", 0),
                    carrier=getattr(torch, grid.carrier.dtype),
                )
    except ValueError:
        return "raises"
    return "warns" if caught else "ok"


def _enforced_storage(grid: Grid, a, st: Storage, elementwise: bool) -> str:
    """:func:`_enforced` for the storage rule, as one op or the other states it."""
    import warnings

    import torch

    from mptorch.number import (
        FormatRangeWarning,
        SaturationMode,
        SubnormalsMode,
        check_binaryK_storage,
        check_fixedpoint_storage,
        check_superfp_storage,
    )

    if grid.family == "fixedpoint" and not elementwise:
        return "n/a"  # there is no fixed-point GEMM, so no GEMM rule
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FormatRangeWarning)
            if grid.family == "fixedpoint":
                check_fixedpoint_storage(
                    a.wl,
                    a.fl,
                    a.signed,
                    a.symmetric,
                    storage=getattr(torch, st.name),
                    elementwise=True,
                )
            elif grid.family == "binaryK":
                check_binaryK_storage(
                    a.K,
                    a.P,
                    a.bias,
                    a.signed,
                    SaturationMode[a.saturation],
                    SubnormalsMode[a.subnormals],
                    storage=getattr(torch, st.name),
                    elementwise=elementwise,
                )
            else:
                check_superfp_storage(
                    a.man,
                    a.exp,
                    a.normal_binades,
                    a.bias,
                    SaturationMode[a.saturation],
                    storage=getattr(torch, st.name),
                    elementwise=elementwise,
                )
    except ValueError:
        return "raises"
    return "warns" if caught else "ok"


def sweep_storage(args) -> int:
    """`sweep` for a result stored in float16 or bfloat16.

    Two ops, so two pairs of columns: what the library enforces for each, and
    what the audit measures through each. The script's own prediction is the
    value-set rule, which is the GEMM's; the elementwise rule is not restated
    here, so the ``quantize`` columns check the library against the kernel
    alone. The ``binary32`` column is `check`'s verdict: where it is not
    ``ok`` the measurement also carries binary32's own partiality, which is
    deliberate only in the bfloat16 top-of-range rows, whose bound *is*
    binary32's.
    """
    from types import SimpleNamespace

    st = storage(args.dtype)
    carrier = CARRIERS[args.carrier]
    rows = []

    def bk(K, P, bias, subnormals="SUBNORMALS", saturation="SAT_NONE", signed=True):
        a = SimpleNamespace(
            K=K, P=P, bias=bias, signed=signed, saturation=saturation, subnormals=subnormals
        )
        run(binaryK_grid(K, P, bias, signed, saturation, subnormals, carrier), a)

    def sfp(man, exp, nb, bias, saturation="SAT_NONE"):
        a = SimpleNamespace(
            man=man,
            exp=exp,
            normal_binades=nb,
            bias=bias,
            signed=True,
            saturation=saturation,
            subnormals="SUBNORMALS",
        )
        run(superfp_grid(man, exp, nb, bias, True, saturation, carrier), a)

    def verdict(q) -> str:
        return "ok" if not sum(q["bad"].values()) else f"{sum(q['bad'].values())} wrong"

    def run(grid, a):
        found = check_storage(grid, st)
        predicted = "ok" if not found else found[0].what.split(" ", 1)[1]
        enforced = (_enforced_storage(grid, a, st, False), _enforced_storage(grid, a, st, True))
        measured = ("", "")
        if args.audit:
            r = audit_storage(grid, a, args.device, st, light=True)
            measured = (verdict(r["gemm"]), verdict(r["quantize"]))
        label = grid.label + ("" if a.saturation == "SAT_NONE" else f" {a.saturation}")
        b32 = check(grid)
        b32_verdict = "ok" if not b32 else b32[0].what
        rows.append(
            (label, b32_verdict, predicted, enforced[0], measured[0], enforced[1], measured[1])
        )

    if st.name == "float16":
        for bias in range(13, 19):  # the top: bias >= 16
            bk(8, 3, bias)
        for bias in range(20, 26):  # the bottom: bias + P <= 26
            bk(8, 4, bias)
        for bias in range(20, 26):
            bk(8, 4, bias, "NORMALS")
        for bias in range(19, 27):  # the hole, which reaches float16 up to bias 24
            bk(8, 4, bias, "EXTENDED_NORMALS")
        for P in (10, 11, 12, 13):  # precision, and the saturated top it moves
            bk(P + 4, P, 8)
            bk(P + 4, P, 8, saturation="SAT_FINITE")
        for bias in range(11, 16):  # superfp's bottom: bias <= 13
            sfp(2, 3, 4, bias)
        for bias in range(-10, -6):  # and its top: bias >= -8
            sfp(2, 3, 4, bias)
        sfp(3, 4, 8, 7)
        sfp(3, 4, 15, 7)
        for K, P, bias in ((16, 11, 15), (8, 4, 7)):
            bk(K, P, bias)
    else:
        for P in (7, 8, 9, 10):
            bk(P + 4, P, 8)
            bk(P + 4, P, 8, saturation="SAT_FINITE")
        for bias in range(-114, -110):  # the top, which is binary32's
            bk(8, 4, bias)
        sfp(3, 4, 8, 7)
        for K, P, bias in ((15, 8, 64), (8, 4, 7), (8, 3, 15)):
            bk(K, P, bias)
    for K, P in ((8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (16, 11), (16, 8)):
        bias = 1 << (K - P - 1)
        if (1 << (K - P)) - 1 - bias <= B32.top_exp and 1 - bias - (P - 1) >= B32.min_simulable_exp:
            bk(K, P, bias)

    w = max(len(r[0]) for r in rows)
    print(f"{st.name}: results stored in it, rounded in {carrier.name}\n")
    print(
        f"{'format':<{w}}  {carrier.name:<16}  {'predicted':<16}  {'gemm: enforced':<14}"
        f"  {'measured':<14}  {'quantize: enforced':<18}  measured"
    )
    for label, b32, predicted, eg, mg, eq, mq in rows:
        print(f"{label:<{w}}  {b32:<16}  {predicted:<16}  {eg:<14}  {mg:<14}  {eq:<18}  {mq}")
    return 0


def sweep(args) -> int:
    """Walk one parameter at a time and print where the audit stops agreeing."""
    from types import SimpleNamespace

    if args.dtype != "float32":
        return sweep_storage(args)

    rows = []

    def run(grid, a):
        predicted = "ok" if not check(grid) else check(grid)[0].level
        measured = ""
        if args.audit and not grid.in_binary64:
            measured = "beyond binary64, not audited"
        elif args.audit:
            try:
                r = audit(grid, a, args.device, light=True)
            except (AssertionError, ValueError, RuntimeError) as exc:
                # the library already refuses some of these -- worth seeing
                # next to the prediction, since a limit it enforces is one
                # the caller cannot walk into
                rows.append(
                    (
                        grid.label,
                        predicted,
                        _enforced(grid, a),
                        f"refused: {str(exc).splitlines()[0][:60]}",
                    )
                )
                return
            n = sum(r["bad"].values())
            lost = r["not_in_carrier"] + r["not_fixed"]
            if n == 0 and lost == 0:
                measured = "ok"
            else:
                modes = ",".join(m for m, c in r["bad"].items() if c)
                measured = f"{lost}/{r['values']} values lost" + (
                    f", {modes} wrong" if modes else ""
                )
        rows.append((grid.label, predicted, _enforced(grid, a), measured))

    if getattr(args, "fixedpoint_only", False):
        _fixedpoint_rows(run, CARRIERS[args.carrier])
    elif args.carrier == "binary64":
        _binary64_rows(run)
        _fixedpoint_rows(run, CARRIERS["binary64"])
    else:
        for bias in range(120, 130):
            a = SimpleNamespace(
                K=8, P=4, bias=bias, signed=True, saturation="SAT_NONE", subnormals="SUBNORMALS"
            )
            run(binaryK_grid(8, 4, bias, True, "SAT_NONE", "SUBNORMALS"), a)
        for sub_mode in ("NORMALS", "EXTENDED_NORMALS"):
            for bias in range(120, 131):
                a = SimpleNamespace(
                    K=8, P=4, bias=bias, signed=True, saturation="SAT_NONE", subnormals=sub_mode
                )
                run(binaryK_grid(8, 4, bias, True, "SAT_NONE", sub_mode), a)
        # the two precisions whose half-floor moves that bottom a binade: P = 1
        # (NORMALS and EXTENDED_NORMALS alike) and P = 24 (EXTENDED_NORMALS), with
        # P = 2, 23 and NORMALS at 24 beside them to show the rest do not move
        for K, P, sub_mode in (
            (8, 1, "NORMALS"),
            (8, 1, "EXTENDED_NORMALS"),
            (9, 2, "EXTENDED_NORMALS"),
            (30, 23, "EXTENDED_NORMALS"),
            (31, 24, "EXTENDED_NORMALS"),
            (31, 24, "NORMALS"),
        ):
            for bias in range(124, 129):
                a = SimpleNamespace(
                    K=K, P=P, bias=bias, signed=True, saturation="SAT_NONE", subnormals=sub_mode
                )
                run(binaryK_grid(K, P, bias, True, "SAT_NONE", sub_mode), a)
        for bias in range(-118, -108):
            a = SimpleNamespace(
                K=8, P=4, bias=bias, signed=True, saturation="SAT_NONE", subnormals="SUBNORMALS"
            )
            run(binaryK_grid(8, 4, bias, True, "SAT_NONE", "SUBNORMALS"), a)
        for P in (22, 23, 24, 25, 26):
            a = SimpleNamespace(
                K=P + 4, P=P, bias=8, signed=True, saturation="SAT_NONE", subnormals="SUBNORMALS"
            )
            run(binaryK_grid(P + 4, P, 8, True, "SAT_NONE", "SUBNORMALS"), a)
        for bias in range(18, 26):
            a = SimpleNamespace(
                man=3, exp=4, normal_binades=1, bias=bias, signed=True, saturation="SAT_NONE"
            )
            run(superfp_grid(3, 4, 1, bias, True, "SAT_NONE"), a)
        for nb in (1, 2, 4, 8, 16):
            a = SimpleNamespace(
                man=4, exp=4, normal_binades=nb, bias=7, signed=True, saturation="SAT_NONE"
            )
            run(superfp_grid(4, 4, nb, 7, True, "SAT_NONE"), a)
        for bias in range(-124, -114):
            a = SimpleNamespace(
                man=2, exp=3, normal_binades=1, bias=bias, signed=True, saturation="SAT_NONE"
            )
            run(superfp_grid(2, 3, 1, bias, True, "SAT_NONE"), a)

        _fixedpoint_rows(run, B32)

        # and the formats a caller would actually write down, at P3109's own bias
        for K, P in ((6, 3), (8, 3), (8, 4), (8, 5), (11, 4), (12, 4), (16, 8), (16, 11), (16, 5)):
            for signed in (True, False):
                bias = (1 << (K - P - 1)) if signed else (1 << (K - P))
                a = SimpleNamespace(
                    K=K,
                    P=P,
                    bias=bias,
                    signed=signed,
                    saturation="SAT_NONE",
                    subnormals="SUBNORMALS",
                )
                run(binaryK_grid(K, P, bias, signed, "SAT_NONE", "SUBNORMALS"), a)

    w = max(len(r[0]) for r in rows)
    if args.carrier != "binary32":
        print(f"rounded in {args.carrier}\n")
    print(f"{'format':<{w}}  {'predicted':<16}  {'enforced':<9}  measured")
    for label, predicted, enforced, measured in rows:
        print(f"{label:<{w}}  {predicted:<16}  {enforced:<9}  {measured}")
    return 0


def _fixedpoint_rows(run, c: Carrier) -> None:
    """The fixed-point bounds in carrier ``c``, each walked across its edge.

    Its precision (``mag_bits`` against the carrier's), its ends against the
    carrier's largest value -- the two's complement bottom, ``-2**(mag_bits -
    fl)``, and the symmetric and unsigned tops a binade higher -- the step's
    floor, and a range below the carrier's normals; then formats a caller would
    write down. The audit is deterministic, so SR is not in it: the cast
    sweeps' ``fixed`` and ``fixed-beyond`` modes hold SR to its definition.
    """
    from types import SimpleNamespace

    def fx(wl, fl, signed=True, symmetric=False):
        a = SimpleNamespace(
            wl=wl,
            fl=fl,
            signed=signed,
            symmetric=symmetric,
            saturation="SAT_NONE",
            subnormals="SUBNORMALS",
        )
        run(fixedpoint_grid(wl, fl, signed, symmetric, c), a)

    precision = c.man_bits + 1
    for mag in (precision - 1, precision, precision + 1):
        fx(mag + 1, mag - 4)
        fx(mag, mag - 4, signed=False)
    # wl 8: the two's complement bottom is -2**(7 - fl), the tops 2**(6 - fl)
    for fl in range(6 - c.top_exp, 9 - c.top_exp):
        fx(8, fl)
    for fl in range(5 - c.top_exp, 8 - c.top_exp):
        fx(8, fl, symmetric=True)
        fx(7, fl, signed=False)
    floor = -c.min_simulable_exp
    for fl in range(floor - 1, floor + 3):
        fx(8, fl)
    below = 6 - c.min_normal_exp  # the largest value at the carrier's smallest normal
    for fl in (below, below + 1):
        fx(8, fl)
    for wl, fl, signed, symmetric in (
        (4, 2, True, False),
        (8, 6, True, True),
        (12, 0, True, False),
        (16, 8, True, False),
        (16, 15, False, False),
    ):
        fx(wl, fl, signed, symmetric)


def _binary64_rows(run) -> None:
    """`sweep`'s rows for binary64: each of its bounds, walked across its edge.

    The edges are binary32's derivation transposed (`mptorch.number`'s
    `_Carrier`), and these rows are what says the kernels agree -- the ones
    `tests/test_format_limits.py` pins.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    c = CARRIERS["binary64"]

    def bk(K, P, bias, sub="SUBNORMALS", signed=True, prng_bits=0):
        a = SimpleNamespace(
            K=K,
            P=P,
            bias=bias,
            signed=signed,
            saturation="SAT_NONE",
            subnormals=sub,
            prng_bits=prng_bits,
        )
        grid = binaryK_grid(K, P, bias, signed, "SAT_NONE", sub, c)
        run(replace(grid, label=grid.label + f" prng={prng_bits}") if prng_bits else grid, a)

    def sfp(man, exp, nb, bias):
        a = SimpleNamespace(
            man=man,
            exp=exp,
            normal_binades=nb,
            bias=bias,
            signed=True,
            saturation="SAT_NONE",
            subnormals="SUBNORMALS",
        )
        run(superfp_grid(man, exp, nb, bias, True, "SAT_NONE", c), a)

    # the bottom under SUBNORMALS: bias + P <= 1023
    for bias in range(1016, 1023):
        bk(14, 4, bias)
    # NORMALS and EXTENDED_NORMALS compare magnitudes: bias <= 1023, and 1022
    for sub in ("NORMALS", "EXTENDED_NORMALS"):
        for bias in range(1019, 1026):
            bk(14, 4, bias, sub)
    # the two half-floor exceptions, a binade short: P = 1 in both modes, and
    # EXTENDED_NORMALS at binary64's full precision -- with P = 2, P = 52 and
    # NORMALS at P = 53 beside them to show the rest do not move
    for K, P, sub in (
        (11, 1, "NORMALS"),
        (11, 1, "EXTENDED_NORMALS"),
        (12, 2, "EXTENDED_NORMALS"),
        (62, 52, "EXTENDED_NORMALS"),
        (63, 53, "EXTENDED_NORMALS"),
        (63, 53, "NORMALS"),
    ):
        for bias in range(1019, 1025):
            bk(K, P, bias, sub)
    # the top: bias >= 2**exp_bits - 1024
    for bias in range(-3, 3):
        bk(14, 4, bias)
    # precision: P <= 53
    for P in (51, 52, 53, 54, 55):
        bk(P + 10, P, 512)
    # the exponent field at P3109's bias: ten bits, and not eleven
    for K, P in ((14, 4), (15, 4), (22, 12), (23, 12), (63, 53), (64, 53)):
        for signed in (True, False):
            e = K - P if signed else K - P + 1
            bk(K, P, 1 << (e - 1), signed=signed)
    # superfp's supernormal floor, 2**-1021, and its top
    for bias in range(914, 921):
        sfp(3, 4, 1, bias)
    for bias in range(-1010, -1005):
        sfp(2, 4, 1, bias)
    # stochastic rounding's bits share binary64's 52 mantissa bits with the
    # format's: the audit is deterministic, so only the enforced column moves
    for prng_bits in (48, 49, 50):
        bk(14, 4, 512, prng_bits=prng_bits)
    # and the formats a caller would write down, at P3109's bias
    for K, P in ((16, 8), (24, 16), (32, 24), (40, 30), (48, 40)):
        bk(K, P, 1 << (K - P - 1))


if __name__ == "__main__":
    sys.exit(main())
