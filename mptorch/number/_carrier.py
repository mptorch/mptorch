"""What a carrier can hold: its bounds, and the range rule binaryK and superfp share.

The casts do not build encodings. They round a value of their *carrier*, the
binary float the arithmetic runs in, onto the format's grid and hand back a
value of the carrier (`docs/source/concepts.rst`, "What the carrier can
hold"). The carrier is the operand's by default, binary64 for a float64
tensor and binary32 for float32, float16 and bfloat16, and binary64 for
those three too where a call asks for it with `carrier=torch.float64`. It is
never narrower than the tensor. A format with more range or precision than
its carrier is quantized *partially*: the result is still a tensor of
plausible numbers, and some of the format's values are simply never among
them. The checks the family modules build on this one are what stop that
being invisible. `dev/benchmarks/formats/format_limits.py` measures the
bounds stated here against the kernels, over ~17 M inputs per format in every
rounding mode, and `tests/test_format_limits.py` pins the boundaries it found.

Two bounds, not one, and they differ by a binade:

  representable  every value of the format is a value of the carrier.
                 Arithmetic alone: its significand bits (24, 53), nothing
                 above its largest finite value, nothing spaced finer than
                 its smallest subnormal (2**-149, 2**-1074).
  faithful       the cast reaches those values, for every input. Stronger,
                 because a cast places its input by the *exponent field*
                 of the word and all of a carrier's subnormals share field
                 0, so nothing below its smallest normal (2**-126, 2**-1022)
                 can be told apart: a binade above that (2**-125, 2**-1021)
                 is the lowest value a format may have, for a cast that
                 reads the field down there. `binaryk`'s `_binaryK_extent`
                 derives the floor per `SubnormalsMode`, since two of them
                 compare magnitudes instead and reach a binade lower.

Together the top and bottom leave 253 binades to spend in binary32 and 2045
in binary64, which allows at most seven and ten exponent bits.

What separates a warning from an error here is not which of those two bounds
is missed but whether the format still *works*. A format whose range outruns
its carrier quantizes correctly over the part that fits (a wide binaryK used
as a precision-only target is a real thing, and `tests/` uses several on
purpose); it is simply partial, and being told so is the whole point. That
is a warning. An error is for a format that cannot function at all: a grid
finer than the carrier's (nothing survives), an exponent field wider than
the kernels shift by, stochastic bits with no significand left to draw
from, a superfp whose regions come out misordered, or one whose largest
finite code is below the carrier's normals, where the largest finite value
is zero and every nonzero result overflows.

Which carrier a format meets belongs to the call, not to the format, so a
carrier's findings are reported per call, from `mptorch.quant.ops`, as the
storage checks (`_storage`) are. All `BinaryK`/`SuperFP` check when they are
built is what no carrier can do (binary64's errors, binary64 being the
wider), and they warn about nothing.

`binaryk` and `superfp` derive their format's `_Extent` and hold it to a
carrier with `_range_findings`; `fixedpoint`, whose format has no exponent
field, holds its own. Each family memoizes what it finds (`_binaryK_findings`
and its twins), and none memoizes the reporting, deliberately. The
elementwise quantizers look their format up on every call, and a flat GEMM
wrapper resolves its formats on every call too, so this sits on a per-call
path: a cache hit is 0.25 us against 1.04 to redo the derivation. Caching
the *warning* instead would make it fire once per process and then go
quiet, which is a worse contract than `warnings`' own once-per-location
default and would make the tests order-dependent.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch


class _Carrier(NamedTuple):
    """The binary float a cast rounds in, as the range checks read it.

    Two instances exist, ``_BINARY32`` and ``_BINARY64``. Every exponent is an
    unbiased power of two: ``man_bits`` is ``precision - 1`` (the stored
    mantissa bits), ``min_normal_exp`` is the exponent of the smallest normal,
    which is also the exponent that field 1 encodes, and ``min_simulable_exp``
    is one binade above it, the lowest a format's smallest value may sit when
    the cast places inputs by their exponent field.
    """

    name: str
    precision: int  # significand bits, the implicit one included
    man_bits: int
    top_exp: int  # exponent of the largest finite value
    min_normal_exp: int
    min_subnormal_exp: int  # exponent of the smallest positive value
    min_simulable_exp: int  # the lowest a format's own smallest value may be


_BINARY32 = _Carrier("binary32", 24, 23, 127, -126, -149, -125)
_BINARY64 = _Carrier("binary64", 53, 52, 1023, -1022, -1074, -1021)
# Keyed on the dtype a `carrier=` argument names, which is the float the
# carrier is; float16 and bfloat16 never carry, they are rounded in binary32.
_CARRIERS: dict[torch.dtype, _Carrier] = {torch.float32: _BINARY32, torch.float64: _BINARY64}
_MAX_EXP_BITS = 30  # `1 << exp_bits` is a C++ `int` in the kernels


def _top_of_range(
    exp_bits: int, man_bits: int, bias: int, reserved: int, carrier: _Carrier
) -> tuple[int, int, int]:
    """The largest finite value of a format, in the kernel's terms.

    Returns ``(exponent field, exponent, mantissa code)`` of the largest
    finite value, transcribed from ``make_normal_range_params`` in
    ``bit_helper.h``. P3109 counts that value in code points: of the top
    binade's codes the last ``reserved`` hold no finite value (an infinity
    outside the finite domain, and the NaN of an unsigned binaryK), and with
    one or two mantissa bits those can outnumber the binade's codes, which
    moves the largest finite value a binade or two down, so the loop borrows
    from lower binades until the code is non-negative. Counted in
    ``min(man_bits, carrier.man_bits)`` codes, as the kernel counts them; a
    format asking for more than that has already been rejected for precision
    before this is read.
    """
    codes = 1 << min(man_bits, carrier.man_bits)
    top_code = codes - 1 - reserved
    top_field = (1 << exp_bits) - 1
    while top_code < 0:
        top_code += codes
        top_field -= 1
    return top_field, top_field - bias, top_code


def _largest_finite(exp_bits: int, man_bits: int, bias: int, reserved: int) -> float:
    """A format's largest finite value, from `_top_of_range` in binary64,
    which holds every format a carrier can simulate. The top binade is a
    normal one in every such format (one with no finite normal value is
    refused), so the value is its implicit bit plus the top code."""
    _, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, _BINARY64)
    return math.ldexp((1 << man_bits) + top_code, top_exp - man_bits)


def _width_error(
    label: str, exp_bits: int, man_bits: int, prng_bits: int, carrier: _Carrier
) -> str | None:
    """The bounds that must hold before anything shifts by ``exp_bits``.

    Returns the message for the first one violated, or ``None``: a negative
    ``prng_bits``, an exponent field wider than ``_MAX_EXP_BITS``, a mantissa
    wider than the carrier's, or stochastic-rounding bits that do not fit next
    to the mantissa in the carrier's significand. Each is a format that cannot
    function, never a partial one.
    """
    if prng_bits < 0:
        # `BinaryK`/`SuperFP` refuse a negative count in `__post_init__`; the
        # plain-integer wrappers in `mptorch.quant.ops` reach here directly
        return f"{label} asks for {prng_bits} stochastic-rounding bits, which cannot be negative"
    if exp_bits > _MAX_EXP_BITS:
        binades = carrier.top_exp - carrier.min_simulable_exp + 1
        return (
            f"{label} has {exp_bits} exponent bits; the kernels derive the format's "
            f"range with `1 << exp_bits` in a 32-bit int, and {carrier.name} spans "
            f"{binades} binades in any case, so at most {binades.bit_length() - 1} are usable"
        )
    if man_bits > carrier.man_bits:
        return (
            f"{label} asks for {man_bits + 1} bits of precision, and it is rounded in "
            f"{carrier.name}, which has {carrier.precision}: the format's grid is finer "
            f"than anything that can be returned"
        )
    if man_bits + prng_bits > carrier.man_bits:
        return (
            f"{label} asks for {prng_bits} stochastic-rounding bits below a "
            f"{man_bits}-bit mantissa; they are drawn from the {carrier.name} significand "
            f"the rounding happens in, which has {carrier.man_bits} bits to share"
        )
    return None


class _Extent(NamedTuple):
    """A format's range, in the only terms the range checks read.

    Derived once from the parameters by `_binaryK_extent` or `_superfp_extent`
    and held against two things: the carrier a cast rounds in, and a result
    narrower than that carrier, which the rounded value is stored in. Every
    exponent is an unbiased power of two.
    """

    label: str
    exp_bits: int
    man_bits: int
    top_field: int  # the largest finite value's exponent field
    top_exp: int  # ... its exponent
    top_code: int  # ... and its mantissa code, of `man_bits`
    bottom_exp: int  # the exponent of the smallest positive value
    step_exp: int  # the exponent of the finest spacing anywhere in the format
    min_bottom: int  # how low `bottom_exp` may go before the carrier's cast loses it
    bottom_why: str  # ... and what goes wrong below that, for the warning
    # What lies below the binades that carry a full mantissa, which start at
    # `normal_exp`: the name of one of `SubnormalsMode`'s three, or
    # "SUPERNORMALS" for superfp's powers of two.
    normal_exp: int
    below: str


def _range_findings(ext: _Extent, carrier: _Carrier) -> tuple[str | None, str | None]:
    """The top and bottom of the range against the carrier's, after the widths.

    Returns ``(error, warning)``, at most one of them set. ``bottom_exp`` is
    the exponent of the format's smallest positive value and ``step_exp`` that
    of its finest spacing; for a format whose lowest binade carries a mantissa
    the two differ by ``man_bits``. ``min_bottom`` is how low that smallest
    value may go, which `_binaryK_extent` derives per subnormals mode.

    Only the first finding is returned, so a format that outruns its carrier
    at both ends says so about the top first and about the bottom once that is
    fixed. The error is the one case that stops the format working at all: no
    finite value the carrier holds, so every nonzero result overflows.
    """
    label, top_exp, bottom_exp, step_exp = ext.label, ext.top_exp, ext.bottom_exp, ext.step_exp
    name = carrier.name
    if ext.top_field < 1 or top_exp < carrier.min_normal_exp:
        return (
            f"{label} has no finite normal value {name} can hold: the largest "
            f"finite code sits at 2^{top_exp}, so every nonzero result overflows"
        ), None
    if top_exp > carrier.top_exp:
        return None, (
            f"{label} reaches 2^{top_exp}, above {name}'s 2^{carrier.top_exp}: the cast "
            f"saturates at {name}'s largest value on the format's grid instead, and "
            f"the codes above it are unreachable (needs bias >= {ext.top_field - carrier.top_exp})"
        )
    if step_exp < carrier.min_subnormal_exp:
        return None, (
            f"{label} spaces its values 2^{step_exp} apart at the bottom, below "
            f"{name}'s 2^{carrier.min_subnormal_exp}: values that far down are not "
            f"{name} values at all, so the bottom of the range is unreachable"
        )
    if bottom_exp < ext.min_bottom:
        return None, (
            f"{label}'s smallest value is 2^{bottom_exp}, below 2^{ext.min_bottom}: "
            f'{ext.bottom_why}. See concepts.rst, "What the carrier can hold".'
        )
    return None, None


# Why a value below `_Extent.min_bottom` is lost, as the sentence
# `_Extent.bottom_why` carries into the warning. This one holds for every cast
# that reads the exponent field down there; `binaryk` has the two that only
# its `NORMALS` and `EXTENDED_NORMALS` floors meet.


def _bottom_by_field(c: _Carrier) -> str:
    """Why a cast that reads the exponent field loses values below the floor."""
    return (
        f"the casts place an input by its {c.name} exponent field, which every {c.name} "
        f"subnormal shares, so this format's values below 2^{c.min_normal_exp} are "
        "rounded onto the wrong grid and some are never returned"
    )


# A binary32 finding that binary64 does not share says so, since the cure is
# then the binary64 carrier rather than a different format.


def _pointing_wider(
    find: Callable[[_Carrier], tuple[str | None, str | None]], carrier: torch.dtype
) -> tuple[str | None, str | None]:
    """Run ``find`` for ``carrier`` and, if binary32 objects where binary64
    would not, append the way out (``carrier=torch.float64``) to its message.
    """
    error, warning = found = find(_CARRIERS[carrier])
    if carrier is torch.float32 and found != (None, None) and find(_BINARY64) == (None, None):

        def wider(message: str) -> str:
            end = "" if message.endswith(".") else "."
            return (
                f"{message}{end} binary64 holds all of it: a float64 tensor's carrier, "
                "which carrier=torch.float64 gives a narrower tensor too."
            )

        return error and wider(error), warning and wider(warning)
    return found
