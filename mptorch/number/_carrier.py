"""The limits of the two carriers, and the range check that binaryK and superfp share.

The casts do not produce bit encodings. They take a value of their
*carrier*, the binary floating-point type the arithmetic runs in, round it to
a value of the format, and return that value as a carrier value
(`docs/source/concepts.rst`, "What the carrier can hold"). By default the
carrier follows the tensor: binary64 for float64, and binary32 for float32,
float16 and bfloat16. A call can ask for binary64 for those three as well,
with `carrier=torch.float64`; the carrier is never narrower than the tensor.

A format with more range or precision than its carrier is quantized only
*partially*: the results still look like plausible numbers, but some of the
format's values never appear among them. The checks in the family modules,
built on this one, exist to report that.
`dev/benchmarks/formats/format_limits.py` measures the limits stated here
against the kernels, over about 17 million inputs per format in every
rounding mode, and `tests/test_format_limits.py` pins the boundaries it
found.

A format can fit a carrier in two senses, and at the bottom of the range they
differ by one binade (the interval between two consecutive powers of two):

  representable  every value of the format is also a value of the carrier.
                 This is plain arithmetic: at most the carrier's significand
                 bits (24, 53), nothing above its largest finite value, and
                 no two values closer together than its smallest subnormal
                 (2**-149, 2**-1074).
  faithful       the cast returns the correct format value for every input.
                 This is stronger. A cast works out where an input lies from
                 the input's *exponent field*, and all of a carrier's
                 subnormals have exponent field 0, so the cast cannot tell
                 which binade an input below the smallest normal (2**-126,
                 2**-1022) lies in.
                 For a cast that reads the field at the bottom of the range,
                 the format's smallest value must therefore be at least one
                 binade higher, 2**-125 (2**-1021). `binaryk`'s
                 `_binaryK_extent` derives the limit for each
                 `SubnormalsMode`: two of the modes compare magnitudes
                 instead of reading the field, and can go one binade lower.

From the top of the range down to that limit, binary32 has 253 binades for a
format to use and binary64 has 2045, which allow at most seven and ten
exponent bits respectively.

Whether a finding is a warning or an error does not depend on which of the
two senses is missed, but on whether the format still *works*. A format whose
range exceeds its carrier still quantizes correctly over the part that fits
(a binaryK with a very wide range, used to study precision alone, is a
legitimate use, and `tests/` uses several on purpose). It is only partial, so
it gets a warning. An error is for a format that cannot work at all: more
precision than the carrier has, an exponent field wider than the kernels can
shift by, stochastic-rounding bits that do not fit in the carrier's
significand next to the mantissa, a superfp that leaves no codes for its
supernormals, or a format with no finite normal value inside the carrier's
range, for which every nonzero result overflows.

The carrier is chosen by each call, not by the format, so these findings are
reported on each call, from `mptorch.quant.ops`, as the storage findings
(`_storage`) are. When a `BinaryK` or `SuperFP` is built, it checks only for
what no carrier can do (binary64's errors, binary64 being the wider carrier),
and it never warns.

`binaryk` and `superfp` compute their format's `_Extent` and compare it with
a carrier using `_range_findings`; `fixedpoint`, whose formats have no
exponent field, makes its own comparison. Each family caches its findings
(`_binaryK_findings` and its counterparts), because they are needed on every
call: the elementwise quantizers and the flat GEMM wrappers look their
formats up on each call, and a cache hit costs 0.25 us against 1.04 us to
recompute. The reporting is deliberately not cached. Caching the *warning*
would make it fire once per process and then never again, which is worse
than the `warnings` module's own once-per-location default, and would make
the tests depend on the order they run in.
"""

import math
from collections.abc import Callable
from typing import NamedTuple

import torch


class _Carrier(NamedTuple):
    """The binary floating-point type a cast rounds in, described by the
    numbers the range checks use.

    There are two instances, ``_BINARY32`` and ``_BINARY64``. Each ``*_exp``
    field is an unbiased exponent ``e``, standing for the power of two
    ``2**e``. ``man_bits`` is ``precision - 1``, the number of stored mantissa
    bits. ``min_normal_exp`` is the exponent of the smallest normal number
    (the one exponent field 1 encodes), and ``min_simulable_exp`` is one
    higher: the lowest that a format's smallest value may be when the cast
    reads the input's exponent field.
    """

    name: str
    precision: int  # significand bits, including the implicit leading bit
    man_bits: int
    top_exp: int  # exponent of the largest finite value
    min_normal_exp: int
    min_subnormal_exp: int  # exponent of the smallest subnormal, the smallest positive value
    min_simulable_exp: int  # the lowest exponent a format's smallest value may have


_BINARY32 = _Carrier("binary32", 24, 23, 127, -126, -149, -125)
_BINARY64 = _Carrier("binary64", 53, 52, 1023, -1022, -1074, -1021)
# Indexed by the dtype that a `carrier=` argument names. float16 and bfloat16
# are never carriers: tensors of those dtypes are rounded in binary32.
_CARRIERS: dict[torch.dtype, _Carrier] = {torch.float32: _BINARY32, torch.float64: _BINARY64}
_MAX_EXP_BITS = 30  # the kernels compute `1 << exp_bits` in a 32-bit C++ `int`


def _top_of_range(
    exp_bits: int, man_bits: int, bias: int, reserved: int, carrier: _Carrier
) -> tuple[int, int, int]:
    """Return a format's largest finite value as ``(exponent field, unbiased
    exponent, mantissa code)``.

    This transcribes ``make_normal_range_params`` in ``bit_helper.h``. P3109
    counts the largest finite value in code points: the last ``reserved``
    codes of the top binade are not finite values (an infinity outside the
    finite domain, and an unsigned binaryK's NaN). With at most one mantissa
    bit there can be more reserved codes than the top binade has, which moves
    the largest finite value one or two binades down, so the loop takes the
    missing codes from the binades below until the mantissa code is
    non-negative. Each binade is counted as ``2**min(man_bits,
    carrier.man_bits)`` codes, as the kernel counts them; a format with more
    mantissa bits than the carrier has been rejected for its precision before
    this is called.
    """
    codes = 1 << min(man_bits, carrier.man_bits)
    top_code = codes - 1 - reserved
    top_field = (1 << exp_bits) - 1
    while top_code < 0:
        top_code += codes
        top_field -= 1
    return top_field, top_field - bias, top_code


def _largest_finite(exp_bits: int, man_bits: int, bias: int, reserved: int) -> float:
    """Return a format's largest finite value as a float.

    It is computed with `_top_of_range` against binary64, which holds every
    format that either carrier can simulate. In every such format the largest
    finite value lies in a normal binade (a format without a finite normal
    value is rejected), so it is the implicit leading bit plus the top
    mantissa code, scaled to that binade.
    """
    _, top_exp, top_code = _top_of_range(exp_bits, man_bits, bias, reserved, _BINARY64)
    return math.ldexp((1 << man_bits) + top_code, top_exp - man_bits)


def _width_error(
    label: str, exp_bits: int, man_bits: int, prng_bits: int, carrier: _Carrier
) -> str | None:
    """Check the bit widths, which must be valid before any code shifts by ``exp_bits``.

    Return the error message for the first violation, or ``None``. The
    violations, in the order checked: a negative ``prng_bits``; an exponent
    field wider than ``_MAX_EXP_BITS``; a mantissa wider than the carrier's;
    and stochastic-rounding bits that do not fit in the carrier's significand
    next to the mantissa. Each makes the format unusable, so each is an
    error, never a warning.
    """
    if prng_bits < 0:
        # `BinaryK` and `SuperFP` reject a negative count in `__post_init__`, but
        # the wrappers in `mptorch.quant.ops` that take plain integers come here
        # unchecked
        return (
            f"{label} has prng_bits={prng_bits}: the number of stochastic-rounding bits "
            "cannot be negative"
        )
    if exp_bits > _MAX_EXP_BITS:
        binades = carrier.top_exp - carrier.min_simulable_exp + 1
        return (
            f"{label} has {exp_bits} exponent bits, more than the kernels support: they "
            f"compute `1 << exp_bits` in a 32-bit int, and in any case {carrier.name}'s range "
            f"has room for at most {binades.bit_length() - 1} exponent bits"
        )
    if man_bits > carrier.man_bits:
        return (
            f"{label} has {man_bits + 1} bits of precision, but it is rounded in "
            f"{carrier.name}, which has {carrier.precision}: {carrier.name} cannot represent "
            "the format's values even in its normal range, so the cast cannot round to them"
        )
    if man_bits + prng_bits > carrier.man_bits:
        return (
            f"{label} has too many stochastic-rounding bits: its {man_bits} mantissa bits "
            f"and {prng_bits} random bits must fit together in the {carrier.man_bits} bits "
            f"of {carrier.name}'s mantissa"
        )
    return None


class _Extent(NamedTuple):
    """A format's range, reduced to the numbers the range checks use.

    `_binaryK_extent` or `_superfp_extent` computes it from the format's
    parameters. It is compared with two things: the carrier the cast rounds in
    (`_range_findings`), and a storage dtype narrower than the carrier, which
    the rounded result is written to (`_storage`). Each ``*_exp`` field is an
    unbiased exponent ``e``, standing for ``2**e``.
    """

    label: str
    exp_bits: int
    man_bits: int
    top_field: int  # the largest finite value's exponent field
    top_exp: int  # ... its exponent
    top_code: int  # ... and its mantissa code (`man_bits` wide)
    bottom_exp: int  # the exponent of the smallest positive value
    step_exp: int  # the exponent of the smallest gap between two neighbouring values
    min_bottom: int  # the lowest `bottom_exp` the carrier's cast still handles correctly
    bottom_why: str  # ... and why, as a sentence for the warning
    # `normal_exp` is the exponent of the lowest binade with a full mantissa,
    # and `below` says what lies under it: the name of a `SubnormalsMode`
    # member, or "SUPERNORMALS" for superfp's powers of two.
    normal_exp: int
    below: str


def _range_findings(ext: _Extent, carrier: _Carrier) -> tuple[str | None, str | None]:
    """Compare the top and bottom of a format's range with the carrier's.

    Call it only after `_width_error` has passed. Return ``(error, warning)``,
    with at most one of them set. The checks, in order: a format with no
    finite normal value that the carrier can hold, for which every nonzero
    result overflows (the only error); values above the carrier's largest
    finite value; a gap between neighbouring values (``step_exp``) below the
    carrier's smallest subnormal; and a smallest value (``bottom_exp``) below
    ``min_bottom``, the limit `_binaryK_extent` derives for each subnormals
    mode.

    Only the first finding is returned: a format that exceeds its carrier at
    both ends is told about the top first, and about the bottom once the top
    is fixed.
    """
    label, top_exp, bottom_exp, step_exp = ext.label, ext.top_exp, ext.bottom_exp, ext.step_exp
    name = carrier.name
    if ext.top_field < 1 or top_exp < carrier.min_normal_exp:
        return (
            f"{label} has no finite normal value in {name}'s normal range: its largest "
            f"finite code is in the 2^{top_exp} binade, so the cast treats every nonzero "
            "result as an overflow"
        ), None
    if top_exp > carrier.top_exp:
        return None, (
            f"{label} reaches 2^{top_exp}, above {name}'s top binade, 2^{carrier.top_exp}: "
            f"the cast never returns the format's values above {name}'s largest finite value "
            f"(a bias of at least {ext.top_field - carrier.top_exp} brings the format's top "
            f"within {name}'s range)"
        )
    if step_exp < carrier.min_subnormal_exp:
        return None, (
            f"{label} has values 2^{step_exp} apart at the bottom of its range, finer than "
            f"{name}'s smallest subnormal, 2^{carrier.min_subnormal_exp}: {name} cannot "
            "represent them, so the cast never returns the bottom of the format's range"
        )
    if bottom_exp < ext.min_bottom:
        return None, (
            f"{label}'s smallest value is in the 2^{bottom_exp} binade, below "
            f"2^{ext.min_bottom}, the lowest the cast handles correctly in {name}: "
            f'{ext.bottom_why}. See "What the carrier can hold" in the documentation.'
        )
    return None, None


# The sentences that `_Extent.bottom_why` puts in the warning, to explain why
# values below `_Extent.min_bottom` are lost. This one applies to every cast
# that reads the exponent field at the bottom of the range; `binaryk` defines
# the two that apply only to its `NORMALS` and `EXTENDED_NORMALS` limits.


def _bottom_by_field(c: _Carrier) -> str:
    """Explain why a cast that reads the exponent field loses values below its limit."""
    return (
        f"the cast locates an input by its {c.name} exponent field, which all {c.name} "
        f"subnormals share, so it rounds inputs below 2^{c.min_normal_exp} onto the wrong "
        "grid and never returns some of the format's values there"
    )


# When binary32 rejects a format that binary64 accepts, the message says so:
# the fix is then to use the binary64 carrier, not to change the format.
_BINARY64_HINT = (
    "binary64 can hold this format: use a float64 tensor, or pass carrier=torch.float64 "
    "to round a narrower tensor in binary64."
)


def _pointing_wider(
    find: Callable[[_Carrier], tuple[str | None, str | None]], carrier: torch.dtype
) -> tuple[str | None, str | None]:
    """Run ``find`` for ``carrier``. If the carrier is binary32 and ``find``
    reports a problem that it does not report for binary64, append the fix
    (``carrier=torch.float64``) to the message.
    """
    error, warning = found = find(_CARRIERS[carrier])
    if carrier is torch.float32 and found != (None, None) and find(_BINARY64) == (None, None):

        def wider(message: str) -> str:
            end = "" if message.endswith(".") else "."
            return f"{message}{end} {_BINARY64_HINT}"

        return error and wider(error), warning and wider(warning)
    return found
