"""What each carrier can hold, and what a narrower tensor can store.

This module guards the format checks of ``mptorch.number`` as ``mptorch.quant``
reports them.

A cast rounds a value of its carrier onto the format's grid and returns a value
of that carrier (binary64 for a float64 tensor or under
``carrier=torch.float64``, binary32 for float32, float16 and bfloat16). A format
with more range or precision than its carrier is therefore quantized
partially: the output is a tensor of plausible numbers in which some of the
format's values never appear. The checks guarded here make that visible, in
two strengths. A format that cannot work at all raises ``ValueError``, and one
that quantizes correctly over the part that fits warns with
``FormatRangeWarning``.

The carrier rule, for binary32: at most 24 significand bits, a largest finite
value no bigger than binary32's, at most seven exponent bits, and a smallest
value no lower than the floor the cast can place. That floor is 2**-125 where
the grid is derived from the input's exponent field, which every binary32
subnormal shares (binaryK's subnormals, superfp's supernormals), and 2**-126
where the bottom is a magnitude compare (``NORMALS``, ``EXTENDED_NORMALS``).
binary64 is the same derivation one carrier wider: 53 bits, 2**1023, floors
2**-1021 and 2**-1022, ten exponent bits.

Only the call knows its carrier, so the checks run per call, against the
carrier the call rounds in. Building a ``BinaryK`` or a ``SuperFP`` raises only
for what neither carrier can do, and warns about nothing.

The storage rule: a result narrower than its carrier (float16 or bfloat16 in
binary32, and float32 too in binary64) is rounded again when it is stored, so
the format the result holds is also checked against that dtype, per call. A
GEMM's sums reach the format's whole value set, so precision above the dtype's
raises and a top or a finest step past the dtype warns. An elementwise
quantizer's inputs are already the dtype's values, so only three edges of the
range can land off the dtype's grid, and those warn.

The pinned boundaries are not derived from the checks under test. They are the
ones ``dev/benchmarks/format_limits.py`` measured against the kernels, with an
oracle built from the format's value set rather than from a second cast:
``sweep --audit`` for binary32, ``sweep --carrier binary64 --audit`` for
binary64, and ``sweep --dtype float16 --audit`` (and ``--dtype bfloat16``) for
storage. A change that moves a bound should move a measurement first.
"""

import itertools
import re
import warnings
from typing import Any

import pytest
import torch

from mptorch import BinaryK, RoundMode, SaturationMode, SubnormalsMode, SuperFP
from mptorch.number import FormatRangeWarning, _binaryK_findings, _superfp_findings
from mptorch.quant import (
    FusedMac,
    QLinear,
    Quant,
    SplitMac,
    binaryK_matmul,
    binaryK_matmul_fma,
    binaryK_matmul_mixed,
    binaryK_quantize,
    qmatmul,
    superfp_matmul_fma,
    superfp_quantize,
)
from mptorch.quant.gemm import binaryK_gemm_formats
from mptorch.quant.mac import spec_for_mac
from mptorch.quant.ops import (
    _binaryK_fma_mixed_spec,
    _binaryK_fma_spec,
    _binaryK_mixed_spec,
    _binaryK_spec,
    _superfp_fma_mixed_spec,
    _superfp_fma_spec,
    _superfp_mixed_spec,
    _superfp_spec,
)

F32, F64 = torch.float32, torch.float64


def _silent(call):
    """Run ``call`` with ``FormatRangeWarning`` promoted to an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


def _in(dtype, build):
    """Return a thunk that builds a format and rounds a ``dtype`` tensor to it.

    Running the thunk is the call that checks the format against that dtype's
    carrier. ``build`` is a zero-argument callable, so construction errors
    surface inside the thunk too.
    """
    return lambda: Quant(build())(torch.ones(4, dtype=dtype))


@pytest.mark.parametrize("dtype", [F32, F64])
@pytest.mark.parametrize(
    "build",
    [
        lambda: BinaryK(8, 4),  # Binary8p4se
        lambda: BinaryK(8, 4, bias=7),  # OCP E4M3
        lambda: BinaryK(8, 3, bias=15),  # OCP E5M2
        lambda: BinaryK(8, 1),  # the one-bit-of-precision end
        lambda: BinaryK(6, 3),
        lambda: BinaryK(11, 4),  # the widest exponent that fits, at the default bias
        lambda: BinaryK(16, 11),
        lambda: BinaryK(8, 4, is_signed=False),
        lambda: BinaryK(8, 4, saturation=SaturationMode.SAT_FINITE),
        lambda: BinaryK(8, 4, subnormals=SubnormalsMode.NORMALS),
        lambda: BinaryK(8, 4, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        # smallest value exactly on the magnitude-compare floor, 2**-126
        lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.NORMALS),
        lambda: BinaryK(8, 4, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        # the exceptions to that floor, a binade higher at 2**-125 (P = 1 in
        # both modes, EXTENDED_NORMALS at P = 24)
        lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.NORMALS),
        lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        lambda: BinaryK(31, 24, bias=125, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        lambda: BinaryK(8, 4, prng_bits=20),  # man_bits 3 + 20 == binary32's 23
        lambda: SuperFP(3, 4, 1, 7),
        lambda: SuperFP(3, 4, 2, 7),
        lambda: SuperFP(3, 4, 8, 7),
        lambda: SuperFP(2, 4, 8, 7),
        lambda: SuperFP(3, 4, 15, 7),  # one supernormal binade left
        lambda: SuperFP(3, 4, 2, 7, prng_bits=8),
    ],
)
def test_formats_in_range_are_silent(build, dtype):
    """Formats the library, docs and tests use, plus formats exactly on a bound.

    All are inside binary32's bound, hence inside binary64's, which is wider at
    every edge. A warning here means a check rejects a format the kernels hold.
    """
    _silent(_in(dtype, build))


@pytest.mark.parametrize(
    ("build", "match", "stored"),
    [
        # Precision: binary32 has 24 significand bits and cannot hold a finer
        # grid, so no part of such a format quantizes correctly.
        (lambda: BinaryK(29, 25, bias=8), "bits of precision", True),
        # No finite normal value binary32 can hold: the largest finite value
        # is zero, so every nonzero result overflows and only 0 and the
        # infinities come back. In binary64 every value of these two formats is
        # below float32's smallest, hence `stored` is False.
        (lambda: BinaryK(8, 4, bias=200), "no finite normal value", False),
        (lambda: SuperFP(3, 4, 1, 200), "no finite normal value", False),
        # The random bits share binary32's 23-bit significand field with the
        # mantissa, and 3 + 21 does not fit.
        (lambda: BinaryK(8, 4, prng_bits=21), "stochastic-rounding bits", True),
        (lambda: SuperFP(3, 4, 2, 7, prng_bits=21), "stochastic-rounding bits", True),
    ],
)
def test_formats_that_cannot_function_in_binary32_raise_at_the_call(build, match, stored):
    """A format of which binary32 quantizes nothing raises at the float32 call.

    Each is whole in binary64, so building it and a float64 call are silent,
    and the float32 error points at ``carrier=torch.float64``. Naming that
    carrier on float32 operands works when float32 can store the result
    (``stored``) and raises the storage error otherwise.
    """
    fmt = _silent(build)
    with pytest.raises(ValueError, match=match) as raised:
        Quant(fmt)(torch.ones(4))
    assert "binary64 holds all of it" in str(raised.value)
    assert "carrier=torch.float64" in str(raised.value)
    _silent(lambda: Quant(fmt)(torch.ones(4, dtype=F64)))
    if stored:
        _silent(lambda: Quant(fmt, carrier=F64)(torch.ones(4)))
    else:
        with pytest.raises(ValueError, match="a float32 result stores none"):
            Quant(fmt, carrier=F64)(torch.ones(4))


def test_a_superfp_past_binary32s_precision_raises_there_and_warns_in_binary64():
    """The binary32 error offers binary64 only when binary64 holds the format.

    With a 24-bit mantissa superfp's supernormal codes are 2**24 binades
    apart, which outruns binary64's bottom as well, so the float32 error
    carries no hint and the float64 call warns.
    """
    fmt = _silent(lambda: SuperFP(24, 4, 1, 7))
    with pytest.raises(ValueError, match="bits of precision") as raised:
        Quant(fmt)(torch.ones(4))
    assert "binary64" not in str(raised.value)
    with pytest.warns(FormatRangeWarning, match="below binary64's 2\\^-1074"):
        Quant(fmt)(torch.ones(4, dtype=F64))


@pytest.mark.parametrize(
    ("build", "match"),
    [
        # precision: binary64 has 53 significand bits
        (lambda: BinaryK(60, 54), "54 bits of precision.*binary64, which has 53"),
        (lambda: SuperFP(53, 4, 1, 7), "54 bits of precision"),
        # no finite normal value binary64 holds
        (lambda: BinaryK(8, 4, bias=2000), "no finite normal value binary64"),
        # 39 exponent bits: the kernels compute `1 << exp_bits` in a 32-bit int
        (lambda: BinaryK(40, 1), "exponent bits"),
        # normal_binades == 2**exp_bits leaves superfp no supernormal codes:
        # its regions would be misordered and the lowest normal binade lost
        (lambda: SuperFP(3, 4, 16, 7), "no supernormal codes"),
        # the random bits share binary64's 52-bit significand field with the
        # mantissa, and 3 + 50 does not fit
        (lambda: BinaryK(8, 4, prng_bits=50), "stochastic-rounding bits.*52 bits"),
        (lambda: BinaryK(8, 4, prng_bits=-1), "non-negative"),
    ],
)
def test_what_no_carrier_can_do_raises_when_the_format_is_built(build, match):
    """Construction raises exactly what binary64, the wider carrier, cannot do."""
    with pytest.raises(ValueError, match=match):
        build()


def test_building_a_format_never_warns():
    """Construction is silent for a range that outruns one carrier or both.

    The carrier is not known until the format is used, so a warning at build
    time would be wrong for one of the two. Resolving a mac or a layer format
    is construction too.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        BinaryK(16, 8)  # below binary32's floor
        BinaryK(24, 8)  # above binary64's top as well
        SuperFP(4, 4, 2, 7)  # finer than binary32 at the bottom
        spec_for_mac(SplitMac(BinaryK(16, 8), BinaryK(24, 8)))
        binaryK_gemm_formats(16, 8)
    assert caught == []


@pytest.mark.parametrize(
    ("build", "match"),
    [
        # The top: codes above binary32's largest finite value are unreachable,
        # but everything below still quantizes. A wide binaryK used as a
        # precision-only target is exactly this case.
        (lambda: BinaryK(8, 4, bias=-113), "above binary32"),
        (lambda: BinaryK(16, 5), "above binary32"),
        (lambda: BinaryK(16, 8, is_signed=False), "above binary32"),
        (lambda: BinaryK(24, 8), "above binary32"),  # other test modules use it
        (lambda: SuperFP(2, 3, 1, -121), "above binary32"),
        # The bottom: values spaced finer than 2**-149, binary32's smallest
        # subnormal, are not binary32 values at all.
        (lambda: BinaryK(11, 4, bias=200), "apart at the bottom"),
        (lambda: SuperFP(4, 4, 2, 7), "apart at the bottom"),
    ],
)
def test_formats_whose_range_outruns_binary32_warn(build, match):
    """A range past binary32's warns: the part binary32 holds still quantizes."""
    with pytest.warns(FormatRangeWarning, match=match):
        _in(F32, build)()


@pytest.mark.parametrize(
    ("build", "smallest"),
    [
        (lambda: BinaryK(16, 8), "2^-134"),  # P3109's own Binary16p8se
        (lambda: BinaryK(12, 4), "2^-130"),
        (lambda: BinaryK(8, 4, bias=124), "2^-126"),
        (lambda: BinaryK(8, 4, bias=128, subnormals=SubnormalsMode.NORMALS), "2^-127"),
        (lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS), "2^-127"),
        (lambda: SuperFP(3, 4, 1, 22), "2^-126"),
    ],
)
def test_formats_below_binary32_normals_warn(build, smallest):
    """A smallest value binary32 holds but the cast cannot place warns, by name.

    Every cast classifies its input by the exponent field of the binary32
    word, and all binary32 subnormals share field 0. A format reaching below
    2**-126 is therefore rounded on a grid one binade too coarse (binaryK) or
    onto powers of two the cast cannot name (superfp). The warning names the
    format's smallest value. binary64 places all of these, so a float64 tensor
    is silent.
    """
    with pytest.warns(FormatRangeWarning, match=re.escape(smallest)):
        _in(F32, build)()
    _silent(_in(F64, build))


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        # SUBNORMALS: the smallest value is 2**(2 - bias - P), and the floor
        # of 2**-125 makes that bias + P <= 127
        (lambda: BinaryK(8, 4, bias=123), lambda: BinaryK(8, 4, bias=124)),
        # NORMALS and EXTENDED_NORMALS decide their bottom by comparing
        # magnitudes rather than by reading an exponent field, so they are
        # exact one binade lower, down to a floor of 2**-126: bias <= 127
        # under NORMALS (floor 2**(1 - bias)) and 126 under EXTENDED_NORMALS
        # (floor just above 2**-bias)
        (
            lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.NORMALS),
            lambda: BinaryK(8, 4, bias=128, subnormals=SubnormalsMode.NORMALS),
        ),
        (
            lambda: BinaryK(8, 4, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        ),
        # The exceptions are where half that floor is out of binary32's
        # reach, which stops these formats a binade short. At P = 1 the round
        # reads the exponent field, sees 2**-127 as a tie between two binades
        # and carries it to the floor ...
        (
            lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.NORMALS),
            lambda: BinaryK(8, 1, bias=127, subnormals=SubnormalsMode.NORMALS),
        ),
        (
            lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            lambda: BinaryK(8, 1, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        ),
        # ... and at P = 24 EXTENDED_NORMALS' half-floor, (1 + 2**-23) * 2**-127,
        # needs a 2**-150 step, below binary32's 2**-149
        (
            lambda: BinaryK(31, 24, bias=125, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            lambda: BinaryK(31, 24, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        ),
        # which NORMALS' power-of-two floor never needs
        (
            lambda: BinaryK(31, 24, bias=127, subnormals=SubnormalsMode.NORMALS),
            lambda: BinaryK(31, 24, bias=128, subnormals=SubnormalsMode.NORMALS),
        ),
        # superfp's supernormals are placed by the exponent field too: 2**-125
        (lambda: SuperFP(3, 4, 1, 21), lambda: SuperFP(3, 4, 1, 22)),
    ],
)
def test_bottom_boundary_is_where_it_was_measured(ok, bad):
    """Each pair straddles binary32's bottom bound by one step of the bias.

    ``format_limits.py sweep --audit`` found no wrong answer for ``ok`` and
    wrong answers for ``bad``. A check that drifts from the kernels fails one
    side of a pair.
    """
    _silent(_in(F32, ok))
    with pytest.warns(FormatRangeWarning):
        _in(F32, bad)()


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: BinaryK(8, 1, bias=127, subnormals=SubnormalsMode.NORMALS), "tie between two"),
        (
            lambda: BinaryK(8, 1, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            "tie between two",
        ),
        (
            lambda: BinaryK(31, 24, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            "round-to-nearest-away",
        ),
        # below 2**-126 the floor itself is out of reach, and the warning says
        # that rather than naming the half-floor
        (lambda: BinaryK(8, 1, bias=128, subnormals=SubnormalsMode.NORMALS), "exponent field"),
    ],
)
def test_bottom_warning_names_the_reason(build, match):
    """The warning text tells the two half-floor limits from the field one.

    A P = 1 tie and a P = 24 half-floor are different defects from a floor in
    the subnormal exponent field, and a user fixes each differently.
    """
    with pytest.warns(FormatRangeWarning, match=match):
        _in(F32, build)()


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        (lambda: BinaryK(8, 4, bias=-112), lambda: BinaryK(8, 4, bias=-113)),
        (lambda: SuperFP(2, 3, 1, -120), lambda: SuperFP(2, 3, 1, -121)),
    ],
)
def test_top_boundary_is_where_it_was_measured(ok, bad):
    """Each pair straddles binary32's largest finite value by one bias step."""
    _silent(_in(F32, ok))
    with pytest.warns(FormatRangeWarning, match="above binary32"):
        _in(F32, bad)()


# --- binary64's edges -----------------------------------------------------------
#
# The same derivation one carrier wider (53 bits, 2**1023, floors 2**-1021 and
# 2**-1022, ten exponent bits), rounded in by a float64 tensor. Measured by
# `dev/benchmarks/format_limits.py sweep --carrier binary64 --audit`: every
# `ok` below is a format whose audit found no wrong answer and no lost value,
# and every `bad` one whose audit found both, one parameter step away.

EXT64 = SubnormalsMode.EXTENDED_NORMALS
NRM64 = SubnormalsMode.NORMALS


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        # SUBNORMALS: smallest value 2**(2 - bias - P) >= 2**-1021, so
        # bias + P <= 1023
        (lambda: BinaryK(14, 4, bias=1019), lambda: BinaryK(14, 4, bias=1020)),
        # the magnitude compares, floor 2**-1022: bias <= 1023 under NORMALS
        # and 1022 under EXTENDED_NORMALS
        (
            lambda: BinaryK(14, 4, bias=1023, subnormals=NRM64),
            lambda: BinaryK(14, 4, bias=1024, subnormals=NRM64),
        ),
        (
            lambda: BinaryK(14, 4, bias=1022, subnormals=EXT64),
            lambda: BinaryK(14, 4, bias=1023, subnormals=EXT64),
        ),
        # the half-floor exceptions, a binade short: P = 1 in both modes ...
        (
            lambda: BinaryK(11, 1, bias=1022, subnormals=NRM64),
            lambda: BinaryK(11, 1, bias=1023, subnormals=NRM64),
        ),
        (
            lambda: BinaryK(11, 1, bias=1022, subnormals=EXT64),
            lambda: BinaryK(11, 1, bias=1023, subnormals=EXT64),
        ),
        # ... and EXTENDED_NORMALS at binary64's full precision, whose half-floor
        # needs a 2**-1075 step, below binary64's 2**-1074
        (
            lambda: BinaryK(63, 53, bias=1021, subnormals=EXT64),
            lambda: BinaryK(63, 53, bias=1022, subnormals=EXT64),
        ),
        # which one bit less precision, and NORMALS' power-of-two floor, never need
        (
            lambda: BinaryK(62, 52, bias=1022, subnormals=EXT64),
            lambda: BinaryK(62, 52, bias=1023, subnormals=EXT64),
        ),
        (
            lambda: BinaryK(63, 53, bias=1023, subnormals=NRM64),
            lambda: BinaryK(63, 53, bias=1024, subnormals=NRM64),
        ),
        # superfp's supernormal floor, 2**-1021
        (lambda: SuperFP(3, 4, 1, 917), lambda: SuperFP(3, 4, 1, 918)),
        # ten exponent bits at P3109's default bias, not eleven
        (lambda: BinaryK(14, 4), lambda: BinaryK(15, 4)),
    ],
)
def test_binary64_bottom_boundary_is_where_it_was_measured(ok, bad):
    """Each pair straddles binary64's bottom bound by one parameter step."""
    _silent(_in(F64, ok))
    with pytest.warns(FormatRangeWarning, match="binary64"):
        _in(F64, bad)()


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        (lambda: BinaryK(14, 4, bias=0), lambda: BinaryK(14, 4, bias=-1)),
        (lambda: SuperFP(2, 4, 1, -1008), lambda: SuperFP(2, 4, 1, -1009)),
    ],
)
def test_binary64_top_boundary_is_where_it_was_measured(ok, bad):
    """Each pair straddles binary64's largest finite value by one bias step."""
    _silent(_in(F64, ok))
    with pytest.warns(FormatRangeWarning, match="above binary64"):
        _in(F64, bad)()


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: BinaryK(11, 1, bias=1023, subnormals=NRM64), "tie between two"),
        (lambda: BinaryK(11, 1, bias=1023, subnormals=EXT64), "tie between two"),
        (lambda: BinaryK(63, 53, bias=1022, subnormals=EXT64), "round-to-nearest-away"),
        (lambda: BinaryK(11, 1, bias=1024, subnormals=NRM64), "binary64 exponent field"),
        (lambda: BinaryK(14, 4, bias=1020), "binary64 exponent field"),
    ],
)
def test_binary64_bottom_warning_names_the_reason(build, match):
    """binary64's bottom warnings tell the same three reasons apart."""
    with pytest.warns(FormatRangeWarning, match=match):
        _in(F64, build)()


def test_binary64_precision_and_stochastic_bits_are_where_they_were_measured():
    """``P = 53`` and ``man_bits + prng_bits = 52`` are the last binary64 rounds.

    One past either raises when the format is built, since no carrier is wider.
    """
    x = torch.ones(4, dtype=F64)
    _silent(lambda: Quant(BinaryK(63, 53, bias=512))(x))
    _silent(lambda: Quant(BinaryK(14, 4, bias=512, prng_bits=49), RoundMode.SR)(x))
    with pytest.raises(ValueError, match="54 bits of precision"):
        BinaryK(64, 54, bias=512)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        BinaryK(14, 4, bias=512, prng_bits=50)


def test_warning_can_be_filtered():
    """The warning has its own category, so a caller who means it can filter it.

    The filtered call still quantizes.
    """
    fmt = BinaryK(16, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", FormatRangeWarning)
        out = Quant(fmt)(torch.ones(4))
    assert caught == []
    assert torch.equal(out, torch.ones(4))


@pytest.mark.parametrize("dtype", [F32, torch.float16, torch.bfloat16])
def test_precision_is_bounded_by_the_carrier_not_by_storage(dtype):
    """Precision is checked against the carrier, not the dtype the tensor stores.

    A float64 operand rounds in binary64, and every other dtype in binary32
    unless the call names binary64. The 41-bit format below holds ``1 + 2**-30``
    exactly and binary32 does not, so the float64 call must return it and the
    narrower calls must raise rather than round it away.
    """
    x = torch.tensor([1.0 + 2.0**-30], dtype=F64)
    assert _silent(lambda: binaryK_quantize(x, 50, 41, bias=256)).item() == 1.0 + 2.0**-30
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_quantize(x.to(dtype), 50, 41, bias=256)
    with pytest.raises(ValueError, match="bits of precision"):
        superfp_quantize(x.to(dtype), 30, 4, 1, 7)
    # 24 bits of precision is the most binary32 can express, and it is allowed
    # whatever the dtype stores
    assert binaryK_quantize(x.to(dtype), 30, 24, bias=8).dtype == dtype
    # Naming binary64 applies its bounds to the narrower tensor too. The
    # inputs are the dtype's own values, which a finer format hands back
    # unchanged, so the storage check has nothing to report.
    y = x.to(dtype)
    assert torch.equal(_silent(lambda: binaryK_quantize(y, 50, 41, bias=256, carrier=F64)), y)
    # and a carrier narrower than the tensor is refused, not used
    with pytest.raises(ValueError, match="narrower than float64"):
        binaryK_quantize(x, 30, 24, bias=8, carrier=F32)


def test_the_carrier_a_call_rounds_in_is_the_one_it_checks():
    """One format gets each carrier's verdict, picked by dtype and ``carrier=``.

    ``BinaryK(16, 8)`` reaches 2**-134, below binary32's floor and inside
    binary64's.
    """
    mac = SplitMac(BinaryK(16, 8), BinaryK(16, 8))
    mac64 = SplitMac(BinaryK(16, 8), BinaryK(16, 8), carrier=F64)
    a = torch.ones(2, 3, dtype=F64)
    _silent(lambda: qmatmul(a, a.mT, mac))
    with pytest.warns(FormatRangeWarning, match="2\\^-134"):
        qmatmul(a.float(), a.float().mT, mac)
    with pytest.warns(FormatRangeWarning, match="2\\^-134"):
        Quant(BinaryK(16, 8))(a.float())
    # binary64 holds all of it, and a float32 result stores all of it: the
    # finest step, 2**-134, is on float32's subnormal grid (step 2**-149)
    _silent(lambda: qmatmul(a.float(), a.float().mT, mac64))
    _silent(lambda: Quant(BinaryK(16, 8), carrier=F64)(a.float()))


def test_binary64_holds_every_format_binary32_holds():
    """No format is clean in binary32 and flagged in binary64.

    The spec builders ask binary64 only about formats binary32 found something
    in (``_format_findings``), which is sound only if binary64 is wider at
    every edge. Swept over widths, biases either side of both carriers' edges
    and every mode that moves an edge.
    """
    b32 = b64 = 0
    for K, P in itertools.product(range(1, 36), range(1, 28)):
        if P > K:
            continue
        for bias, signed, sat, sub in itertools.product(
            (-130, -1, 0, 1, 7, 100, 124, 126, 127, 128, 200, 1000, 1022, 1024),
            (True, False),
            (SaturationMode.OVF_INF, SaturationMode.SAT_FINITE),
            list(SubnormalsMode),
        ):
            for prng_bits in (0, 20):
                args = (K, P, bias, signed, sat, sub, prng_bits)
                if _binaryK_findings(*args, F32) == (None, None):
                    b32 += 1
                    b64 += _binaryK_findings(*args, F64) == (None, None)
    for man_bits, exp_bits, nb, bias in itertools.product(
        range(0, 24), range(1, 11), (1, 2, 7, 30, 255), (-130, 0, 7, 22, 127, 500)
    ):
        for sat, prng_bits in itertools.product(
            (SaturationMode.OVF_INF, SaturationMode.SAT_FINITE), (0, 12)
        ):
            args = (man_bits, exp_bits, nb, bias, sat, prng_bits)
            if _superfp_findings(*args, F32) == (None, None):
                b32 += 1
                b64 += _superfp_findings(*args, F64) == (None, None)
    assert b32 > 1000
    assert b64 == b32


# --- the plain-integer wrappers reach the same rule ---------------------------
#
# The wrappers in `mptorch.quant.ops` take a format as loose integers and never
# build a `BinaryK`/`SuperFP`, so without their own checks they would quantize
# onto a grid they cannot reach and say nothing. `BinaryK(16, 8)` and
# `SuperFP(3, 4, 1, 22)` are the two formats used below: the first reaches
# 2**-134, the second 2**-126, both inside binary32 at the top and inside
# binary64 everywhere.


@pytest.mark.parametrize(
    "build",
    [
        # the multiply slot, the accumulate slot, and the fused one
        lambda: _binaryK_spec(mul_K=16, mul_P=8),
        lambda: _binaryK_spec(mul_K=8, mul_P=4, acc_K=16, acc_P=8),
        lambda: _binaryK_fma_spec(fma_K=16, fma_P=8),
        # per palette entry, not just the first
        lambda: _binaryK_mixed_spec(mul_K=[8, 16], mul_P=[4, 8]),
        lambda: _binaryK_mixed_spec(mul_K=[8, 8], mul_P=[4, 4], acc_K=[8, 16], acc_P=[4, 8]),
        lambda: _binaryK_fma_mixed_spec(fma_K=[8, 16], fma_P=[4, 8]),
        lambda: _superfp_spec(mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=22),
        lambda: _superfp_spec(
            mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=7, acc_bias=22
        ),
        lambda: _superfp_fma_spec(
            fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=1, fma_bias=22
        ),
        lambda: _superfp_mixed_spec(
            mul_man_bits=[3, 3],
            mul_exp_bits=[4, 4],
            mul_normal_binades=[1, 1],
            mul_bias=[7, 22],
        ),
        lambda: _superfp_fma_mixed_spec(
            fma_man_bits=[3, 3],
            fma_exp_bits=[4, 4],
            fma_normal_binades=[1, 1],
            fma_bias=[7, 22],
        ),
    ],
)
def test_every_spec_builder_finds_every_slot(build):
    """Each builder checks every format slot against both carriers, silently.

    Each case hides one out-of-range format in a different slot (multiply,
    accumulate, fused, a later palette entry). ``findings`` is a (binary32,
    binary64) pair, each a deduplicated tuple of ``(error, warning)`` entries,
    reported only when a call picks a carrier.
    """
    spec = _silent(build)
    (b32, b64) = spec.findings
    assert len(b32) == 1 and b32[0][0] is None and "smallest value" in str(b32[0][1])
    assert b64 == ()


def test_a_spec_builder_raises_what_no_carrier_can_do():
    """A builder raises what ``BinaryK(60, 54)`` raises when built.

    It takes plain integers and never constructs the format object. A
    ``carrier`` that is not ``torch.float32`` or ``torch.float64`` raises too.
    """
    with pytest.raises(ValueError, match="54 bits of precision"):
        _binaryK_spec(mul_K=8, mul_P=4, acc_K=60, acc_P=54)
    with pytest.raises(ValueError, match="54 bits of precision"):
        binaryK_gemm_formats(60, 54)
    with pytest.raises(TypeError, match="carrier must be a torch.dtype"):
        _binaryK_spec(mul_K=8, mul_P=4, carrier="binary64")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="carrier must be torch.float32"):
        _binaryK_spec(mul_K=8, mul_P=4, carrier=torch.float16)


def test_quantize_wrappers_check_their_format():
    """The two elementwise wrappers warn and raise per call, from plain integers."""
    x = torch.zeros(4)
    with pytest.warns(FormatRangeWarning, match=re.escape("2^-134")):
        binaryK_quantize(x, 16, 8)
    with pytest.warns(FormatRangeWarning, match=re.escape("2^-126")):
        superfp_quantize(x, 3, 4, 1, 22)
    with pytest.raises(ValueError, match="no finite normal value"):
        binaryK_quantize(x, 8, 4, bias=200)


def test_matmul_wrapper_checks_its_format():
    """A flat GEMM wrapper reports the verdict of the carrier its operands pick."""
    a, b = torch.randn(2, 3), torch.randn(3, 2)
    with pytest.warns(FormatRangeWarning, match=re.escape("2^-134")):
        binaryK_matmul(a, b, mul_K=16, mul_P=8)
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_matmul(a, b, mul_K=40, mul_P=30, mul_bias=512)
    _silent(lambda: binaryK_matmul(a.double(), b.double(), mul_K=40, mul_P=30, mul_bias=512))


def test_a_format_object_is_checked_once_per_call_at_the_callers_line():
    """A format object is reported once, at the call, naming the caller's line.

    Neither building it nor resolving its mac knows the carrier. The warning
    is raised deep inside mptorch, and ``skip_file_prefixes`` attributes it to
    the first frame outside mptorch and torch.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        fmt = BinaryK(16, 8)
        spec_for_mac(SplitMac(fmt, fmt))
        assert caught == []
        qmatmul(torch.ones(2, 3), torch.ones(3, 2), SplitMac(fmt, fmt))
    # both halves round in the one format, and it is reported once
    assert len(caught) == 1
    assert caught[0].filename == __file__


# --- what a tensor narrower than its carrier can store --------------------------
#
# A half-width tensor rounds in binary32, its carrier, and the result is
# rounded again when it is written. Under binary64 a float32 one is too. Only
# the call knows the dtype, so the check runs per call, against the format
# whose values the result holds, and by one of two rules, because the two ops
# reach different parts of that format. float16's bounds are 11 bits of
# precision, a top of 65504 and a finest step of 2**-24, and bfloat16's
# precision is 8 bits. `dev/benchmarks/format_limits.py sweep --dtype float16
# --audit` measured every boundary below through both ops: a fused GEMM over
# operands of the dtype, and the elementwise quantizer over every value of it.

F16, BF16 = torch.float16, torch.bfloat16


def _fma(dtype, **fmt):
    """Return a thunk running a fused GEMM over all-ones ``dtype`` operands.

    The result holds ``fmt``'s values. superfp keywords (``fma_man_bits``)
    select ``superfp_matmul_fma``, anything else ``binaryK_matmul_fma``.
    """
    a, b = torch.ones(2, 3, dtype=dtype), torch.ones(3, 2, dtype=dtype)
    if "fma_man_bits" in fmt:
        return lambda: superfp_matmul_fma(a, b, **fmt)
    return lambda: binaryK_matmul_fma(a, b, **fmt)


def _quant(dtype, *fmt, **kw):
    """Return a thunk quantizing a ``dtype`` tensor elementwise.

    Four positional format arguments select ``superfp_quantize``, two select
    ``binaryK_quantize``.
    """
    x = torch.ones(4, dtype=dtype)
    if len(fmt) == 4:
        return lambda: superfp_quantize(x, *fmt, **kw)
    return lambda: binaryK_quantize(x, *fmt, **kw)


EXT: dict[str, Any] = dict(subnormals_mode=SubnormalsMode.EXTENDED_NORMALS)
NRM: dict[str, Any] = dict(subnormals_mode=SubnormalsMode.NORMALS)
SATF: dict[str, Any] = dict(saturation_mode=SaturationMode.SAT_FINITE)


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        # the top: a result above 65504 is stored as infinity, and this 5-bit
        # exponent field stays under it only for bias >= 16
        (
            _fma(F16, fma_K=8, fma_P=3, fma_bias=16),
            _fma(F16, fma_K=8, fma_P=3, fma_bias=15),
            "above float16",
        ),
        # the finest step float16 stores, 2**-24, in each subnormals mode and
        # for superfp, followed by superfp's top
        (
            _fma(F16, fma_K=8, fma_P=4, fma_bias=22),
            _fma(F16, fma_K=8, fma_P=4, fma_bias=23),
            "apart at the bottom",
        ),
        (
            _fma(F16, fma_K=8, fma_P=4, fma_bias=22, **NRM),
            _fma(F16, fma_K=8, fma_P=4, fma_bias=23, **NRM),
            "apart at the bottom",
        ),
        (
            _fma(F16, fma_K=8, fma_P=4, fma_bias=21, **EXT),
            _fma(F16, fma_K=8, fma_P=4, fma_bias=22, **EXT),
            "apart at the bottom",
        ),
        (
            _fma(F16, fma_man_bits=2, fma_exp_bits=3, fma_normal_binades=4, fma_bias=13),
            _fma(F16, fma_man_bits=2, fma_exp_bits=3, fma_normal_binades=4, fma_bias=14),
            "apart at the bottom",
        ),
        (
            _fma(F16, fma_man_bits=2, fma_exp_bits=3, fma_normal_binades=4, fma_bias=-8),
            _fma(F16, fma_man_bits=2, fma_exp_bits=3, fma_normal_binades=4, fma_bias=-9),
            "above float16",
        ),
    ],
)
def test_gemm_storage_range_boundaries_warn(ok, bad, match):
    """A GEMM whose last format tops or undercuts the dtype's range warns.

    Its sums land anywhere on that format's grid, so every value of the format
    is a possible result, stored or not.
    """
    _silent(ok)
    with pytest.warns(FormatRangeWarning, match=match):
        bad()


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        (
            _fma(F16, fma_K=15, fma_P=11, fma_bias=8),
            _fma(F16, fma_K=16, fma_P=12, fma_bias=8),
            "a float16 result holds 11",
        ),
        (
            _fma(BF16, fma_K=12, fma_P=8, fma_bias=8),
            _fma(BF16, fma_K=13, fma_P=9, fma_bias=8),
            "a bfloat16 result holds 8",
        ),
        # nothing of the format is stored: its largest value is under 2**-24,
        # float16's smallest
        (
            _fma(F16, fma_K=8, fma_P=4, fma_bias=22),
            _fma(F16, fma_K=8, fma_P=4, fma_bias=60),
            "none of the format.s nonzero",
        ),
    ],
)
def test_gemm_storage_precision_boundaries_raise(ok, bad, match):
    """One bit past the dtype's precision (float16 11, bfloat16 8) raises.

    Storing would round every sum a second time, so the format does not
    survive. A format the dtype stores no nonzero value of raises as well.
    """
    _silent(ok)
    with pytest.raises(ValueError, match=match):
        bad()


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        # Edge 1, the top: 65504 is off the grid of a format that reaches past
        # it, and rounds up out of float16's range.
        (_quant(F16, 8, 3, bias=16), _quant(F16, 8, 3, bias=15), "rounds up out of float16"),
        (_quant(F16, 2, 3, 4, -8), _quant(F16, 2, 3, 4, -9), "rounds up out of float16"),
        # Edge 2, EXTENDED_NORMALS' hole: 2**-bias is an input float16 has and
        # the format does not, and the value above it, (1 + 2**-3) * 2**-bias,
        # is float16's only while its 2**(-bias - 3) step is, which is up to
        # bias 21. From bias 25 float16 no longer has the hole either.
        (
            _quant(F16, 8, 4, bias=21, **EXT),
            _quant(F16, 8, 4, bias=22, **EXT),
            "no value at 2\\^-22",
        ),
        (
            _quant(F16, 8, 4, bias=25, **EXT),
            _quant(F16, 8, 4, bias=24, **EXT),
            "no value at 2\\^-24",
        ),
        # Edge 3: a largest value finer than float16's grid, which an input
        # reaches only by saturating onto it.
        (_quant(F16, 16, 12, bias=8), _quant(F16, 16, 12, bias=8, **SATF), "saturates onto it"),
        (
            _quant(F16, 15, 11, bias=8, **SATF),
            _quant(F16, 16, 12, bias=8, **SATF),
            "saturates onto it",
        ),
    ],
)
def test_quantizer_storage_edges_warn(ok, bad, match):
    """The three range edges where a quantizer leaves the dtype's grid warn.

    An elementwise quantizer's inputs are the dtype's values already, and a
    format at least as fine hands them back unchanged, so an off-grid result
    can only come from an edge of the format's range. The audit measured each
    ``ok`` clean and each ``bad`` wrong.
    """
    _silent(ok)
    with pytest.warns(FormatRangeWarning, match=match):
        bad()


@pytest.mark.parametrize(
    ("quantize", "gemm"),
    [
        # more precision than float16: the quantizer hands its input back
        (_quant(F16, 16, 12, bias=8), _fma(F16, fma_K=16, fma_P=12, fma_bias=8)),
        (_quant(BF16, 13, 9, bias=8), _fma(BF16, fma_K=13, fma_P=9, fma_bias=8)),
        # a bottom finer than float16's, which no float16 input rounds into
        (_quant(F16, 8, 4, bias=23), _fma(F16, fma_K=8, fma_P=4, fma_bias=23)),
        (_quant(F16, 16, 11), _fma(F16, fma_K=16, fma_P=11)),
        (
            _quant(F16, 3, 4, 8, 7),
            _fma(F16, fma_man_bits=3, fma_exp_bits=4, fma_normal_binades=8, fma_bias=7),
        ),
        # a top past 65504 whose grid holds 65504 itself
        (_quant(F16, 16, 11, bias=15), _fma(F16, fma_K=16, fma_P=11, fma_bias=15)),
    ],
)
def test_what_a_gemm_reaches_a_quantizer_does_not(quantize, gemm):
    """Formats a quantizer accepts silently and a GEMM warns or raises about.

    The two rules differ exactly where the audit found the two ops differ. One
    rule shared by both ops fails one half of every pair.
    """
    _silent(quantize)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        try:
            gemm()
        except ValueError:
            return
    assert caught, "the same format as a GEMM's last rounding should not be silent"


def test_the_storage_rule_is_what_the_kernel_does():
    """One kernel run per rule, so a warning's text is not the only evidence."""
    # BinaryK(8, 3, bias=15) is E5M2's layout, but P3109 reserves only the top
    # code rather than the top binade, so it reaches 98304. 60000 rounds up to
    # 65536 on its grid, and float16 stores that as infinity even though the
    # format saturates.
    x = torch.tensor([60000.0], dtype=F16)
    with pytest.warns(FormatRangeWarning):
        y = binaryK_quantize(x, 8, 3, bias=15, rounding_mode=RoundMode.RU, **SATF)
    assert torch.isinf(y).all()
    assert binaryK_quantize(x.float(), 8, 3, bias=15, rounding_mode=RoundMode.RU, **SATF) == 65536.0
    # the quantizer returns float16's own values for a format finer than it...
    x = torch.tensor([1.0 + 2.0**-10, 3.0 * 2.0**-12, 255.875], dtype=F16)
    assert torch.equal(_silent(lambda: binaryK_quantize(x, 16, 12, bias=8)), x)
    # ...and a GEMM, whose products are not float16's, does not
    a = torch.tensor([[1.0 + 2.0**-10]], dtype=F16)
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_matmul_fma(a, a, fma_K=16, fma_P=12, fma_bias=8)


def test_only_the_last_rounding_of_a_gemm_is_stored():
    """Only the format a GEMM rounds last is checked against the dtype.

    The multiply format's products are intermediates in the carrier, and the
    accumulate (or fused) rounding is what the result holds.
    """
    a, b = torch.ones(2, 3, dtype=F16), torch.ones(3, 2, dtype=F16)
    # a multiply format float16 could not store, under an accumulate it can
    _silent(lambda: binaryK_matmul(a, b, mul_K=20, mul_P=16, mul_bias=8, acc_K=8, acc_P=4))
    with pytest.raises(ValueError, match="float16 result holds 11"):
        binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=12, acc_bias=8)
    # with the last rounding off, no format's values are stored and there is
    # nothing to check against the dtype
    _silent(lambda: binaryK_matmul(a, b, mul_K=16, mul_P=12, mul_bias=8, accumulate_quant=False))
    _silent(lambda: binaryK_matmul_fma(a, b, fma_K=16, fma_P=12, fma_bias=8, fma_quant=False))


def test_every_palette_entry_is_held_against_the_dtype():
    """A mixed GEMM checks each palette entry, not just the first, for storage."""
    a, b = torch.ones(2, 3, dtype=F16), torch.ones(3, 2, dtype=F16)
    prec_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    _silent(lambda: binaryK_matmul_mixed(a, b, prec_idx, mul_K=[8, 8], mul_P=[4, 3]))
    with pytest.raises(ValueError, match="float16 result holds 11"):
        binaryK_matmul_mixed(a, b, prec_idx, mul_K=[8, 16], mul_P=[4, 12], mul_bias=[8, 8])


def test_float32_and_float64_results_are_not_held_against_a_dtype():
    """A result as wide as its carrier stores every value, so nothing is checked."""
    for dtype in (torch.float32, torch.float64):
        _silent(_quant(dtype, 16, 12, bias=8, **SATF))
        _silent(_quant(dtype, 8, 3, bias=15))
    _silent(_fma(torch.float32, fma_K=16, fma_P=12, fma_bias=8))
    _silent(_fma(torch.float64, fma_K=16, fma_P=12, fma_bias=8, carrier=F64))


# The same two rules apply to a float32 result under binary64, at float32's
# bounds: 24 bits of precision, 2**128 and a 2**-149 step for a GEMM, and the
# edges of the range for the quantizer. binary64 carries every format below.
C64: dict[str, Any] = dict(carrier=F64)


@pytest.mark.parametrize(
    ("ok", "bad", "match", "error"),
    [
        (
            _fma(F32, fma_K=31, fma_P=24, fma_bias=8, **C64),
            _fma(F32, fma_K=32, fma_P=25, fma_bias=8, **C64),
            "a float32 result holds 24",
            True,
        ),
        (
            _fma(F32, fma_K=16, fma_P=8, fma_bias=128, **C64),
            _fma(F32, fma_K=16, fma_P=8, fma_bias=127, **C64),
            "above float32",
            False,
        ),
        (
            _fma(F32, fma_K=12, fma_P=4, fma_bias=147, **C64),
            _fma(F32, fma_K=12, fma_P=4, fma_bias=148, **C64),
            "apart at the bottom",
            False,
        ),
        (
            _quant(F32, 12, 4, bias=128, **C64),
            _quant(F32, 12, 4, bias=127, **C64),
            "rounds up out of float32",
            False,
        ),
        (
            _quant(F32, 40, 30, bias=897, **C64),
            _quant(F32, 40, 30, bias=897, **C64, **SATF),
            "saturates onto it",
            False,
        ),
        # float16's bounds hold for a float16 result whichever carrier made it
        (
            _fma(F16, fma_K=15, fma_P=11, fma_bias=8, **C64),
            _fma(F16, fma_K=16, fma_P=12, fma_bias=8, **C64),
            "a float16 result holds 11",
            True,
        ),
        (
            _quant(F16, 8, 3, bias=16, **C64),
            _quant(F16, 8, 3, bias=15, **C64),
            "rounds up out of float16",
            False,
        ),
    ],
)
def test_a_binary64_result_is_held_against_the_narrower_dtype(ok, bad, match, error):
    """The storage rule reads the result's dtype, not the carrier that rounded it."""
    _silent(ok)
    if error:
        with pytest.raises(ValueError, match=match):
            bad()
    else:
        with pytest.warns(FormatRangeWarning, match=match):
            bad()


def test_the_binary64_storage_rule_is_what_the_call_does():
    """The kernel run behind the float32 top-edge warning.

    float32's largest value is off a 4-bit grid that reaches past it, so
    rounding it up in binary64 lands on 2**128, which float32 stores as
    infinity.
    """
    x = torch.tensor([3.4028234663852886e38], dtype=F32)
    with pytest.warns(FormatRangeWarning, match="rounds up out of float32"):
        y = binaryK_quantize(x, 12, 4, bias=127, rounding_mode=RoundMode.RU, carrier=F64)
    assert y.dtype is F32 and torch.isinf(y).all()
    assert binaryK_quantize(x.double(), 12, 4, bias=127, rounding_mode=RoundMode.RU) == 2.0**128


def test_a_format_object_is_held_against_the_dtype_at_the_call():
    """The mac tier applies the storage rule per call as well.

    A ``BinaryK`` does not know the dtype its values will be stored in.
    """
    mac = SplitMac(BinaryK(8, 4), BinaryK(8, 3, bias=15))
    a = torch.ones(2, 3, dtype=F16)
    _silent(lambda: qmatmul(a.float(), a.float().mT, mac))
    with pytest.warns(FormatRangeWarning, match="above float16"):
        qmatmul(a, a.mT, mac)
    with pytest.raises(ValueError, match="float16 result holds 11"):
        qmatmul(a, a.mT, FusedMac(BinaryK(16, 12, bias=8)))


def test_stochastic_bits_are_bounded_by_the_carrier_not_the_dtype():
    """``man_bits + prng_bits`` is bounded by the carrier's significand field.

    The random bits are drawn against the carrier value the kernel rounds, not
    against the dtype the tensor stores.
    """
    x = torch.rand(64, dtype=BF16)
    sr: dict[str, Any] = dict(rounding_mode=RoundMode.SR)
    # man_bits 3 + 20 == binary32's 23, far past bfloat16's 7
    _silent(lambda: binaryK_quantize(x, 8, 4, prng_bits=20, **sr))
    _silent(
        lambda: binaryK_matmul_fma(
            x[:8].reshape(2, 4), x[:8].reshape(4, 2), fma_K=8, fma_P=4, fma_prng_bits=20, **sr
        )
    )
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_quantize(x, 8, 4, prng_bits=21, **sr)
    with pytest.raises(ValueError, match="cannot be negative"):
        binaryK_quantize(x, 8, 4, prng_bits=-1, **sr)
    # binary64 has 52 bits to share, on a float64 tensor or a narrower one
    # that names it
    x64 = x.double()
    _silent(lambda: binaryK_quantize(x64, 8, 4, prng_bits=49, **sr))
    _silent(lambda: binaryK_quantize(x, 8, 4, prng_bits=49, carrier=F64, **sr))
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_quantize(x64, 8, 4, prng_bits=50, **sr)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_quantize(x, 8, 4, prng_bits=50, carrier=F64, **sr)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_matmul_fma(
            x64[:8].reshape(2, 4), x64[:8].reshape(4, 2), fma_K=8, fma_P=4, fma_prng_bits=50, **sr
        )


def test_a_storage_warning_names_the_callers_line():
    """A storage warning names the caller's line at whatever depth it is raised.

    A wrapper, a layer hook and a backward pass reach the check at different
    stack depths, so a fixed ``stacklevel`` would point into mptorch or torch.
    ``skip_file_prefixes`` names the first frame outside both trees.
    """
    lin = QLinear(
        3, 3, formats=binaryK_gemm_formats(mul_K=8, mul_P=4, acc_K=8, acc_P=3, acc_bias=15)
    )
    lin = lin.to(F16)
    x = torch.ones(2, 3, dtype=F16, requires_grad=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        binaryK_quantize(x.detach(), 8, 3, bias=15)
        y = lin(x)
        y.sum().backward()
    assert len(caught) >= 4  # the quantizer, the forward, and both gradients
    assert {c.filename for c in caught} == {__file__}
