"""What each carrier can hold, and what float16 and bfloat16 can store (T4, T6).

The casts round a value of their carrier onto the format's grid and return a
value of it -- binary64 for a float64 tensor, binary32 for float32, float16 and
bfloat16 -- so a format with more range or precision than its carrier is
quantized *partially*: a tensor of plausible numbers, some of the format's
values never among them. These checks are what stops that being invisible, and
they come in two strengths: a format that cannot function raises, and one that
works over part of its range warns.

Which carrier a format meets is the call's to know, so the checks run per
call, against the carrier the call rounds in. Building a ``BinaryK`` or a
``SuperFP`` raises only for what neither carrier can do, and warns about
nothing.

A float16 or bfloat16 result is rounded once more when it is stored, so the
format is held against that dtype too, per call, by one of two rules: a
GEMM's result can be any value of its last format, and an elementwise
quantizer's can only be the format's answer to a value of the dtype.

The boundaries asserted here are the ones
``dev/benchmarks/format_limits.py sweep --audit`` (and ``sweep --dtype float16
--audit``, ``--dtype bfloat16``) measured against the kernels, so a change
that moves a bound should move a measurement first.
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
    """Run a call with `FormatRangeWarning` promoted to an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


def _in(dtype, build):
    """Build a format and round a tensor of ``dtype`` to it: the call that
    holds the format to that dtype's carrier."""
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
        # exactly on the floor those two reach, 2**-126
        lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.NORMALS),
        lambda: BinaryK(8, 4, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        # and on the two exceptions, a binade higher at 2**-125
        lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.NORMALS),
        lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        lambda: BinaryK(31, 24, bias=125, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        lambda: BinaryK(8, 4, prng_bits=20),  # 3 + 20 == binary32's 23
        lambda: SuperFP(3, 4, 1, 7),
        lambda: SuperFP(3, 4, 2, 7),
        lambda: SuperFP(3, 4, 8, 7),
        lambda: SuperFP(2, 4, 8, 7),
        lambda: SuperFP(3, 4, 15, 7),  # one supernormal binade left
        lambda: SuperFP(3, 4, 2, 7, prng_bits=8),
    ],
)
def test_formats_in_range_are_silent(build, dtype):
    """Every format the library, its docs and its tests use is inside binary32's
    bound, and so inside binary64's, which is wider at every edge."""
    _silent(_in(dtype, build))


@pytest.mark.parametrize(
    ("build", "match"),
    [
        # precision: binary32 has 24 bits and cannot hold a finer grid, so
        # nothing of such a format survives
        (lambda: BinaryK(29, 25, bias=8), "bits of precision"),
        # no finite normal value binary32 can hold: max_num is zero, so every
        # nonzero result overflows and only 0 and the infinities come back
        (lambda: BinaryK(8, 4, bias=200), "no finite normal value"),
        (lambda: SuperFP(3, 4, 1, 200), "no finite normal value"),
        # the random bits share binary32's significand with the mantissa
        (lambda: BinaryK(8, 4, prng_bits=21), "stochastic-rounding bits"),
        (lambda: SuperFP(3, 4, 2, 7, prng_bits=21), "stochastic-rounding bits"),
    ],
)
def test_formats_that_cannot_function_in_binary32_raise_at_the_call(build, match):
    """The error half: a format no part of which quantizes -- in binary32. Each
    is whole in binary64, so building it says nothing, a float64 call is
    silent, and the float32 call raises and points at float64."""
    fmt = _silent(build)
    with pytest.raises(ValueError, match=match) as raised:
        Quant(fmt)(torch.ones(4))
    assert "binary64, holds all of it" in str(raised.value)
    _silent(lambda: Quant(fmt)(torch.ones(4, dtype=F64)))


def test_a_superfp_past_binary32s_precision_raises_there_and_warns_in_binary64():
    """No hint here: a 24-bit mantissa spreads superfp's supernormals over 2**24
    binades a code, which outruns binary64's bottom as well."""
    fmt = _silent(lambda: SuperFP(24, 4, 1, 7))
    with pytest.raises(ValueError, match="bits of precision") as raised:
        Quant(fmt)(torch.ones(4))
    assert "binary64" not in str(raised.value)
    with pytest.warns(FormatRangeWarning, match="below binary64's 2\\^-1074"):
        Quant(fmt)(torch.ones(4, dtype=F64))


@pytest.mark.parametrize(
    ("build", "match"),
    [
        # precision: binary64 has 53 bits
        (lambda: BinaryK(60, 54), "54 bits of precision.*binary64, which has 53"),
        (lambda: SuperFP(53, 4, 1, 7), "54 bits of precision"),
        # no finite value binary64 holds
        (lambda: BinaryK(8, 4, bias=2000), "no finite normal value binary64"),
        # the exponent field the kernels shift a 32-bit int by
        (lambda: BinaryK(40, 1), "exponent bits"),
        # the regions would be misordered and the lowest normal binade lost
        (lambda: SuperFP(3, 4, 16, 7), "no supernormal codes"),
        # the random bits share binary64's significand with the mantissa
        (lambda: BinaryK(8, 4, prng_bits=50), "stochastic-rounding bits.*52 bits"),
        (lambda: BinaryK(8, 4, prng_bits=-1), "non-negative"),
    ],
)
def test_what_no_carrier_can_do_raises_when_the_format_is_built(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_building_a_format_never_warns():
    """Neither for a range that outruns binary32 nor for one that outruns both:
    which carrier a format meets is not known until it is used."""
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
        # the top: above binary32's largest finite value the codes are
        # unreachable, but everything below still quantizes -- a wide binaryK
        # used as a precision-only target is exactly this
        (lambda: BinaryK(8, 4, bias=-113), "above binary32"),
        (lambda: BinaryK(16, 5), "above binary32"),
        (lambda: BinaryK(16, 8, is_signed=False), "above binary32"),
        (lambda: BinaryK(24, 8), "above binary32"),  # tests/ uses this one
        (lambda: SuperFP(2, 3, 1, -121), "above binary32"),
        # the bottom: a spacing finer than 2**-149 is not binary32 at all
        (lambda: BinaryK(11, 4, bias=200), "apart at the bottom"),
        (lambda: SuperFP(4, 4, 2, 7), "apart at the bottom"),
    ],
)
def test_formats_whose_range_outruns_binary32_warn(build, match):
    """The warning half: quantizes over the part binary32 holds, partially."""
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
    """Held by binary32, but the casts cannot place an input that far down.

    Every cast classifies its input by the exponent field of the binary32
    word, and all binary32 subnormals share field 0 -- so a format reaching
    below 2**-126 is rounded on a grid one binade coarse (binaryK) or onto
    powers of two the cast cannot name (superfp). binary64 places all of
    these, and a float64 tensor is silent.
    """
    with pytest.warns(FormatRangeWarning, match=re.escape(smallest)):
        _in(F32, build)()
    _silent(_in(F64, build))


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        # the bottom, at the exact bias the audit found: bias + P <= 127
        (lambda: BinaryK(8, 4, bias=123), lambda: BinaryK(8, 4, bias=124)),
        # NORMALS and EXTENDED_NORMALS decide their bottom by comparing
        # magnitudes rather than by reading an exponent field, so they are
        # exact one binade lower -- down to a floor of 2**-126 (T5)
        (
            lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.NORMALS),
            lambda: BinaryK(8, 4, bias=128, subnormals=SubnormalsMode.NORMALS),
        ),
        (
            lambda: BinaryK(8, 4, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        ),
        # -- except where half that floor is out of binary32's reach, which
        # stops two of their formats a binade short. At P = 1 the round reads
        # 2**-127 as a tie between binades and carries it to the floor ...
        (
            lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.NORMALS),
            lambda: BinaryK(8, 1, bias=127, subnormals=SubnormalsMode.NORMALS),
        ),
        (
            lambda: BinaryK(8, 1, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            lambda: BinaryK(8, 1, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        ),
        # ... and at P = 24 EXTENDED_NORMALS' half-floor needs a 2**-150 step
        (
            lambda: BinaryK(31, 24, bias=125, subnormals=SubnormalsMode.EXTENDED_NORMALS),
            lambda: BinaryK(31, 24, bias=126, subnormals=SubnormalsMode.EXTENDED_NORMALS),
        ),
        # which NORMALS' power-of-two floor never does
        (
            lambda: BinaryK(31, 24, bias=127, subnormals=SubnormalsMode.NORMALS),
            lambda: BinaryK(31, 24, bias=128, subnormals=SubnormalsMode.NORMALS),
        ),
        # superfp's supernormal floor, likewise
        (lambda: SuperFP(3, 4, 1, 21), lambda: SuperFP(3, 4, 1, 22)),
    ],
)
def test_bottom_boundary_is_where_it_was_measured(ok, bad):
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
        # below 2**-126 the floor itself is out of reach, and that is what is said
        (lambda: BinaryK(8, 1, bias=128, subnormals=SubnormalsMode.NORMALS), "exponent field"),
    ],
)
def test_bottom_warning_names_the_reason(build, match):
    """The two half-floor limits are not the exponent-field one, and say so."""
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
    _silent(_in(F32, ok))
    with pytest.warns(FormatRangeWarning, match="above binary32"):
        _in(F32, bad)()


# --- binary64's edges, where `sweep --carrier binary64 --audit` measured them --
#
# The same derivation one carrier wider, rounded in by a float64 tensor. Every
# `ok` below is a format whose float64 audit found no wrong answer and no lost
# value, and every `bad` one whose audit found both, a parameter step away.

EXT64 = SubnormalsMode.EXTENDED_NORMALS
NRM64 = SubnormalsMode.NORMALS


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        # SUBNORMALS: bias + P <= 1023
        (lambda: BinaryK(14, 4, bias=1019), lambda: BinaryK(14, 4, bias=1020)),
        # the magnitude compares: bias <= 1023 under NORMALS, 1022 under EXTENDED_NORMALS
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
        # needs a 2**-1075 step
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
        # ten exponent bits at P3109's bias, not eleven
        (lambda: BinaryK(14, 4), lambda: BinaryK(15, 4)),
    ],
)
def test_binary64_bottom_boundary_is_where_it_was_measured(ok, bad):
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
    with pytest.warns(FormatRangeWarning, match=match):
        _in(F64, build)()


def test_binary64_precision_and_stochastic_bits_are_where_they_were_measured():
    """P = 53 and ``man_bits + prng_bits = 52`` are the last the kernel rounds
    faithfully; one past either raises when the format is built."""
    x = torch.ones(4, dtype=F64)
    _silent(lambda: Quant(BinaryK(63, 53, bias=512))(x))
    _silent(lambda: Quant(BinaryK(14, 4, bias=512, prng_bits=49), RoundMode.SR)(x))
    with pytest.raises(ValueError, match="54 bits of precision"):
        BinaryK(64, 54, bias=512)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        BinaryK(14, 4, bias=512, prng_bits=50)


def test_warning_can_be_filtered():
    """The warning names a category so a caller who means it can silence it."""
    fmt = BinaryK(16, 8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", FormatRangeWarning)
        out = Quant(fmt)(torch.ones(4))
    assert caught == []
    assert torch.equal(out, torch.ones(4))


@pytest.mark.parametrize("dtype", [F32, torch.float16, torch.bfloat16])
def test_precision_is_bounded_by_the_carrier_not_by_storage(dtype):
    """A float64 operand rounds in binary64, and every other dtype in binary32.

    The format below holds ``1 + 2**-30`` exactly and binary32 does not; before
    binary64 was a carrier the float64 call was refused too.
    """
    x = torch.tensor([1.0 + 2.0**-30], dtype=F64)
    assert _silent(lambda: binaryK_quantize(x, 50, 41, bias=256)).item() == 1.0 + 2.0**-30
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_quantize(x.to(dtype), 50, 41, bias=256)
    # carrier="binary32" is binary32's arithmetic, and binary32's bounds
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_quantize(x, 50, 41, bias=256, carrier="binary32")
    with pytest.raises(ValueError, match="bits of precision"):
        superfp_quantize(x.to(dtype), 30, 4, 1, 7)
    # 24 bits of precision is the most binary32 can express, and it is allowed
    assert binaryK_quantize(x.to(dtype), 30, 24, bias=8).dtype == dtype


def test_the_carrier_a_call_rounds_in_is_the_one_it_checks():
    """One format, both verdicts: the operand dtype and `carrier=` pick which."""
    mac = SplitMac(BinaryK(16, 8), BinaryK(16, 8))
    mac32 = SplitMac(BinaryK(16, 8), BinaryK(16, 8), carrier="binary32")
    a = torch.ones(2, 3, dtype=F64)
    _silent(lambda: qmatmul(a, a.mT, mac))
    with pytest.warns(FormatRangeWarning, match="2\\^-134"):
        qmatmul(a, a.mT, mac32)
    with pytest.warns(FormatRangeWarning, match="2\\^-134"):
        qmatmul(a.float(), a.float().mT, mac)
    with pytest.warns(FormatRangeWarning, match="2\\^-134"):
        Quant(BinaryK(16, 8), carrier="binary32")(a)


def test_binary64_holds_every_format_binary32_holds():
    """The spec builders ask binary64 only about formats binary32 has found
    something in (`_format_findings`), which is sound only if binary64 is wider
    at every edge. Swept over widths, biases either side of both carriers'
    edges and every mode that moves an edge."""
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
                if _binaryK_findings(*args, "binary32") == (None, None):
                    b32 += 1
                    b64 += _binaryK_findings(*args, "binary64") == (None, None)
    for man_bits, exp_bits, nb, bias in itertools.product(
        range(0, 24), range(1, 11), (1, 2, 7, 30, 255), (-130, 0, 7, 22, 127, 500)
    ):
        for sat, prng_bits in itertools.product(
            (SaturationMode.OVF_INF, SaturationMode.SAT_FINITE), (0, 12)
        ):
            args = (man_bits, exp_bits, nb, bias, sat, prng_bits)
            if _superfp_findings(*args, "binary32") == (None, None):
                b32 += 1
                b64 += _superfp_findings(*args, "binary64") == (None, None)
    assert b32 > 1000
    assert b64 == b32


# --- the plain-integer wrappers reach the same rule ---------------------------
#
# The wrappers in `mptorch.quant.ops` take a format as loose integers and never
# build a `BinaryK`/`SuperFP`, so without these they would quantize onto a grid
# they cannot reach and say nothing. `BinaryK(16, 8)` and `SuperFP(3, 4, 1, 22)`
# are the two formats used below: the first reaches 2**-134, the second
# 2**-126, both inside binary32 at the top and inside binary64 everywhere.


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
    """Each builder holds every format it rounds with to both carriers, once,
    and says nothing until a call picks one of them."""
    spec = _silent(build)
    (b32, b64) = spec.findings
    assert len(b32) == 1 and b32[0][0] is None and "smallest value" in str(b32[0][1])
    assert b64 == ()


def test_a_spec_builder_raises_what_no_carrier_can_do():
    """The plain-integer spelling of what `BinaryK(60, 54)` raises when built."""
    with pytest.raises(ValueError, match="54 bits of precision"):
        _binaryK_spec(mul_K=8, mul_P=4, acc_K=60, acc_P=54)
    with pytest.raises(ValueError, match="54 bits of precision"):
        binaryK_gemm_formats(60, 54)
    with pytest.raises(ValueError, match="carrier must be"):
        _binaryK_spec(mul_K=8, mul_P=4, carrier="float64")


def test_quantize_wrappers_check_their_format():
    x = torch.zeros(4)
    with pytest.warns(FormatRangeWarning, match=re.escape("2^-134")):
        binaryK_quantize(x, 16, 8)
    with pytest.warns(FormatRangeWarning, match=re.escape("2^-126")):
        superfp_quantize(x, 3, 4, 1, 22)
    with pytest.raises(ValueError, match="no finite normal value"):
        binaryK_quantize(x, 8, 4, bias=200)


def test_matmul_wrapper_checks_its_format():
    a, b = torch.randn(2, 3), torch.randn(3, 2)
    with pytest.warns(FormatRangeWarning, match=re.escape("2^-134")):
        binaryK_matmul(a, b, mul_K=16, mul_P=8)
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_matmul(a, b, mul_K=40, mul_P=30, mul_bias=512)
    _silent(lambda: binaryK_matmul(a.double(), b.double(), mul_K=40, mul_P=30, mul_bias=512))


def test_a_format_object_is_checked_once_per_call_at_the_callers_line():
    """Not when it is built, not when its mac is resolved: at the call, which
    is the first to know the carrier, naming the line that made it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        fmt = BinaryK(16, 8)
        spec_for_mac(SplitMac(fmt, fmt))
        assert caught == []
        qmatmul(torch.ones(2, 3), torch.ones(3, 2), SplitMac(fmt, fmt))
    # both halves round in the one format, and it is reported once
    assert len(caught) == 1
    assert caught[0].filename == __file__


# --- what float16 and bfloat16 can store (T6) ----------------------------------
#
# A half-width tensor rounds in binary32, its carrier, and the result is
# rounded again when it is written. The dtype is the call's, so the check runs per call, against the
# format whose values the result holds -- and by one of two rules, because the
# two ops reach different parts of that format. `sweep --dtype float16 --audit`
# measured every boundary below through both: a fused GEMM over operands of the
# dtype, and the elementwise quantizer over every value of it.

F16, BF16 = torch.float16, torch.bfloat16


def _fma(dtype, **fmt):
    """A fused GEMM whose result holds `fmt`'s values, over operands of `dtype`."""
    a, b = torch.ones(2, 3, dtype=dtype), torch.ones(3, 2, dtype=dtype)
    if "fma_man_bits" in fmt:
        return lambda: superfp_matmul_fma(a, b, **fmt)
    return lambda: binaryK_matmul_fma(a, b, **fmt)


def _quant(dtype, *fmt, **kw):
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
        # the top: a result above 65504 is stored as infinity (bias >= 16)
        (
            _fma(F16, fma_K=8, fma_P=3, fma_bias=16),
            _fma(F16, fma_K=8, fma_P=3, fma_bias=15),
            "above float16",
        ),
        # the finest step, 2**-24, in each subnormals mode
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
    """A GEMM's sums land anywhere, so every value of its last format is a result."""
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
        # nothing of the format is stored: its largest value is under 2**-24
        (
            _fma(F16, fma_K=8, fma_P=4, fma_bias=22),
            _fma(F16, fma_K=8, fma_P=4, fma_bias=60),
            "none of the format.s nonzero",
        ),
    ],
)
def test_gemm_storage_precision_boundaries_raise(ok, bad, match):
    _silent(ok)
    with pytest.raises(ValueError, match=match):
        bad()


@pytest.mark.parametrize(
    ("ok", "bad", "match"),
    [
        # the top, where it is also the value set's: 65504 is off the grid of
        # a format that reaches past it, and rounds up out of float16's range
        (_quant(F16, 8, 3, bias=16), _quant(F16, 8, 3, bias=15), "rounds up out of float16"),
        (_quant(F16, 2, 3, 4, -8), _quant(F16, 2, 3, 4, -9), "rounds up out of float16"),
        # EXTENDED_NORMALS' hole: 2**-bias is an input float16 has and the
        # format does not, and the value above it is float16's only while its
        # step is -- up to bias 21, and again from 25, where float16 no longer
        # has the hole either
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
        # a largest value finer than float16, reached only by saturating onto it
        (_quant(F16, 16, 12, bias=8), _quant(F16, 16, 12, bias=8, **SATF), "saturates onto it"),
        (
            _quant(F16, 15, 11, bias=8, **SATF),
            _quant(F16, 16, 12, bias=8, **SATF),
            "saturates onto it",
        ),
    ],
)
def test_quantizer_storage_edges_warn(ok, bad, match):
    """An elementwise quantizer's inputs are the dtype's values already.

    So what it can return off the dtype's grid is only at the edges of the
    format's range -- the audit measured every one of these clean on one side
    and wrong on the other.
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
    """The two rules differ exactly where the audit found the two ops differ."""
    _silent(quantize)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        try:
            gemm()
        except ValueError:
            return
    assert caught, "the same format as a GEMM's last rounding should not be silent"


def test_the_storage_rule_is_what_the_kernel_does():
    """One case per rule, straight from the kernel, so the text is not the evidence."""
    # E5M2 as this library spells it reaches 98304; 60000 rounds up to 65536,
    # and float16 stores that as infinity even though the format saturates
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
    """The multiply format's products are binary32 intermediates; the sum is the result."""
    a, b = torch.ones(2, 3, dtype=F16), torch.ones(3, 2, dtype=F16)
    # a multiply format float16 could not store, under an accumulate it can
    _silent(lambda: binaryK_matmul(a, b, mul_K=20, mul_P=16, mul_bias=8, acc_K=8, acc_P=4))
    with pytest.raises(ValueError, match="float16 result holds 11"):
        binaryK_matmul(a, b, mul_K=8, mul_P=4, acc_K=16, acc_P=12, acc_bias=8)
    # nothing rounded last, nothing to hold against the dtype
    _silent(lambda: binaryK_matmul(a, b, mul_K=16, mul_P=12, mul_bias=8, accumulate_quant=False))
    _silent(lambda: binaryK_matmul_fma(a, b, fma_K=16, fma_P=12, fma_bias=8, fma_quant=False))


def test_every_palette_entry_is_held_against_the_dtype():
    a, b = torch.ones(2, 3, dtype=F16), torch.ones(3, 2, dtype=F16)
    prec_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    _silent(lambda: binaryK_matmul_mixed(a, b, prec_idx, mul_K=[8, 8], mul_P=[4, 3]))
    with pytest.raises(ValueError, match="float16 result holds 11"):
        binaryK_matmul_mixed(a, b, prec_idx, mul_K=[8, 16], mul_P=[4, 12], mul_bias=[8, 8])


def test_float32_and_float64_results_are_not_held_against_a_dtype():
    """They store every value of their carrier, so there is nothing more to say."""
    for dtype in (torch.float32, torch.float64):
        _silent(_quant(dtype, 16, 12, bias=8, **SATF))
        _silent(_quant(dtype, 8, 3, bias=15))
    _silent(_fma(torch.float32, fma_K=16, fma_P=12, fma_bias=8))


def test_a_format_object_is_held_against_the_dtype_at_the_call():
    """A `BinaryK` does not know its dtype, so the mac tier checks per call too."""
    mac = SplitMac(BinaryK(8, 4), BinaryK(8, 3, bias=15))
    a = torch.ones(2, 3, dtype=F16)
    _silent(lambda: qmatmul(a.float(), a.float().mT, mac))
    with pytest.warns(FormatRangeWarning, match="above float16"):
        qmatmul(a, a.mT, mac)
    with pytest.raises(ValueError, match="float16 result holds 11"):
        qmatmul(a, a.mT, FusedMac(BinaryK(16, 12, bias=8)))


def test_stochastic_bits_are_bounded_by_the_carrier_not_the_dtype():
    """The bits are drawn in the carrier's value the kernel rounds, not in the dtype."""
    x = torch.rand(64, dtype=BF16)
    sr: dict[str, Any] = dict(rounding_mode=RoundMode.SR)
    # 3 + 20 == binary32's 23, far past bfloat16's 7
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
    # binary64 has 52 bits to share, and binary32's arithmetic on a float64
    # tensor has binary32's 23
    x64 = x.double()
    _silent(lambda: binaryK_quantize(x64, 8, 4, prng_bits=49, **sr))
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_quantize(x64, 8, 4, prng_bits=50, **sr)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_quantize(x64, 8, 4, prng_bits=21, carrier="binary32", **sr)
    with pytest.raises(ValueError, match="stochastic-rounding bits"):
        binaryK_matmul_fma(
            x64[:8].reshape(2, 4), x64[:8].reshape(4, 2), fma_K=8, fma_P=4, fma_prng_bits=50, **sr
        )


def test_a_storage_warning_names_the_callers_line():
    """Reached at different depths -- a wrapper, a hook, a backward pass -- it
    names the first frame outside mptorch and torch, which is the caller's."""
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
