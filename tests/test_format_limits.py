"""What binary32 can carry, and what float16 and bfloat16 can store (T4, T6).

The casts round a binary32 value onto the format's grid and return a binary32
value, so a format with more range or precision than binary32 is quantized
*partially* -- a tensor of plausible numbers, some of the format's values
never among them. These checks are what stops that being invisible, and they
come in two strengths: a format that cannot function raises, and one that
works over part of its range warns.

A float16 or bfloat16 result is rounded once more when it is stored, so the
format is held against that dtype too, per call, by one of two rules: a
GEMM's result can be any value of its last format, and an elementwise
quantizer's can only be the format's answer to a value of the dtype.

The boundaries asserted here are the ones
``dev/benchmarks/format_limits.py sweep --audit`` (and ``sweep --dtype float16
--audit``, ``--dtype bfloat16``) measured against the kernels, so a change
that moves a bound should move a measurement first.
"""

import re
import warnings
from typing import Any

import pytest
import torch

from mptorch import BinaryK, RoundMode, SaturationMode, SubnormalsMode, SuperFP
from mptorch.number import FormatRangeWarning
from mptorch.quant import (
    FusedMac,
    QLinear,
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


def _construct(call):
    """Build a format with `FormatRangeWarning` promoted to an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


@pytest.mark.parametrize(
    "call",
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
def test_formats_in_range_are_silent(call):
    """Every format the library, its docs and its tests use is inside the bound."""
    _construct(call)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        # precision: binary32 has 24 bits and cannot hold a finer grid, so
        # nothing of such a format survives
        (lambda: BinaryK(29, 25, bias=8), "bits of precision"),
        (lambda: SuperFP(24, 4, 1, 7), "bits of precision"),
        # no finite normal value binary32 can hold: max_num is zero, so every
        # nonzero result overflows and only 0 and the infinities come back
        (lambda: BinaryK(8, 4, bias=200), "no finite normal value"),
        # the exponent field the kernels shift a 32-bit int by
        (lambda: BinaryK(40, 1), "exponent bits"),
        # the regions would be misordered and the lowest normal binade lost
        (lambda: SuperFP(3, 4, 16, 7), "no supernormal codes"),
        # the random bits share binary32's significand with the mantissa
        (lambda: BinaryK(8, 4, prng_bits=21), "stochastic-rounding bits"),
    ],
)
def test_formats_that_cannot_function_raise(call, match):
    """The error half: a format no part of which quantizes."""
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    ("call", "match"),
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
def test_formats_whose_range_outruns_binary32_warn(call, match):
    """The warning half: quantizes over the part binary32 holds, partially."""
    with pytest.warns(FormatRangeWarning, match=match):
        call()


@pytest.mark.parametrize(
    ("call", "smallest"),
    [
        (lambda: BinaryK(16, 8), "2^-134"),  # P3109's own Binary16p8se
        (lambda: BinaryK(12, 4), "2^-130"),
        (lambda: BinaryK(8, 4, bias=124), "2^-126"),
        (lambda: BinaryK(8, 4, bias=128, subnormals=SubnormalsMode.NORMALS), "2^-127"),
        (lambda: BinaryK(8, 4, bias=127, subnormals=SubnormalsMode.EXTENDED_NORMALS), "2^-127"),
        (lambda: SuperFP(3, 4, 1, 22), "2^-126"),
    ],
)
def test_formats_below_binary32_normals_warn(call, smallest):
    """Held by binary32, but the casts cannot place an input that far down.

    Every cast classifies its input by the exponent field of the binary32
    word, and all binary32 subnormals share field 0 -- so a format reaching
    below 2**-126 is rounded on a grid one binade coarse (binaryK) or onto
    powers of two the cast cannot name (superfp).
    """
    with pytest.warns(FormatRangeWarning, match=re.escape(smallest)):
        call()


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
    _construct(ok)
    with pytest.warns(FormatRangeWarning):
        bad()


@pytest.mark.parametrize(
    ("call", "match"),
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
def test_bottom_warning_names_the_reason(call, match):
    """The two half-floor limits are not the exponent-field one, and say so."""
    with pytest.warns(FormatRangeWarning, match=match):
        call()


@pytest.mark.parametrize(
    ("ok", "bad"),
    [
        (lambda: BinaryK(8, 4, bias=-112), lambda: BinaryK(8, 4, bias=-113)),
        (lambda: SuperFP(2, 3, 1, -120), lambda: SuperFP(2, 3, 1, -121)),
    ],
)
def test_top_boundary_is_where_it_was_measured(ok, bad):
    _construct(ok)
    with pytest.warns(FormatRangeWarning, match="above binary32"):
        bad()


def test_warning_can_be_filtered():
    """The warning names a category so a caller who means it can silence it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("ignore", FormatRangeWarning)
        fmt = BinaryK(16, 8)
    assert caught == []
    assert fmt.K == 16


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_precision_is_bounded_by_binary32_not_by_storage(dtype):
    """A float64 operand is still rounded in binary32, so 23 bits is the bound.

    The format below holds ``1 + 2**-30`` exactly and binary32 does not; before
    this bound the call was accepted and returned 1.0.
    """
    x = torch.tensor([1.0 + 2.0**-30], dtype=dtype)
    with pytest.raises(ValueError, match="bits of precision"):
        binaryK_quantize(x, 60, 41, bias=8)
    with pytest.raises(ValueError, match="bits of precision"):
        superfp_quantize(x, 30, 4, 1, 7)
    # 24 bits of precision is the most binary32 can express, and it is allowed
    assert binaryK_quantize(x, 30, 24, bias=8).dtype == dtype


# --- the plain-integer wrappers reach the same rule ---------------------------
#
# The wrappers in `mptorch.quant.ops` take a format as loose integers and never
# build a `BinaryK`/`SuperFP`, so without these they would quantize onto a grid
# they cannot reach and say nothing. `BinaryK(16, 8)` and `SuperFP(3, 4, 1, 22)`
# are the two formats used below: the first reaches 2**-134, the second
# 2**-126, and both are inside binary32 at the top.


@pytest.mark.parametrize(
    "call",
    [
        # the multiply slot, the accumulate slot, and the fused one
        lambda: _binaryK_spec(mul_K=16, mul_P=8),
        lambda: _binaryK_spec(mul_K=8, mul_P=4, acc_K=16, acc_P=8),
        lambda: _binaryK_fma_spec(fma_K=16, fma_P=8),
        # per palette entry, not just the first
        lambda: _binaryK_mixed_spec(mul_K=[8, 16], mul_P=[4, 8]),
        lambda: _binaryK_fma_mixed_spec(fma_K=[8, 16], fma_P=[4, 8]),
        lambda: _superfp_spec(mul_man_bits=3, mul_exp_bits=4, mul_normal_binades=1, mul_bias=22),
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
def test_every_spec_builder_checks_every_slot(call):
    with pytest.warns(FormatRangeWarning):
        call()


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


def test_a_format_object_is_not_checked_twice():
    """The value tier opts out: its formats checked themselves when built.

    Without that the same format would warn once at the line that named it and
    once from inside the resolver, where the second warning points at mptorch
    rather than at the caller.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FormatRangeWarning)
        fmt = BinaryK(16, 8)
        assert len(caught) == 1, "the format object itself should warn"
        spec_for_mac(SplitMac(fmt, fmt))
    assert len(caught) == 1, "and the resolver should not warn again"


# --- what float16 and bfloat16 can store (T6) ----------------------------------
#
# The cast rounds in binary32 and a half-width result is rounded again when it
# is written. The dtype is the call's, so the check runs per call, against the
# format whose values the result holds -- and by one of two rules, because the
# two ops reach different parts of that format. `sweep --dtype float16 --audit`
# measured every boundary below through both: a fused GEMM over operands of the
# dtype, and the elementwise quantizer over every value of it.

F16, BF16 = torch.float16, torch.bfloat16


def _silent(call):
    """Run a call with `FormatRangeWarning` promoted to an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FormatRangeWarning)
        return call()


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
    """They store every value binary32 does, so there is nothing more to say."""
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


def test_stochastic_bits_are_bounded_by_binary32_whatever_the_dtype():
    """The bits are drawn in the binary32 value the kernel rounds, not in the dtype."""
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
