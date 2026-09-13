"""What binary32 can carry: the range checks `BinaryK` and `SuperFP` run (T4).

The casts round a binary32 value onto the format's grid and return a binary32
value, so a format with more range or precision than binary32 is quantized
*partially* -- a tensor of plausible numbers, some of the format's values
never among them. These checks are what stops that being invisible, and they
come in two strengths: a format binary32 cannot hold at all raises, and one it
can hold but cannot round onto faithfully warns.

The boundaries asserted here are the ones
``dev/benchmarks/format_limits.py sweep --audit`` measured against the
kernels, so a change that moves a bound should move a measurement first.
"""

import re
import warnings

import pytest
import torch

from mptorch import BinaryK, SaturationMode, SubnormalsMode, SuperFP
from mptorch.number import FormatRangeWarning
from mptorch.quant import SplitMac, binaryK_matmul, binaryK_quantize, superfp_quantize
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
        # superfp's supernormal floor, likewise
        (lambda: SuperFP(3, 4, 1, 21), lambda: SuperFP(3, 4, 1, 22)),
    ],
)
def test_bottom_boundary_is_where_it_was_measured(ok, bad):
    _construct(ok)
    with pytest.warns(FormatRangeWarning):
        bad()


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
    with pytest.raises(AssertionError, match="binary32 significand|torch.float32"):
        binaryK_quantize(x, 60, 41, bias=8)
    with pytest.raises(AssertionError, match="binary32 significand|torch.float32"):
        superfp_quantize(x, 30, 4, 1, 7)
    # 24 bits of precision is the most binary32 can express, and it is allowed
    assert binaryK_quantize(x, 30, 24, bias=8).dtype == dtype


def test_narrow_storage_keeps_its_own_tighter_bound():
    """float16 carries 10 mantissa bits, and that is still the binding one."""
    x = torch.tensor([1.0], dtype=torch.float16)
    with pytest.raises(AssertionError, match="torch.float16"):
        binaryK_quantize(x, 14, 13, bias=16)
    assert binaryK_quantize(x, 14, 11, bias=16).dtype == torch.float16


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
