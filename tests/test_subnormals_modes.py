"""Underflow below a binaryK format's smallest value, in all three floors.

``SUBNORMALS`` is IEEE P3109's: below the smallest normal the exponent-zero
codes hold subnormals. ``NORMALS`` drops the subnormals and
``EXTENDED_NORMALS`` spends their codes on one more binade of normals, so each
mode has a different smallest value (its floor). Below that floor all three
round the same way: there are two candidates, zero and the floor, and the
rounding mode picks between them. ``bit_helper.h`` states that rule once, as
``UnderflowMode``, and this module holds the kernels to it with one
expectation table run against all three floors.

The failure the table pins is a single threshold at 0.75 of the floor applied
identically in every rounding mode, under which ``RU`` and ``RD`` agree below
the floor and ``RD`` rounds away from zero. A reference cast that carries the
same arm agrees with such a kernel, so the exhaustive cast sweeps cannot see
it. The table is therefore written from the definition of each rounding mode
over the two candidates, not from a second cast, which is also how
``dev/benchmarks/format_limits.py`` checks the casts (against the format's own
value set).

The table runs twice: in binary32, on a format whose floor is well inside it,
and in binary64 on float64 inputs, at the lowest floor each mode may have
there (``bias + P = 1023`` under ``SUBNORMALS``, ``bias = 1023`` under
``NORMALS`` and ``1022`` under ``EXTENDED_NORMALS``), where every input below
the floor is a binary64 subnormal.
"""

import math

import pytest
import torch

from mptorch import RoundMode, SaturationMode, SubnormalsMode
from mptorch.quant import binaryK_quantize
from tests.markers import available_devices

DETERMINISTIC = [rm for rm in RoundMode if rm is not RoundMode.SR]

K, P, BIAS = 8, 4, 8
MAN = P - 1
MIN_EXP = 1 - BIAS

# The bias that puts each mode's floor at the lowest binary64 can carry, for a
# 14-bit format with P = 4: 2**-1021 where the grid comes from the input's
# exponent field (SUBNORMALS, smallest value 2**(2 - bias - P)) and 2**-1022
# where the bottom is a magnitude compare (NORMALS at 2**(1 - bias),
# EXTENDED_NORMALS just above 2**-bias).
K64 = 14
BIAS64 = {
    SubnormalsMode.SUBNORMALS: 1023 - P,
    SubnormalsMode.NORMALS: 1023,
    SubnormalsMode.EXTENDED_NORMALS: 1022,
}


def floor_of(subnormals: SubnormalsMode, bias: int = BIAS, man: int = MAN) -> float:
    """Return the smallest positive value of the format, by mode.

    ``SUBNORMALS`` gives ``2**(min_exp - man)`` and ``NORMALS`` ``2**min_exp``,
    with ``min_exp = 1 - bias``. ``EXTENDED_NORMALS`` adds the binade below
    ``min_exp`` but keeps that binade's mantissa-zero code for the zero, so its
    values start one step up, at ``(1 + 2**-man) * 2**(min_exp - 1)``. With no
    mantissa bits the added binade holds only that code, and the floor is the
    one ``NORMALS`` has.
    """
    min_exp = 1 - bias
    if subnormals is SubnormalsMode.SUBNORMALS:
        return math.ldexp(1.0, min_exp - man)
    if subnormals is SubnormalsMode.NORMALS or man == 0:
        return math.ldexp(1.0, min_exp)
    return math.ldexp(1.0 + 2.0**-man, min_exp - 1)


def _format(dtype, subnormals):
    """Return ``(K, bias, floor)`` of the format the table runs on for ``dtype``."""
    if dtype is torch.float64:
        bias = BIAS64[subnormals]
        return K64, bias, floor_of(subnormals, bias)
    return K, BIAS, floor_of(subnormals)


# (input as a multiple of the floor, what each mode returns as a multiple of
# the floor). Below the floor the candidates are 0 and 1: nearest picks by the
# midpoint, with the tie to 0 for RNE (code 0 is the even one) and to 1 for
# RNA, RU and RO (to odd) take 1 for any nonzero input, RD and RZ take 0. The
# multiples are exact binary fractions, so each input is exactly representable
# in the carrier and its own rounding cannot affect the answer.
TABLE = [
    #  x/floor   RNE  RNA   RU   RD   RZ   RO
    (0.0, {"RNE": 0, "RNA": 0, "RU": 0, "RD": 0, "RZ": 0, "RO": 0}),
    (0.0625, {"RNE": 0, "RNA": 0, "RU": 1, "RD": 0, "RZ": 0, "RO": 1}),
    (0.25, {"RNE": 0, "RNA": 0, "RU": 1, "RD": 0, "RZ": 0, "RO": 1}),
    (0.5, {"RNE": 0, "RNA": 1, "RU": 1, "RD": 0, "RZ": 0, "RO": 1}),
    (0.75, {"RNE": 1, "RNA": 1, "RU": 1, "RD": 0, "RZ": 0, "RO": 1}),
    (1.0, {"RNE": 1, "RNA": 1, "RU": 1, "RD": 1, "RZ": 1, "RO": 1}),
]


DTYPES = [torch.float32, torch.float64]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("subnormals", list(SubnormalsMode))
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_underflow_is_the_same_rule_in_every_subnormals_mode(device, dtype, subnormals, mode):
    """Every deterministic mode follows the table at each mode's floor.

    A threshold shared by all rounding modes fails the ``RU``, ``RD``, ``RZ``
    and ``RO`` columns.
    """
    k, bias, floor = _format(dtype, subnormals)
    x = torch.tensor([f * floor for f in (f for f, _ in TABLE)], dtype=dtype, device=device)
    want = torch.tensor([want[mode.name] * floor for _, want in TABLE], dtype=dtype, device=device)
    got = binaryK_quantize(
        x,
        k,
        P,
        bias=bias,
        rounding_mode=mode,
        saturation_mode=SaturationMode.OVF_INF,
        subnormals_mode=subnormals,
    )
    assert torch.equal(got, want), f"{got.tolist()} != {want.tolist()}"


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("subnormals", list(SubnormalsMode))
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_underflow_is_signed_the_same_way_below_zero(device, dtype, subnormals, mode):
    """Negative inputs mirror the table, with ``RU`` and ``RD`` swapped.

    A result that flushes to zero carries no sign bit, since P3109's zero is
    unsigned. The sign is checked with ``signbit`` because ``-0.0 == 0.0``.
    """
    k, bias, floor = _format(dtype, subnormals)
    mirror = {"RNE": "RNE", "RNA": "RNA", "RU": "RD", "RD": "RU", "RZ": "RZ", "RO": "RO"}
    x = torch.tensor([-f * floor for f in (f for f, _ in TABLE)], dtype=dtype, device=device)
    want = torch.tensor(
        [-want[mirror[mode.name]] * floor for _, want in TABLE], dtype=dtype, device=device
    )
    got = binaryK_quantize(
        x,
        k,
        P,
        bias=bias,
        rounding_mode=mode,
        saturation_mode=SaturationMode.OVF_INF,
        subnormals_mode=subnormals,
    )
    assert torch.equal(got, want), f"{got.tolist()} != {want.tolist()}"
    assert not torch.signbit(got[want == 0]).any(), "P3109's zero is unsigned"


@pytest.mark.parametrize("device", available_devices)
def test_extended_normals_keeps_its_zero_code(device):
    """``EXTENDED_NORMALS`` never returns the power of two under its floor.

    The floor binade's mantissa-zero code is the zero, not a value, so the
    power of two at the bottom of that binade is a hole in the grid and the
    binade's smallest value is one step above. A kernel that returns the hole
    would produce a value with no code point.
    """
    hole = math.ldexp(1.0, MIN_EXP - 1)
    floor = floor_of(SubnormalsMode.EXTENDED_NORMALS)
    assert floor > hole

    def q(vals, mode):
        return binaryK_quantize(
            torch.tensor(vals, dtype=torch.float32, device=device),
            K,
            P,
            bias=BIAS,
            rounding_mode=mode,
            subnormals_mode=SubnormalsMode.EXTENDED_NORMALS,
        ).tolist()

    # the hole is 8/9 of the floor, above the midpoint, so nearest lifts it
    assert q([hole], RoundMode.RNE) == [floor]
    assert q([hole], RoundMode.RD) == [0.0]
    assert q([hole], RoundMode.RU) == [floor]
    # and no mode returns it
    for mode in DETERMINISTIC:
        assert hole not in q([hole, hole * 1.01, floor, 0.0], mode)
    # the zero still round-trips, and the format's own values are fixed points
    for mode in DETERMINISTIC:
        assert q([0.0, floor], mode) == [0.0, floor]


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("subnormals", list(SubnormalsMode))
def test_stochastic_underflow_is_unbiased(device, dtype, subnormals):
    """Below the floor ``SR`` takes the floor with probability ``|x| / floor``.

    The mean of the results is then ``|x|``. The tolerance of 0.01 is about ten
    standard errors: ``sqrt(0.25 * 0.75 / 200_000)`` is 0.001.
    """
    k, bias, floor = _format(dtype, subnormals)
    frac = 0.25
    x = torch.full((200_000,), frac * floor, dtype=dtype, device=device)
    got = binaryK_quantize(
        x,
        k,
        P,
        bias=bias,
        prng_bits=40 if dtype is torch.float64 else 12,
        rounding_mode=RoundMode.SR,
        subnormals_mode=subnormals,
    )
    assert set(got.unique().tolist()) <= {0.0, floor}
    assert got.mean().item() / floor == pytest.approx(frac, abs=0.01)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("subnormals", [SubnormalsMode.NORMALS, SubnormalsMode.EXTENDED_NORMALS])
@pytest.mark.parametrize(("P", "frac"), [(1, 0.625), (1, 0.75), (2, 0.8125), (4, 0.96875)])
def test_stochastic_underflow_is_unbiased_off_the_grid(device, dtype, subnormals, P, frac):
    """``SR`` stays unbiased for an input off the format's grid.

    ``frac * floor`` sits in the grid cell just under the floor, so ``SR``'s
    rounding can carry it onto the floor. An underflow arm that asks where the
    rounding landed, and then draws again with the same random word, returns
    the floor with probability ``P(carry) + P(no carry) * frac``, which is 1.0
    for 0.75 at ``P = 1``. The arm must ask about the input and draw once.
    ``test_stochastic_underflow_is_unbiased`` cannot see this, because its
    input is on the grid, where the rounding is exact. In binary64 the floor is
    the lowest each mode may have there, a binade higher at ``P = 1`` and under
    ``EXTENDED_NORMALS``.
    """
    K = P + 4
    if dtype is torch.float64:
        bias = 1023 if subnormals is SubnormalsMode.NORMALS and P > 1 else 1022
    else:
        bias = 8
    floor = floor_of(subnormals, bias, P - 1)
    x = torch.full((400_000,), frac * floor, dtype=dtype, device=device)
    got = binaryK_quantize(
        x,
        K,
        P,
        bias=bias,
        prng_bits=40 if dtype is torch.float64 else 12,
        rounding_mode=RoundMode.SR,
        subnormals_mode=subnormals,
    )
    assert set(got.unique().tolist()) <= {0.0, floor}
    # 5 sigma at p = 0.5 over 400k draws is 0.004 (sigma = 0.5 / sqrt(400_000)),
    # and the double-draw bias described above is 0.03 to 0.25 for these cases
    assert got.double().mean().item() / floor == pytest.approx(frac, abs=0.004)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_extended_normals_is_normals_at_one_bit_of_precision(device, mode):
    """With ``P == 1`` both modes quantize identically, floor included.

    The extra binade is then a single code, and it is the zero, so the binade
    holds nothing and the floor is the one ``NORMALS`` has.
    ``make_normal_range_params`` gets there by adding the step to the binade's
    first value with ``+`` rather than ``|``: at ``man_bits == 0`` that step is
    the exponent field's own low bit and has to carry. An ``|`` does nothing
    for every odd exponent field, which leaves half of all formats with a
    floor one binade low.
    """
    lo = math.ldexp(1.0, -63)  # every value a power of two, and this is the floor
    x = torch.tensor(
        [0.0, lo / 8, lo / 2, lo * 0.75, lo, lo * 1.5, 1.0, 3.0, 2.0**40],
        dtype=torch.float32,
        device=device,
    )

    def q(subnormals):
        return binaryK_quantize(x, 8, 1, bias=64, rounding_mode=mode, subnormals_mode=subnormals)

    assert torch.equal(q(SubnormalsMode.EXTENDED_NORMALS), q(SubnormalsMode.NORMALS))


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", list(RoundMode))
def test_extended_normals_floor_at_the_smallest_normal(device, mode):
    """The same agreement where the floor is binary32's smallest normal, 2**-126.

    ``EXTENDED_NORMALS`` lowers the stored minimum exponent by a binade, and at
    this bias that takes it to 0, while the carry at ``P == 1`` puts the floor
    right back on a normal. A range test that reads the lowered exponent field
    instead of the floor leaves the format with no floor at all. Only ``SR``
    can tell: every other mode's rounding lands on zero or the floor by itself
    down there, whereas ``SR`` needs the underflow arm's draw against the
    floor, which then never runs. Hence ``SR`` is in the mode list and the
    seeds are fixed, so both calls see the same random words. The format is a
    binade outside ``mptorch.number``'s bound (the round-to-nearest-even of
    ``2**-127`` happens in binary32), so the call warns, but the floor is a
    kernel fact and the two modes must agree on it.
    """
    lo = math.ldexp(1.0, -126)
    vals = [0.0, lo / 8, lo / 4, lo / 2, lo * 0.75, lo, lo * 1.5, 1.0, 3.0]
    x = torch.tensor(vals * 512, dtype=torch.float32, device=device)

    def q(subnormals):
        torch.manual_seed(20260913)
        torch.cuda.manual_seed_all(20260913)
        return binaryK_quantize(
            x, 8, 1, bias=127, prng_bits=8, rounding_mode=mode, subnormals_mode=subnormals
        )

    assert torch.equal(q(SubnormalsMode.EXTENDED_NORMALS), q(SubnormalsMode.NORMALS))
