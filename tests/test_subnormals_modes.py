"""The bottom of the range under the two non-P3109 subnormals modes.

``SUBNORMALS`` is the standard's: below the smallest normal the exponent-zero
codes hold subnormals, and below the smallest of *those* the only two answers
are zero and it. ``NORMALS`` drops the subnormals and ``EXTENDED_NORMALS``
spends their codes on one more binade of normals, so each has a different
smallest value -- but the region below it is the same shape, two candidates and
a rounding mode, and the rule is the same one. That is what this file asserts:
one expectation table, run against all three floors.

The rule had been a single threshold at 0.75 of the floor, applied identically
in every mode -- so ``RU`` and ``RD`` agreed there and ``RD`` rounded away from
zero. ``reference_casts.h`` carried the same arm, so the exhaustive cast sweeps
agreed with it and always would; it was ``dev/benchmarks/format_limits.py``,
which checks the casts against the format's own value set, that found it (T4).

The table runs twice: in binary32, on a format whose floor is well inside it,
and in binary64 on float64 inputs, at the lowest floor each mode may have
there -- ``bias + P = 1023`` under ``SUBNORMALS``, ``bias = 1023`` and
``1022`` for the other two -- where every input below the floor is a binary64
subnormal.
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

# binary64's lowest floor per mode, for a 14-bit format with P = 4
K64 = 14
BIAS64 = {
    SubnormalsMode.SUBNORMALS: 1023 - P,
    SubnormalsMode.NORMALS: 1023,
    SubnormalsMode.EXTENDED_NORMALS: 1022,
}


def floor_of(subnormals: SubnormalsMode, bias: int = BIAS, man: int = MAN) -> float:
    """The smallest magnitude the format represents, by mode.

    ``EXTENDED_NORMALS``' floor binade keeps its mantissa-zero code for the
    zero, as every binade-zero code is kept, so its values start one step up.
    """
    min_exp = 1 - bias
    if subnormals is SubnormalsMode.SUBNORMALS:
        return math.ldexp(1.0, min_exp - man)
    if subnormals is SubnormalsMode.NORMALS or man == 0:
        return math.ldexp(1.0, min_exp)
    return math.ldexp(1.0 + 2.0**-man, min_exp - 1)


def _format(dtype, subnormals):
    """(K, bias, floor) of the format a carrier's table runs on."""
    if dtype is torch.float64:
        bias = BIAS64[subnormals]
        return K64, bias, floor_of(subnormals, bias)
    return K, BIAS, floor_of(subnormals)


# (multiple of the floor, what each mode returns as a multiple of the floor).
# Exact binary fractions, so the input is a float32 value and the answer is
# not a question about the input's own rounding.
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
    """Nearest with the tie to zero, the directed modes their direction, odd away."""
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
    """The directed modes swap, and the zero stays unsigned whichever side it came from."""
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
    """The floor binade's mantissa-zero code is the zero, not a value.

    So the power of two that would sit at the bottom of that binade is a hole
    in the grid: rounding never lands on it, and the binade's smallest value is
    one step above.
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

    # the hole itself is above the midpoint to the floor, so nearest lifts it
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
    """SR picks the floor with probability |x| / floor, so the mean is |x|."""
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
    """The same, for an input the format's grid does not hold.

    ``frac * floor`` sits in the grid cell just under the floor, so SR's
    rounding can carry it onto the floor -- and the arm below the floor used to
    ask where the rounding landed, then draw again with the same random word,
    so the floor came out with probability ``P(carry) + P(no carry) * frac``:
    1.0 for 0.75 at ``P = 1``. It asks about the input now, and draws once.
    ``test_stochastic_underflow_is_unbiased`` could not see it: its input is on
    the grid, where the rounding is exact. In binary64 the floor is the lowest
    each mode may have there, a binade higher at ``P = 1`` and under
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
    # 5 sigma at p = 0.5 over 400k draws is 0.004; the bias was 0.03-0.25
    assert got.double().mean().item() / floor == pytest.approx(frac, abs=0.004)


@pytest.mark.parametrize("device", available_devices)
@pytest.mark.parametrize("mode", DETERMINISTIC)
def test_extended_normals_is_normals_at_one_bit_of_precision(device, mode):
    """With ``P == 1`` the extra binade is a single code, and it is the zero.

    So that binade holds nothing and the floor is the one ``NORMALS`` has --
    which in ``make_normal_range_params`` is a ``+`` and not an ``|``, because
    at ``man_bits == 0`` the step to the binade's first value is the exponent
    field's own low bit and has to carry. With ``|`` it silently did nothing
    for every odd exponent field, which is half of all formats; the exhaustive
    cast sweep is what caught it.
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
    """The same, where that floor is binary32's smallest normal, 2**-126.

    ``EXTENDED_NORMALS`` lowers ``min_exponent_store`` by a binade, and at this
    bias that takes it to 0 -- where the range test used to read it, leaving
    the format with no floor at all, when the carry at ``P == 1`` puts the floor
    right back on a normal. Only ``SR`` could tell: every other mode's rounding
    lands on zero or the floor by itself down there, and ``SR``'s second draw,
    against the floor, never ran. The image sweep found it (binary64 has the
    floor either way). The format is a binade outside ``mptorch.number``'s
    bound -- round-to-nearest-even of ``2**-127`` is binary32's -- but the
    floor is a kernel fact, and the two modes must agree on it.
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
