#pragma once

/*
What the exponent-zero codes hold, and so where the bottom of the range is.

SUBNORMALS:
Subnormal values are supported. This is IEEE P3109's behaviour: every P3109
format with more than one bit of precision has subnormals.
NORMALS:
Only normal values are supported. Not a P3109 format.
EXTENDED_NORMALS:
The binade used to encode subnormals is used as an extra binade to encode normal
values. Not a P3109 format. Its mantissa-zero code is *not* one of them: that
code is the format's zero, and with the sign bit its NaN, as in every binaryK
format, so the extra binade's values start one step above the power of two it
would otherwise hold. With man_bits == 0 that step is the whole binade, and the
mode is NORMALS with extra steps.

All three modes round the same way below whatever their smallest value is,
because the shape of that region is the same in each: two candidates, zero and
that value, and the rounding mode picks between them. Nearest takes the nearer
and a tie the zero, the directed modes take their own direction, round-to-odd
takes the nonzero one, and stochastic takes it with probability |x| divided by
it. See bit_helper.h's UnderflowMode, which is where that is written down.
*/

enum class SubnormalsMode
{
    SUBNORMALS,
    NORMALS,
    EXTENDED_NORMALS
};

/*
The three are IEEE P3109's saturation modes, and they also select P3109's
domain: SAT_FINITE is the finite domain, in which the code points the extended
domain spends on the infinities hold finite values, and the other two are the
extended domain. Which codes those are, and so what the largest finite value
is, is make_normal_range_params' business (bit_helper.h). In an unsigned
format every mode returns 0 for a negative value.

SAT_FINITE:
P3109 SatFinite. All return values are clamped to the representable finite
range. NaN inputs pass unchanged.

SAT_PROPAGATE:
P3109 SatPropagate. Finite return values are clamped to the representable
range, whereas infinite values are preserved.

OVF_INF:
P3109 SatNone. Out-of-range values become positive or negative infinity.
Rounding comes before saturation, so this holds under every rounding mode --
unlike IEEE 754, where rounding toward zero, for one, overflows to the largest
finite value.
 */

enum class SaturationMode
{
    SAT_FINITE,
    SAT_PROPAGATE,
    OVF_INF
};

/*
Each is one of IEEE P3109's rounding modes, named in parentheses.

RNE:
Round to nearest, ties to even (NearestTiesToEven)

RNA:
Round to nearest, ties to away from zero (NearestTiesToAway)

RU:
Round up (TowardPositive)

RD:
Round down (TowardNegative)

RZ:
Round towards zero (TowardZero)

RO:
Round to odd (ToOdd)

SR:
Stochastic rounding (StochasticA, with N = prng_bits random bits added below
the retained significand before it is truncated)
*/
enum class RoundMode
{
    RNE,
    RNA,
    RU,
    RD,
    RZ,
    RO,
    SR
};

/*
NAIVE:
The running sum of a dot product is quantized (to the accumulate format)
after every addition.

KAHAN:
Kahan-compensated summation of the (quantized) partial products, with the
compensation term itself tracked in the accumulate format. Not yet
implemented.

BLOCK:
Two-level (FABSum-style) block summation: partial products are locally
summed in blocks (inner precision) before each block sum is folded (and
quantized, at a possibly different, outer precision) into the running
total. Not yet implemented.

TREE:
Partial products are locally combined pairwise (tree reduction) within a
block before the block's reduced value is folded into the running total.
Not yet implemented.
*/
enum class AccumulateAlgorithm
{
    NAIVE,
    KAHAN,
    BLOCK,
    TREE
};