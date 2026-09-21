#pragma once

/*
SubnormalsMode says what the exponent-zero codes of a binaryK format hold,
and so where the bottom of the format's range is.

SUBNORMALS:
The exponent-zero codes hold subnormal values, as in IEEE 754. This is IEEE
P3109's behaviour: every P3109 format with more than one bit of precision has
subnormals.

NORMALS:
Only normal values are representable. The exponent-zero codes hold nothing
but the zero (and, with the sign bit, the NaN). Not a P3109 format.

EXTENDED_NORMALS:
The exponent-zero binade holds one more binade of normal values instead of
subnormals, which extends the range downward by a factor of two. Not a P3109
format. The binade's mantissa-zero code is still the format's zero, and with
the sign bit its NaN, as in every binaryK format, so the extra binade's values
start one step above the power of two that code would otherwise hold. With
man_bits == 0 that step is the whole binade, and the mode degenerates to
NORMALS.

All three modes round the same way below whatever their smallest value is,
because that region has the same shape in each: two candidates, zero and the
smallest value, and the rounding mode picks between them. Nearest takes the
nearer and a tie the zero, the directed modes take their own direction,
round-to-odd takes the nonzero one, and stochastic takes it with probability
|x| divided by it. bit_helper.h's UnderflowMode is where that rule is
implemented.
*/

enum class SubnormalsMode
{
    SUBNORMALS,
    NORMALS,
    EXTENDED_NORMALS
};

/*
The three saturation modes are IEEE P3109's, and they also select P3109's
domain: SAT_FINITE is the finite domain, in which the code points the extended
domain spends on the infinities hold finite values, and the other two are the
extended domain. Which codes those are, and so what the largest finite value
is, is decided by make_normal_range_params in bit_helper.h. In an unsigned
format every mode returns 0 for a negative value.

SAT_FINITE:
P3109 SatFinite. Every return value is clamped to the representable finite
range, an infinite input included. NaN inputs pass through unchanged.

SAT_PROPAGATE:
P3109 SatPropagate. Finite values that overflow are clamped to the
representable range, whereas infinite inputs stay infinite.

OVF_INF:
P3109 SatNone. Out-of-range values become positive or negative infinity.
Rounding comes before saturation, so this holds under every rounding mode,
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
How a GEMM's dot product is accumulated. gemm_accumulate.h states the three
past NAIVE step by step, and holds their policies; they are implemented for
the single-format ops, in binary32, on the CPU and CUDA.

NAIVE:
The running sum is quantized (to the accumulate format) after every addition.

KAHAN:
Kahan-compensated summation of the (quantized) partial products, with the
compensation term itself tracked in the accumulate format.

BLOCK:
Two-level (FABSum-style) block summation: partial products are locally
summed in blocks (inner precision) before each block sum is folded (and
quantized, at a possibly different, outer precision) into the running
total.

TREE:
Partial products are locally combined pairwise (tree reduction) within a
block before the block's reduced value is folded into the running total.
Split macs only: a fused multiply-add has no product term.
*/
enum class AccumulateAlgorithm
{
    NAIVE,
    KAHAN,
    BLOCK,
    TREE
};
