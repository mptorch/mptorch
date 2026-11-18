#pragma once

/*
SubnormalsMode: what the exponent-zero codes of a binaryK format hold, and so
where the bottom of the format's range lies.

SUBNORMALS:
The exponent-zero codes hold subnormal values, as in IEEE 754. This is IEEE
P3109's behaviour: every P3109 format with more than one bit of precision has
subnormals.

NORMALS:
Only normal values are representable. The exponent-zero codes hold only the
zero and, with the sign bit set, the NaN. Not a P3109 format.

EXTENDED_NORMALS:
The exponent-zero codes hold one more binade (the values that share one
exponent) of normal values instead of subnormals, which extends the range
down by a factor of two. Not a P3109 format. As in every binaryK format, the
code with a zero mantissa in that binade is still the format's zero, and with
the sign bit set its NaN, so the extra binade's values start one step above
the power of two that code would otherwise hold. With man_bits == 0 that step
is the whole binade, and the mode behaves as NORMALS.

Below its smallest value every mode rounds the same way. A magnitude |x|
there has two candidates, zero and the smallest value x_min, and the rounding
mode chooses between them: round to nearest takes the nearer one, and on a
tie RNE takes zero and RNA takes x_min; the directed modes take the candidate
in their direction; round to odd takes x_min, the nonzero one; and stochastic
rounding takes x_min with probability |x| / x_min. bit_helper.h's
UnderflowMode implements this rule.
*/

enum class SubnormalsMode
{
    SUBNORMALS,
    NORMALS,
    EXTENDED_NORMALS
};

/*
SaturationMode: what becomes of a value beyond the format's largest finite
value. Each mode is one of IEEE P3109's saturation modes together with the
format's domain, which P3109 specifies separately: the finite domain uses for
finite values the code points the extended domain uses for the infinities,
and allows only SatFinite. SAT_FINITE is SatFinite in the finite domain; the
other three are in the extended domain. make_normal_range_params in
bit_helper.h decides which codes are reserved, and so what the largest finite
value is. In an unsigned format every mode returns 0 for a negative number.
On the Python side SAT_FINITE_EXTENDED can also be written as SAT_FINITE with
domain=Domain.EXTENDED on a format object, which the format converts to this
mode when it is built (mptorch.number's _resolve_domain).

SAT_FINITE:
P3109 SatFinite, finite domain. Every result is clamped to the representable
finite range, an infinite input included. A NaN input passes through
unchanged.

SAT_PROPAGATE:
P3109 SatPropagate. A finite value that overflows is clamped to the largest
finite value, while an infinite input stays infinite.

SAT_NONE:
P3109 SatNone. A value that overflows becomes an infinity of its sign.
Rounding comes before saturation, so this holds under every rounding mode,
unlike in IEEE 754, where rounding toward zero, for one, overflows to the
largest finite value.

SAT_FINITE_EXTENDED:
P3109 SatFinite, extended domain. The format keeps its infinity codes, so its
largest finite value is the one it has under SAT_PROPAGATE and SAT_NONE, but
every result is clamped to that value, an infinite input included, as under
SAT_FINITE.
 */

enum class SaturationMode
{
    SAT_FINITE,
    SAT_PROPAGATE,
    SAT_NONE,
    SAT_FINITE_EXTENDED
};

/*
RoundMode: each is one of IEEE P3109's rounding modes, whose P3109 name is in
parentheses.

RNE:
Round to nearest, ties to even (NearestTiesToEven)

RNA:
Round to nearest, ties away from zero (NearestTiesToAway)

RU:
Round up (TowardPositive)

RD:
Round down (TowardNegative)

RZ:
Round toward zero (TowardZero)

RO:
Round to odd (ToOdd)

SR, SR_B, SR_C:
Stochastic rounding with N = prng_bits random bits, read as an integer R in
[0, 2^N) (StochasticA, StochasticB and StochasticC). With eta the fraction of
the way |x| lies from its lower neighbour in the format to its upper one, the
magnitude rounds away from zero if and only if floor(eta * 2^N) + R >= 2^N
(A), floor(eta * 2^(N+1)) + 2R + 1 >= 2^(N+1) (B), or RNITE(eta * 2^N) + R >=
2^N (C, where RNITE rounds to nearest, ties to even), in every region of every
format (common/stochastic.h). Only the entry points see SR_B and SR_C: they
turn them into SR and record the variant next to N in the format's prng_bits,
so that each kernel has a single stochastic instantiation.
*/
enum class RoundMode
{
    RNE,
    RNA,
    RU,
    RD,
    RZ,
    RO,
    SR,
    SR_B,
    SR_C
};

/*
AccumulateAlgorithm: how a GEMM sums its products along K. gemm_accumulate.h
states the three algorithms other than NAIVE step by step, and holds their
policies. Every GEMM and conv computation can use all four (a GEMM past NAIVE
runs through its op's *_accumulated twin), except the palette ops, which run
NAIVE only; the odd ops run the other three only in a build with
MPTORCH_ODD_ACCUMULATED=1.

NAIVE:
The running sum is rounded to the accumulate format after every addition.

KAHAN:
Kahan-compensated summation of the (rounded) products: a compensation term,
itself kept in the accumulate format, carries the error of each addition into
the next one.

BLOCK:
Two-level block summation, in the style of FABsum: the products of each block
of K are summed into a block sum (the inner precision), which is then added,
and rounded, possibly in a different outer format, to the running total.

TREE:
The products of each block are summed pairwise, as a binary tree, before the
block's sum is added to the running total. Split macs only: a fused
multiply-add has no separate product to put in the tree.
*/
enum class AccumulateAlgorithm
{
    NAIVE,
    KAHAN,
    BLOCK,
    TREE
};
