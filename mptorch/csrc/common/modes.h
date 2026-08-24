#pragma once

/*
SUBNORMALS:
Subnormal values are supported.
NORMALS:
Only normal values are supported.
EXTENDED_NORMALS:
The binade used to encode subnormals is used as an extra binade to encode normal
values.
*/

enum class SubnormalsMode
{
    SUBNORMALS,
    NORMALS,
    EXTENDED_NORMALS
};

/*
SAT_FINITE:
All return values are clamped to the representable finite range. NaN inputs
pass unchanged.

SAT_PROPAGATE:
Finite return values are clamped to the representable range, whereas infinite
values are preserved.

OVF_INF:
Just like with IEEE-754 2019, out-of-range values are replaced with: the
extremal finite value, positive or negative infinity, as indicated by the
rounding mode, and the signedness of the target format.
 */

enum class SaturationMode
{
    SAT_FINITE,
    SAT_PROPAGATE,
    OVF_INF
};

/*
RNE:
Round to nearest, ties to even

RNA:
Round to nearest, ties to away from zero

RU:
Round up

RD:
Round down

RZ:
Round towards zero

RO:
Round to odd

SR:
Stochastic rounding
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