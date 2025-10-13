#pragma once

/*
SUBNORMALS:
Subnormal values are supported.
NORMALS:
Only normal values are supported.
EXTENDED_NORMALS:
The binade used to encode subnormals is used as an extra binade to encode normal values.
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
Just like with IEEE-754 2019, out-of-range values are replaced with: the extremal
finite value, positive or negative infinity, as indicated by the rounding mode, and
the signedness of the target format.
 */

enum class SaturationMode
{
    SAT_FINITE,
    SAT_PROPAGATE,
    OVF_INF
};