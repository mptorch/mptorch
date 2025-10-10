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