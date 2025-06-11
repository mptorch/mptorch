#pragma once

/*
SATURATE_INFTY:
Finite binary32 input values that exceed the maximum float value of the current binary8 format will map to infinity.
maximum float value : 0xfe or 0xfd in unsigned
Infinite inputs will still map to infinities in this mode.

SATURATE_MAXFLOAT:
Both finite and infinite binary32 inputs exceeding the maximum float value of the binary8 format will saturate to the
maximum float value represented by 0x7e/0xfe or 0xfd in unsigned.
This number system has an encoding reserved for infinity (0x7f/0xff).

SATURATE_MAXFLOAT2:
Both finite and infinite binary32 inputs exceeding the maximum float value of the binary8 format will saturate to the
maximum float represented by 0x7f/0xff or 0xfe in unsigned.
This number system does not have an encoding reserved for infinity (0x7f/0xff).
*/
enum class OverflowPolicy
{
    SATURATE_INFTY,
    SATURATE_MAXFLOAT,
    SATURATE_MAXFLOAT2
};