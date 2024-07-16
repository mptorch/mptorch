#pragma once

enum class OverflowPolicy {
    SATURATE_INFTY,
    SATURATE_ER,
    SATURATE_RE
};

/*
SATURATE_INFTY: 
Finite input values of binary32, exceeding the maximum float value of the binary8 format, will map to infinity.
maximum float value : 0xfe or 0xfd in unsigned
Infinite inputs will still map to infinities in this mode.

SATURATE_ER:
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, 
will saturate to the maximum float represented by 0x7e/0xfe or 0xfd in unsigned.
This number system has an encoding reserved for infinity (0x7f/0xff).

SATURATE_RE:
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, 
will saturate to the maximum float represented by 0x7f/0xff or 0xfe in unsigned
This number system does not have an encoding reserved for infinity (0x7f/0xff).
*/