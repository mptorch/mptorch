#pragma once

enum class OverflowPolicy {
    OVERFLOW_INFTY,
    OVERFLOW_MAXFLOAT_EXT_REALS,
    OVERFLOW_MAXFLOAT_REALS
    
    // SATURATE,
    // NO_OVERFLOW,
    // OVERFLOWS
};

/*
OVERFLOW_INFTY: 
Finite input values of binary32, exceeding the maximum float value of the binary8 format, will map to infinity.
Infinite inputs will still map to infinities in this mode.

OVERFLOW_MAXFLOAT_EXT_REALS:
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float represented by 0x7e/0xfe.
This number system has an encoding reserved for infinity (0x7f/0xff).

OVERFLOW_MAXFLOAT_REALS:
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float represented by 0x7f/0xff.
This number system does not have an encoding reserved for infinity (0x7f/0xff).
*/