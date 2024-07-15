#pragma once

enum class OverflowPolicy {
    OVERFLOW_INFTY,
    OVERFLOW_MAXFLOAT_FE,
    OVERFLOW_MAXFLOAT_FF
    
    // SATURATE,
    // NO_OVERFLOW,
    // OVERFLOWS
};

/*
OVERFLOW_INFTY: 
Finite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float.
Infinite inputs will still map to infinities in this mode.

OVERFLOW_MAXFLOAT_FF:
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float represented by 0x7f/0xff.
This number system does not have an encoding reserved for infinity.

OVERFLOW_MAXFLOAT_FE:
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float represented by 0x7e/0xfe.
This number system has an encoding reserved for infinity (0x7f/0xff).


past functionalities:
NO_OVERFLOW: 
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float.

OVERFLOWS:
All input values of binary32, exceeding the maximum float value of the binary8 format, will map to infinity.
Infinite inputs will still map to infinities in this mode.
*/