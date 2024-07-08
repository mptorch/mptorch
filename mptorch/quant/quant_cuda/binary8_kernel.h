#pragma once

enum class SaturationMode {
    SATURATE,
    NO_OVERFLOW,
    OVERFLOWS
};

/*
SATURATE: 
Finite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float.
Infinite inputs will still map to infinities in this mode.

NO_OVERFLOW: 
Both finite and infinite input values of binary32, exceeding the maximum float value of the binary8 format, will saturate to the maximum float.

OVERFLOWS:
All input values of binary32, exceeding the maximum float value of the binary8 format, will map to infinity.
Infinite inputs will still map to infinities in this mode.
*/