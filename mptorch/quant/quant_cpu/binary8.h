#pragma once

enum class OverflowPolicy {
    SATURATE_INFTY,
    SATURATE_MAXFLOAT,
    SATURATE_MAXFLOAT2
};

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

/**
 * Converts an array of floating-point numbers to binary8 format using nearest rounding for signed values.
 * 
 * @param a                Input array of floating-point numbers.
 * @param o                Output array to store the converted binary8 numbers.
 * @param size             The number of elements in the input array.
 * @param P                The precision parameter for binary8 format.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 */
void binary8_signed_nearest(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Converts an array of floating-point numbers to binary8 format using nearest rounding for unsigned values.
 * 
 * @param a                Input array of floating-point numbers.
 * @param o                Output array to store the converted binary8 numbers.
 * @param size             The number of elements in the input array.
 * @param P                The precision parameter for binary8 format.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 */
void binary8_unsigned_nearest(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Converts an array of floating-point numbers to binary8 format using stochastic rounding for signed values.
 * 
 * @param a                Input array of floating-point numbers.
 * @param r                Array of random probabilities for stochastic rounding.
 * @param o                Output array to store the converted binary8 numbers.
 * @param size             The number of elements in the input array.
 * @param P                The precision parameter for binary8 format.
 * @param prng_bits        The number of bits used for the pseudo-random number generator.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 */
void binary8_signed_stochastic(float *a, int *r, float *o, int size, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Converts an array of floating-point numbers to binary8 format using stochastic rounding for unsigned values.
 * 
 * @param a                Input array of floating-point numbers.
 * @param r                Array of random probabilities for stochastic rounding.
 * @param o                Output array to store the converted binary8 numbers.
 * @param size             The number of elements in the input array.
 * @param P                The precision parameter for binary8 format.
 * @param prng_bits        The number of bits used for the pseudo-random number generator.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 */
void binary8_unsigned_stochastic(float *a, int *r, float *o, int size, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Converts an array of floating-point numbers to binary8 format using truncation for signed values.
 * 
 * @param a                Input array of floating-point numbers.
 * @param o                Output array to store the converted binary8 numbers.
 * @param size             The number of elements in the input array.
 * @param P                The precision parameter for binary8 format.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 */
void binary8_signed_truncate(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Converts an array of floating-point numbers to binary8 format using truncation for unsigned values.
 * 
 * @param a                Input array of floating-point numbers.
 * @param o                Output array to store the converted binary8 numbers.
 * @param size             The number of elements in the input array.
 * @param P                The precision parameter for binary8 format.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 */
void binary8_unsigned_truncate(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals);
