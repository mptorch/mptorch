#include "bit_helper.cpp"
#include "quant_kernel.h"
#include "sim_helper.cpp"
#include <cmath>
#include "binary8_kernel.h"

float cast_binary8_signed_nearest(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && (exp_val < min_exp)) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                              : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits - 1)) - 1;
        int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

    uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_signed_truncate(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = uval32 >> (23 - man_bits + subnormal_shift) << (23 - man_bits + subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_nearest(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    if (origin_float < 0) return NAN;

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 9 - P;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = (P == 1) - max_exp;

        if ((min_exp - exp_val) <= man_bits && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                               : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    if (origin_float < 0) return NAN;

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 8 - P + 1;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

    uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

    return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_truncate(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    if (origin_float < 0) return NAN;

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }

    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = uval32 >> (23 - man_bits + subnormal_shift) << (23 - man_bits + subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

void binary8_signed_kernel_nearest(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_signed_nearest(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_unsigned_kernel_nearest(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_unsigned_nearest(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_signed_kernel_stochastic(float *a, int *r, float *o, int size, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
    }
}

void binary8_unsigned_kernel_stochastic(float *a, int *r, float *o, int size, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
    }
}

void binary8_signed_kernel_truncate(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_signed_truncate(a[idx], P, overflow_policy, subnormals);
    }
}

void binary8_unsigned_kernel_truncate(float *a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
    for (int idx = 0; idx < size; ++idx) {
        o[idx] = cast_binary8_unsigned_truncate(a[idx], P, overflow_policy, subnormals);
    }
}
