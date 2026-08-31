#pragma once

#include "modes.h"
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define CUDA_HOST_DEVICE_INLINE inline
#endif

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

CUDA_HOST_DEVICE_INLINE uint32_t extract_exponent(float *a)
{
    uint32_t temp = *(reinterpret_cast<uint32_t *>(a));
    // extract exponent bits (single precision, 1 sign bit, 23 mantissa bits)
    temp = (temp >> 23) & 0xFFu;
    // adjust for exponent bias and virtual bit
    return temp - 127 + 1;
}

// rounds to nearest, ties to even
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_even(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t tie = 1 << (22 - man_bits);
    uint32_t add_r = target + tie;
    uint32_t quantized = add_r & ~mask;
    uint32_t is_tie = (target & mask) == tie;
    uint32_t odd = (man_bits == 0) ? 0 : 1; // if man_bits == 0, implicit bit is 1 (odd) so we always round up (carry to exponent)
    return quantized & ~((is_tie & odd) << (23 - man_bits));
}

CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_even(uint32_t target)
{
    uint32_t tie = 0x00400000;
    uint32_t quantized = (target + tie) & ~0x007FFFFF;
    uint32_t is_tie = (target & 0x007FFFFF) == tie;
    return quantized - ((is_tie << 23) & ~quantized);
}

// Precomputed-parameter overloads below: same bit-twiddling as the
// man_bits/exp_bits/bias-taking functions above, but reading values derived
// from those format parameters instead of recomputing them every call. A
// GEMM Multiplier/Adder (gemm_policy.h) precomputes these once at
// construction and reuses them across its whole kernel launch (up to M*N*K
// calls); the *_bits-taking originals stay in use everywhere else
// (elementwise binaryK_quantize/superfp_quantize). Deliberately flat
// (no nested structs) -- see dev/gemm_core_roadmap.md item 6 for the
// compile-time blowup that motivated this. RoundMode::SR is out of scope
// here (needs a per-call random value, not a format constant).

// Precomputed for round_bitwise_nearest_even(target, man_bits) and reused by
// nearest-away/up/down/odd's precomputed overloads below (they share the
// same bypass/mask/tie formulas). Not valid for the man_bits == 0 case,
// which uses the structurally different zero-arg overload above.
struct RoundParams
{
    bool bypass; // man_bits >= 23: round_bitwise_nearest_even returns target unchanged
    uint32_t mask;
    uint32_t tie;
    int shift; // 23 - man_bits
};

CUDA_HOST_DEVICE_INLINE RoundParams make_round_params(int man_bits)
{
    RoundParams p;
    p.bypass = man_bits >= 23;
    if (p.bypass)
    {
        p.mask = 0u;
        p.tie = 0u;
        p.shift = 0;
        return p;
    }
    p.shift = 23 - man_bits;
    p.mask = (1u << p.shift) - 1u;
    p.tie = 1u << (p.shift - 1);
    return p;
}

CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_even(uint32_t target, bool bypass, uint32_t mask,
                                                            uint32_t tie, int shift)
{
    if (bypass)
        return target;
    uint32_t add_r = target + tie;
    uint32_t quantized = add_r & ~mask;
    uint32_t is_tie = (target & mask) == tie;
    return quantized & ~(is_tie << shift);
}

// rounds to nearest, ties to away
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_away(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t tie = 1 << (22 - man_bits);
    uint32_t add_r = target + tie;
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_nearest_away above, via RoundParams
// (never needs the `shift` field).
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_nearest_away(uint32_t target, bool bypass, uint32_t mask, uint32_t tie)
{
    if (bypass)
        return target;
    uint32_t add_r = target + tie;
    return add_r & ~mask;
}

// rounds up, towards positive infinity
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_up(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? 0 : mask);
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_up above: `sign` is per-value data,
// not a format constant, so only bypass/mask (man_bits-derived) precompute.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_up(uint32_t target, bool bypass, uint32_t mask)
{
    if (bypass)
        return target;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? 0 : mask);
    return add_r & ~mask;
}

// rounds down, towards negative infinity
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_down(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? mask : 0);
    return add_r & ~mask;
}

// Precomputed sibling of round_bitwise_down above -- see round_bitwise_up's.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_down(uint32_t target, bool bypass, uint32_t mask)
{
    if (bypass)
        return target;
    uint32_t sign = target >> 31;
    uint32_t add_r = target + (sign ? mask : 0);
    return add_r & ~mask;
}

// rounds to odd: truncates towards zero to man_bits, then ORs the "sticky
// bit" of the discarded bits into the kept LSB. When man_bits == 0 the kept
// "bit" is the exponent's LSB (no explicit significand), so a nonzero
// sticky bit carries into the exponent.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_odd(uint32_t target, int man_bits)
{
    if (man_bits >= 23)
        return target;
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t lsb = 1 << (23 - man_bits);
    uint32_t sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// Precomputed sibling of round_bitwise_odd above. lsb = mask + 1, so no
// separate field is needed for it.
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_odd(uint32_t target, bool bypass, uint32_t mask)
{
    if (bypass)
        return target;
    uint32_t lsb = mask + 1u;
    uint32_t sticky = (target & mask) != 0;
    return (target & ~mask) | (sticky * lsb);
}

// stochastic rounding
CUDA_HOST_DEVICE_INLINE uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
{ // passing number of random bits as second parameter
    // (all the bits after the least significant bit which is based on prng);
    // target is the original number
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    // adding random bits to target (which is not masked)
    uint32_t add_r = target + (rand_prob & mask);
    // masking out bits on the right hand side of the significant bits (truncating)
    uint32_t quantized = add_r & ~mask;
    return quantized;
}

// clips the exponent of a floating point format with subnormal values
CUDA_HOST_DEVICE_INLINE uint32_t clip_subnormal_range_exponent(uint32_t old_num, uint32_t quantized_num,
                                                               int exp_bits, int man_bits, int bias)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);
    int min_exponent_store = -(bias - 1) - man_bits + 127;

    uint32_t old_sign = old_num & 0x80000000u;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num |= old_sign;
        quantized_num *= offset;
    }

    return quantized_num;
}

// Precomputed for clip_subnormal_range_exponent -- exp_bits is accepted by
// the original for signature symmetry with its sibling clip functions but
// unused by its body, so only min_exponent_store (derived from man_bits/
// bias) needs precomputing.
struct SubnormalRangeParams
{
    int min_exponent_store;
};

CUDA_HOST_DEVICE_INLINE SubnormalRangeParams make_subnormal_range_params(int man_bits, int bias)
{
    SubnormalRangeParams p;
    p.min_exponent_store = -(bias - 1) - man_bits + 127;
    return p;
}

CUDA_HOST_DEVICE_INLINE uint32_t clip_subnormal_range_exponent(uint32_t old_num, uint32_t quantized_num,
                                                                int min_exponent_store)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);

    uint32_t old_sign = old_num & 0x80000000u;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num |= old_sign;
        quantized_num *= offset;
    }

    return quantized_num;
}

CUDA_HOST_DEVICE_INLINE uint32_t clip_subnormal_range_exponent_up(uint32_t old_num, uint32_t quantized_num,
                                                                  int exp_bits, int man_bits, int bias)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);
    int min_exponent_store = -(bias - 1) - man_bits + 127;

    uint32_t old_sign = old_num & 0x80000000u;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        quantized_num = min_exponent_store << 23;
        quantized_num |= old_sign;
    }

    return quantized_num;
}

// Precomputed sibling of clip_subnormal_range_exponent_up above -- shares
// SubnormalRangeParams/make_subnormal_range_params with the plain
// clip_subnormal_range_exponent overload.
CUDA_HOST_DEVICE_INLINE uint32_t clip_subnormal_range_exponent_up(uint32_t old_num, uint32_t quantized_num,
                                                                  int min_exponent_store)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);

    uint32_t old_sign = old_num & 0x80000000u;
    if (quantized_exponent_store < min_exponent_store)
    {
        quantized_num = min_exponent_store << 23;
        quantized_num |= old_sign;
    }

    return quantized_num;
}

// clips the exponent of a floating point format without subnormal values (binaryK version)
CUDA_HOST_DEVICE_INLINE uint32_t clip_normal_range_exponent(uint32_t old_num, uint32_t quantized_num,
                                                            int exp_bits, int man_bits, int bias,
                                                            SaturationMode saturation_mode, bool extended_normals = false)
{
    if (quantized_num == 0)
        return quantized_num;

    uint32_t sign = old_num & 0x80000000u;
    quantized_num &= 0x7FFFFFFFu;
    if ((quantized_num == 0x7F800000 && saturation_mode != SaturationMode::SAT_FINITE) || (quantized_num > 0x7F800000))
        return sign | quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);
    int max_exponent_store = ((1 << exp_bits) - 1 - bias) + 126 + (man_bits > 1);
    int min_exponent_store = -(bias - 1) + 127 - extended_normals;
    int finite = (saturation_mode == SaturationMode::SAT_FINITE);

    uint32_t max_man = ((0x007FFFFF >> (23 - man_bits)) - 1 + finite) << (23 - man_bits);
    uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;

    // handle overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        switch (saturation_mode)
        {
        case SaturationMode::SAT_FINITE:
            quantized_num = max_num;
            break;

        case SaturationMode::SAT_PROPAGATE:
            quantized_num = max_num;
            break;

        default:
            quantized_num = 0x7F800000;
            break;
        }
    }
    else if (quantized_exponent_store == max_exponent_store)
    {
        // handle overflow
        if (quantized_num > max_num && saturation_mode == SaturationMode::OVF_INF)
            quantized_num = 0x7F800000;
    }
    // handle underflow
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
    }

    quantized_num |= sign;

    return quantized_num;
}

// Precomputed for clip_normal_range_exponent above (the overload
// cast_binaryK_nearest_even/cast_superfp_nearest_even use).
struct NormalRangeParams
{
    SaturationMode saturation_mode; // still needed raw: selects the overflow branch's outcome
    int max_exponent_store;
    int min_exponent_store;
    uint32_t max_num;
};

CUDA_HOST_DEVICE_INLINE NormalRangeParams make_normal_range_params(int exp_bits, int man_bits, int bias,
                                                                    SaturationMode saturation_mode,
                                                                    bool extended_normals = false)
{
    NormalRangeParams p;
    p.saturation_mode = saturation_mode;
    p.max_exponent_store = ((1 << exp_bits) - 1 - bias) + 126 + (man_bits > 1);
    p.min_exponent_store = -(bias - 1) + 127 - extended_normals;
    int finite = (saturation_mode == SaturationMode::SAT_FINITE);
    uint32_t max_man = ((0x007FFFFF >> (23 - man_bits)) - 1 + finite) << (23 - man_bits);
    p.max_num = ((uint32_t)p.max_exponent_store << 23) | max_man;
    return p;
}

CUDA_HOST_DEVICE_INLINE uint32_t clip_normal_range_exponent(uint32_t old_num, uint32_t quantized_num,
                                                             SaturationMode saturation_mode,
                                                             int max_exponent_store, int min_exponent_store,
                                                             uint32_t max_num)
{
    if (quantized_num == 0)
        return quantized_num;

    uint32_t sign = old_num & 0x80000000u;
    quantized_num &= 0x7FFFFFFFu;
    if ((quantized_num == 0x7F800000 && saturation_mode != SaturationMode::SAT_FINITE) || (quantized_num > 0x7F800000))
        return sign | quantized_num;

    int quantized_exponent_store = (int)((quantized_num >> 23) & 0xFF);

    // handle overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        switch (saturation_mode)
        {
        case SaturationMode::SAT_FINITE:
            quantized_num = max_num;
            break;

        case SaturationMode::SAT_PROPAGATE:
            quantized_num = max_num;
            break;

        default:
            quantized_num = 0x7F800000;
            break;
        }
    }
    else if (quantized_exponent_store == max_exponent_store)
    {
        // handle overflow
        if (quantized_num > max_num && saturation_mode == SaturationMode::OVF_INF)
            quantized_num = 0x7F800000;
    }
    // handle underflow
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
    }

    quantized_num |= sign;

    return quantized_num;
}
