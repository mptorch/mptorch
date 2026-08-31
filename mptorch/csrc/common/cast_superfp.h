#pragma once

#include "bit_helper.h"
#include "modes.h"

// superfp has no subnormals: below the "supernormal" region (which repurposes
// what would otherwise be subnormal encoding space to extend the exponent
// range with implicit-mantissa-1.0 powers of two) everything flushes to
// zero. These two cutoffs classify a float32 unbiased exponent into one of
// three regions: normal (>= normal_cutoff), supernormal ([supernormal_cutoff,
// normal_cutoff)), or underflow (< supernormal_cutoff).
struct SuperfpCutoffs
{
    int normal_cutoff;
    int supernormal_cutoff;
};

CUDA_HOST_DEVICE_INLINE SuperfpCutoffs superfp_region_cutoffs(int man_bits, int exp_bits,
                                                               int normal_binades, int bias)
{
    SuperfpCutoffs c;
    c.normal_cutoff = ((1 << exp_bits) - 1 - bias) - normal_binades + 1;
    c.supernormal_cutoff = c.normal_cutoff - ((1 << exp_bits) - normal_binades) * (1 << man_bits) + 1;
    return c;
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_even(float origin_float,
                                                        int man_bits, int exp_bits, int normal_binades,
                                                        int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized{0.0f};

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;
    if (target_exp == 128)
    {
        // handle NaN/inf inputs
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_even(target);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-even between 0 and the smallest supernormal magnitude
        // (2^supernormal_cutoff): only the topmost underflow exponent
        // (supernormal_cutoff - 1) can be at or above the halfway point;
        // anything strictly below always flushes to (signed) zero.
        uint32_t sign_bit = target & 0x80000000u;
        quantize_bits = sign_bit;
        if (target_exp == co.supernormal_cutoff - 1 && (target & 0x007FFFFFu) != 0)
        {
            // strictly above half -> round up to the smallest supernormal;
            // exactly at half (mantissa all zero) ties to even = zero.
            quantize_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = (man_bits > 0)
                            ? round_bitwise_nearest_even(target, man_bits)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_nearest_even -- see the
// analogous BinaryKParams/cast_binaryK_nearest_even overload in
// cast_binaryK.h. Only covers RNE, matching the GEMM core's current scope.
// Flat by design -- see dev/gemm_core_roadmap.md item 6.
struct SuperfpParams
{
    int man_bits; // kept raw: drives the man_bits > 0 round-formula selection
    int bias;     // kept raw: cast_superfp_odd's man_bits == 0 branch needs it directly
    // region classification (normal/supernormal/underflow)
    int normal_cutoff;
    int supernormal_cutoff;
    // shared round-to-nearest/up/down/odd bitwise constants
    bool round_bypass;
    uint32_t round_mask;
    uint32_t round_tie;
    int round_shift;
    // for clip_normal_range_exponent
    SaturationMode saturation_mode;
    int max_exponent_store;
    int min_exponent_store;
    uint32_t max_num;
};

CUDA_HOST_DEVICE_INLINE SuperfpParams make_superfp_params(int man_bits, int exp_bits, int normal_binades,
                                                          int bias, SaturationMode saturation_mode)
{
    SuperfpParams p;
    p.man_bits = man_bits;
    p.bias = bias;

    SuperfpCutoffs cutoffs = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);
    p.normal_cutoff = cutoffs.normal_cutoff;
    p.supernormal_cutoff = cutoffs.supernormal_cutoff;

    RoundParams round_p = make_round_params(man_bits);
    p.round_bypass = round_p.bypass;
    p.round_mask = round_p.mask;
    p.round_tie = round_p.tie;
    p.round_shift = round_p.shift;

    NormalRangeParams normal_p = make_normal_range_params(exp_bits, man_bits, bias, saturation_mode);
    p.saturation_mode = normal_p.saturation_mode;
    p.max_exponent_store = normal_p.max_exponent_store;
    p.min_exponent_store = normal_p.min_exponent_store;
    p.max_num = normal_p.max_num;
    return p;
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_even(float origin_float, bool is_signed,
                                                         const SuperfpParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized{0.0f};

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;
    if (target_exp == 128)
    {
        // handle NaN/inf inputs
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_even(target);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-even between 0 and the smallest supernormal magnitude
        // (2^supernormal_cutoff): only the topmost underflow exponent
        // (supernormal_cutoff - 1) can be at or above the halfway point;
        // anything strictly below always flushes to (signed) zero.
        uint32_t sign_bit = target & 0x80000000u;
        quantize_bits = sign_bit;
        if (target_exp == p.supernormal_cutoff - 1 && (target & 0x007FFFFFu) != 0)
        {
            // strictly above half -> round up to the smallest supernormal;
            // exactly at half (mantissa all zero) ties to even = zero.
            quantize_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = (p.man_bits > 0)
                            ? round_bitwise_nearest_even(target, p.round_bypass, p.round_mask, p.round_tie, p.round_shift)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_away(float origin_float,
                                                         int man_bits, int exp_bits, int normal_binades,
                                                         int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_away(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // ties-to-away: bump to the smallest supernormal magnitude as soon
        // as the halfway point is reached (unlike nearest-even, the exact
        // tie also rounds away from zero).
        uint32_t sign_bit = target & 0x80000000u;
        if (target_exp >= co.supernormal_cutoff - 1)
        {
            quantize_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        else
        {
            quantize_bits = sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_nearest_away above.
CUDA_HOST_DEVICE_INLINE float cast_superfp_nearest_away(float origin_float, bool is_signed,
                                                         const SuperfpParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_nearest_away(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        uint32_t sign_bit = target & 0x80000000u;
        if (target_exp >= p.supernormal_cutoff - 1)
        {
            quantize_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
        }
        else
        {
            quantize_bits = sign_bit;
        }
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, p.round_bypass, p.round_mask, p.round_tie);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// rounds to odd
CUDA_HOST_DEVICE_INLINE float cast_superfp_odd(float origin_float,
                                               int man_bits, int exp_bits, int normal_binades,
                                               int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        // supernormal codepoints have no explicit mantissa bits, so
        // "oddness" refers to the parity of the supernormal grid index
        // itself (1 for the smallest magnitude, 2 for the next, ...), a
        // different reference frame than the float32 exponent field's
        // parity used by nearest-even, and different again from the
        // *target format's own* biased exponent parity used in the normal-
        // region man_bits==0 case below.
        int stored_index = target_exp - co.supernormal_cutoff + 1;
        uint32_t mask = 0x007FFFFFu;
        bool sticky = (target & mask) != 0;
        bool already_odd = (stored_index & 1) != 0;
        quantize_bits = target & ~mask;
        if (sticky && !already_odd)
            quantize_bits += (1u << 23);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // round to odd never flushes a nonzero input to zero; the smallest
        // supernormal codepoint's grid index is always 1 (odd), so a bare
        // bump (no sticky-driven adjustment) always yields an odd result.
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            uint32_t sign_bit = target & 0x80000000u;
            quantize_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }
    else
    {
        if (man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, man_bits);
        }
        else
        {
            // no explicit significand bits: oddness refers to the parity of
            // the target format's own biased exponent (target_exp + bias),
            // mirroring cast_binaryK_odd's man_bits==0 handling.
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (1u << 23);
        }
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_odd above.
CUDA_HOST_DEVICE_INLINE float cast_superfp_odd(float origin_float, bool is_signed, const SuperfpParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        int stored_index = target_exp - p.supernormal_cutoff + 1;
        uint32_t mask = 0x007FFFFFu;
        bool sticky = (target & mask) != 0;
        bool already_odd = (stored_index & 1) != 0;
        quantize_bits = target & ~mask;
        if (sticky && !already_odd)
            quantize_bits += (1u << 23);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            uint32_t sign_bit = target & 0x80000000u;
            quantize_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }
    else
    {
        if (p.man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, p.round_bypass, p.round_mask);
        }
        else
        {
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + p.bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (1u << 23);
        }
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// unsigned helper for cast_superfp_up/cast_superfp_down: assumes
// origin_float >= 0.
CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_up(float origin_float,
                                                        int man_bits, int exp_bits, int normal_binades,
                                                        int bias, SaturationMode saturation_mode)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_up(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // rounding up (toward +inf) never flushes a nonzero magnitude to
        // zero: any nonzero underflowing input bumps up to the smallest
        // supernormal magnitude.
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            quantize_bits = (uint32_t)(co.supernormal_cutoff + 127) << 23;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }
    else
    {
        quantize_bits = round_bitwise_up(target, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_absolute_up above.
CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_up(float origin_float, const SuperfpParams &p)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_up(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        if ((target & 0x7FFFFFFFu) == 0)
        {
            quantized = origin_float;
        }
        else
        {
            quantize_bits = (uint32_t)(p.supernormal_cutoff + 127) << 23;
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }
    else
    {
        quantize_bits = round_bitwise_up(target, p.round_bypass, p.round_mask);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_down(float origin_float,
                                                          int man_bits, int exp_bits, int normal_binades,
                                                          int bias, SaturationMode saturation_mode)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_down(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // rounding down (toward -inf, on a nonnegative magnitude) always
        // flushes to zero: there is no partial precision in this region to
        // create a boundary case worth preserving.
        quantize_bits = target & 0x80000000u;
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_down(target, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_absolute_down above.
CUDA_HOST_DEVICE_INLINE float cast_superfp_absolute_down(float origin_float, const SuperfpParams &p)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_down(target, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        quantize_bits = target & 0x80000000u;
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else
    {
        quantize_bits = round_bitwise_down(target, p.round_bypass, p.round_mask);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_up(float origin_float,
                                              int man_bits, int exp_bits, int normal_binades,
                                              int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    // NaN comparisons are always false, so `origin_float >= 0` alone would
    // route every NaN through the sign-flip dispatch below; unary negation
    // of a NaN is not guaranteed to preserve its exact bit pattern on every
    // backend (observed on CUDA, where -(-NaN) can canonicalize the
    // payload), so NaN/Inf must be passed through directly instead.
    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_up(origin_float, man_bits, exp_bits, normal_binades, bias,
                                        saturation_mode);
    else
        return -cast_superfp_absolute_down(-origin_float, man_bits, exp_bits, normal_binades, bias,
                                           saturation_mode);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_down(float origin_float,
                                                int man_bits, int exp_bits, int normal_binades,
                                                int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    // see cast_superfp_up for why NaN/Inf must be special-cased before the
    // sign-flip dispatch rather than relying on it to preserve the payload.
    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_down(origin_float, man_bits, exp_bits, normal_binades, bias,
                                          saturation_mode);
    else
        return -cast_superfp_absolute_up(-origin_float, man_bits, exp_bits, normal_binades, bias,
                                         saturation_mode);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_zero(float origin_float,
                                                int man_bits, int exp_bits, int normal_binades,
                                                int bias, bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float >= 0.0f)
        return cast_superfp_down(origin_float, man_bits, exp_bits, normal_binades, bias, is_signed,
                                 saturation_mode);
    else
        return cast_superfp_up(origin_float, man_bits, exp_bits, normal_binades, bias, is_signed,
                               saturation_mode);
}

// Precomputed-parameter overloads of cast_superfp_up/_down/_zero above.
CUDA_HOST_DEVICE_INLINE float cast_superfp_up(float origin_float, bool is_signed, const SuperfpParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    // see the *_bits-taking cast_superfp_up above for why NaN/Inf must be
    // special-cased before the sign-flip dispatch.
    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_up(origin_float, p);
    else
        return -cast_superfp_absolute_down(-origin_float, p);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_down(float origin_float, bool is_signed, const SuperfpParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target = FLOAT_TO_BITS(&origin_float);
    if ((int)((target >> 23) & 0xFF) - 127 == 128)
        return origin_float;

    if (origin_float >= 0)
        return cast_superfp_absolute_down(origin_float, p);
    else
        return -cast_superfp_absolute_up(-origin_float, p);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_zero(float origin_float, bool is_signed, const SuperfpParams &p)
{
    if (origin_float >= 0.0f)
        return cast_superfp_down(origin_float, is_signed, p);
    else
        return cast_superfp_up(origin_float, is_signed, p);
}

CUDA_HOST_DEVICE_INLINE float cast_superfp_stochastic(float origin_float, uint32_t rand_prob,
                                                       int rand_bits, int man_bits, int exp_bits,
                                                       int normal_binades, int bias, bool is_signed,
                                                       SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    SuperfpCutoffs co = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias);

    bool supernormal = (target_exp < co.normal_cutoff && target_exp >= co.supernormal_cutoff);
    bool underflow = target_exp < co.supernormal_cutoff;

    uint32_t rand_bits_raw = rand_prob & 0x007FFFFFu;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        // the supernormal grid has zero explicit precision (man_bits acts
        // as 0 here) regardless of the format's actual man_bits, so the
        // full 23 bits of randomness are used rather than the narrower
        // rand_bits-relative mask used in the normal region below.
        quantize_bits = round_bitwise_stochastic(target, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        // shift the value up by the smallest supernormal magnitude so it
        // lands in [smallest, 2*smallest), stochastically round it there
        // at 0-bit (whole power-of-two) precision, then shift back --
        // mirrors cast_binaryK_stochastic's subnormal shift trick.
        uint32_t sign_bit = target & 0x80000000u;
        uint32_t shift_bits = ((uint32_t)(co.supernormal_cutoff + 127) << 23) | sign_bit;
        float shift_float = BITS_TO_FLOAT(&shift_bits);
        float val = origin_float + shift_float;
        uint32_t target2 = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target2, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else
    {
        uint32_t rand_prob_man = rand_bits_raw & ~((1u << (23 - man_bits - rand_bits)) - 1u);
        quantize_bits = round_bitwise_stochastic(target, rand_prob_man, man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, exp_bits, man_bits,
                                                   bias, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Precomputed-parameter overload of cast_superfp_stochastic. Like
// cast_binaryK_stochastic's precomputed overload, rand_prob/rand_bits stay
// per-call arguments -- only the region-cutoff and round-to-nearest-even
// constants precompute away.
CUDA_HOST_DEVICE_INLINE float cast_superfp_stochastic(float origin_float, uint32_t rand_prob, int rand_bits,
                                                       bool is_signed, const SuperfpParams &p)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;

    bool supernormal = (target_exp < p.normal_cutoff && target_exp >= p.supernormal_cutoff);
    bool underflow = target_exp < p.supernormal_cutoff;

    uint32_t rand_bits_raw = rand_prob & 0x007FFFFFu;

    if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else if (supernormal)
    {
        quantize_bits = round_bitwise_stochastic(target, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    else if (underflow)
    {
        uint32_t sign_bit = target & 0x80000000u;
        uint32_t shift_bits = ((uint32_t)(p.supernormal_cutoff + 127) << 23) | sign_bit;
        float shift_float = BITS_TO_FLOAT(&shift_bits);
        float val = origin_float + shift_float;
        uint32_t target2 = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target2, rand_bits_raw, 0);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else
    {
        uint32_t rand_prob_man = rand_bits_raw & ~((1u << (23 - p.man_bits - rand_bits)) - 1u);
        quantize_bits = round_bitwise_stochastic(target, rand_prob_man, p.man_bits);
        quantize_bits = clip_normal_range_exponent(target, quantize_bits, p.saturation_mode, p.max_exponent_store,
                                                   p.min_exponent_store, p.max_num);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}
