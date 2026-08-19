#pragma once

#include "bit_helper.h"
#include "modes.h"

CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_even(float origin_float, int man_bits, int exp_bits,
                                                        int bias, bool is_signed,
                                                        SaturationMode saturation_mode,
                                                        SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
        uint32_t rounded_val = (man_bits > 0)
                                   ? round_bitwise_nearest_even(target, exp_diff)
                                   : round_bitwise_nearest_even(target);
        quantize_bits = not_uflow * rounded_val;
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits,
                                                      exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = (man_bits > 0)
                            ? round_bitwise_nearest_even(target, man_bits)
                            : round_bitwise_nearest_even(target);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_nearest_away(float origin_float, int man_bits, int exp_bits,
                                                        int bias, bool is_signed,
                                                        SaturationMode saturation_mode,
                                                        SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff >= -1;
        quantize_bits = not_uflow * round_bitwise_nearest_away(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits,
                                                      exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = round_bitwise_nearest_away(target, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// rounds to odd
CUDA_HOST_DEVICE_INLINE float cast_binaryK_odd(float origin_float, int man_bits, int exp_bits,
                                               int bias, bool is_signed,
                                               SaturationMode saturation_mode,
                                               SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        // round to odd never flushes a nonzero input to zero (like round-up
        // in magnitude), so the "_up" subnormal clip is used to clamp
        // underflowing nonzero values to the smallest subnormal rather than
        // to zero
        int exp_diff = man_bits - (min_exp - target_exp);
        if (exp_diff > 0)
        {
            // exp_diff explicit subnormal fraction bits remain: these map
            // directly onto real bits of the origin value, so the generic
            // bitwise formula applies unchanged
            quantize_bits = round_bitwise_odd(target, exp_diff);
        }
        else
        {
            // effective precision has collapsed to zero: every value in
            // this range rounds down to exactly the smallest subnormal,
            // whose significand is always 1 (odd), so no sticky-driven
            // carry is ever needed here
            quantize_bits = target & ~0x007FFFFFu;
        }
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits,
                                                          exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        if (man_bits > 0)
        {
            quantize_bits = round_bitwise_odd(target, man_bits);
        }
        else
        {
            // no explicit significand bits: representable values are exact
            // powers of two, and "odd" refers to the parity of the target
            // format's own biased exponent (not the float32 exponent field,
            // whose bias generally differs in parity from the target's)
            uint32_t mask = 0x007FFFFFu;
            bool sticky = (target & mask) != 0;
            bool already_odd = ((target_exp + bias) & 1) != 0;
            quantize_bits = target & ~mask;
            if (sticky && !already_odd)
                quantize_bits += (1u << 23);
        }
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_absolute_up(float origin_float, int man_bits, int exp_bits, int bias,
                                               SaturationMode saturation_mode,
                                               SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        quantize_bits = round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
        quantize_bits = clip_subnormal_range_exponent_up(target, quantize_bits,
                                                         exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = round_bitwise_up(target, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_absolute_down(float origin_float, int man_bits, int exp_bits,
                                                 int bias, SaturationMode saturation_mode,
                                                 SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal && subnormals == SubnormalsMode::SUBNORMALS)
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff > -1;
        quantize_bits = not_uflow * round_bitwise_down(target, exp_diff);
        quantize_bits = clip_subnormal_range_exponent(target, quantize_bits,
                                                      exp_bits, man_bits, bias);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    // normal value range or overflow
    else
    {
        quantize_bits = round_bitwise_down(target, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_up(float origin_float, int man_bits, int exp_bits, int bias,
                                              bool is_signed, SaturationMode saturation_mode,
                                              SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_up(origin_float, man_bits, exp_bits, bias,
                                saturation_mode, subnormals);
    else
        return -cast_absolute_down(-origin_float, man_bits, exp_bits, bias,
                                   saturation_mode, subnormals);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_down(float origin_float, int man_bits, int exp_bits,
                                                int bias, bool is_signed,
                                                SaturationMode saturation_mode,
                                                SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_down(origin_float, man_bits, exp_bits, bias,
                                  saturation_mode, subnormals);
    else
        return -cast_absolute_up(-origin_float, man_bits, exp_bits, bias,
                                 saturation_mode, subnormals);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_zero(float origin_float, int man_bits, int exp_bits,
                                                int bias, bool is_signed,
                                                SaturationMode saturation_mode,
                                                SubnormalsMode subnormals)
{
    if (origin_float >= 0.0f)
        return cast_binaryK_down(origin_float, man_bits, exp_bits, bias, is_signed,
                                 saturation_mode, subnormals);
    else
        return cast_binaryK_up(origin_float, man_bits, exp_bits, bias, is_signed,
                               saturation_mode, subnormals);
}

CUDA_HOST_DEVICE_INLINE float cast_binaryK_stochastic(float origin_float, uint32_t rand_prob,
                                                      int rand_bits, int man_bits, int exp_bits, int bias,
                                                      bool is_signed, SaturationMode saturation_mode,
                                                      SubnormalsMode subnormals)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (int)((target >> 23) & 0xFF) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    rand_prob = rand_prob & 0x007FFFFFu;
    rand_prob = rand_prob & ~((1u << (23 - man_bits - rand_bits)) - 1u);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        float shift_float, val;
        int shift_bits = ((127 + min_exp) << 23) | (target & 0x80000000u);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val = origin_float + shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    // handle NaN/inf inputs
    else if (target_exp == 128)
    {
        quantized = origin_float;
    }
    else
    {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantize_bits = clip_normal_range_exponent(
            target, quantize_bits, exp_bits, man_bits, bias, saturation_mode,
            subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}