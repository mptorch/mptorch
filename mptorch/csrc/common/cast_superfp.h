#pragma once

#include "bit_helper.h"
#include "modes.h"

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
    int normal_cutoff = ((1 << exp_bits) - 1 - bias) - normal_binades + 1;
    int supernormal_cutoff = normal_cutoff - ((1 << exp_bits) - normal_binades) * (1 << man_bits) + 1;

    bool supernormal = (target_exp < normal_cutoff && target_exp >= supernormal_cutoff);
    bool underflow = target_exp < supernormal_cutoff;
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
        quantized = 0.0f;
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