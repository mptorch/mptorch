#include "bit_helper.h"
#include "binaryK.h"
#include "softmax.h"
#include "quant.h"
#include "layernorm.h"
#include <ATen/ATen.h>
#include <cmath>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

uint32_t round_bitwise_nearest_even(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    int shift_value = man_bits == 0 ? 1 << (23 - man_bits + offset) : 1 << std::min<int>((23 - man_bits + offset), 23);
    return add_r & ~(shift_value - 1);
}

uint32_t round_bitwise_nearest_away(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    int shift_value = man_bits == 0 ? 1 << (23 - man_bits + offset) : 1 << std::min<int>((23 - man_bits + offset), 23);
    return (add_r & ~(shift_value - 1)) + offset * (machine_eps << 1);
}

uint32_t clip_subnormal_range_exponent(int exp_bits, int man_bits, int bias,
                                       uint32_t old_num, uint32_t quantized_num)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int min_exponent_store = -(bias - 1) - man_bits + 127;

    uint32_t old_sign = old_num >> 31 << 31;
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

uint32_t clip_normal_range_exponent(int exp_bits, int man_bits, int bias,
                                    uint32_t old_num, uint32_t quantized_num,
                                    SaturationMode saturation_mode)
{
    if (quantized_num == 0)
        return quantized_num;

    uint32_t sign = old_num >> 31 << 31;
    if ((quantized_num == 0x7F800000 && saturation_mode != SaturationMode::SAT_FINITE) || (quantized_num > 0x7F800000))
        return sign | quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int max_exponent_store = (bias - 1) + 126 + (man_bits > 1);
    int min_exponent_store = -(bias - 1) + 127;
    int finite = (saturation_mode == SaturationMode::SAT_FINITE);

    uint32_t max_man = ((0x007FFFFF >> (23 - man_bits)) - 1 + finite) << (23 - man_bits);
    uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;

    // handle overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        switch (saturation_mode)
        {
        case SaturationMode::SAT_FINITE:
            quantized_num = sign | max_num;
            break;

        case SaturationMode::SAT_PROPAGATE:
            quantized_num = sign | max_num;
            break;

        default:
            quantized_num = sign | 0x7F800000;
            break;
        }
    }
    else if (quantized_exponent_store == max_exponent_store)
    {
        // handle overflow
        if (quantized_num > max_num && saturation_mode == SaturationMode::OVF_INF)
            quantized_num = sign | 0x7F800000;
    }
    // handle underflow
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= sign;
    }

    return quantized_num;
}

float cast_binaryK_nearest_even(float origin_float,
                                int man_bits, int exp_bits,
                                int bias,
                                bool is_signed,
                                SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    // handle subnormal inputs (if subnormal mode is active)
    if (subnormal)
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
        quantize_bits = not_uflow * round_bitwise_nearest_even(target, exp_diff);
        quantize_bits =
            clip_subnormal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits);
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
        quantize_bits = round_bitwise_nearest_even(target, man_bits);
        quantize_bits =
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

void binaryK_kernel_nearest_even(float *a, float *o, int size, int K, int P, int bias, bool is_signed, SaturationMode saturation_mode)
{
    int man_bits, exp_bits;
    if (is_signed)
    {
        man_bits = P - 1;
        exp_bits = K - P;
    }
    else
    {
        man_bits = P - 1;
        exp_bits = K - P + 1;
    }

    for (int idx = 0; idx < size; ++idx)
    {
        o[idx] = cast_binaryK_nearest_even(a[idx], man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}