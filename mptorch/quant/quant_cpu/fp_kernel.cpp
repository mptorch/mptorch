#include "bit_helper.h"
#include "fp_kernel.h"
#include <random>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

float cast_fp_nearest_even(float origin_float,
                           int man_bits, int exp_bits,
                           int bias,
                           bool saturate,
                           SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -(bias - 1);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
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
            quantize_bits = clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                                       saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

float cast_fp_nearest_away(float origin_float,
                           int man_bits, int exp_bits, int bias,
                           bool saturate,
                           SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -(bias - 1);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest_away(target, exp_diff);
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
            quantize_bits = round_bitwise_nearest_away(target, man_bits);
            quantize_bits =
                clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                           saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

float cast_absolute_up(float origin_float,
                       int man_bits, int exp_bits, int bias,
                       bool saturate,
                       SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
            quantize_bits =
                clip_subnormal_range_exponent_up(exp_bits, man_bits, bias, target, quantize_bits);
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
            quantize_bits =
                clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                           saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

float cast_absolute_down(float origin_float,
                         int man_bits, int exp_bits, int bias,
                         bool saturate,
                         SubnormalsMode subnormals)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1;
            quantize_bits = not_uflow * round_bitwise_down(target, exp_diff);
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
            quantize_bits = round_bitwise_down(target, man_bits);
            quantize_bits =
                clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                           saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

float cast_fp_up(float origin_float,
                 int man_bits, int exp_bits, int bias,
                 bool saturate,
                 SubnormalsMode subnormals)
{
    if (origin_float >= 0.0f)
        return cast_absolute_up(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
    else
        return -cast_absolute_down(-origin_float, man_bits, exp_bits, bias, saturate, subnormals);
}

float cast_fp_down(float origin_float,
                   int man_bits, int exp_bits, int bias,
                   bool saturate,
                   SubnormalsMode subnormals)
{
    if (origin_float >= 0.0f)
        return cast_absolute_down(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
    else
        return -cast_absolute_up(-origin_float, man_bits, exp_bits, bias, saturate, subnormals);
}

float cast_fp_zero(float origin_float,
                   int man_bits, int exp_bits, int bias,
                   bool saturate,
                   SubnormalsMode subnormals)
{
    if (origin_float >= 0.0f)
        return cast_fp_down(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
    else
        return cast_fp_up(origin_float, man_bits, exp_bits, bias, saturate, subnormals);
}

float cast_fp_stochastic(float origin_float,
                         int man_bits, int exp_bits, int bias,
                         bool saturate,
                         SubnormalsMode subnormals)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0);

    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t rand_prob = (dis(gen)) & mask;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -(bias - 1);
    bool subnormal = (target_exp < min_exp);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        float shift_float, val;
        int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val = origin_float + shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else
    {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantize_bits =
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                       saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

float cast_fp_stochastic(float origin_float,
                         int man_bits, int exp_bits, int rand_bits, int bias,
                         bool saturate,
                         SubnormalsMode subnormals)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0);

    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t rand_prob = (dis(gen)) & mask;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -(bias - 1);
    bool subnormal = (target_exp < min_exp);

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~((1 << (23 - man_bits - rand_bits)) - 1);

    if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
    {
        float shift_float, val;
        int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val = origin_float + shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else
    {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantize_bits =
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits,
                                       saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}