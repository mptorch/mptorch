#include <iostream>
#include <cmath>
#include <set>
#include <cstdint>
#include <random>
#include <stdfloat>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

#define NC "\e[0m"
#define RED "\e[0;31m"
#define GRN "\e[0;32m"
#define CYN "\e[0;36m"
#define REDB "\e[41m"

using std::cout;
using std::endl;

enum class SaturationMode
{
    SAT_FINITE,
    SAT_PROPAGATE,
    OVF_INF
};

auto print_uint32 = [](uint32_t u)
{
    cout << CYN << (u >> 31) << " ";
    for (int i{1}; i < 9; ++i)
        cout << GRN << ((u << i) >> 31);
    cout << " ";
    for (int i{9}; i < 32; ++i)
        cout << RED << ((u << i) >> 31);
    cout << NC << endl;
};

auto print_float = [](float x)
{
    uint32_t u = FLOAT_TO_BITS(&x);
    print_uint32(u);
};

uint32_t round_bitwise_nearest_even(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    int shift_value = man_bits == 0 ? 1 << (23 - man_bits + offset) : 1 << std::min<int>((23 - man_bits + offset), 23);
    return add_r & ~(shift_value - 1) + offset * (man_bits == 0) * (machine_eps << 1);
}

uint32_t round_bitwise_nearest_away(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 0x7FFFFFFF & (1 << (22 - man_bits));
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    int shift_value = man_bits == 0 ? 1 << (23 - man_bits + offset) : 1 << std::min<int>((23 - man_bits + offset), 23);
    return (add_r & ~(shift_value - 1)) + offset * (man_bits > -1) * (machine_eps << 1);
}

uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
{
    uint32_t mask = (1 << (23 - man_bits)) - 1;

    // add masked random bits to an unmasked target
    uint32_t add_r = target + (rand_prob & mask);

    // mask out bits on the right hand side of the least significant bit
    uint32_t quantized = add_r & ~mask;
    return quantized;
}

// rounds up, towards positive infinity
uint32_t round_bitwise_up(uint32_t target, int man_bits)
{
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
    uint32_t sign = target >> 31;
    uint32_t rand_prob = (nexact & ~sign) << (23 - man_bits);
    uint32_t add_r = target + rand_prob;
    uint32_t quantized = add_r & ~mask;
    return quantized;
}

// rounds down, towards negative infinity
uint32_t round_bitwise_down(uint32_t target, int man_bits)
{
    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t nexact = ((target << 1 >> 1) & mask) > 0u ? 1u : 0u;
    uint32_t sign = target >> 31;
    uint32_t rand_prob = (nexact & sign) << (23 - man_bits);
    uint32_t add_r = target + rand_prob;
    uint32_t quantized = add_r & ~mask;
    return quantized;
}

// clips the exponent of a binary8 float value
uint32_t clip_exponent(int exp_bits, int man_bits, int bias, uint32_t old_num,
                       uint32_t quantized_num, bool saturate)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int max_exponent_store = bias + 127;
    int min_exponent_store = -(bias - 1) + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // handle overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        if (saturate)
        {
            uint32_t max_man =
                (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
            uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
            quantized_num = old_sign | max_num;
        }
        else
        {
            quantized_num = old_sign | 0x7f800000;
        }
        // handle underflow
    }
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t min_num = ((uint32_t)min_exponent_store << 23);
        uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
        uint32_t unsigned_quantized_num = quantized_num << 1 >> 1;
        if (unsigned_quantized_num > middle_num)
        {
            uint32_t old_sign = old_num >> 31 << 31;
            quantized_num = old_sign | min_num;
        }
        else
        {
            quantized_num = 0;
        }
    }
    return quantized_num;
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

float cast_binaryK_nearest_away(float origin_float,
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
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) >= 0));
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
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

float cast_binaryK_up(float origin_float,
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
        quantize_bits = not_uflow * round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
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
        quantize_bits = round_bitwise_up(target, man_bits);
        quantize_bits =
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

float cast_binaryK_down(float origin_float,
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
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

float cast_binaryK_zero(float origin_float,
                        int man_bits, int exp_bits,
                        int bias,
                        bool is_signed,
                        SaturationMode saturation_mode)
{
    if (origin_float >= 0.0f)
        return cast_binaryK_down(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode);
    else
        return cast_binaryK_up(origin_float, man_bits, exp_bits, bias, is_signed, saturation_mode);
}

float cast_binaryK_stochastic(float origin_float,
                              int man_bits, int exp_bits,
                              int rand_bits, int bias,
                              bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0);

    uint32_t mask = (1 << (23 - man_bits)) - 1;
    uint32_t rand_prob = (dis(gen)) & mask;

    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -bias + 1;
    bool subnormal = (target_exp < min_exp);

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - man_bits - rand_bits) - 1);

    if (subnormal)
    {
        float shift_float, val;
        int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
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
        quantize_bits =
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

int main()
{
    SaturationMode sat_mode = SaturationMode::OVF_INF;
    bool is_signed = true;
    int man_bits = 2;
    int exp_bits = 4;
    int bias = 1 << (exp_bits - 1);                // for signed
    std::float32_t fval = 0.00097656250000000000f; // 0.00292968750000000000f; // -powf(2.0f, -5) * (1 + 1.0 / 2 + 1.0 / 8);
    std::float32_t qval1 = cast_binaryK_nearest_even(fval, man_bits, exp_bits, bias, is_signed, sat_mode);
    std::float32_t qval2 = cast_binaryK_nearest_away(fval, man_bits, exp_bits, bias, is_signed, sat_mode);
    std::float32_t qval3 = cast_binaryK_stochastic(fval, man_bits, exp_bits, 23 - man_bits, bias, is_signed, sat_mode);
    std::cout << "fval      = " << fval << std::endl;
    print_float(fval);
    std::cout << "qval_even = " << qval1 << std::endl;
    print_float(qval1);
    std::cout << "qval_away = " << qval2 << std::endl;
    print_float(qval2);
    std::cout << "qval_rand = " << qval3 << std::endl;
    print_float(qval3);
}
