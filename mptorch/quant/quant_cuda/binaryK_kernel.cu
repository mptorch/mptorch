#include "quant_kernel.h"
#include "bit_helper.cu"
#include "binaryK_kernel.h"
#include "softmax_kernel.h"
#include "layernorm_kernel.h"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

__host__ __device__ float cast_binaryK_nearest_even(float origin_float,
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
        uint32_t rounded_val = (man_bits > 0) ? round_bitwise_nearest_even(target, exp_diff) : round_bitwise_nearest_even(target);
        quantize_bits = not_uflow * rounded_val;
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
        quantize_bits = (man_bits > 0) ? round_bitwise_nearest_even(target, man_bits) : round_bitwise_nearest_even(target);
        quantize_bits =
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

__host__ __device__ float cast_binaryK_nearest_away(float origin_float,
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
        int not_uflow = exp_diff > -1 || (exp_diff == -1);
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

__host__ __device__ __inline__ float cast_absolute_up(float origin_float, int man_bits, int exp_bits,
                                                      int bias,
                                                      SaturationMode saturation_mode)
{
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
        // int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
        quantize_bits = round_bitwise_up(target, exp_diff < 0 ? 0 : exp_diff);
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
            clip_normal_range_exponent(exp_bits, man_bits, bias, target, quantize_bits, saturation_mode);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

__host__ __device__ __inline__ float cast_absolute_down(float origin_float, int man_bits, int exp_bits,
                                                        int bias,
                                                        SaturationMode saturation_mode)
{
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

__host__ __device__ float cast_binaryK_up(float origin_float,
                                          int man_bits, int exp_bits,
                                          int bias,
                                          bool is_signed,
                                          SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_up(origin_float, man_bits, exp_bits, bias, saturation_mode);
    else
        return -cast_absolute_down(-origin_float, man_bits, exp_bits, bias, saturation_mode);
}

__host__ __device__ float cast_binaryK_down(float origin_float,
                                            int man_bits, int exp_bits,
                                            int bias,
                                            bool is_signed,
                                            SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

    if (origin_float >= 0)
        return cast_absolute_down(origin_float, man_bits, exp_bits, bias, saturation_mode);
    else
        return -cast_absolute_up(-origin_float, man_bits, exp_bits, bias, saturation_mode);
}

__host__ __device__ float cast_binaryK_zero(float origin_float,
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

__host__ __device__ float cast_binaryK_stochastic(float origin_float, uint32_t rand_prob,
                                                  int rand_bits, int man_bits, int exp_bits, int bias,
                                                  bool is_signed, SaturationMode saturation_mode)
{
    if (origin_float < 0.0f && !is_signed)
        return 0.0f;

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

__global__ void binaryK_kernel_nearest_even(
    float *__restrict__ a, float *o, int size,
    int man_bits, int exp_bits, int bias, bool is_signed, SaturationMode saturation_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = cast_binaryK_nearest_even(a[idx], man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}

__global__ void binaryK_kernel_nearest_away(
    float *__restrict__ a, float *o, int size,
    int man_bits, int exp_bits, int bias, bool is_signed, SaturationMode saturation_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = cast_binaryK_nearest_away(a[idx], man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}

__global__ void binaryK_kernel_up(
    float *__restrict__ a, float *o, int size,
    int man_bits, int exp_bits, int bias, bool is_signed, SaturationMode saturation_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = cast_binaryK_up(a[idx], man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}

__global__ void binaryK_kernel_down(
    float *__restrict__ a, float *o, int size,
    int man_bits, int exp_bits, int bias, bool is_signed, SaturationMode saturation_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = cast_binaryK_down(a[idx], man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}

__global__ void binaryK_kernel_zero(
    float *__restrict__ a, float *o, int size,
    int man_bits, int exp_bits, int bias, bool is_signed, SaturationMode saturation_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = cast_binaryK_zero(a[idx], man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}

__global__ void binaryK_kernel_stochastic(
    float *__restrict__ a, int *__restrict__ r, float *o, int size,
    int man_bits, int exp_bits, int bias, int prng_bits, bool is_signed, SaturationMode saturation_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = cast_binaryK_stochastic(a[idx], (uint32_t)r[idx], prng_bits, man_bits, exp_bits, bias, is_signed, saturation_mode);
    }
}