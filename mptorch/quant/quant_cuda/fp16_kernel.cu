#include "bit_helper_16.cu"
#include "modes.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __half cast_fp16_nearest_even(__half origin, int man_bits, int exp_bits,
                                       int bias, bool saturate, SubnormalsMode subnormals)
{
    uint16_t target = HALF_TO_BITS(origin);
    uint16_t quantize_bits;
    __half quantized;

    int target_exp = ((target & 0x7FFF) >> 10) - 15;
    int min_exp = -(bias - 1);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 10) && (exp_bits >= 5);

    if (noquantize)
    {
        quantized = origin;
    }
    else
    {
        // handle subnormal inputs
        if (subnormal && (subnormals == SubnormalsMode::SUBNORMALS))
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target & 0x03FF) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest_even_fp16(target, exp_diff);
            quantize_bits = clip_subnormal_range_exponent_fp16(exp_bits, man_bits, bias, target, quantize_bits);
            quantized = BITS_TO_HALF(quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 16)
        {
            quantized = origin;
        }
        else
        {
            quantize_bits = round_bitwise_nearest_even_fp16(target, man_bits);
            quantize_bits = clip_normal_range_exponent_fp16(exp_bits, man_bits, bias, target, quantize_bits,
                                                            saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
            quantized = BITS_TO_HALF(quantize_bits);
        }
    }

    return quantized;
}

// Vectorized Kernel
__global__ void fp16_kernel_nearest_even_packed(
    const float4 *__restrict__ a_packed,
    float4 *__restrict__ o_packed,
    int packed_size,
    int man_bits, int exp_bits,
    bool saturate,
    SubnormalsMode subnormals)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bias = (1 << (exp_bits - 1)) - 1;
    if (index < packed_size)
    {
        float4 in_val = a_packed[index];
        float4 out_val;

        __half *in_halfs = reinterpret_cast<__half *>(&in_val);
        __half *out_halfs = reinterpret_cast<__half *>(&out_val);

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            out_halfs[i] = cast_fp16_nearest_even(in_halfs[i], man_bits, exp_bits, bias, saturate, subnormals);
        }

        o_packed[index] = out_val;
    }
}

__global__ void fp16_kernel_nearest_even_scalar(
    const __half *__restrict__ a,
    __half *__restrict__ o,
    int size,
    int man_bits, int exp_bits,
    bool saturate,
    SubnormalsMode subnormals)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bias = (1 << (exp_bits - 1)) - 1;
    if (index < size)
    {
        o[index] = cast_fp16_nearest_even(a[index], man_bits, exp_bits, bias, saturate, subnormals);
    }
}
