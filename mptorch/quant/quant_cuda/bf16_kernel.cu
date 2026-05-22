#include "bit_helper_16.cu"
#include "modes.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __nv_bfloat16 cast_bf16_nearest_even(__nv_bfloat16 origin, int man_bits, int exp_bits,
                                       int bias, bool saturate, SubnormalsMode subnormals)
{
    uint16_t target = BFLOAT16_TO_BITS(origin);
    uint16_t quantize_bits;
    __nv_bfloat16 quantized;

    int target_exp = ((target & 0x7FFF) >> 7) - 127;
    int min_exp = -(bias - 1);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 7) && (exp_bits >= 8);

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
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target & 0x007F) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest_even_bf16(target, exp_diff);
            quantize_bits = clip_subnormal_range_exponent_bf16(exp_bits, man_bits, bias, target, quantize_bits);
            quantized = BITS_TO_BFLOAT16(quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 128)
        {
            quantized = origin;
        }
        else
        {
            quantize_bits = round_bitwise_nearest_even_bf16(target, man_bits);
            quantize_bits = clip_normal_range_exponent_bf16(exp_bits, man_bits, bias, target, quantize_bits,
                                                            saturate, subnormals == SubnormalsMode::EXTENDED_NORMALS);
            quantized = BITS_TO_BFLOAT16(quantize_bits);
        }
    }

    return quantized;
}

// Vectorized Kernel
__global__ void bfloat16_kernel_nearest_even_packed(
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

        __nv_bfloat16 *in_halfs = reinterpret_cast<__nv_bfloat16 *>(&in_val);
        __nv_bfloat16 *out_halfs = reinterpret_cast<__nv_bfloat16 *>(&out_val);

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            out_halfs[i] = cast_bf16_nearest_even(in_halfs[i], man_bits, exp_bits, bias, saturate, subnormals);
        }

        o_packed[index] = out_val;
    }
}

__global__ void bfloat16_kernel_nearest_even_scalar(
    const __nv_bfloat16 *__restrict__ a,
    __nv_bfloat16 *__restrict__ o,
    int size,
    int man_bits, int exp_bits,
    bool saturate,
    SubnormalsMode subnormals)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int bias = (1 << (exp_bits - 1)) - 1;
    if (index < size)
    {
        o[index] = cast_bf16_nearest_even(a[index], man_bits, exp_bits, bias, saturate, subnormals);
    }
}
