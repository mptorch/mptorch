#include "bit_helper_16.cu"
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ __nv_bfloat16 cast_bf16_nearest_even(__nv_bfloat16 origin, int man_bits, int exp_bits)
{
    uint16_t target = BFLOAT16_TO_BITS(origin);

    // Perform bit masking and exponent clipping on the 16-bit representation
    uint16_t quantize_bits = round_bitwise_nearest_even_bf16(target, man_bits);

    // (Add exponent clipping logic similar to fp32 but scaled for bf16)

    return BITS_TO_BFLOAT16(quantize_bits);
}

// Vectorized Kernel
__global__ void bfloat16_kernel_nearest_even_packed(
    const float4 *__restrict__ a_packed,
    float4 *__restrict__ o_packed,
    int packed_size,
    int man_bits, int exp_bits)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < packed_size)
    {
        float4 in_val = a_packed[index];
        float4 out_val;

        __nv_bfloat16 *in_halfs = reinterpret_cast<__nv_bfloat16 *>(&in_val);
        __nv_bfloat16 *out_halfs = reinterpret_cast<__nv_bfloat16 *>(&out_val);

#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            out_halfs[i] = cast_bf16_nearest_even(in_halfs[i], man_bits, exp_bits);
        }

        o_packed[index] = out_val;
    }
}

__global__ void bfloat16_kernel_nearest_even_scalar(
    const __nv_bfloat16 *__restrict__ a,
    __nv_bfloat16 *__restrict__ o,
    int size,
    int man_bits, int exp_bits)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size)
    {
        o[index] = cast_bf16_nearest_even(a[index], man_bits, exp_bits);
    }
}
