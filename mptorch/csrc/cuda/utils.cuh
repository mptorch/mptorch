#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>

template <class RandType>
__device__ __forceinline__ RandType gen_rand(curandState_t *state, int sidx)
{
    return curand(&state[sidx]);
}

template <>
__device__ __forceinline__ float gen_rand<float>(curandState_t *state,
                                                 int sidx)
{
    return 1.0f - curand_uniform(&state[sidx]);
}

template <typename scalar_t, class Quant>
__global__ void quant_kernel_vec(scalar_t *__restrict__ a, scalar_t *o, int vec_size, Quant quant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size)
    {
        reinterpret_cast<float4*>(o)[idx] = quant.vec(reinterpret_cast<const float4*>(a)[idx]);
    }
}

template <typename scalar_t, class Quant>
__global__ void quant_kernel_vec_sr(scalar_t *__restrict__ a, int *__restrict__ r, scalar_t *o, int vec_size, Quant quant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vec_size)
    {
        int offset = idx * (sizeof(float4) / sizeof(scalar_t));
        reinterpret_cast<float4*>(o)[idx] = quant.vec_sr(reinterpret_cast<const float4*>(a)[idx], r + offset);
    }
}

template <typename scalar_t, class Quant>
__global__ void quant_kernel_rem(scalar_t *__restrict__ a, scalar_t *o, int rem_size, Quant quant)
{
    int idx = threadIdx.x; // only 1 block launched
    if (idx < rem_size)
    {
        o[idx] = quant.scalar(a[idx]);
    }
}

template <typename scalar_t, class Quant>
__global__ void quant_kernel_rem_sr(scalar_t *__restrict__ a, int *__restrict__ r, scalar_t *o, int rem_size, Quant quant)
{
    int idx = threadIdx.x;
    if (idx < rem_size)
    {
        o[idx] = quant.scalar_sr(a[idx], (uint32_t)r[idx]);
    }
}

// Fallback scalar kernels for unsupported types or sizes
template <typename scalar_t, class Quant>
__global__ void quant_kernel(scalar_t *__restrict__ a, scalar_t *o, int size, Quant quant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = quant.scalar(a[idx]);
    }
}

template <typename scalar_t, class Quant>
__global__ void quant_kernel_sr(scalar_t *__restrict__ a, int *__restrict__ r, scalar_t *o, int size, Quant quant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = quant.scalar_sr(a[idx], (uint32_t)r[idx]);
    }
}