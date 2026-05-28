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

template <class Quant>
__global__ void quant_kernel(float *__restrict__ a, float *o, int size, Quant quant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = quant(a[idx]);
    }
}

template <class Quant>
__global__ void quant_kernel(float *__restrict__ a, int *__restrict__ r, float *o, int size, Quant quant)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        o[idx] = quant(a[idx], (uint32_t)(r[idx]));
    }
}