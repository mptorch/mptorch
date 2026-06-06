#pragma once

#include <cstdint>
#include <cuda_runtime.h>

template <typename scalar_t, class Quant>
__global__ __launch_bounds__(256, 2)
void quant_kernel_all(scalar_t *__restrict__ a,
                      int *__restrict__ r,
                      scalar_t *o,
                      int size,
                      Quant quant)
{
    constexpr int vec_elems = SIMDTraits<scalar_t>::vec_elems;
    int vec_size = size / vec_elems;
    int rem_size = size % vec_elems;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < vec_size; i += stride)
    {
        // Force 128-bit load using int4 to avoid any aliasing issues with c10::Half/BFloat16
        const int4* a_vec = reinterpret_cast<const int4*>(a);
        int4 in_i4 = a_vec[i];
        float4 in = reinterpret_cast<const float4&>(in_i4);
        
        float4 out = (r == nullptr) ? quant.vec(in) : quant.vec_sr(in, r + i * (sizeof(float4) / sizeof(scalar_t)));
        
        int4 out_i4 = reinterpret_cast<const int4&>(out);
        reinterpret_cast<int4*>(o)[i] = out_i4;
    }

    if (idx < rem_size)
    {
        int rem_idx = vec_size * vec_elems + idx;
        if (r == nullptr)
        {
            o[rem_idx] = quant.scalar(a[rem_idx]);
        }
        else
        {
            o[rem_idx] = quant.scalar_sr(a[rem_idx], static_cast<uint32_t>(r[rem_idx]));
        }
    }
}