#pragma once

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include "simd_traits.cuh"

// Blocks of `block` threads needed to cover `work` items. grid.x tops out at
// 2^31 - 1; the loop below is grid-stride, so a grid clamped there still
// covers a larger tensor, in more iterations rather than incorrectly.
inline int grid_for(int64_t work, int block)
{
    constexpr int64_t max_grid = 2147483647;
    int64_t grid = (work + block - 1) / block;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(grid, max_grid)));
}

// Indices are 64-bit: `size` comes straight from Tensor::numel(), which an int
// would truncate past 2^31 elements. The kernel is memory bound on every
// tensor large enough for the wider arithmetic to be reachable, and measures
// the same as the 32-bit version on ones small enough to sit in L2, so there
// is no separate narrow-index path.
template <typename scalar_t, class Quant>
__global__ __launch_bounds__(256, 2)
void quant_kernel_all(const scalar_t *__restrict__ a,
                      const int *__restrict__ r,
                      scalar_t *o,
                      int64_t size,
                      Quant quant)
{
    constexpr int vec_elems = SIMDTraits<scalar_t>::vec_elems;
    int64_t vec_size = size / vec_elems;
    int64_t rem_size = size % vec_elems;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < vec_size; i += stride)
    {
        // Force 128-bit load using int4 to avoid any aliasing issues with c10::Half/BFloat16
        const int4* a_vec = reinterpret_cast<const int4*>(a);
        int4 in_i4 = a_vec[i];
        float4 in = reinterpret_cast<const float4&>(in_i4);
        
        float4 out = (r == nullptr) ? quant.vec(in) : quant.vec_sr(in, r + i * vec_elems);
        
        int4 out_i4 = reinterpret_cast<const int4&>(out);
        reinterpret_cast<int4*>(o)[i] = out_i4;
    }

    if (idx < rem_size)
    {
        int64_t rem_idx = vec_size * vec_elems + idx;
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
