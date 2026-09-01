#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include "../common/philox.h"
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

// Draws (seed, offset) from ATen's default CUDA generator for one elementwise
// RoundMode::SR launch, reserving a single 128-bit Philox block per
// subsequence so a subsequent unrelated RNG-consuming op doesn't reuse the
// same pair. The same idiom as custom_matmul_kernel.cu's
// matmul_rng_engine_inputs, and the same generator the SR path used to draw
// from through randint_like, so torch.manual_seed still governs the result.
// One block is enough because each subsequence here is consumed by exactly
// four elements (see PhiloxBlock in common/philox.h).
inline at::PhiloxCudaState quant_rng_engine_inputs()
{
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    return gen->philox_cuda_state(1);
}

// Indices are 64-bit: `size` comes straight from Tensor::numel(), which an int
// would truncate past 2^31 elements. The kernel is memory bound on every
// tensor large enough for the wider arithmetic to be reachable, and measures
// the same as the 32-bit version on ones small enough to sit in L2, so there
// is no separate narrow-index path.
template <typename scalar_t, class Quant>
__global__ __launch_bounds__(256, 2)
void quant_kernel_all(const scalar_t *__restrict__ a,
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

        float4 out = quant.vec(in);

        int4 out_i4 = reinterpret_cast<const int4&>(out);
        reinterpret_cast<int4*>(o)[i] = out_i4;
    }

    if (idx < rem_size)
    {
        int64_t rem_idx = vec_size * vec_elems + idx;
        o[rem_idx] = quant.scalar(a[rem_idx]);
    }
}

// RoundMode::SR twin of the above. It used to be the same kernel reading a
// third pointer -- an int32 tensor of draws that binaryK_quantize_cuda /
// superfp_quantize_cuda materialized with randint_like before every launch,
// which cost an allocation the size of the input, a second kernel to fill it,
// and two extra passes over DRAM on an operation that is otherwise purely
// bandwidth bound. The draws are generated here instead (finding E1 in
// dev/gemm_perf_audit.md).
//
// Element `j` takes word `j & 3` of Philox block `j >> 2`, so its value is a
// function of its own index and nothing else: independent of block size, of
// the grid, and of how many elements the dtype packs into a vector, exactly
// as the tensor of draws was. It is *not* the same value the tensor held --
// this is the one change in that document that alters results rather than
// only their cost.
template <typename scalar_t, class Quant>
__global__ __launch_bounds__(256, 2)
void quant_kernel_all_sr(const scalar_t *__restrict__ a,
                         scalar_t *o,
                         int64_t size,
                         Quant quant,
                         at::PhiloxCudaState rng_args)
{
    constexpr int vec_elems = SIMDTraits<scalar_t>::vec_elems;
    int64_t vec_size = size / vec_elems;
    int64_t rem_size = size % vec_elems;

    auto seeds = at::cuda::philox::unpack(rng_args);
    const uint64_t seed = std::get<0>(seeds);
    const uint64_t offset = std::get<1>(seeds);

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < vec_size; i += stride)
    {
        const int4* a_vec = reinterpret_cast<const int4*>(a);
        int4 in_i4 = a_vec[i];
        float4 in = reinterpret_cast<const float4&>(in_i4);

        // The whole iteration's randomness up front: one 10-round generate
        // for float and double, two for half/bfloat16, which is all the
        // elements this iteration touches can span. SIMDTraits::process
        // passes `lane` as a literal, so for the widths whose base index is a
        // multiple of 4 the word selection folds to a constant; for double it
        // stays a select over four named registers, which is what PhiloxBlock
        // holds rather than an array.
        const int64_t base = (int64_t)i * vec_elems;
        const uint64_t blk = (uint64_t)(base >> 2);
        PhiloxBlock b0 = philox_block(seed, blk, offset);
        PhiloxBlock b1;
        if constexpr (vec_elems > 4)
            b1 = philox_block(seed, blk + 1, offset);

        float4 out = SIMDTraits<scalar_t>::process(in, [&](float val, int lane)
                                                   {
            const int w = (int)((base + lane) & 3);
            uint32_t rv;
            if constexpr (vec_elems > 4)
                rv = (lane < 4) ? b0.word(w) : b1.word(w);
            else
                rv = b0.word(w);
            return quant.eval_sr(val, rv); });

        int4 out_i4 = reinterpret_cast<const int4&>(out);
        reinterpret_cast<int4*>(o)[i] = out_i4;
    }

    if (idx < rem_size)
    {
        int64_t rem_idx = vec_size * vec_elems + idx;
        PhiloxBlock b = philox_block(seed, (uint64_t)(rem_idx >> 2), offset);
        o[rem_idx] = quant.scalar_sr(a[rem_idx], b.word((int)(rem_idx & 3)));
    }
}
