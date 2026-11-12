#pragma once

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <tuple>
#include <type_traits>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include "../common/philox.h"
#include "simd_traits.cuh"

// Blocks of `block` threads needed to cover `work` items, at least one and
// at most grid.x's limit of 2^31 - 1. The kernels below are grid-stride, so
// a grid clamped at the limit still covers a larger tensor, in more
// iterations per thread rather than incorrectly.
inline int grid_for(int64_t work, int block)
{
    constexpr int64_t max_grid = 2147483647;
    int64_t grid = (work + block - 1) / block;
    return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(grid, max_grid)));
}

// Draws (seed, offset) from ATen's default CUDA generator for one
// elementwise RoundMode::SR launch, so torch.manual_seed governs the result,
// and advances the generator's offset by one 128-bit Philox block so that a
// later RNG-consuming op does not reuse the same pair. The GEMM's
// make_context (gemm_backend.cpp) is the same idiom. One block is enough
// because the kernel keys the Philox subsequence on the element index and
// reads only the first block of each subsequence, which at most four
// elements share (PhiloxBlock in common/philox.h).
inline at::PhiloxCudaState quant_rng_engine_inputs()
{
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    return gen->philox_cuda_state(1);
}

// Elementwise quantization over a flat array: `size` elements at `a` are
// rounded through `quant` into `o`. Each thread moves 16 bytes per iteration
// through an int4 (four floats, two doubles, or eight halves or bfloat16s,
// per SIMDTraits), which is what lets the kernel run at memory bandwidth;
// the int4 rather than a typed vector sidesteps aliasing rules around
// c10::Half and c10::BFloat16. The scalar tail handles the size % vec_elems
// elements left over, indexed by the same thread ids.
//
// `a` and `o` must start on a 16-byte boundary: a 16-byte vector load of an
// address that is not raises `misaligned address`, a sticky error that costs
// the process its CUDA context, and a contiguous view (`x[1:]`) can start
// anywhere in its storage. The entry points therefore pass their input
// through mptorch::vector_loadable (vector_load.h), which copies such a
// view; the output is a fresh allocation and is aligned by construction.
// The in-place ops check their tensor instead (check_vector_loadable_in_place).
//
// `a` and `o` may be the same pointer, which is what the in-place ops pass,
// so neither carries __restrict__. That is value-safe because a thread reads
// vector i (or tail element i) whole before it writes it, and no thread
// touches another's.
//
// Indices are 64-bit because `size` comes straight from Tensor::numel(),
// which an int would truncate past 2^31 elements. The kernel is memory bound
// on any tensor large enough for the wider index arithmetic to matter, and
// measures the same as a 32-bit version on ones small enough to sit in L2,
// so there is no separate narrow-index path.
template <typename scalar_t, class Quant>
__global__ __launch_bounds__(256, 2)
void quant_kernel_all(const scalar_t *a,
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
        // One 128-bit load through int4, reinterpreted for the lane function.
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

// The RoundMode::SR twin of quant_kernel_all: the same loads and stores,
// with one random word generated per element for the stochastic cast.
// Generating in the kernel replaces an int32 tensor of draws materialized
// before every launch, which cost an allocation the size of the input, a
// fill kernel and two extra passes over DRAM on an op that is otherwise
// purely bandwidth bound.
//
// Each element's word is a function of its own index and nothing else.
// Element `j` takes word `j & 3` of Philox block `j >> 2` of the launch's
// (seed, offset) stream; a float64 element, which rounds in binary64 and
// draws 64 bits, takes words `2 * (j & 1)` and the next of block `j >> 1`
// (PhiloxBlock, common/philox.h). The result is therefore independent of the
// block size, the grid, and how many elements the dtype packs into a vector,
// and one 10-round Philox generate is amortized over the four (or two)
// elements that share a block, which is one vector's worth of lanes.
template <typename scalar_t, class Quant>
__global__ __launch_bounds__(256, 2)
void quant_kernel_all_sr(const scalar_t *a,
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

    if constexpr (std::is_same_v<scalar_t, double>)
    {
        // binary64: a double2 vector is elements 2i and 2i + 1, which are
        // block i's two 64-bit draws, so the lane selects the word pair and
        // one generate covers the iteration.
        for (int64_t i = idx; i < vec_size; i += stride)
        {
            const int4* a_vec = reinterpret_cast<const int4*>(a);
            int4 in_i4 = a_vec[i];
            float4 in = reinterpret_cast<const float4&>(in_i4);

            PhiloxBlock b0 = philox_block(seed, (uint64_t)i, offset);
            float4 out = SIMDTraits<scalar_t>::process(in, [&](double val, int lane)
                                                       { return quant.eval_sr(val, b0.word64(2 * lane)); });

            int4 out_i4 = reinterpret_cast<const int4&>(out);
            reinterpret_cast<int4*>(o)[i] = out_i4;
        }
    }
    else
    {
    for (int64_t i = idx; i < vec_size; i += stride)
    {
        const int4* a_vec = reinterpret_cast<const int4*>(a);
        int4 in_i4 = a_vec[i];
        float4 in = reinterpret_cast<const float4&>(in_i4);

        // The whole iteration's randomness up front: one 10-round generate
        // for float (four elements, one block) and two for half/bfloat16
        // (eight elements, two consecutive blocks). SIMDTraits::process
        // passes `lane` as a literal and `base` is a multiple of 4, so the
        // word selection folds to a constant.
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
    }

    if (idx < rem_size)
    {
        int64_t rem_idx = vec_size * vec_elems + idx;
        if constexpr (std::is_same_v<scalar_t, double>)
        {
            PhiloxBlock b = philox_block(seed, (uint64_t)(rem_idx >> 1), offset);
            o[rem_idx] = quant.scalar_sr(a[rem_idx], b.word64(2 * (int)(rem_idx & 1)));
        }
        else
        {
            PhiloxBlock b = philox_block(seed, (uint64_t)(rem_idx >> 2), offset);
            o[rem_idx] = quant.scalar_sr(a[rem_idx], b.word((int)(rem_idx & 3)));
        }
    }
}
