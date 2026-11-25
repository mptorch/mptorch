// The CUDA kernel of block_project (common/block_project.h states the op;
// cuda/block_project_entry.cpp handles the tensors).
//
// One thread takes a group of consecutive elements of a row whose codes
// fill whole bytes (project_group: 2 elements of 4 bits, 1 of 8, 4 of 6), so
// that no two threads write one byte, and assembles the group's bytes as it
// encodes, low bits first. A block holds a whole number of groups, so a
// thread decodes one scale code; past the row's last element the codes are
// zero, as block_pack leaves the end of a row's last block. The grid strides
// over the groups of every row, and a group's row and batch come from 64-bit
// divisions: this op is not on a training step's path, and its loads are
// one element at a time through x's strides, as block_pack's are, so a
// transposed view is projected without a copy.
//
// Under SR, element i = (b * rows + r) * cols + c draws block_pack's word:
// word i & 3 of Philox block i >> 2 of the call's (seed, offset) in binary32,
// and the 64-bit word 2 (i & 1) of block i >> 1 in binary64.

#include "block_project.h"
#include "../common/dispatch.h"
#include "../common/philox.h"
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mptorch::block_cuda
{
    using namespace mptorch::block;

    namespace
    {
        constexpr int THREADS = 256;
        constexpr int64_t MAX_GRID = int64_t(1) << 20;

        template <class T, typename scalar_t, RoundMode RM>
        __global__ __launch_bounds__(THREADS) void block_project_kernel(BlockProjectJob j, BlockFormatParamsT<T> p,
                                                                        BlockCastT<T> c, at::PhiloxCudaState rng)
        {
            using F = FloatTraits<T>;
            uint64_t seed = 0, offset = 0;
            if constexpr (RM == RoundMode::SR)
            {
                auto seeds = at::cuda::philox::unpack(rng);
                seed = std::get<0>(seeds);
                offset = std::get<1>(seeds);
            }
            const scalar_t *x = static_cast<const scalar_t *>(j.x);
            const int G = project_group(p.elem_bits);
            const int group_bytes = G * p.elem_bits / 8;
            const int64_t per_row = p.n_blocks * (p.block_size / G);
            const int64_t groups = j.batch * j.rows * per_row;
            bool bad = false;
            for (int64_t t = int64_t(blockIdx.x) * THREADS + threadIdx.x; t < groups;
                 t += int64_t(gridDim.x) * THREADS)
            {
                const int64_t g = t / per_row;
                const int64_t gi = t - g * per_row;
                const int64_t b = g / j.rows;
                const int64_t r = g - b * j.rows;
                const int64_t c0 = gi * G;
                const int64_t blk = c0 >> p.log2_block_size;
                const uint32_t sc = p.scale_cols != 0
                                        ? static_cast<uint32_t>(
                                              j.scales[(b * p.row_tiles + (r >> p.log2_block_rows)) * p.scale_cols + blk])
                                        : 0u;
                const ProjectScale<T> s = project_scale<T>(sc, p);
                const scalar_t *xr = x + b * j.xs0 + r * j.xs1;
                const int64_t row_idx = g * j.cols;
                uint8_t *dst = j.data + g * p.row_bytes + gi * group_bytes;
                uint32_t acc = 0; // the codes not yet written, low bits first
                int held = 0;     // how many bits of them
                PhiloxBlock draws;
                int64_t drawn = -1; // the index of the Philox block in `draws`
                for (int e = 0; e < G; ++e)
                {
                    const int64_t cc = c0 + e;
                    uint32_t code = 0;
                    if (cc < j.cols)
                    {
                        const T v = static_cast<T>(xr[cc * j.xs2]);
                        typename F::word_t rv = 0;
                        if constexpr (RM == RoundMode::SR)
                        {
                            const int64_t i = row_idx + cc;
                            if ((i >> block_draw_shift<T>) != drawn)
                            {
                                drawn = i >> block_draw_shift<T>;
                                draws = philox_block(seed, static_cast<uint64_t>(drawn), offset);
                            }
                            if constexpr (std::is_same_v<T, double>)
                                rv = draws.word64(2 * static_cast<int>(i & 1));
                            else
                                rv = draws.word(static_cast<int>(i & 3));
                        }
                        code = project_code<T, RM>(v, s, c.elem_signed, rv, p, c, bad);
                    }
                    acc |= code << held;
                    held += p.elem_bits;
                    while (held >= 8)
                    {
                        *dst++ = static_cast<uint8_t>(acc);
                        acc >>= 8;
                        held -= 8;
                    }
                }
            }
            if (bad)
                *j.unstored = 1;
        }

        inline unsigned grid_of(int64_t work)
        {
            if (work < 1)
                work = 1;
            return static_cast<unsigned>(work < MAX_GRID ? work : MAX_GRID);
        }

        template <class T, typename scalar_t>
        void launch_as(const BlockProjectJob &j, const BlockFormatParamsT<T> &p, const BlockCastT<T> &c,
                       at::PhiloxCudaState rng, cudaStream_t stream)
        {
            const int G = project_group(p.elem_bits);
            const int64_t groups = j.batch * j.rows * p.n_blocks * (p.block_size / G);
            const unsigned grid = grid_of((groups + THREADS - 1) / THREADS);
            mptorch::dispatch_round_mode(j.rm, [&](auto rm_c)
            {
                constexpr RoundMode RM = decltype(rm_c)::value;
                block_project_kernel<T, scalar_t, RM><<<grid, THREADS, 0, stream>>>(j, p, c, rng);
            });
        }
    } // namespace

    void launch_block_project(const BlockProjectJob &j, const BlockFormatParams &p, const BlockCast &c,
                              at::PhiloxCudaState rng, cudaStream_t stream)
    {
        switch (j.dt)
        {
        case mptorch::GemmDtype::Half:
            return launch_as<float, at::Half>(j, p, c, rng, stream);
        case mptorch::GemmDtype::BFloat16:
            return launch_as<float, at::BFloat16>(j, p, c, rng, stream);
        default:
            return launch_as<float, float>(j, p, c, rng, stream);
        }
    }

#if !defined(MPTORCH_NO_FP64)
    void launch_block_project(const BlockProjectJob &j, const BlockFormatParamsT<double> &p,
                              const BlockCastT<double> &c, at::PhiloxCudaState rng, cudaStream_t stream)
    {
        launch_as<double, double>(j, p, c, rng, stream);
    }
#endif
} // namespace mptorch::block_cuda
