// The CUDA kernels of block_pack, block_unpack, block_quant and block_quant_
// (common/block_decode.h has the format; block_entry.cpp the tensors).
//
// A 1D format (block_rows = 1) packs one row segment of a block per group of
// G = block_size / 4 lanes of a warp (one lane below a block of 4), each lane
// taking four elements at a stride of G, so a warp's loads of a contiguous
// row coalesce. Four elements a lane rather than one: the group's reduction
// and its scale are per segment, and over one element a lane they held the
// kernel to 45 GB/s of float32 against the 1D quantizer's 105 (the 128-wide
// blocks, four elements a lane already, ran at 100).
// The group's largest magnitude is a shuffle reduction (a max of
// non-negative words, exact in any order), its scale is computed by every
// lane alike, and each lane encodes its elements. block_pack stages the codes
// in shared memory and has the group's lanes assemble the block's bytes, lane
// j taking bytes j, j + G, ... from the codes whose bits meet each; the
// quantizers write the decoded values in place of the codes.
//
// A 2D format (block_rows > 1) takes block_pack_tile_kernel: one CTA per tile,
// an amax pass over the whole tile combined through shared memory, a barrier,
// and then the tile's row segments encoded by the 1D code. The barrier is
// what keeps the in-place quantizer sound: every read of the amax pass
// completes before any write, and the encode pass reads element i before it
// writes it. A tile of up to 16,384 elements does not fit in registers, so
// the encode pass reads its input a second time (from L2 for the largest).
//
// Every element's stochastic draw is word (i & 3) of Philox block i >> 2 of
// the call's (seed, offset), i its logical index in the packed orientation,
// as on the CPU; no draw depends on the launch geometry.
//
// Loads are per element through the input's strides, not 16-byte vectors, so
// a transposed view packs without a copy and no input needs
// mptorch::vector_loadable. block_unpack is a grid-stride kernel over the
// result's elements: SIMDTraits does not apply to a packed input.

#include "block_kernel.h"
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
        constexpr int WARPS = THREADS / 32;
        // A CTA pass stages four codes a thread: a group of G lanes holds a
        // block of 4 G codes (a block of 2 is one lane's two).
        constexpr int MAX_CODES = THREADS * 4;
        constexpr int64_t MAX_GRID = int64_t(1) << 20;

        template <typename scalar_t>
        __device__ __forceinline__ float load_as_float(const void *x, int64_t i)
        {
            return static_cast<float>(static_cast<const scalar_t *>(x)[i]);
        }

        template <typename scalar_t>
        __device__ __forceinline__ void store_from_float(void *o, int64_t i, float v)
        {
            static_cast<scalar_t *>(o)[i] = static_cast<scalar_t>(v);
        }

        // What a lane keeps of its group: its index in the group, the group's
        // width G and each lane's element count E.
        struct Group
        {
            int gl;
            int G;
            int E;
        };

        __device__ __forceinline__ Group group_of(const BlockFormatParams &p)
        {
            Group g;
            g.G = p.block_size >= 4 ? p.block_size / 4 : 1;
            g.E = p.block_size / g.G;
            g.gl = static_cast<int>(threadIdx.x % 32) % g.G;
            return g;
        }

        // The largest magnitude among a lane's words, and whether any is a
        // NaN, reduced over its group of G lanes (a power of two, aligned in
        // the warp). Every lane of the warp takes part.
        __device__ __forceinline__ void reduce_group(uint32_t &amax, bool &nan, int G)
        {
            unsigned n = nan ? 1u : 0u;
            for (int off = G / 2; off > 0; off >>= 1)
            {
                const uint32_t other = __shfl_xor_sync(0xffffffffu, amax, off);
                amax = other > amax ? other : amax;
                n |= __shfl_xor_sync(0xffffffffu, n, off);
            }
            nan = n != 0;
        }

        // One row segment's encode, by its group: the E elements `v` a lane
        // loaded (cols past the row's end are not elements), their codes, and
        // either the decoded values written back (quantize) or the block's
        // bytes (pack). `codes` is the group's staging area in shared memory.
        template <typename scalar_t, RoundMode RM>
        __device__ __forceinline__ void encode_segment(const BlockPackJob &j, const BlockFormatParams &p,
                                                       const BlockCast &c, const TileScale<float> &s,
                                                       uint64_t seed, uint64_t offset, bool live, int64_t b,
                                                       int64_t r, int64_t blk, const Group &g, const float *v,
                                                       uint16_t *codes)
        {
            const int64_t row_idx = (b * j.rows + r) * j.cols;
            for (int e = 0; e < g.E; ++e)
            {
                const int i = g.gl + e * g.G;
                const int64_t cc = blk * p.block_size + i;
                uint32_t code = 0;
                if (live && cc < j.cols)
                {
                    uint32_t rv = 0;
                    if constexpr (RM == RoundMode::SR)
                    {
                        const int64_t idx = row_idx + cc;
                        rv = philox_block(seed, static_cast<uint64_t>(idx >> 2), offset).word(static_cast<int>(idx & 3));
                    }
                    const float q = block_cast<float, RM>(scaled_elem(v[e], s, p), c.elem_signed, rv, c);
                    code = encode_elem(v[e], q, p);
                    if (j.quant)
                        store_from_float<scalar_t>(j.o, b * j.os0 + r * j.os1 + cc * j.os2,
                                                   decode_value<float>(code, s.code, p));
                }
                if (!j.quant)
                    codes[i] = static_cast<uint16_t>(code);
            }
            if (!j.quant)
            {
                __syncwarp();
                if (live)
                {
                    uint8_t *dst = j.data + (b * j.rows + r) * p.row_bytes + blk * p.block_bytes;
                    for (int jb = g.gl; jb < p.block_bytes; jb += g.G)
                        dst[jb] = assemble_byte(codes, jb, p.elem_bits);
                }
                __syncwarp();
            }
        }

        template <typename scalar_t, RoundMode RM>
        __global__ __launch_bounds__(THREADS) void block_pack_kernel(BlockPackJob j, BlockFormatParams p,
                                                                     BlockCast c, at::PhiloxCudaState rng)
        {
            __shared__ uint16_t codes_sh[MAX_CODES];
            uint64_t seed = 0, offset = 0;
            if constexpr (RM == RoundMode::SR)
            {
                auto seeds = at::cuda::philox::unpack(rng);
                seed = std::get<0>(seeds);
                offset = std::get<1>(seeds);
            }
            const Group g = group_of(p);
            const int groups_per_warp = 32 / g.G;
            const int my_group = static_cast<int>(threadIdx.x / 32) * groups_per_warp +
                                 static_cast<int>(threadIdx.x % 32) / g.G;
            const int64_t groups_per_cta = int64_t(WARPS) * groups_per_warp;
            uint16_t *codes = codes_sh + my_group * p.block_size;
            const int64_t segs = j.batch * j.rows * p.n_blocks;

            // The bound is the CTA's, so every lane of a warp runs every
            // iteration and every shuffle.
            for (int64_t base = int64_t(blockIdx.x) * groups_per_cta; base < segs;
                 base += int64_t(gridDim.x) * groups_per_cta)
            {
                const int64_t seg = base + my_group;
                const bool live = seg < segs;
                int64_t row_g = 0, blk = 0, b = 0, r = 0;
                if (live && j.fast)
                {
                    int32_t q, rem;
                    j.by_blocks.divmod(static_cast<int32_t>(seg), q, rem);
                    row_g = q;
                    blk = rem;
                    j.by_rows.divmod(q, q, rem);
                    b = q;
                    r = rem;
                }
                else if (live)
                {
                    row_g = seg / p.n_blocks;
                    blk = seg - row_g * p.n_blocks;
                    b = row_g / j.rows;
                    r = row_g - b * j.rows;
                }

                float v[4];
                uint32_t amax = 0;
                bool nan = false;
                for (int e = 0; e < g.E; ++e)
                {
                    const int64_t cc = blk * p.block_size + g.gl + e * g.G;
                    v[e] = (live && cc < j.cols) ? load_as_float<scalar_t>(j.x, b * j.xs0 + r * j.xs1 + cc * j.xs2)
                                                 : 0.0f;
                    const uint32_t w = block_word(v[e]) & FloatTraits<float>::ABS_MASK;
                    if (w > FloatTraits<float>::INF_BITS)
                        nan = true;
                    else if (w > amax)
                        amax = w;
                }
                reduce_group(amax, nan, g.G);
                const TileScale<float> s = tile_scale<float>(block_float<float>(amax), nan, p, c);
                encode_segment<scalar_t, RM>(j, p, c, s, seed, offset, live, b, r, blk, g, v, codes);
                if (!j.quant && live && g.gl == 0 && p.scale_cols != 0)
                    j.scales[(b * p.row_tiles + r) * p.scale_cols + blk] = static_cast<uint8_t>(s.code);
            }
        }

        template <typename scalar_t, RoundMode RM>
        __global__ __launch_bounds__(THREADS) void block_pack_tile_kernel(BlockPackJob j, BlockFormatParams p,
                                                                          BlockCast c, at::PhiloxCudaState rng)
        {
            __shared__ uint16_t codes_sh[MAX_CODES];
            __shared__ uint32_t warp_amax[WARPS];
            __shared__ uint32_t warp_nan[WARPS];
            uint64_t seed = 0, offset = 0;
            if constexpr (RM == RoundMode::SR)
            {
                auto seeds = at::cuda::philox::unpack(rng);
                seed = std::get<0>(seeds);
                offset = std::get<1>(seeds);
            }
            // The CTA is sized to the tile (launch_pack_as): four elements a
            // thread, between one warp and THREADS.
            const int threads = static_cast<int>(blockDim.x);
            const int warps = threads / 32;
            const Group g = group_of(p);
            const int groups_per_warp = 32 / g.G;
            const int my_group = static_cast<int>(threadIdx.x / 32) * groups_per_warp +
                                 static_cast<int>(threadIdx.x % 32) / g.G;
            const int groups_per_cta = warps * groups_per_warp;
            uint16_t *codes = codes_sh + my_group * p.block_size;
            const int64_t per_batch = p.row_tiles * p.n_blocks;
            const int tile_elems = p.block_rows * p.block_size;

            for (int64_t t = blockIdx.x; t < j.batch * per_batch; t += gridDim.x)
            {
                const int64_t b = t / per_batch;
                const int64_t rem = t - b * per_batch;
                const int64_t rt = rem / p.n_blocks;
                const int64_t blk = rem - rt * p.n_blocks;
                const int64_t r0 = rt * p.block_rows;
                const int64_t c0 = blk * p.block_size;
                const int64_t nrows = j.rows - r0 < p.block_rows ? j.rows - r0 : p.block_rows;
                const int64_t ncols = j.cols - c0 < p.block_size ? j.cols - c0 : p.block_size;

                // The amax pass over the whole tile.
                uint32_t amax = 0;
                bool nan = false;
                for (int i = threadIdx.x; i < tile_elems; i += threads)
                {
                    const int rr = i >> p.log2_block_size;
                    const int cc = i & (p.block_size - 1);
                    if (rr < nrows && cc < ncols)
                    {
                        const uint32_t w =
                            block_word(load_as_float<scalar_t>(j.x, b * j.xs0 + (r0 + rr) * j.xs1 + (c0 + cc) * j.xs2)) &
                            FloatTraits<float>::ABS_MASK;
                        if (w > FloatTraits<float>::INF_BITS)
                            nan = true;
                        else if (w > amax)
                            amax = w;
                    }
                }
                reduce_group(amax, nan, 32);
                if (threadIdx.x % 32 == 0)
                {
                    warp_amax[threadIdx.x / 32] = amax;
                    warp_nan[threadIdx.x / 32] = nan ? 1u : 0u;
                }
                __syncthreads();
                for (int w = 0; w < warps; ++w)
                {
                    amax = warp_amax[w] > amax ? warp_amax[w] : amax;
                    nan = nan || warp_nan[w] != 0;
                }
                const TileScale<float> s = tile_scale<float>(block_float<float>(amax), nan, p, c);

                // The encode pass: the tile's rows as row segments, a group
                // each, as the 1D kernel encodes them.
                for (int rr0 = 0; rr0 < nrows; rr0 += groups_per_cta)
                {
                    const int rr = rr0 + my_group;
                    const bool live = rr < nrows;
                    float v[4];
                    for (int e = 0; e < g.E; ++e)
                    {
                        const int64_t cc = c0 + g.gl + e * g.G;
                        v[e] = (live && cc < j.cols)
                                   ? load_as_float<scalar_t>(j.x, b * j.xs0 + (r0 + rr) * j.xs1 + cc * j.xs2)
                                   : 0.0f;
                    }
                    encode_segment<scalar_t, RM>(j, p, c, s, seed, offset, live, b, r0 + rr, blk, g, v, codes);
                }
                if (!j.quant && threadIdx.x == 0 && p.scale_cols != 0)
                    j.scales[(b * p.row_tiles + rt) * p.scale_cols + blk] = static_cast<uint8_t>(s.code);
                // The shared amax slots are the next tile's.
                __syncthreads();
            }
        }

        template <typename scalar_t>
        __global__ __launch_bounds__(THREADS) void block_unpack_kernel(BlockUnpackJob j, BlockFormatParams p)
        {
            const int64_t n = j.batch * j.rows * j.cols;
            for (int64_t i = int64_t(blockIdx.x) * THREADS + threadIdx.x; i < n; i += int64_t(gridDim.x) * THREADS)
            {
                const int64_t g = i / j.cols;
                const int64_t cc = i - g * j.cols;
                const int64_t b = g / j.rows;
                const int64_t r = g - b * j.rows;
                const int64_t blk = cc >> p.log2_block_size;
                const uint32_t code = extract_code(j.data + g * p.row_bytes + blk * p.block_bytes,
                                                   cc & (p.block_size - 1), p.elem_bits);
                const uint32_t sc =
                    p.scale_cols != 0
                        ? static_cast<uint32_t>(j.scales[(b * p.row_tiles + (r >> p.log2_block_rows)) * p.scale_cols + blk])
                        : 0u;
                store_from_float<scalar_t>(j.o, i, decode_value<float>(code, sc, p));
            }
        }

        inline unsigned grid_of(int64_t work)
        {
            if (work < 1)
                work = 1;
            return static_cast<unsigned>(work < MAX_GRID ? work : MAX_GRID);
        }

        template <typename scalar_t, RoundMode RM>
        void launch_pack_as(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c,
                            at::PhiloxCudaState rng, cudaStream_t stream)
        {
            if (p.block_rows == 1)
            {
                const int G = p.block_size >= 4 ? p.block_size / 4 : 1;
                const int64_t groups_per_cta = int64_t(WARPS) * (32 / G);
                const int64_t segs = j.batch * j.rows * p.n_blocks;
                block_pack_kernel<scalar_t, RM>
                    <<<grid_of((segs + groups_per_cta - 1) / groups_per_cta), THREADS, 0, stream>>>(j, p, c, rng);
            }
            else
            {
                // Four elements a thread: a small tile (NVFP4's 16 x 16) in a
                // CTA of THREADS would leave three quarters of it idle through
                // both passes and the barrier between them.
                const int tile_elems = p.block_rows * p.block_size;
                const int threads = tile_elems / 4 < 32 ? 32 : (tile_elems / 4 > THREADS ? THREADS : tile_elems / 4);
                block_pack_tile_kernel<scalar_t, RM>
                    <<<grid_of(j.batch * p.row_tiles * p.n_blocks), threads, 0, stream>>>(j, p, c, rng);
            }
        }

        template <typename scalar_t>
        void launch_pack_dtype(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c,
                               at::PhiloxCudaState rng, cudaStream_t stream)
        {
            mptorch::dispatch_round_mode(j.rm, [&](auto rm_c)
            {
                constexpr RoundMode RM = decltype(rm_c)::value;
                launch_pack_as<scalar_t, RM>(j, p, c, rng, stream);
            });
        }
    } // namespace

    void launch_block_pack(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c,
                           at::PhiloxCudaState rng, cudaStream_t stream)
    {
        switch (j.dt)
        {
        case mptorch::GemmDtype::Half:
            return launch_pack_dtype<at::Half>(j, p, c, rng, stream);
        case mptorch::GemmDtype::BFloat16:
            return launch_pack_dtype<at::BFloat16>(j, p, c, rng, stream);
        default:
            return launch_pack_dtype<float>(j, p, c, rng, stream);
        }
    }

    void launch_block_unpack(const BlockUnpackJob &j, const BlockFormatParams &p, cudaStream_t stream)
    {
        const unsigned grid = grid_of((j.batch * j.rows * j.cols + THREADS - 1) / THREADS);
        switch (j.dt)
        {
        case mptorch::GemmDtype::Half:
            return block_unpack_kernel<at::Half><<<grid, THREADS, 0, stream>>>(j, p);
        case mptorch::GemmDtype::BFloat16:
            return block_unpack_kernel<at::BFloat16><<<grid, THREADS, 0, stream>>>(j, p);
        default:
            return block_unpack_kernel<float><<<grid, THREADS, 0, stream>>>(j, p);
        }
    }
} // namespace mptorch::block_cuda
