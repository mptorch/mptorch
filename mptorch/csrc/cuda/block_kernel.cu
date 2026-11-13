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
// quantizers write the decoded values in place of the codes. Under a Four
// Over Six rule the group first compares the errors of two candidate scales,
// reduced by shuffles as well (four_over_six, below).
//
// A 2D format (block_rows > 1) takes block_pack_tile_kernel: one CTA per tile,
// an amax pass over the whole tile combined through shared memory, a barrier,
// and then the tile's row segments encoded by the 1D code. The barrier is
// what keeps the in-place quantizer sound: every read of the amax pass
// completes before any write, and the encode pass reads element i before it
// writes it. A tile of up to 16,384 elements does not fit in registers, so
// the encode pass reads its input a second time (from L2 for the largest).
// Under a Four Over Six rule a third pass sits between the two, behind a
// barrier of its own (four_over_six_tile): the whole tile's errors under the
// two candidate scales, combined as the amax is, so the tile makes one choice.
//
// Every element's stochastic draw is word (i & 3) of Philox block i >> 2 of
// the call's (seed, offset), i its logical index in the packed orientation,
// as on the CPU; no draw depends on the launch geometry.
//
// Loads are per element through the input's strides, not 16-byte vectors, so
// a transposed view packs without a copy and no input needs
// mptorch::vector_loadable. block_unpack is a grid-stride kernel over the
// result's elements: SIMDTraits does not apply to a packed input.
//
// The kernels are templates on the carrier in block_kernel.cuh; this file
// instantiates the binary32 ones, and block_kernel_f64.cu the binary64 ones.

#include "block_kernel.cuh"

namespace mptorch::block_cuda
{
    void launch_block_pack(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c,
                           at::PhiloxCudaState rng, cudaStream_t stream)
    {
        switch (j.dt)
        {
        case mptorch::GemmDtype::Half:
            return launch_pack_dtype<float, at::Half>(j, p, c, rng, stream);
        case mptorch::GemmDtype::BFloat16:
            return launch_pack_dtype<float, at::BFloat16>(j, p, c, rng, stream);
        default:
            return launch_pack_dtype<float, float>(j, p, c, rng, stream);
        }
    }

    void launch_block_unpack(const BlockUnpackJob &j, const BlockFormatParams &p, cudaStream_t stream)
    {
        const unsigned grid = grid_of((j.batch * j.rows * j.cols + THREADS - 1) / THREADS);
        switch (j.dt)
        {
        case mptorch::GemmDtype::Half:
            return block_unpack_kernel<float, at::Half><<<grid, THREADS, 0, stream>>>(j, p);
        case mptorch::GemmDtype::BFloat16:
            return block_unpack_kernel<float, at::BFloat16><<<grid, THREADS, 0, stream>>>(j, p);
        default:
            return block_unpack_kernel<float, float><<<grid, THREADS, 0, stream>>>(j, p);
        }
    }
} // namespace mptorch::block_cuda
