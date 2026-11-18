#pragma once

// What an MPS kernel takes per launch besides its tensors: the shape, the
// strides and the stochastic-rounding seed, passed as one small `constant`
// buffer. The host code (gemm_backend.cpp, quantize.cpp, block_entry.cpp)
// fills these structs and the kernels (gemm.metal, quantize.metal,
// block.metal) read them, and both compile this one definition, so the two
// layouts cannot drift apart; the size checks pin them anyway. Everything in
// them varies per call, which is why they are a buffer and not part of a
// kernel's source. The formats are part of the source, so that their
// constants are known at compile time (gemm.metal says why that matters).

#if !defined(__METAL_VERSION__)
#include <cstdint>
#endif

namespace mptorch_mps
{

  // A GEMM, as common/gemm_args.h's GemmShape describes it: C[b] =
  // op(A[b]) @ op(B[b]), written densely as [batch, M, N], with per-operand
  // batch strides in elements (a stride of 0 broadcasts that operand). The
  // idx_* strides index a mixed op's precision map, and are 0 on the
  // single-format ops.
  struct GemmLaunch
  {
    uint32_t M, K, N, batch;
    uint32_t trans_a, trans_b, use_rng, unused;
    uint64_t stride_a, stride_b;
    uint64_t seed;
    uint64_t idx_row_stride, idx_col_stride, idx_batch_stride;
  };
  static_assert(sizeof(GemmLaunch) == 80, "GemmLaunch must have one layout on the host and the GPU");

  // The GEMM's tile (gemm.metal): a threadgroup of GEMM_TILE x GEMM_TILE
  // threads computes that block of C, one element per thread, copying op(A)
  // and op(B) into threadgroup memory GEMM_TILE steps of K at a time. The
  // host sizes the grid with it.
  enum : uint32_t
  {
    GEMM_TILE = 16
  };

  // The batch strides of the block GEMM's two scale tensors, in scale bytes
  // (a stride of 0 broadcasts that operand's scales). common/gemm_block.h's
  // BlockOperand carries them beside its scales; the kernel binds them as a
  // buffer of their own.
  struct BlockGemmLaunch
  {
    int64_t scale_stride_a, scale_stride_b;
  };
  static_assert(sizeof(BlockGemmLaunch) == 16, "BlockGemmLaunch must have one layout on the host and the GPU");

  // A block_pack, block_quant or block_quant_ call (block.metal's
  // mpt_block_pack and mpt_block_pack_tile): the tensor's [batch, rows, cols]
  // extents and element strides, the result's strides (block_quant only),
  // the SR seed, and the number of tiles, each of which gets one thread or
  // one threadgroup.
  struct BlockPackLaunch
  {
    int64_t batch, rows, cols;
    int64_t xs0, xs1, xs2;
    int64_t os0, os1, os2;
    uint64_t seed;
    uint64_t tiles;
  };
  static_assert(sizeof(BlockPackLaunch) == 88, "BlockPackLaunch must have one layout on the host and the GPU");

  // The most threads a tile's threadgroup has (mpt_block_pack_tile). The
  // host gives each thread 64 of a tile's elements, with at least one
  // simdgroup and at most this many threads, which a 128 x 128 tile
  // reaches.
  enum : uint32_t
  {
    BLOCK_TILE_THREADS = 256
  };

  // A block_unpack call (mpt_block_unpack): the dense result's extents.
  struct BlockUnpackLaunch
  {
    int64_t batch, rows, cols;
  };
  static_assert(sizeof(BlockUnpackLaunch) == 24, "BlockUnpackLaunch must have one layout on the host and the GPU");

  // An elementwise quantizer over `n` contiguous elements.
  struct QuantLaunch
  {
    uint64_t n;
    uint64_t seed;
  };
  static_assert(sizeof(QuantLaunch) == 16, "QuantLaunch must have one layout on the host and the GPU");

} // namespace mptorch_mps
