#pragma once

// The elementwise block ops (block_pack, block_unpack, block_quantize and its
// in-place version) as their kernels see them: raw pointers, extents and
// element strides, with no ATen, so that the CUDA kernels compile without
// ATen's headers (the tensors stay in cuda/block_entry.cpp, as the GEMM's
// stay in cuda/custom_matmul_entry.cpp). The format arrives as a
// BlockFormatParamsT plus the casts' constants, both built once per call in
// the call's carrier (block_host.h): binary64 for a float64 tensor, binary32
// otherwise.

#include "block_decode.h"
#include "gemm_dtype.h"
#include "gemm_gather.h" // FastDivmod
#include "philox.h"
#include <cstdint>

namespace mptorch::block
{

  // The element format's cast and the MINIFLOAT scale's, in either family
  // (block_decode.h's BlockCastT), in binary32.
  using BlockCast = BlockCastT<float>;

  // Under RoundMode::SR in carrier T, element i's random word comes from
  // Philox block i >> block_draw_shift<T> of the call's stream (see
  // philox.h's layout): word i & 3 of block i >> 2 in binary32, and in
  // binary64 the 64-bit word made of words 2*(i & 1) and 2*(i & 1) + 1 of
  // block i >> 1. The kernels read the word with an `if constexpr` per
  // carrier rather than through a helper that takes the block by reference:
  // such a helper changed the stack layout of the CPU's binary32 SR kernels.
  template <class T>
  inline constexpr int block_draw_shift = 2;

  template <>
  inline constexpr int block_draw_shift<double> = 1;

  // block_pack (quant = false: the element codes into data, the scale codes
  // into scales) and block_quantize (quant = true: the decoded values into o,
  // which is x itself for the in-place op). x is [batch, rows, cols] with the
  // packed axis last, read through its element strides, so a transposed view
  // packs without a copy; o is written through its own strides. data is
  // [batch, rows, row_bytes] and scales [batch, row_tiles, scale_cols], both
  // contiguous. Element (b, r, c)'s random word is the one described above
  // for the carrier, at index i = (b * rows + r) * cols + c of the stream
  // (seed, offset): its logical index in the packed orientation, which is
  // the same in all three ops.
  struct BlockPackJob
  {
    const void *x = nullptr;
    void *o = nullptr;
    uint8_t *data = nullptr;
    uint8_t *scales = nullptr;
    mptorch::GemmDtype dt = mptorch::GemmDtype::Float;
    int64_t batch = 1, rows = 0, cols = 0;
    int64_t xs0 = 0, xs1 = 0, xs2 = 1;
    int64_t os0 = 0, os1 = 0, os2 = 1;
    bool quant = false;
    RoundMode rm = RoundMode::RNE;
    // The CUDA kernel for 1D blocks splits each segment index into (row,
    // block), and each row into (batch, row), with these multiply-high
    // divisors when every index fits in 31 bits (`fast`), at a few
    // instructions each, and otherwise with 64-bit division, which is a
    // software routine on the device.
    bool fast = false;
    mptorch::gemm::FastDivmod by_blocks{}, by_rows{};
  };

  // block_unpack: data and scales as block_pack writes them, and o
  // [batch, rows, cols], contiguous, in dtype dt.
  struct BlockUnpackJob
  {
    const uint8_t *data = nullptr;
    const uint8_t *scales = nullptr;
    void *o = nullptr;
    mptorch::GemmDtype dt = mptorch::GemmDtype::Float;
    int64_t batch = 1, rows = 0, cols = 0;
  };

} // namespace mptorch::block
