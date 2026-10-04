#pragma once

// The elementwise block ops (block_pack, block_unpack, block_quantize and its
// in-place twin) as their kernels see them: raw pointers, extents and element
// strides, with no ATen, so the CUDA kernels compile without its front end
// (the tensors stay in cuda/block_entry.cpp, as the GEMM's stay in
// cuda/custom_matmul_entry.cpp). The format arrives as a BlockFormatParams
// and the casts' constants, built once per call (common/block_host.h).

#include "block_decode.h"
#include "gemm_dtype.h"
#include "gemm_gather.h" // FastDivmod
#include <cstdint>

namespace mptorch::block
{

  // The element format's cast and the MINIFLOAT scale's, of either family
  // (block_decode.h's BlockCastT), in binary32.
  using BlockCast = BlockCastT<float>;

  // block_pack (quant = false: codes into data, scale codes into scales) and
  // block_quantize (quant = true: the decoded values into o, which is x
  // itself for the in-place op). x is [batch, rows, cols] with the packed
  // axis last, read through its element strides, so a transposed view packs
  // with no copy; o is written through its own. data is [batch, rows,
  // row_bytes] and scales [batch, row_tiles, scale_cols], contiguous. The
  // stochastic draw of element (b, r, c) is word (i & 3) of Philox block
  // i >> 2 of (seed, offset), i = (b * rows + r) * cols + c: the packed
  // orientation's logical index, the same in all three ops.
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
    // The CUDA 1D kernel splits a segment index into (row, block) and a row
    // into (batch, row) per segment, by these multiply-high divisors when
    // every index fits in 31 bits (`fast`), a few instructions each, and by
    // 64-bit division, a software routine on the device, otherwise.
    bool fast = false;
    mptorch::gemm::FastDivmod by_blocks{}, by_rows{};
  };

  // block_unpack: data and scales as block_pack writes them, o [batch, rows,
  // cols] contiguous in dtype dt.
  struct BlockUnpackJob
  {
    const uint8_t *data = nullptr;
    const uint8_t *scales = nullptr;
    void *o = nullptr;
    mptorch::GemmDtype dt = mptorch::GemmDtype::Float;
    int64_t batch = 1, rows = 0, cols = 0;
  };

} // namespace mptorch::block
