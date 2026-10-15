#pragma once

// The CPU pack and unpack kernels of the elementwise block ops, by every
// scale rule but Four Over Six's (block_pack_tiles_fos.h), as templates on
// the storage dtype, whose carrier (carrier_t: binary64 for double, binary32
// for the other three) they round in. block_kernel.cpp instantiates them for
// float32, float16 and bfloat16, and block_kernel_f64.cpp for float64, so a
// build with MPTORCH_NO_FP64 leaves the binary64 ones out with that file.
// block_kernel.cpp says how they work.
//
// The bodies are block_kernel.cpp's from before binary64 had block kernels,
// with the carrier spelled T: a binary32 instantiation compiles to what it
// compiled to there (dev/gemm_roadmap.md, phase G). In an unnamed namespace,
// as they were, so that each including file keeps its own copies.

#include "../common/block_kernels.h"
#include "../common/dispatch.h"
#include "../common/philox.h"
#include "utils.h" // quant_grain_size
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cstring>

namespace
{
  using namespace mptorch::block;

  template <typename scalar_t, RoundMode RM, bool IsSigned>
  void pack_tiles(const BlockPackJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p,
                  const BlockCastT<carrier_t<scalar_t>> &c, uint64_t seed)
  {
    using T = carrier_t<scalar_t>;
    using F = FloatTraits<T>;
    const scalar_t *x = static_cast<const scalar_t *>(j.x);
    scalar_t *o = static_cast<scalar_t *>(j.o);
    const int64_t per_batch = p.row_tiles * p.n_blocks;
    const int64_t tiles = j.batch * per_batch;
    const int64_t tile_elems = int64_t(p.block_rows) * p.block_size;
    const int64_t grain = std::max<int64_t>(1, mptorch_cpu::quant_grain_size / tile_elems);

    at::parallel_for(0, tiles, grain, [&](int64_t t_begin, int64_t t_end)
    {
      uint8_t buf[256]; // one row of a block: at most 128 codes of 16 bits
      for (int64_t t = t_begin; t < t_end; ++t)
      {
        const int64_t b = t / per_batch;
        const int64_t rem = t - b * per_batch;
        const int64_t rt = rem / p.n_blocks;
        const int64_t blk = rem - rt * p.n_blocks;
        const int64_t r0 = rt * p.block_rows, r1 = std::min<int64_t>(r0 + p.block_rows, j.rows);
        const int64_t c0 = blk * p.block_size, c1 = std::min<int64_t>(c0 + p.block_size, j.cols);
        const scalar_t *xb = x + b * j.xs0;

        // The largest magnitude, compared as words (a non-negative float's
        // order is its word's), with the NaNs left out and noted.
        typename F::word_t amax = 0;
        bool nan = false;
        for (int64_t r = r0; r < r1; ++r)
          for (int64_t cc = c0; cc < c1; ++cc)
          {
            const typename F::word_t w = block_word(static_cast<T>(xb[r * j.xs1 + cc * j.xs2])) & F::ABS_MASK;
            if (w > F::INF_BITS)
              nan = true;
            else if (w > amax)
              amax = w;
          }
        const TileScale<T> s = tile_scale<T>(block_float<T>(amax), nan, p, c);
        if (!j.quant && p.scale_cols != 0)
          j.scales[(b * p.row_tiles + rt) * p.scale_cols + blk] = static_cast<uint8_t>(s.code);

        for (int64_t r = r0; r < r1; ++r)
        {
          if (!j.quant)
            std::memset(buf, 0, static_cast<size_t>(p.block_bytes));
          const int64_t row_idx = (b * j.rows + r) * j.cols;
          PhiloxBlock draws;
          int64_t drawn = -1;
          for (int64_t cc = c0; cc < c1; ++cc)
          {
            const T v = static_cast<T>(xb[r * j.xs1 + cc * j.xs2]);
            typename F::word_t rv = 0;
            if constexpr (RM == RoundMode::SR)
            {
              const int64_t i = row_idx + cc;
              if ((i >> block_draw_shift<T>) != drawn)
              {
                drawn = i >> block_draw_shift<T>;
                draws = philox_block(seed, static_cast<uint64_t>(drawn), 0);
              }
              if constexpr (std::is_same_v<T, double>)
                rv = draws.word64(2 * static_cast<int>(i & 1));
              else
                rv = draws.word(static_cast<int>(i & 3));
            }
            const T q = block_cast<T, RM>(scaled_elem(v, s, p), IsSigned, rv, c);
            const uint32_t code = encode_elem(v, q, p);
            if (j.quant)
              o[b * j.os0 + r * j.os1 + cc * j.os2] = static_cast<scalar_t>(decode_value<T>(code, s.code, p));
            else
              put_code(buf, cc - c0, code, p.elem_bits);
          }
          if (!j.quant)
            std::memcpy(j.data + (b * j.rows + r) * p.row_bytes + blk * p.block_bytes, buf,
                        static_cast<size_t>(p.block_bytes));
        }
      }
    });
  }

  template <typename scalar_t>
  void pack_by_mode(const BlockPackJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p,
                    const BlockCastT<carrier_t<scalar_t>> &c, uint64_t seed)
  {
    mptorch::dispatch_round_mode(j.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      if (c.elem_signed)
        pack_tiles<scalar_t, RM, true>(j, p, c, seed);
      else
        pack_tiles<scalar_t, RM, false>(j, p, c, seed);
    });
  }

  // block_unpack: one task per row of the data, each element decoded with
  // its tile's scale into a contiguous result.
  template <typename scalar_t>
  void unpack_rows(const BlockUnpackJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p)
  {
    scalar_t *o = static_cast<scalar_t *>(j.o);
    const int64_t grain = std::max<int64_t>(1, mptorch_cpu::quant_grain_size / std::max<int64_t>(j.cols, 1));
    at::parallel_for(0, j.batch * j.rows, grain, [&](int64_t g_begin, int64_t g_end)
    {
      for (int64_t g = g_begin; g < g_end; ++g)
      {
        const int64_t b = g / j.rows;
        const int64_t r = g - b * j.rows;
        const uint8_t *drow = j.data + g * p.row_bytes;
        const uint8_t *srow = j.scales + (b * p.row_tiles + (r >> p.log2_block_rows)) * p.scale_cols;
        for (int64_t cc = 0; cc < j.cols; ++cc)
        {
          const int64_t blk = cc >> p.log2_block_size;
          const uint32_t code =
              extract_code(drow + blk * p.block_bytes, cc & (p.block_size - 1), p.elem_bits);
          const uint32_t sc = p.scale_cols != 0 ? srow[blk] : 0u;
          o[g * j.cols + cc] = static_cast<scalar_t>(decode_value<carrier_t<scalar_t>>(code, sc, p));
        }
      }
    });
  }

} // namespace
