#pragma once

// The CPU kernel of block_pack, block_quant and block_quant_ under a Four Over
// Six rule, as templates on the storage dtype, whose carrier (carrier_t) it
// rounds in: block_kernel_fos.cpp instantiates it for float32, float16 and
// bfloat16, and block_kernel_f64.cpp for float64. block_kernel_fos.cpp says
// how it works. The body is that file's from before binary64 had block
// kernels, with the carrier spelled T, in an unnamed namespace as it was.

#include "../common/block_kernels.h"
#include "../common/dispatch.h"
#include "../common/philox.h"
#include "utils.h" // quant_grain_size
#include <ATen/Parallel.h>
#include <algorithm>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cstring>
#include <memory>

namespace
{
  using namespace mptorch::block;

  template <typename scalar_t, RoundMode RM, bool IsSigned>
  void pack_tiles_fos(const BlockPackJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p,
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
      // A tile's codes under each candidate, row-major: on the stack for a 1D
      // block, and for a 2D tile (up to 16,384 elements) in a buffer this
      // task allocates once.
      uint16_t row_codes[2][128];
      std::unique_ptr<uint16_t[]> tile_codes;
      uint16_t *kept[2] = {row_codes[0], row_codes[1]};
      if (tile_elems > 128)
      {
        tile_codes.reset(new uint16_t[static_cast<size_t>(2 * tile_elems)]);
        kept[0] = tile_codes.get();
        kept[1] = tile_codes.get() + tile_elems;
      }
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
        TileScale<T> s = tile_scale<T>(block_float<T>(amax), nan, p, c);

        // The tile under both candidate scales, and the second candidate if
        // its error is strictly the smaller. `chosen` is the codes the encode
        // pass writes, when this pass made them.
        const uint16_t *chosen = nullptr;
        const TileScale<T> low = four_over_six_low<T>(block_float<T>(amax), s, p, c);
        if (low.code != s.code)
        {
          const int amax_exp = block_floor_log2(block_float<T>(amax));
          uint64_t err_six = 0, err_low = 0;
          for (int64_t r = r0; r < r1; ++r)
          {
            const int64_t row_idx = (b * j.rows + r) * j.cols;
            uint16_t *row_six = kept[0] + (r - r0) * p.block_size;
            uint16_t *row_low = kept[1] + (r - r0) * p.block_size;
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
              uint32_t code_six, code_low;
              err_six = add_block_error(
                  err_six, candidate_error_units<T, RM>(v, s, amax_exp, IsSigned, rv, p, c, code_six),
                  c.scale_rounding);
              err_low = add_block_error(
                  err_low, candidate_error_units<T, RM>(v, low, amax_exp, IsSigned, rv, p, c, code_low),
                  c.scale_rounding);
              row_six[cc - c0] = static_cast<uint16_t>(code_six);
              row_low[cc - c0] = static_cast<uint16_t>(code_low);
            }
          }
          chosen = kept[err_low < err_six ? 1 : 0];
          if (err_low < err_six)
            s = low;
        }
        if (!j.quant && p.scale_cols != 0)
          j.scales[(b * p.row_tiles + rt) * p.scale_cols + blk] = static_cast<uint8_t>(s.code);

        for (int64_t r = r0; r < r1; ++r)
        {
          if (!j.quant)
            std::memset(buf, 0, static_cast<size_t>(p.block_bytes));
          const int64_t row_idx = (b * j.rows + r) * j.cols;
          const uint16_t *row_chosen = chosen != nullptr ? chosen + (r - r0) * p.block_size : nullptr;
          PhiloxBlock draws;
          int64_t drawn = -1;
          for (int64_t cc = c0; cc < c1; ++cc)
          {
            uint32_t code;
            if (row_chosen != nullptr)
              code = row_chosen[cc - c0];
            else
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
              code = encode_elem(v, q, p);
            }
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
  void pack_fos_by_mode(const BlockPackJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p,
                        const BlockCastT<carrier_t<scalar_t>> &c, uint64_t seed)
  {
    mptorch::dispatch_round_mode(j.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      if (c.elem_signed)
        pack_tiles_fos<scalar_t, RM, true>(j, p, c, seed);
      else
        pack_tiles_fos<scalar_t, RM, false>(j, p, c, seed);
    });
  }

} // namespace
