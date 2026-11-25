// The CPU kernel and entry point of block_project (common/block_project.h
// states the op; mptorch/quant/ops/block.py is the Python side).
//
// One task per row of x on ATen's thread pool: for each block of the row,
// the tile's scale code is decoded once, and the block's elements are
// projected into a zeroed buffer of its bytes, as block_pack's encode pass
// writes them (cpu/block_pack_tiles.h), then copied out. The rounding mode
// and the element format's sign are template parameters, as in block_pack.
// Each element's stochastic draw is keyed on its own index, as in
// block_pack, so the codes do not depend on the thread count.
//
// The kernel is a template on the storage dtype, rounding in its carrier:
// binary32 for float32, float16 and bfloat16, binary64 for float64, which a
// build with MPTORCH_NO_FP64 leaves out (block_dtype refuses float64 there).

#include "../common/block_project_host.h"
#include "../common/dispatch.h"
#include "../common/philox.h"
#include "../quant_ops.h"
#include "utils.h" // draw_cpu_seed, quant_grain_size
#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>
#include <algorithm>
#include <atomic>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cstring>

using at::Tensor;
using namespace mptorch::block;

namespace
{

  template <typename scalar_t, RoundMode RM, bool IsSigned>
  void project_rows(const BlockProjectJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p,
                    const BlockCastT<carrier_t<scalar_t>> &c, uint64_t seed, std::atomic<bool> &unstored)
  {
    using T = carrier_t<scalar_t>;
    using F = FloatTraits<T>;
    const scalar_t *x = static_cast<const scalar_t *>(j.x);
    const int64_t grain = std::max<int64_t>(1, mptorch_cpu::quant_grain_size / std::max<int64_t>(j.cols, 1));

    at::parallel_for(0, j.batch * j.rows, grain, [&](int64_t g_begin, int64_t g_end)
    {
      uint8_t buf[256]; // one block row's codes: at most 128 codes of 16 bits
      bool bad = false;
      for (int64_t g = g_begin; g < g_end; ++g)
      {
        const int64_t b = g / j.rows;
        const int64_t r = g - b * j.rows;
        const scalar_t *xr = x + b * j.xs0 + r * j.xs1;
        const uint8_t *srow =
            p.scale_cols != 0 ? j.scales + (b * p.row_tiles + (r >> p.log2_block_rows)) * p.scale_cols : nullptr;
        const int64_t row_idx = g * j.cols;
        PhiloxBlock draws;
        int64_t drawn = -1;
        for (int64_t blk = 0; blk < p.n_blocks; ++blk)
        {
          std::memset(buf, 0, static_cast<size_t>(p.block_bytes));
          const ProjectScale<T> s = project_scale<T>(srow != nullptr ? srow[blk] : 0u, p);
          const int64_t c0 = blk * p.block_size, c1 = std::min<int64_t>(c0 + p.block_size, j.cols);
          for (int64_t cc = c0; cc < c1; ++cc)
          {
            const T v = static_cast<T>(xr[cc * j.xs2]);
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
            put_code(buf, cc - c0, project_code<T, RM>(v, s, IsSigned, rv, p, c, bad), p.elem_bits);
          }
          std::memcpy(j.data + g * p.row_bytes + blk * p.block_bytes, buf, static_cast<size_t>(p.block_bytes));
        }
      }
      if (bad)
        unstored.store(true, std::memory_order_relaxed);
    });
  }

  template <typename scalar_t>
  void project_by_mode(const BlockProjectJob &j, const BlockFormatParamsT<carrier_t<scalar_t>> &p,
                       const BlockCastT<carrier_t<scalar_t>> &c, uint64_t seed, std::atomic<bool> &unstored)
  {
    mptorch::dispatch_round_mode(j.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      if (c.elem_signed)
        project_rows<scalar_t, RM, true>(j, p, c, seed, unstored);
      else
        project_rows<scalar_t, RM, false>(j, p, c, seed, unstored);
    });
  }

} // namespace

Tensor block_project_cpu(Tensor x, Tensor scales, c10::IntArrayRef fmt, double elem_max, double scale_max,
                         double tensor_scale, int64_t round_mode)
{
  constexpr const char *op = "block_project";
  bool wide = false;
  const Fmt f = check_block_project(x, scales, fmt, elem_max, scale_max, tensor_scale, round_mode, op, wide);
  Tensor data = at::empty_symint(block_project_sizes(x, f), x.options().dtype(at::kByte));
  if (x.numel() == 0)
    return data;
  const Tensor s = scales.contiguous();
  BlockProjectJob j = project_job(x, s, block_dtype(x.scalar_type(), op), round_mode);
  j.data = data.data_ptr<uint8_t>();
  const uint64_t seed = j.rm == RoundMode::SR ? draw_cpu_seed() : 0;
  std::atomic<bool> unstored{false};
  const auto run = [&](auto t)
  {
    using T = decltype(t);
    const BlockFormatParamsT<T> p = block_params<T>(f, elem_max, scale_max, tensor_scale, j.cols, j.rows);
    const BlockCastT<T> c = make_block_cast<T>(f, elem_max);
    if constexpr (std::is_same_v<T, double>)
      project_by_mode<double>(j, p, c, seed, unstored);
    else if (j.dt == mptorch::GemmDtype::Half)
      project_by_mode<at::Half>(j, p, c, seed, unstored);
    else if (j.dt == mptorch::GemmDtype::BFloat16)
      project_by_mode<at::BFloat16>(j, p, c, seed, unstored);
    else
      project_by_mode<float>(j, p, c, seed, unstored);
  };
#if !defined(MPTORCH_NO_FP64)
  if (wide)
    run(double{});
  else
#endif
    run(float{});
  refuse_unstored_nan(unstored.load(), op);
  return data;
}
