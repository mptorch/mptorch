// The CPU kernels and entry points of the elementwise block ops: block_pack,
// block_unpack, block_quant and block_quant_ (common/block_decode.h has the
// format; mptorch/quant/block.py the Python side).
//
// A pack or quantize call is one task per tile, a tile being block_rows rows
// of one block (block_rows = 1 for a 1D format), over ATen's thread pool: an
// amax pass over the tile, its scale, then an encode pass over the same
// elements, which reads element i before it writes it and touches no other,
// so the in-place quantizer is sound. The encode rounds through the element
// format's cast in the call's rounding mode, a template parameter as for the
// GEMM, and with the sign as a template parameter as in binaryK_kernel.cpp,
// which folds the cast's unsigned early return away (2.5x there). Inputs are
// read through their strides, so a transposed view packs without a copy.
//
// Tiles are disjoint and each element's stochastic draw is keyed on its own
// index (common/block_kernels.h), so results do not depend on the thread
// count.

#include "../common/block_host.h"
#include "../common/dispatch.h"
#include "../common/philox.h"
#include "../quant_ops.h"
#include "utils.h" // draw_cpu_seed, quant_grain_size
#include <ATen/Parallel.h>
#include <ATen/ops/empty_like.h>
#include <algorithm>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cstring>

using at::Tensor;
using namespace mptorch::block;

namespace
{

  template <typename scalar_t, RoundMode RM, bool IsSigned>
  void pack_tiles(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c, uint64_t seed)
  {
    using F = FloatTraits<float>;
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
        uint32_t amax = 0;
        bool nan = false;
        for (int64_t r = r0; r < r1; ++r)
          for (int64_t cc = c0; cc < c1; ++cc)
          {
            const uint32_t w = block_word(static_cast<float>(xb[r * j.xs1 + cc * j.xs2])) & F::ABS_MASK;
            if (w > F::INF_BITS)
              nan = true;
            else if (w > amax)
              amax = w;
          }
        const TileScale<float> s = tile_scale<float>(block_float<float>(amax), nan, p, c);
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
            const float v = static_cast<float>(xb[r * j.xs1 + cc * j.xs2]);
            uint32_t rv = 0;
            if constexpr (RM == RoundMode::SR)
            {
              const int64_t i = row_idx + cc;
              if ((i >> 2) != drawn)
              {
                drawn = i >> 2;
                draws = philox_block(seed, static_cast<uint64_t>(drawn), 0);
              }
              rv = draws.word(static_cast<int>(i & 3));
            }
            const float q = block_cast<float, RM>(scaled_elem(v, s, p), IsSigned, rv, c);
            const uint32_t code = encode_elem(v, q, p);
            if (j.quant)
              o[b * j.os0 + r * j.os1 + cc * j.os2] = static_cast<scalar_t>(decode_value<float>(code, s.code, p));
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
  void pack_by_mode(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c, uint64_t seed)
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

  void run_pack(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c)
  {
    const uint64_t seed = j.rm == RoundMode::SR ? draw_cpu_seed() : 0;
    switch (j.dt)
    {
    case mptorch::GemmDtype::Half:
      return pack_by_mode<at::Half>(j, p, c, seed);
    case mptorch::GemmDtype::BFloat16:
      return pack_by_mode<at::BFloat16>(j, p, c, seed);
    default:
      return pack_by_mode<float>(j, p, c, seed);
    }
  }

  // block_unpack: one task per row of the data, each element decoded with
  // its tile's scale into a contiguous result.
  template <typename scalar_t>
  void unpack_rows(const BlockUnpackJob &j, const BlockFormatParams &p)
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
          o[g * j.cols + cc] = static_cast<scalar_t>(decode_value<float>(code, sc, p));
        }
      }
    });
  }

  void run_unpack(const BlockUnpackJob &j, const BlockFormatParams &p)
  {
    switch (j.dt)
    {
    case mptorch::GemmDtype::Half:
      return unpack_rows<at::Half>(j, p);
    case mptorch::GemmDtype::BFloat16:
      return unpack_rows<at::BFloat16>(j, p);
    default:
      return unpack_rows<float>(j, p);
    }
  }

  // The body of block_quant and block_quant_: every element of x rounded
  // into o, which is x itself for the in-place op.
  void quantize_into(const Tensor &x, Tensor &o, c10::IntArrayRef fmt, double elem_max,
                     double scale_max, double tensor_scale, int64_t round_mode, const char *op)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
    const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op);
    const BlockExtents e = block_extents(x, op);
    const mptorch::GemmDtype dt = block_dtype(x.scalar_type(), op);
    if (x.numel() == 0)
      return;
    const BlockFormatParams p = block_params(f, elem_max, scale_max, tensor_scale, e.cols, e.rows);
    BlockPackJob j = pack_job(x, e, dt, round_mode);
    set_output(j, o, e);
    run_pack(j, p, make_block_cast(f, elem_max));
  }

} // namespace

std::tuple<Tensor, Tensor> block_pack_cpu(Tensor x, c10::IntArrayRef fmt, double elem_max,
                                          double scale_max, double tensor_scale, int64_t round_mode)
{
  constexpr const char *op = "block_pack";
  TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
  const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op);
  const BlockExtents e = block_extents(x, op);
  const mptorch::GemmDtype dt = block_dtype(x.scalar_type(), op);
  const BlockFormatParams p = block_params(f, elem_max, scale_max, tensor_scale, e.cols, e.rows);
  Tensor data = at::empty(block_sizes(e, e.rows, p.row_bytes), x.options().dtype(at::kByte));
  Tensor scales = at::empty(block_sizes(e, p.row_tiles, p.scale_cols), x.options().dtype(at::kByte));
  if (x.numel() == 0)
    return {data, scales};
  BlockPackJob j = pack_job(x, e, dt, round_mode);
  j.data = data.data_ptr<uint8_t>();
  j.scales = scales.data_ptr<uint8_t>();
  run_pack(j, p, make_block_cast(f, elem_max));
  return {data, scales};
}

Tensor block_unpack_cpu(Tensor data, Tensor scales, int64_t cols, c10::IntArrayRef fmt,
                        double elem_max, double scale_max, double tensor_scale, at::ScalarType dtype)
{
  constexpr const char *op = "block_unpack";
  const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op);
  const mptorch::GemmDtype dt = block_dtype(dtype, op);
  BlockFormatParams p;
  const BlockExtents e = unpack_extents(data, scales, cols, f, elem_max, scale_max, tensor_scale, p, op);
  Tensor o = at::empty(block_sizes(e, e.rows, e.cols), data.options().dtype(dtype));
  if (o.numel() == 0)
    return o;
  const Tensor d = data.contiguous(), s = scales.contiguous();
  BlockUnpackJob j;
  j.data = d.data_ptr<uint8_t>();
  j.scales = s.data_ptr<uint8_t>();
  j.o = o.data_ptr();
  j.dt = dt;
  j.batch = e.batch;
  j.rows = e.rows;
  j.cols = e.cols;
  run_unpack(j, p);
  return o;
}

Tensor block_quantize_cpu(Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                          double tensor_scale, int64_t round_mode)
{
  // A result laid out as x is (empty_like keeps a dense view's strides), so
  // a view with its packed axis moved last comes back as the caller's layout
  // once the axis is moved back.
  Tensor o = at::empty_like(x);
  quantize_into(x, o, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant");
  return o;
}

Tensor &block_quantize_cpu_(Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                            double tensor_scale, int64_t round_mode)
{
  TORCH_CHECK(x.is_non_overlapping_and_dense(), "block_quant_ writes its argument in place and "
              "needs a tensor whose elements do not overlap, got strides ", x.strides(),
              " for sizes ", x.sizes(), ": use block_quant, which writes a new tensor");
  quantize_into(x, x, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant_");
  return x;
}
