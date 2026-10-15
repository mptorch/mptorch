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
// A Four Over Six rule's calls leave for block_kernel_fos.cpp before any of
// that (run_pack). The rule reads a tile's elements between the two passes,
// and written into pack_tiles, even behind a template parameter, it moved the
// register and stack assignment of every other rule's instantiation
// (dev/gemm_roadmap.md, R-3), so pack_tiles is the text it was before the
// rule and is not the place for the next one either.
//
// Tiles are disjoint and each element's stochastic draw is keyed on its own
// index (common/block_kernels.h), so results do not depend on the thread
// count.
//
// The kernels themselves are templates on the storage dtype in
// block_pack_tiles.h and block_pack_tiles_fos.h, which round in its carrier:
// this file instantiates the binary32 ones (float32, float16, bfloat16), and
// block_kernel_f64.cpp the binary64 ones (float64), so that MPTORCH_NO_FP64
// can leave those out with the file. The entry points pick the carrier from
// the dtype and check the format against it.

#include "../common/block_host.h"
#include "../quant_ops.h"
#include "block_kernel_f64.h"
#include "block_kernel_fos.h"
#include "block_pack_tiles.h"
#include "utils.h" // draw_cpu_seed
#include <ATen/ops/empty_like.h>

using at::Tensor;
using namespace mptorch::block;

namespace
{

  void run_pack(const BlockPackJob &j, const BlockFormatParams &p, const BlockCast &c)
  {
    const uint64_t seed = j.rm == RoundMode::SR ? draw_cpu_seed() : 0;
    if (is_four_over_six(c.scale_rounding))
      return mptorch::block_cpu::run_pack_fos(j, p, c, seed);
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

#if !defined(MPTORCH_NO_FP64)
  // binary64: a float64 tensor, whose kernels block_kernel_f64.cpp holds.
  void run_pack(const BlockPackJob &j, const BlockFormatParamsT<double> &p, const BlockCastT<double> &c)
  {
    mptorch::block_cpu::run_pack_f64(j, p, c, j.rm == RoundMode::SR ? draw_cpu_seed() : 0);
  }

  void run_unpack(const BlockUnpackJob &j, const BlockFormatParamsT<double> &p)
  {
    mptorch::block_cpu::run_unpack_f64(j, p);
  }
#endif

  // Calls `f` with a value of the carrier dtype `dt` rounds in: double for
  // float64 (block_dtype has refused it in a build without binary64
  // kernels), float otherwise.
  template <class F>
  void with_carrier(mptorch::GemmDtype dt, F &&f)
  {
#if !defined(MPTORCH_NO_FP64)
    if (dt == mptorch::GemmDtype::Double)
      return f(double{});
#endif
    f(float{});
  }

  // The body of block_quant and block_quant_: every element of x rounded
  // into o, which is x itself for the in-place op.
  void quantize_into(const Tensor &x, Tensor &o, c10::IntArrayRef fmt, double elem_max,
                     double scale_max, double tensor_scale, int64_t round_mode, const char *op)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
    const BlockExtents e = block_extents(x, op);
    const mptorch::GemmDtype dt = block_dtype(x.scalar_type(), op);
    const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, dt == mptorch::GemmDtype::Double);
    if (x.numel() == 0)
      return;
    with_carrier(dt, [&](auto t)
    {
      using T = decltype(t);
      const BlockFormatParamsT<T> p = block_params<T>(f, elem_max, scale_max, tensor_scale, e.cols, e.rows);
      BlockPackJob j = pack_job(x, e, dt, round_mode);
      set_output(j, o, e);
      run_pack(j, p, make_block_cast<T>(f, elem_max));
    });
  }

} // namespace

std::tuple<Tensor, Tensor> block_pack_cpu(Tensor x, c10::IntArrayRef fmt, double elem_max,
                                          double scale_max, double tensor_scale, int64_t round_mode)
{
  constexpr const char *op = "block_pack";
  TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
  const BlockExtents e = block_extents(x, op);
  const mptorch::GemmDtype dt = block_dtype(x.scalar_type(), op);
  const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, dt == mptorch::GemmDtype::Double);
  std::tuple<Tensor, Tensor> out;
  with_carrier(dt, [&](auto t)
  {
    using T = decltype(t);
    const BlockFormatParamsT<T> p = block_params<T>(f, elem_max, scale_max, tensor_scale, e.cols, e.rows);
    Tensor data = at::empty(block_sizes(e, e.rows, p.row_bytes), x.options().dtype(at::kByte));
    Tensor scales = at::empty(block_sizes(e, p.row_tiles, p.scale_cols), x.options().dtype(at::kByte));
    out = {data, scales};
    if (x.numel() == 0)
      return;
    BlockPackJob j = pack_job(x, e, dt, round_mode);
    j.data = data.data_ptr<uint8_t>();
    j.scales = scales.data_ptr<uint8_t>();
    run_pack(j, p, make_block_cast<T>(f, elem_max));
  });
  return out;
}

Tensor block_unpack_cpu(Tensor data, Tensor scales, int64_t cols, c10::IntArrayRef fmt,
                        double elem_max, double scale_max, double tensor_scale, at::ScalarType dtype)
{
  constexpr const char *op = "block_unpack";
  const mptorch::GemmDtype dt = block_dtype(dtype, op);
  const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, dt == mptorch::GemmDtype::Double);
  Tensor o;
  with_carrier(dt, [&](auto t)
  {
    using T = decltype(t);
    BlockFormatParamsT<T> p;
    const BlockExtents e = unpack_extents(data, scales, cols, f, elem_max, scale_max, tensor_scale, p, op);
    o = at::empty(block_sizes(e, e.rows, e.cols), data.options().dtype(dtype));
    if (o.numel() == 0)
      return;
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
  });
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
