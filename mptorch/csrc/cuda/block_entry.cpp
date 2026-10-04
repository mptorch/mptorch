// The CUDA entry points of the elementwise block ops: the tensors, the stream
// and the generator draw, and nothing nvcc has to see (block_kernel.h says
// why). A .cpp inside csrc/cuda/, so a CPU-only build skips it along with the
// .cu it drives. The checks and shapes are common/block_host.h's, shared
// with the CPU entry points in cpu/block_kernel.cpp.

#include "../common/block_host.h"
#include "../quant_ops.h"
#include "block_kernel.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/empty_like.h>
#include <algorithm>
#include <mutex>

using at::Tensor;
using namespace mptorch::block;

namespace
{
  // The 1D kernel's divisors (common/block_kernels.h), when every segment
  // and row index fits in 31 bits.
  void set_divisors(BlockPackJob &j, const BlockFormatParams &p)
  {
    constexpr int64_t LIM = (int64_t(1) << 31) - 1;
    j.fast = j.batch * j.rows * p.n_blocks <= LIM && p.n_blocks <= LIM && j.rows <= LIM;
    if (j.fast)
    {
      j.by_blocks = mptorch::gemm::make_fast_divmod(static_cast<int32_t>(std::max<int64_t>(p.n_blocks, 1)));
      j.by_rows = mptorch::gemm::make_fast_divmod(static_cast<int32_t>(std::max<int64_t>(j.rows, 1)));
    }
  }

  // (seed, offset) from ATen's default CUDA generator under RoundMode::SR,
  // reserving one Philox block per subsequence: an element draws word i & 3
  // of block `offset` of subsequence i >> 2, as the elementwise quantizers
  // do (utils.cuh's quant_rng_engine_inputs).
  at::PhiloxCudaState draw_rng(RoundMode rm)
  {
    if (rm != RoundMode::SR)
      return at::PhiloxCudaState{};
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    return gen->philox_cuda_state(1);
  }

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
    set_divisors(j, p);
    mptorch::block_cuda::launch_block_pack(j, p, make_block_cast(f, elem_max), draw_rng(j.rm),
                                           at::cuda::getCurrentCUDAStream());
  }
} // namespace

std::tuple<Tensor, Tensor> block_pack_cuda(Tensor x, c10::IntArrayRef fmt, double elem_max,
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
  set_divisors(j, p);
  mptorch::block_cuda::launch_block_pack(j, p, make_block_cast(f, elem_max), draw_rng(j.rm),
                                         at::cuda::getCurrentCUDAStream());
  return {data, scales};
}

Tensor block_unpack_cuda(Tensor data, Tensor scales, int64_t cols, c10::IntArrayRef fmt,
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
  mptorch::block_cuda::launch_block_unpack(j, p, at::cuda::getCurrentCUDAStream());
  return o;
}

Tensor block_quantize_cuda(Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                           double tensor_scale, int64_t round_mode)
{
  Tensor o = at::empty_like(x);
  quantize_into(x, o, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant");
  return o;
}

Tensor &block_quantize_cuda_(Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                             double tensor_scale, int64_t round_mode)
{
  TORCH_CHECK(x.is_non_overlapping_and_dense(), "block_quant_ writes its argument in place and "
              "needs a tensor whose elements do not overlap, got strides ", x.strides(),
              " for sizes ", x.sizes(), ": use block_quant, which writes a new tensor");
  quantize_into(x, x, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant_");
  return x;
}
