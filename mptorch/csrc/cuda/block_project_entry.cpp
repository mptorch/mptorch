// The CUDA entry point of block_project (common/block_project.h): the
// tensors, the stream and the generator draw, so that nvcc sees none of them
// (cuda/block_kernel.h says why). The checks and sizes are
// common/block_project_host.h's, shared with the CPU, MPS and Meta entry
// points, in the CPU's order.

#include "../common/block_project_host.h"
#include "../quant_ops.h"
#include "block_project.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#include <mutex>

using at::Tensor;
using namespace mptorch::block;

namespace
{
  // block_pack's draw (cuda/block_entry.cpp): one Philox block per
  // subsequence of ATen's default CUDA generator, under RoundMode::SR only.
  at::PhiloxCudaState draw_rng(RoundMode rm)
  {
    if (rm != RoundMode::SR)
      return at::PhiloxCudaState{};
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    std::lock_guard<std::mutex> lock(gen->mutex_);
    return gen->philox_cuda_state(1);
  }
} // namespace

Tensor block_project_cuda(Tensor x, Tensor scales, c10::IntArrayRef fmt, double elem_max, double scale_max,
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
  const at::PhiloxCudaState rng = draw_rng(j.rm);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  Tensor flag;
  const auto run = [&](auto t)
  {
    using T = decltype(t);
    const BlockFormatParamsT<T> p = block_params<T>(f, elem_max, scale_max, tensor_scale, j.cols, j.rows);
    if (project_can_refuse(p))
    {
      flag = at::zeros({1}, x.options().dtype(at::kInt));
      j.unstored = flag.data_ptr<int32_t>();
    }
    mptorch::block_cuda::launch_block_project(j, p, make_block_cast<T>(f, elem_max), rng, stream);
  };
#if !defined(MPTORCH_NO_FP64)
  if (wide)
    run(double{});
  else
#endif
    run(float{});
  // Reading the flag waits for the kernel: only a format that can refuse a
  // NaN pays for it.
  if (flag.defined())
    refuse_unstored_nan(flag.item<int32_t>() != 0, op);
  return data;
}
