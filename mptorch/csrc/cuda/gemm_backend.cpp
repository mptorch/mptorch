#include "gemm_backend.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <mutex>

namespace mptorch::gemm_cuda
{

  // Draws (seed, offset) from ATen's default CUDA generator (respecting
  // torch.manual_seed, same as the elementwise binaryK_quantize/
  // superfp_quantize SR path's quant_rng_engine_inputs in cuda/utils.cuh),
  // reserving `counter_offset` 128-bit Philox blocks so a subsequent
  // unrelated RNG-consuming op doesn't reuse the same (seed, offset) pair --
  // the standard native CUDA RNG kernel idiom (see e.g. native/cuda/Dropout.cu
  // upstream). Only drawn when RoundMode::SR is selected. draws_per_thread is
  // a safe upper bound on how many random values any single output element's
  // thread may draw over its K-step reduction (2*K for SplitMac's independent
  // mul/add draws, K for FusedMac's single draw per step -- Args::
  // draws_per_k_step times K, applied by the driver) -- Philox batches 4
  // draws per 128-bit block.
  CudaBackend::LaunchContext CudaBackend::make_context(bool use_rng, uint64_t draws_per_thread)
  {
    LaunchContext ctx;
    ctx.stream = at::cuda::getCurrentCUDAStream();
    if (!use_rng)
      return ctx;
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
    uint64_t counter_offset = (draws_per_thread + 3) / 4;
    std::lock_guard<std::mutex> lock(gen->mutex_);
    ctx.rng = gen->philox_cuda_state(counter_offset);
    return ctx;
  }

} // namespace mptorch::gemm_cuda
