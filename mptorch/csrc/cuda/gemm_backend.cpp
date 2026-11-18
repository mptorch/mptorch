#include "gemm_backend.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <mutex>

namespace mptorch::gemm_cuda
{

  // Takes the current stream and, under RoundMode::SR, reserves random
  // numbers from ATen's default CUDA generator, so that torch.manual_seed
  // governs the GEMM's draws exactly as it governs the elementwise
  // quantizers' (quant_rng_engine_inputs in cuda/utils.cuh). The generator's
  // offset advances by counter_offset Philox blocks of 128 bits, so that no
  // later op that draws random numbers reuses the same (seed, offset) pair;
  // ATen's own CUDA random kernels do the same (native/cuda/Dropout.cu, for
  // one). draws_per_thread is the driver's upper bound on the 32-bit words
  // that one output element's thread consumes over its reduction:
  // Args::draws_per_k_step (2 for a SplitMac, which draws for its multiply
  // and for its add, and 1 for a FusedMac) times K, doubled in binary64,
  // where a draw is two words. Philox yields four words per block.
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
