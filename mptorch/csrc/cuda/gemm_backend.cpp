#include "gemm_backend.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <mutex>

namespace mptorch::gemm_cuda
{

  // Picks up the current stream and, under RoundMode::SR, draws (seed,
  // offset) from ATen's default CUDA generator, so torch.manual_seed governs
  // the GEMM's draws exactly as it governs the elementwise quantizers'
  // (quant_rng_engine_inputs in utils.cuh). The generator's offset is
  // advanced by `counter_offset` 128-bit Philox blocks so that no later
  // RNG-consuming op reuses the same (seed, offset) pair; this is the idiom
  // of ATen's own CUDA RNG kernels (native/cuda/Dropout.cu, for one).
  // draws_per_thread is an upper bound on the 32-bit words one output
  // element's thread consumes over its K-step reduction, as the driver
  // computes it: Args::draws_per_k_step (2 for SplitMac's separate multiply
  // and add draws, 1 for FusedMac's one per step) times K, doubled in
  // binary64 where a draw is two words. Philox yields four words per block.
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
