#pragma once

// The CUDA backend as common/gemm_host.h's driver sees it: a launch context
// and one `launch` per op. Nothing here reaches nvcc-only code, so the .cpp
// that owns the tensors can include it; the `launch` bodies live in
// custom_matmul_kernel.cuh and are instantiated by the four .cu files.
//
// That split is the point (finding H1). A .cu pays ~25 s of nvcc just to put
// ATen's headers through the front end, against 3.2 s for one that includes
// only the policies and the CUDA fp16 headers -- and the only thing the GEMM
// .cu files ever wanted from ATen was at::Tensor in a signature and
// getCurrentCUDAStream() in a launch. Both now happen on the g++ side of this
// header.

#include "../common/gemm_args.h"
#include <ATen/cuda/PhiloxCudaState.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mptorch::gemm_cuda
{

  struct CudaBackend
  {
    // Whatever the kernel needs beyond the shape. The Philox state is drawn
    // even when use_rng is false (as an empty state) so the launch signature
    // does not branch; only RoundMode::SR consumes it.
    struct LaunchContext
    {
      at::PhiloxCudaState rng{};
      cudaStream_t stream = nullptr;
    };

    // gemm_backend.cpp: draws (seed, offset) from ATen's default CUDA
    // generator and picks up the current stream.
    static LaunchContext make_context(bool use_rng, uint64_t draws_per_thread);

    // custom_matmul_kernel.cuh defines this; each custom_matmul_*.cu
    // instantiates it for the two ops it owns, which is what keeps every
    // kernel specialization in exactly one object (finding B2).
    template <class Args>
    static void launch(const mptorch::gemm::GemmShape &s, const Args &args,
                       const LaunchContext &ctx);
  };

} // namespace mptorch::gemm_cuda
