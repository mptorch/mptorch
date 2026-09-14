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
    // instantiates it for the two ops it owns in binary32, and its
    // custom_matmul_*_f64.cu twin in binary64, which is what keeps every
    // kernel specialization in exactly one object (finding B2) and every
    // existing object the size it was (dev/binary64_carrier_plan.md, phase 4).
    template <class T, class Args>
    static void launch_as(const mptorch::gemm::GemmShape &s, const Args &args,
                          const LaunchContext &ctx);

    // The carrier is chosen here, on the host, once per call: a float64 GEMM
    // runs the binary64 kernels and every other dtype the binary32 ones, whose
    // loads convert on the device. A build with MPTORCH_NO_FP64 has no
    // binary64 objects to call, and gemm_dtype_of has already refused the
    // dtype by the time this runs.
    template <class Args>
    static void launch(const mptorch::gemm::GemmShape &s, const Args &args,
                       const LaunchContext &ctx)
    {
#if !defined(MPTORCH_NO_FP64)
      if (s.dt == mptorch::GemmDtype::Double)
        return launch_as<double>(s, args, ctx);
#endif
      launch_as<float>(s, args, ctx);
    }
  };

} // namespace mptorch::gemm_cuda
