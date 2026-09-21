#pragma once

// The CUDA backend as common/gemm_host.h's driver sees it: a launch context
// and one `launch` per op. Nothing here needs nvcc, so the .cpp that owns the
// tensors (custom_matmul_entry.cpp) includes it; the `launch_as` bodies are
// in custom_matmul_kernel.cuh and are instantiated by the eight
// custom_matmul_*.cu files.
//
// The split exists for build time. Under nvcc a translation unit that
// includes ATen's tensor headers pays about 25 s of fixed front-end cost,
// against about 3 s for one that includes only the policy headers and the
// CUDA fp16 headers. The only things the GEMM .cu files ever needed from ATen
// were at::Tensor in a signature and getCurrentCUDAStream() in a launch, and
// both now happen on the g++ side of this header.

#include "../common/gemm_accumulate.h"
#include "../common/gemm_args.h"
#include <ATen/cuda/PhiloxCudaState.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mptorch::gemm_cuda
{

  struct CudaBackend
  {
    // What the kernel needs beyond the shape. The Philox state is present
    // even when use_rng is false (as an empty state) so the launch signature
    // does not branch on it; only RoundMode::SR reads it.
    struct LaunchContext
    {
      at::PhiloxCudaState rng{};
      cudaStream_t stream = nullptr;
    };

    // gemm_backend.cpp: draws (seed, offset) from ATen's default CUDA
    // generator and picks up the current stream.
    static LaunchContext make_context(bool use_rng, uint64_t draws_per_thread);

    // Defined in custom_matmul_kernel.cuh. Each custom_matmul_*.cu
    // instantiates it for the two ops it owns in binary32 (T = float), and
    // its custom_matmul_*_f64.cu twin for the same two ops in binary64
    // (T = double), so every kernel specialization lives in exactly one
    // object and the binary64 kernels can be left out of a build.
    template <class T, class Args>
    static void launch_as(const mptorch::gemm::GemmShape &s, const Args &args,
                          const LaunchContext &ctx);

    // The carrier is chosen here, on the host, once per call: a float64 GEMM
    // runs the binary64 kernels, and every other dtype runs the binary32
    // ones, whose loads convert on the device. A build with MPTORCH_NO_FP64
    // has no binary64 objects to call, and gemm_dtype_of has already refused
    // float64 operands by the time this runs.
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

    // KAHAN, BLOCK and TREE (common/gemm_accumulate.h). The algorithm is a
    // template parameter of the launch, as the round mode is of the policies,
    // and for the same reason: each is its own accumulate step, and a switch
    // between them inside the K-loop would keep all of them live in it. It
    // is resolved here, on the host, once per call. Defined in the kernel
    // header next to launch_as and instantiated, in binary32 only so far
    // (the driver has refused float64 operands by now), by the six
    // custom_matmul_{binaryK,superfp}_{kahan,block,tree}.cu
    // files, one per family and algorithm, so that the eight NAIVE files
    // compile what they always compiled.
    template <class T, AccumulateAlgorithm ALG, class Base>
    static void launch_accumulated_as(const mptorch::gemm::GemmShape &s,
                                      const mptorch::gemm::AccumulateArgs<Base> &args,
                                      const LaunchContext &ctx);

    template <class Base>
    static void launch(const mptorch::gemm::GemmShape &s,
                       const mptorch::gemm::AccumulateArgs<Base> &args, const LaunchContext &ctx)
    {
      switch (args.alg)
      {
      case AccumulateAlgorithm::KAHAN:
        return launch_accumulated_as<float, AccumulateAlgorithm::KAHAN>(s, args, ctx);
      case AccumulateAlgorithm::BLOCK:
        return launch_accumulated_as<float, AccumulateAlgorithm::BLOCK>(s, args, ctx);
      case AccumulateAlgorithm::TREE:
        // A fused mac has no tree (no product term): check_accumulate_algorithm
        // has refused it, and no such kernel is instantiated to call.
        if constexpr (mptorch::gemm::AccumulateArgs<Base>::has_product)
          return launch_accumulated_as<float, AccumulateAlgorithm::TREE>(s, args, ctx);
        return;
      case AccumulateAlgorithm::NAIVE:
        return; // refused by run_custom_matmul_accumulated: NAIVE is the twin op's
      }
    }
  };

} // namespace mptorch::gemm_cuda
