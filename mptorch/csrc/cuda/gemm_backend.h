#pragma once

// The CUDA backend as the shared drivers (common/gemm_host.h and its
// siblings) see it: a launch context, and one `launch` overload per kind of
// Args, which picks the carrier and the accumulate algorithm and calls a
// kernel template. Nothing here needs nvcc, so the entry files that own the
// tensors (cuda/custom_*_entry.cpp, compiled by g++) include it. The kernel
// templates are only declared here: cuda/custom_matmul_kernel.cuh defines
// them and the units in cuda/inst/ instantiate them.
//
// The split exists for build time. Under nvcc a translation unit that
// includes ATen's tensor headers pays about 25 s of fixed front-end cost,
// against about 3 s for one that includes only the policy headers and the
// CUDA fp16 headers. All the GEMM .cu files ever needed from ATen was
// at::Tensor in a signature and getCurrentCUDAStream() in a launch, and both
// are now on the g++ side of this header.

#include "../common/gemm_accumulate.h"
#include "../common/gemm_args.h"
#include "../common/gemm_block.h"
#include "../common/gemm_cross.h"
#include "../common/gemm_gather.h"
#include <ATen/cuda/PhiloxCudaState.h>
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mptorch::gemm_cuda
{

  struct CudaBackend
  {
    // What the kernel needs beyond the shape. The Philox state is there even
    // when use_rng is false (as an empty state), so that the launch signature
    // does not depend on it; only RoundMode::SR reads it.
    struct LaunchContext
    {
      at::PhiloxCudaState rng{};
      cudaStream_t stream = nullptr;
    };

    // Defined in cuda/gemm_backend.cpp: takes the current stream and, under
    // SR, reserves draws_per_thread random words per thread from ATen's
    // default CUDA generator.
    static LaunchContext make_context(bool use_rng, uint64_t draws_per_thread);

    // Runs the kernel in carrier T (float or double) for one Args type.
    // Defined in cuda/custom_matmul_kernel.cuh and explicitly instantiated in
    // cuda/inst/, once per Args type and carrier, so every kernel is compiled
    // into exactly one object and the binary64 kernels can be left out of a
    // build.
    template <class T, class Args>
    static void launch_as(const mptorch::gemm::GemmShape &s, const Args &args,
                          const LaunchContext &ctx);

    // Picks the carrier on the host, once per call: a float64 GEMM runs the
    // binary64 kernel, and every other dtype the binary32 one, whose loads
    // convert the operands on the device. Under MPTORCH_NO_FP64 the binary64
    // instantiations do not exist, and gemm_dtype_of has already refused a
    // float64 operand with an error that names the flag.
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
    // template parameter of the launch, as the round mode is of the
    // policies, and for the same reason: each algorithm has its own
    // accumulate step, and a switch between them inside the K-loop would
    // keep the code of all of them in the loop. It is chosen here, on the
    // host, once per call, after the carrier. Defined in the kernel header
    // next to launch_as and instantiated by the
    // cuda/inst/custom_matmul_*_{kahan,block,tree}* units.
    template <class T, AccumulateAlgorithm ALG, class Base>
    static void launch_accumulated_as(const mptorch::gemm::GemmShape &s,
                                      const mptorch::gemm::AccumulateArgs<Base> &args,
                                      const LaunchContext &ctx);

    template <class T, class Base>
    static void launch_accumulated(const mptorch::gemm::GemmShape &s,
                                   const mptorch::gemm::AccumulateArgs<Base> &args,
                                   const LaunchContext &ctx)
    {
      switch (args.alg)
      {
      case AccumulateAlgorithm::KAHAN:
        return launch_accumulated_as<T, AccumulateAlgorithm::KAHAN>(s, args, ctx);
      case AccumulateAlgorithm::BLOCK:
        return launch_accumulated_as<T, AccumulateAlgorithm::BLOCK>(s, args, ctx);
      case AccumulateAlgorithm::TREE:
        // TREE sums products, and a fused mac has no separate product to sum:
        // check_accumulate_algorithm has refused it, and no such kernel is
        // instantiated.
        if constexpr (mptorch::gemm::AccumulateArgs<Base>::has_product)
          return launch_accumulated_as<T, AccumulateAlgorithm::TREE>(s, args, ctx);
        return;
      case AccumulateAlgorithm::NAIVE:
        return; // run_custom_matmul_accumulated has refused it: NAIVE is the twin op's
      }
    }

    template <class Base>
    static void launch(const mptorch::gemm::GemmShape &s,
                       const mptorch::gemm::AccumulateArgs<Base> &args, const LaunchContext &ctx)
    {
#if !defined(MPTORCH_NO_FP64)
      if (s.dt == mptorch::GemmDtype::Double)
        return launch_accumulated<double>(s, args, ctx);
#endif
      launch_accumulated<float>(s, args, ctx);
    }

    // The conv ops (common/gemm_gather.h): the same kernel over a Gathered
    // policy built from `Inner`, the Args of the GEMM op whose arithmetic the
    // conv op runs (a single-format op's Args, its AccumulateArgs, or a
    // palette op's Args). ALG is NAIVE unless Inner is an AccumulateArgs.
    // Defined in the kernel header next to launch_as and instantiated by the
    // cuda/inst/custom_conv_* units, the cross-family ones included.
    template <class T, AccumulateAlgorithm ALG, class Inner>
    static void launch_conv_as(const mptorch::gemm::GemmShape &s,
                               const mptorch::gemm::ConvArgs<Inner> &args, const LaunchContext &ctx);

    template <class T, class Inner>
    static void launch_conv(const mptorch::gemm::GemmShape &s, const mptorch::gemm::ConvArgs<Inner> &args,
                            const LaunchContext &ctx)
    {
      if constexpr (requires { args.inner.alg; })
      {
        switch (args.inner.alg)
        {
        case AccumulateAlgorithm::KAHAN:
          return launch_conv_as<T, AccumulateAlgorithm::KAHAN>(s, args, ctx);
        case AccumulateAlgorithm::BLOCK:
          return launch_conv_as<T, AccumulateAlgorithm::BLOCK>(s, args, ctx);
        case AccumulateAlgorithm::TREE:
          if constexpr (Inner::has_product)
            return launch_conv_as<T, AccumulateAlgorithm::TREE>(s, args, ctx);
          return;
        case AccumulateAlgorithm::NAIVE:
          return; // a NAIVE conv call is launched with plain Args, not these
        }
      }
      else
      {
        launch_conv_as<T, AccumulateAlgorithm::NAIVE>(s, args, ctx);
      }
    }

    template <class Inner>
    static void launch(const mptorch::gemm::GemmShape &s, const mptorch::gemm::ConvArgs<Inner> &args,
                       const LaunchContext &ctx)
    {
#if !defined(MPTORCH_NO_FP64)
      if (s.dt == mptorch::GemmDtype::Double)
        return launch_conv<double>(s, args, ctx);
#endif
      launch_conv<float>(s, args, ctx);
    }

    // The block GEMM (common/gemm_block.h): the same kernel over a Blocked
    // policy built from `Inner`, the Args of the mac it runs (BlockSplitArgs,
    // BinaryKFusedArgs, or an AccumulateArgs of either). Defined in the kernel
    // header and instantiated by the cuda/inst/custom_matmul_block* units.
    template <class T, AccumulateAlgorithm ALG, class Inner>
    static void launch_block_as(const mptorch::gemm::GemmShape &s,
                                const mptorch::gemm::BlockGemmArgs<Inner> &args, const LaunchContext &ctx);

    template <class T, class Inner>
    static void launch_block(const mptorch::gemm::GemmShape &s,
                             const mptorch::gemm::BlockGemmArgs<Inner> &args, const LaunchContext &ctx)
    {
      if constexpr (requires { args.inner.alg; })
      {
        switch (args.inner.alg)
        {
        case AccumulateAlgorithm::KAHAN:
          return launch_block_as<T, AccumulateAlgorithm::KAHAN>(s, args, ctx);
        case AccumulateAlgorithm::BLOCK:
          return launch_block_as<T, AccumulateAlgorithm::BLOCK>(s, args, ctx);
        case AccumulateAlgorithm::TREE:
          if constexpr (Inner::has_product)
            return launch_block_as<T, AccumulateAlgorithm::TREE>(s, args, ctx);
          return;
        case AccumulateAlgorithm::NAIVE:
          return; // a NAIVE call is launched with the mac's plain Args
        }
      }
      else
      {
        launch_block_as<T, AccumulateAlgorithm::NAIVE>(s, args, ctx);
      }
    }

    template <class Inner>
    static void launch(const mptorch::gemm::GemmShape &s, const mptorch::gemm::BlockGemmArgs<Inner> &args,
                       const LaunchContext &ctx)
    {
#if !defined(MPTORCH_NO_FP64)
      if (s.dt == mptorch::GemmDtype::Double)
        return launch_block<double>(s, args, ctx);
#endif
      launch_block<float>(s, args, ctx);
    }
  };

} // namespace mptorch::gemm_cuda
