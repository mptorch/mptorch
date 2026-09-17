#pragma once

// The CPU backend as common/gemm_host.h's driver sees it: a launch context
// and one `launch` per Args type, with the kernels behind it declared rather
// than defined. cpu/custom_matmul_kernel.h defines launch_as and the eight
// custom_matmul_*.cpp files explicitly instantiate it, four in binary32 and
// their four *_f64.cpp twins in binary64, so each kernel is compiled into
// exactly one object, the entry points' translation unit
// (cpu/custom_matmul_entry.cpp) never instantiates a kernel, and a build
// with MPTORCH_NO_FP64=1 leaves the binary64 kernels out by leaving four
// files out. The shape mirrors cuda/gemm_backend.h, where the same split
// also keeps at::Tensor out of the .cu files, which is what makes them
// cheap to compile through nvcc.

#include "../common/gemm_args.h"
#include "utils.h" // draw_cpu_seed
#include <cstdint>

namespace mptorch::gemm_cpu
{

  struct CpuBackend
  {
    struct LaunchContext
    {
      uint64_t seed = 0;
    };

    // One 64-bit seed from ATen's CPU generator when the round mode is SR,
    // and 0 otherwise. Unlike the CUDA generator there is no counter to
    // reserve: NaiveTile::seed_rng keys each output element's Philox stream
    // on this one seed plus the element's global linear index, so a call
    // consumes exactly one draw from the generator however many random
    // values its K-reductions go on to need.
    static LaunchContext make_context(bool use_rng, uint64_t /*draws_per_thread*/)
    {
      return LaunchContext{use_rng ? draw_cpu_seed() : 0};
    }

    // Runs the kernel in carrier T (float or double) for one Args type.
    // Defined in cpu/custom_matmul_kernel.h and explicitly instantiated in
    // the eight custom_matmul_*.cpp files; nothing else instantiates it.
    template <class T, class Args>
    static void launch_as(const mptorch::gemm::GemmShape &s, const Args &args,
                          const LaunchContext &ctx);

    // Picks the carrier once per call, exactly as the device side does: a
    // float64 GEMM runs the binary64 kernel and every other dtype the
    // binary32 one, whose tile packing converts the operands on load. Under
    // MPTORCH_NO_FP64 the binary64 instantiations do not exist, and
    // gemm_dtype_of has already refused a float64 operand with an error that
    // names the flag.
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

} // namespace mptorch::gemm_cpu
