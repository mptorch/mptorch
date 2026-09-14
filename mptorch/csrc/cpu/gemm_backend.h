#pragma once

// The CPU backend as common/gemm_host.h's driver sees it: a launch context and
// one `launch` per op, with the kernels behind it declared rather than
// defined. The shape is the CUDA side's (cuda/gemm_backend.h), for a different
// reason: there a .cu must not see at::Tensor (finding H1); here the kernels
// come in two carriers, and the translation unit that holds the entry points
// must not instantiate either of them. cpu/custom_matmul_kernel.h defines
// launch_as, and the eight custom_matmul_*.cpp files instantiate it -- four in
// binary32 and their four *_f64.cpp twins in binary64 -- so each kernel is
// compiled into exactly one object, and a build with MPTORCH_NO_FP64=1 leaves
// the binary64 ones out by leaving four files out
// (dev/binary64_carrier_plan.md, phase 4).

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

    // No counter to reserve, unlike the CUDA generator: NaiveTile::seed_rng
    // keys each output element's stream on this one seed plus the element's
    // global linear index, so a call consumes exactly one draw however many
    // values its K-reduction goes on to need.
    static LaunchContext make_context(bool use_rng, uint64_t /*draws_per_thread*/)
    {
      return LaunchContext{use_rng ? draw_cpu_seed() : 0};
    }

    // cpu/custom_matmul_kernel.h defines this; see the top of this file for
    // where it is instantiated.
    template <class T, class Args>
    static void launch_as(const mptorch::gemm::GemmShape &s, const Args &args,
                          const LaunchContext &ctx);

    // The carrier is chosen here, once per call, exactly as on the device: a
    // float64 GEMM runs the binary64 kernels and every other dtype the
    // binary32 ones, whose tile packing converts. Under MPTORCH_NO_FP64,
    // gemm_dtype_of has already refused a float64 operand by now.
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
