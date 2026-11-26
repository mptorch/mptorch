// Explicit instantiations of the CPU GEMM kernel for the counted op
// custom_matmul_fixedpoint_fma_counted (common/fixedpoint_counted.h), every
// Args its twin's plan can run, in the binary32 carrier (float32, float16 and bfloat16 operands).
// Built with MPTORCH_COUNTING=1 only. The entry points are in
// ../custom_matmul_counted_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../../common/fixedpoint_counted.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::FixedPointIntFusedArgs;
  using mptorch::gemm::FixedPointCountingArgs;

  template void CpuBackend::launch_as<float, FixedPointCountingArgs<FixedPointIntFusedArgs<false>>>(
    const GemmShape &, const FixedPointCountingArgs<FixedPointIntFusedArgs<false>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, FixedPointCountingArgs<FixedPointIntFusedArgs<true>>>(
    const GemmShape &, const FixedPointCountingArgs<FixedPointIntFusedArgs<true>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
