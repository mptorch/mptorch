// superfp formats, fused mac, in binary64: the float64 kernels of
// custom_matmul_superfp_fma and its mixed-format twin, whose binary32 kernels are
// custom_matmul_superfp_fma.cpp.
//
// A separate object so that no existing translation unit grows and a build
// with MPTORCH_NO_FP64=1 can leave the binary64 kernels out by leaving these
// four files out (setup.py); see dev/binary64_carrier_plan.md (phase 4).

#include "custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::SuperfpFusedMixedArgs;

  template void CpuBackend::launch_as<double, SuperfpFusedArgs>(
      const GemmShape &, const SuperfpFusedArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<double, SuperfpFusedMixedArgs>(
      const GemmShape &, const SuperfpFusedMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
