// binaryK formats, split mac, in binary64: the float64 kernels of
// custom_matmul_binaryK and its mixed-format twin, whose binary32 kernels are
// custom_matmul_binaryK.cpp.
//
// A separate object so that no existing translation unit grows and a build
// with MPTORCH_NO_FP64=1 can leave the binary64 kernels out by leaving these
// four files out (setup.py); see dev/binary64_carrier_plan.md (phase 4).

#include "custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKSplitMixedArgs;

  template void CpuBackend::launch_as<double, BinaryKSplitArgs>(
      const GemmShape &, const BinaryKSplitArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<double, BinaryKSplitMixedArgs>(
      const GemmShape &, const BinaryKSplitMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
