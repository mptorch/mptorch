// superfp formats, split mac: the binary32 kernels of custom_matmul_superfp and
// its mixed-format twin, for float32, float16 and bfloat16 operands.
//
// Still one translation unit per (format family x mac mode), so the four
// objects compile in parallel rather than one of them being the pole (finding
// B2). The entry points that used to share this file are
// custom_matmul_entry.cpp; what is left is what puts these kernels in this
// object, and custom_matmul_superfp_f64.cpp is its binary64 twin
// (dev/binary64_carrier_plan.md, phase 4).

#include "custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpSplitMixedArgs;

  template void CpuBackend::launch_as<float, SuperfpSplitArgs>(
      const GemmShape &, const SuperfpSplitArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, SuperfpSplitMixedArgs>(
      const GemmShape &, const SuperfpSplitMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
