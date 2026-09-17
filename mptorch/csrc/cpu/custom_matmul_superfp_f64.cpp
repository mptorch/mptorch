// Explicit instantiation of the CPU GEMM kernel for superfp formats with a
// split mac (a rounding after the multiply and another after the add) in the
// binary64 carrier: the kernels behind custom_matmul_superfp and
// custom_matmul_superfp_mixed for float64 operands. The binary32 twin is
// custom_matmul_superfp.cpp. A separate translation unit so the binary64
// kernels add no compile time to the binary32 object, and so a build with
// MPTORCH_NO_FP64=1 can leave them out by leaving the four *_f64.cpp files
// out (setup.py).

#include "custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::SuperfpSplitArgs;
  using mptorch::gemm::SuperfpSplitMixedArgs;

  template void CpuBackend::launch_as<double, SuperfpSplitArgs>(
      const GemmShape &, const SuperfpSplitArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<double, SuperfpSplitMixedArgs>(
      const GemmShape &, const SuperfpSplitMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
