// Explicit instantiation of the CPU GEMM kernel for binaryK formats with a
// split mac (a rounding after the multiply and another after the add) in the
// binary32 carrier: the kernels behind custom_matmul_binaryK and
// custom_matmul_binaryK_mixed for float32, float16 and bfloat16 operands.
// The kernel template (custom_matmul_kernel.h) is heavy to compile, one body
// per round mode per policy, so each (format family x mac mode x carrier)
// has its own translation unit and the eight objects build in parallel with
// none of them the critical path. The binary64 twin is
// custom_matmul_binaryK_f64.cpp; the entry points are in
// custom_matmul_entry.cpp.

#include "custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKSplitMixedArgs;

  template void CpuBackend::launch_as<float, BinaryKSplitArgs>(
      const GemmShape &, const BinaryKSplitArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, BinaryKSplitMixedArgs>(
      const GemmShape &, const BinaryKSplitMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
