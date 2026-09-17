// Explicit instantiation of the CPU GEMM kernel for binaryK formats with a
// fused mac (one rounding per multiply-add) in the binary32 carrier: the
// kernels behind custom_matmul_binaryK_fma and
// custom_matmul_binaryK_fma_mixed for float32, float16 and bfloat16
// operands. The kernel template (custom_matmul_kernel.h) is heavy to
// compile, one body per round mode per policy, so each (format family x mac
// mode x carrier) has its own translation unit and the eight objects build
// in parallel with none of them the critical path. The binary64 twin is
// custom_matmul_binaryK_fma_f64.cpp; the entry points are in
// custom_matmul_entry.cpp.

#include "custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKFusedMixedArgs;

  template void CpuBackend::launch_as<float, BinaryKFusedArgs>(
      const GemmShape &, const BinaryKFusedArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, BinaryKFusedMixedArgs>(
      const GemmShape &, const BinaryKFusedMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
