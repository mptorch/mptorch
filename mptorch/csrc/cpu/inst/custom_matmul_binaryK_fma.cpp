// Explicit instantiations of the CPU GEMM kernel for custom_matmul_binaryK_fma
// and custom_matmul_binaryK_fma_mixed. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier
// (float32, float16 and bfloat16 operands). The entry points are in
// ../custom_matmul_entry.cpp, and ../custom_matmul_kernel.h explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKFusedMixedArgs;

  template void CpuBackend::launch_as<float, BinaryKFusedArgs>(
      const GemmShape &, const BinaryKFusedArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, BinaryKFusedMixedArgs>(
      const GemmShape &, const BinaryKFusedMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
