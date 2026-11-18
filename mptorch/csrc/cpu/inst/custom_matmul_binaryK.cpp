// Explicit instantiations of the CPU GEMM kernel for custom_matmul_binaryK and
// custom_matmul_binaryK_mixed. They accumulate under AccumulateAlgorithm::NAIVE
// (one running sum) in the binary32 carrier (float32, float16 and bfloat16
// operands). The entry points are in ../custom_matmul_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::BinaryKSplitMixedArgs;

  template void CpuBackend::launch_as<float, BinaryKSplitArgs>(
      const GemmShape &, const BinaryKSplitArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, BinaryKSplitMixedArgs>(
      const GemmShape &, const BinaryKSplitMixedArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
