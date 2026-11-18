// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_binaryK_superfp and custom_matmul_superfp_binaryK. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary32
// carrier (float32, float16 and bfloat16 operands). The entry points are in
// ../custom_matmul_cross_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_as<float, BinaryKSuperfpArgs>(
      const GemmShape &, const BinaryKSuperfpArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, SuperfpBinaryKArgs>(
      const GemmShape &, const SuperfpBinaryKArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
