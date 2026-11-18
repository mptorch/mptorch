// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_binaryK_superfp and custom_matmul_superfp_binaryK. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary64
// carrier (float64 operands). The entry points are in
// ../custom_matmul_cross_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_as<double, BinaryKSuperfpArgs>(
      const GemmShape &, const BinaryKSuperfpArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<double, SuperfpBinaryKArgs>(
      const GemmShape &, const SuperfpBinaryKArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
