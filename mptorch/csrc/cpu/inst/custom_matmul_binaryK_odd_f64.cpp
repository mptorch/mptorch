// Explicit instantiations of the CPU GEMM kernel for custom_matmul_binaryK_odd.
// They accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the
// binary64 carrier (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::OddArgs;

  template void CpuBackend::launch_as<double, OddArgs<BinaryKSplitArgs>>(
      const GemmShape &, const OddArgs<BinaryKSplitArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
