// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_binaryK_accumulated_odd. They accumulate under
// AccumulateAlgorithm::TREE (pairwise sums within each block of products) in
// the binary64 carrier (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::OddArgs;
  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::TREE, OddArgs<BinaryKSplitArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKSplitArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
