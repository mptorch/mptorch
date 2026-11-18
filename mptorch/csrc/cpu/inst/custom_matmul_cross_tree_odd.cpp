// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_binaryK_superfp_accumulated_odd and
// custom_matmul_superfp_binaryK_accumulated_odd. They accumulate under
// AccumulateAlgorithm::TREE (pairwise sums within each block of products) in
// the binary32 carrier (float32, float16 and bfloat16 operands), and round each
// carrier step to odd (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::TREE, OddArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKSuperfpArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::TREE, OddArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpBinaryKArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
