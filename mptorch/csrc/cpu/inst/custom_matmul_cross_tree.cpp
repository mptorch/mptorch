// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_binaryK_superfp_accumulated and
// custom_matmul_superfp_binaryK_accumulated. They accumulate under
// AccumulateAlgorithm::TREE (pairwise sums within each block of products) in
// the binary32 carrier (float32, float16 and bfloat16 operands). The entry
// points are in ../custom_matmul_cross_entry.cpp, and ../custom_matmul_kernel.h
// explains how the instantiations are divided among the files in this
// directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::TREE, BinaryKSuperfpArgs>(
    const GemmShape &, const AccumulateArgs<BinaryKSuperfpArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::TREE, SuperfpBinaryKArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpBinaryKArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
