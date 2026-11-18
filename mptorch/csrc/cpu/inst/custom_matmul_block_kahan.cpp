// Explicit instantiations of the CPU GEMM kernel's block-format form, which
// decodes its operands from packed block-format codes, for custom_matmul_block
// with a split and with a fused mac. They accumulate under
// AccumulateAlgorithm::KAHAN (compensated summation) in the binary32 carrier.
// The entry points are in ../custom_matmul_block_entry.cpp, and
// ../custom_matmul_kernel.h explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BlockGemmArgs;
  using mptorch::gemm::BlockSplitArgs;

  template void CpuBackend::launch_block_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<BlockSplitArgs>>(
      const GemmShape &, const BlockGemmArgs<AccumulateArgs<BlockSplitArgs>> &,
      const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_block_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<BinaryKFusedArgs>>(
      const GemmShape &, const BlockGemmArgs<AccumulateArgs<BinaryKFusedArgs>> &,
      const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
