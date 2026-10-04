#include "custom_matmul_kernel.h"

// The block GEMM (custom_matmul_block; common/gemm_block.h) under
// AccumulateAlgorithm::NAIVE, in binary32: the blocked kernels of its split
// mac (the carrier's product, the sum rounded or not) and its fused one.
//
// One of four such files, one per algorithm, mirroring the GEMM's: a blocked
// kernel is its GEMM kernel's instantiation over a Blocked policy, and keeping
// these apart from the GEMM files is what leaves those compiling exactly what
// they did. binary32 only until dev/continuation_plan.md's phase G.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BlockGemmArgs;
  using mptorch::gemm::BlockSplitArgs;

  template void CpuBackend::launch_block_as<float, AccumulateAlgorithm::NAIVE, BlockSplitArgs>(
      const GemmShape &, const BlockGemmArgs<BlockSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_block_as<float, AccumulateAlgorithm::NAIVE, BinaryKFusedArgs>(
      const GemmShape &, const BlockGemmArgs<BinaryKFusedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
