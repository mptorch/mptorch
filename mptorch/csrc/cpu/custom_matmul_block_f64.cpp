#include "custom_matmul_kernel.h"

// The binary64 twin of custom_matmul_block.cpp: its kernels in the binary64
// carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cpp files,
// so that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BlockGemmArgs;
  using mptorch::gemm::BlockSplitArgs;

  template void CpuBackend::launch_block_as<double, AccumulateAlgorithm::NAIVE, BlockSplitArgs>(
      const GemmShape &, const BlockGemmArgs<BlockSplitArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_block_as<double, AccumulateAlgorithm::NAIVE, BinaryKFusedArgs>(
      const GemmShape &, const BlockGemmArgs<BinaryKFusedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
