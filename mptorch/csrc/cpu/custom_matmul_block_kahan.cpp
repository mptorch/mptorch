#include "custom_matmul_kernel.h"

// The block GEMM (custom_matmul_block; common/gemm_block.h) under
// AccumulateAlgorithm::KAHAN (Kahan-compensated summation;
// common/gemm_accumulate.h), in binary32. One of four block GEMM
// instantiation files (custom_matmul_block.cpp says why).

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
