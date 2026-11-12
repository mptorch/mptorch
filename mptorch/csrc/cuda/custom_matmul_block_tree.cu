#include "custom_matmul_kernel.cuh"

// The block GEMM (custom_matmul_block; common/gemm_block.h) under
// AccumulateAlgorithm::TREE (pairwise summation within a block; split mac only;
// common/gemm_accumulate.h), in binary32. One of four block GEMM
// instantiation files (custom_matmul_block.cu says why).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
      using mptorch::gemm::BlockGemmArgs;
    using mptorch::gemm::BlockSplitArgs;

    template void CudaBackend::launch_block_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<BlockSplitArgs>>(
        const GemmShape &, const BlockGemmArgs<AccumulateArgs<BlockSplitArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
