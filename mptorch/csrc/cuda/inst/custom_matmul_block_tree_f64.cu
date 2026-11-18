// Explicit instantiations of the CUDA GEMM kernel's block-format form, which
// decodes its operands from packed block-format codes, for custom_matmul_block.
// They accumulate under AccumulateAlgorithm::TREE (pairwise sums within each
// block of products) in the binary64 carrier. The entry points are in
// ../custom_matmul_block_entry.cpp, and ../custom_matmul_kernel.cuh explains
// how the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
      using mptorch::gemm::BlockGemmArgs;
    using mptorch::gemm::BlockSplitArgs;

    template void CudaBackend::launch_block_as<double, AccumulateAlgorithm::TREE, AccumulateArgs<BlockSplitArgs>>(
        const GemmShape &, const BlockGemmArgs<AccumulateArgs<BlockSplitArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
