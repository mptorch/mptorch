// Explicit instantiations of the CUDA GEMM kernel's block-format form, which
// decodes its operands from packed block-format codes, for custom_matmul_block
// with a split and with a fused mac. They accumulate under
// AccumulateAlgorithm::BLOCK (a sum per block of products, added to the running
// total) in the binary64 carrier. The entry points are in
// ../custom_matmul_block_entry.cpp, and ../custom_matmul_kernel.cuh explains
// how the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::BlockGemmArgs;
    using mptorch::gemm::BlockSplitArgs;

    template void CudaBackend::launch_block_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<BlockSplitArgs>>(
        const GemmShape &, const BlockGemmArgs<AccumulateArgs<BlockSplitArgs>> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_block_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<BinaryKFusedArgs>>(
        const GemmShape &, const BlockGemmArgs<AccumulateArgs<BinaryKFusedArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
