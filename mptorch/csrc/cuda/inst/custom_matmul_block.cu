// Explicit instantiations of the CUDA GEMM kernel's block-format form, which
// decodes its operands from packed block-format codes, for custom_matmul_block
// with a split and with a fused mac. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier. The
// entry points are in ../custom_matmul_block_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::BlockGemmArgs;
    using mptorch::gemm::BlockSplitArgs;

    template void CudaBackend::launch_block_as<float, AccumulateAlgorithm::NAIVE, BlockSplitArgs>(
        const GemmShape &, const BlockGemmArgs<BlockSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_block_as<float, AccumulateAlgorithm::NAIVE, BinaryKFusedArgs>(
        const GemmShape &, const BlockGemmArgs<BinaryKFusedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
