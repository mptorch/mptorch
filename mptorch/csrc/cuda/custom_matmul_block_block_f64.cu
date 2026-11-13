#include "custom_matmul_kernel.cuh"

// The binary64 twin of custom_matmul_block_block.cu: its kernels in the
// binary64 carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cu files, so
// that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

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
