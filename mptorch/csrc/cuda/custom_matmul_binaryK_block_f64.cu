#include "custom_matmul_kernel.cuh"

// The binary64 twin of custom_matmul_binaryK_block.cu: its kernels in the
// binary64 carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cu files, so
// that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::AccumulateArgs;

    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, BinaryKSplitArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, BinaryKFusedArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKFusedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
