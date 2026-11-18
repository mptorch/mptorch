// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_accumulated and custom_matmul_binaryK_fma_accumulated.
// They accumulate under AccumulateAlgorithm::BLOCK (a sum per block of
// products, added to the running total) in the binary64 carrier (float64
// operands). The entry points are in ../custom_matmul_accumulated_entry.cpp,
// and ../custom_matmul_kernel.cuh explains how the instantiations are divided
// among the files in this directory.

#include "../custom_matmul_kernel.cuh"

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
