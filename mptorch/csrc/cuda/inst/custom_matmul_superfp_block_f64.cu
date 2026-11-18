// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_superfp_accumulated and custom_matmul_superfp_fma_accumulated.
// They accumulate under AccumulateAlgorithm::BLOCK (a sum per block of
// products, added to the running total) in the binary64 carrier (float64
// operands). The entry points are in ../custom_matmul_accumulated_entry.cpp,
// and ../custom_matmul_kernel.cuh explains how the instantiations are divided
// among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::AccumulateArgs;

    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, SuperfpSplitArgs>(
        const GemmShape &, const AccumulateArgs<SuperfpSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, SuperfpFusedArgs>(
        const GemmShape &, const AccumulateArgs<SuperfpFusedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
