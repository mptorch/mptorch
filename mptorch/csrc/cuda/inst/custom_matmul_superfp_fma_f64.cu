// Explicit instantiations of the CUDA GEMM kernel for custom_matmul_superfp_fma
// and custom_matmul_superfp_fma_mixed. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier (float64
// operands). The entry points are in ../custom_matmul_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::SuperfpFusedMixedArgs;

    template void CudaBackend::launch_as<double, SuperfpFusedArgs>(
        const GemmShape &, const SuperfpFusedArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, SuperfpFusedMixedArgs>(
        const GemmShape &, const SuperfpFusedMixedArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
