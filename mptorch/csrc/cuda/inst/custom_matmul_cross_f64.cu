// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_superfp and custom_matmul_superfp_binaryK. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary64
// carrier (float64 operands). The entry points are in
// ../custom_matmul_cross_entry.cpp, and ../custom_matmul_kernel.cuh explains
// how the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_as<double, BinaryKSuperfpArgs>(
        const GemmShape &, const BinaryKSuperfpArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, SuperfpBinaryKArgs>(
        const GemmShape &, const SuperfpBinaryKArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
