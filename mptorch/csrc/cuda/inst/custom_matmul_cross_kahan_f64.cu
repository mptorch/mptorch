// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_superfp_accumulated and
// custom_matmul_superfp_binaryK_accumulated. They accumulate under
// AccumulateAlgorithm::KAHAN (compensated summation) in the binary64 carrier
// (float64 operands). The entry points are in ../custom_matmul_cross_entry.cpp,
// and ../custom_matmul_kernel.cuh explains how the instantiations are divided
// among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, BinaryKSuperfpArgs>(
      const GemmShape &, const AccumulateArgs<BinaryKSuperfpArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, SuperfpBinaryKArgs>(
      const GemmShape &, const AccumulateArgs<SuperfpBinaryKArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
