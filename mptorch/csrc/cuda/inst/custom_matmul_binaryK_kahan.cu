// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_accumulated and custom_matmul_binaryK_fma_accumulated.
// They accumulate under AccumulateAlgorithm::KAHAN (compensated summation) in
// the binary32 carrier (float32, float16 and bfloat16 operands). The entry
// points are in ../custom_matmul_accumulated_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::AccumulateArgs;

    template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, BinaryKSplitArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, BinaryKFusedArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKFusedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
