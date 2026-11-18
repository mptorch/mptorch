// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_binaryK
// and custom_conv_binaryK_fma. They accumulate under AccumulateAlgorithm::KAHAN
// (compensated summation) in the binary64 carrier (float64 operands). The entry
// points are in ../custom_conv_entry.cpp, and ../custom_matmul_kernel.cuh
// explains how the instantiations are divided among the files in this
// directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::KAHAN, AccumulateArgs<BinaryKSplitArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::KAHAN, AccumulateArgs<BinaryKFusedArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKFusedArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
