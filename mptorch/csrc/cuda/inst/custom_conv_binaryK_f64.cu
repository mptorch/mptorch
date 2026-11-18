// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_binaryK
// and custom_conv_binaryK_mixed. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier (float64
// operands). The entry points are in ../custom_conv_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKSplitMixedArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, BinaryKSplitArgs>(
        const GemmShape &, const ConvArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, BinaryKSplitMixedArgs>(
        const GemmShape &, const ConvArgs<BinaryKSplitMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
