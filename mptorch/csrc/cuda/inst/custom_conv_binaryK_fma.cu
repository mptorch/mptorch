// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_fma and custom_conv_binaryK_fma_mixed. They accumulate
// under AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier
// (float32, float16 and bfloat16 operands). The entry points are in
// ../custom_conv_entry.cpp, and ../custom_matmul_kernel.cuh explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::BinaryKFusedMixedArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, BinaryKFusedArgs>(
        const GemmShape &, const ConvArgs<BinaryKFusedArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, BinaryKFusedMixedArgs>(
        const GemmShape &, const ConvArgs<BinaryKFusedMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
