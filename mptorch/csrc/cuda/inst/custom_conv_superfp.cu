// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_superfp
// and custom_conv_superfp_mixed. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier
// (float32, float16 and bfloat16 operands). The entry points are in
// ../custom_conv_entry.cpp, and ../custom_matmul_kernel.cuh explains how the
// instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpSplitMixedArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, SuperfpSplitArgs>(
        const GemmShape &, const ConvArgs<SuperfpSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, SuperfpSplitMixedArgs>(
        const GemmShape &, const ConvArgs<SuperfpSplitMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
