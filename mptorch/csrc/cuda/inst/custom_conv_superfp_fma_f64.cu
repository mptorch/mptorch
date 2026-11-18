// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_superfp_fma and custom_conv_superfp_fma_mixed. They accumulate
// under AccumulateAlgorithm::NAIVE (one running sum) in the binary64 carrier
// (float64 operands). The entry points are in ../custom_conv_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::SuperfpFusedArgs;
    using mptorch::gemm::SuperfpFusedMixedArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, SuperfpFusedArgs>(
        const GemmShape &, const ConvArgs<SuperfpFusedArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, SuperfpFusedMixedArgs>(
        const GemmShape &, const ConvArgs<SuperfpFusedMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
