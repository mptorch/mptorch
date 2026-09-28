#include "custom_matmul_kernel.cuh"

// The conv ops over binaryK formats with a fused mac under AccumulateAlgorithm::NAIVE,
// in binary32: the gathered kernels (common/gemm_gather.h) of custom_conv_binaryK_fma
// and custom_conv_binaryK_fma_mixed, the conv twins of custom_matmul_binaryK_fma and its
// palette op.
//
// One of ten such files, one per (format family x mac mode) for NAIVE and one
// per (format family x algorithm) for KAHAN, BLOCK and TREE, mirroring the
// GEMM's: a gathered kernel is its GEMM kernel's instantiation over a
// Gathered policy, so it costs what its GEMM kernel costs to compile, and
// keeping these apart from the GEMM files is what leaves those compiling
// exactly what they did. binary32 only until dev/continuation_plan.md's
// phase G.

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
