#include "custom_matmul_kernel.cuh"

// The conv ops over the cross-family split macs (common/gemm_cross.h)
// under AccumulateAlgorithm::NAIVE, in binary32: the gathered kernels
// (common/gemm_gather.h) of custom_conv_binaryK_superfp and
// custom_conv_superfp_binaryK, the conv twins of
// custom_matmul_binaryK_superfp and custom_matmul_superfp_binaryK.
//
// One of four such files, one per algorithm, mirroring the cross-family
// GEMM's (custom_matmul_cross.cu): a gathered kernel is its GEMM kernel's
// instantiation over a Gathered policy, and keeping the two apart is what
// leaves every earlier file compiling exactly what it did. binary32 only
// until dev/continuation_plan.md's phase G.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, BinaryKSuperfpArgs>(
      const GemmShape &, const ConvArgs<BinaryKSuperfpArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, SuperfpBinaryKArgs>(
      const GemmShape &, const ConvArgs<SuperfpBinaryKArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
