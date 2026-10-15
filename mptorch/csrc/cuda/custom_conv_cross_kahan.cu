#include "custom_matmul_kernel.cuh"

// The conv ops over the cross-family split macs (common/gemm_cross.h)
// under AccumulateAlgorithm::KAHAN (Kahan-compensated summation;
// common/gemm_accumulate.h), in binary32: the gathered kernels of
// custom_conv_binaryK_superfp and custom_conv_superfp_binaryK. One of four
// cross-family conv instantiation files (custom_conv_cross.cu says why).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSuperfpArgs>> &,
      const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpBinaryKArgs>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
