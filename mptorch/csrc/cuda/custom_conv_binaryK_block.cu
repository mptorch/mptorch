#include "custom_matmul_kernel.cuh"

// The conv ops over binaryK formats under AccumulateAlgorithm::BLOCK
// (blocked summation with an outer format; common/gemm_accumulate.h), in binary32: the
// gathered kernels (common/gemm_gather.h) of custom_conv_binaryK and custom_conv_binaryK_fma.
// One of ten conv instantiation files (custom_conv_binaryK.cu says why).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::BLOCK, AccumulateArgs<BinaryKSplitArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::BLOCK, AccumulateArgs<BinaryKFusedArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKFusedArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
