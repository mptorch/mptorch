#include "custom_matmul_kernel.cuh"

// The conv ops over superfp formats under AccumulateAlgorithm::KAHAN
// (Kahan-compensated summation; common/gemm_accumulate.h), in binary32: the
// gathered kernels (common/gemm_gather.h) of custom_conv_superfp and custom_conv_superfp_fma.
// One of ten conv instantiation files (custom_conv_superfp.cu says why).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpFusedArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<SuperfpSplitArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpSplitArgs>> &,
        const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<SuperfpFusedArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpFusedArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
