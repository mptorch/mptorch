#include "custom_matmul_kernel.cuh"

// The conv ops over binaryK formats under AccumulateAlgorithm::TREE
// (pairwise summation over blocks; common/gemm_accumulate.h), in binary32: the
// gathered kernels (common/gemm_gather.h) of custom_conv_binaryK (a fused mac has no tree).
// One of ten conv instantiation files (custom_conv_binaryK.cu says why).

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSplitArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<BinaryKSplitArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
