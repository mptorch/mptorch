// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for custom_conv_binaryK.
// They accumulate under AccumulateAlgorithm::TREE (pairwise sums within each
// block of products) in the binary32 carrier (float32, float16 and bfloat16
// operands). The entry points are in ../custom_conv_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided among
// the files in this directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSplitArgs;

    template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<BinaryKSplitArgs>>(
        const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSplitArgs>> &,
        const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
