// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_superfp and custom_conv_superfp_binaryK. They accumulate
// under AccumulateAlgorithm::BLOCK (a sum per block of products, added to the
// running total) in the binary64 carrier (float64 operands). The entry points
// are in ../custom_matmul_cross_entry.cpp, and ../custom_matmul_kernel.cuh
// explains how the instantiations are divided among the files in this
// directory.

#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSuperfpArgs>> &,
      const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpBinaryKArgs>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
