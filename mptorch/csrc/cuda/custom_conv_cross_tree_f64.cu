#include "custom_matmul_kernel.cuh"

// The binary64 twin of custom_conv_cross_tree.cu: its kernels in the binary64
// carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cu files, so
// that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::AccumulateArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::TREE, AccumulateArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<BinaryKSuperfpArgs>> &,
      const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::TREE, AccumulateArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<SuperfpBinaryKArgs>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
