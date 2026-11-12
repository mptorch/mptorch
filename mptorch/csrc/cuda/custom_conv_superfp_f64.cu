#include "custom_matmul_kernel.cuh"

// The binary64 twin of custom_conv_superfp.cu: its kernels in the binary64
// carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cu files, so
// that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::SuperfpSplitArgs;
    using mptorch::gemm::SuperfpSplitMixedArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, SuperfpSplitArgs>(
        const GemmShape &, const ConvArgs<SuperfpSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, SuperfpSplitMixedArgs>(
        const GemmShape &, const ConvArgs<SuperfpSplitMixedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
