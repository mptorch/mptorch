#include "custom_matmul_kernel.cuh"

// The binary64 twin of custom_matmul_cross.cu: its kernels in the binary64
// carrier, for float64 operands and for calls that name
// carrier=torch.float64. A file of its own, like the NAIVE *_f64.cu files, so
// that MPTORCH_NO_FP64=1 leaves these kernels out by leaving the file out,
// and no other translation unit grows.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSuperfpArgs;
    using mptorch::gemm::SuperfpBinaryKArgs;

    template void CudaBackend::launch_as<double, BinaryKSuperfpArgs>(
        const GemmShape &, const BinaryKSuperfpArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, SuperfpBinaryKArgs>(
        const GemmShape &, const SuperfpBinaryKArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
