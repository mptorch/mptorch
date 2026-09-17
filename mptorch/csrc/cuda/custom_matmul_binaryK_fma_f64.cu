#include "custom_matmul_kernel.cuh"

// binaryK formats, fused mac, in binary64: the float64 kernels of
// custom_matmul_binaryK_fma and its mixed-format twin, whose binary32 kernels
// are custom_matmul_binaryK_fma.cu.
//
// A separate file rather than two more lines in that one, so that
// MPTORCH_NO_FP64=1 can leave the binary64 kernels out of the build by leaving
// the four *_f64.cu files out, and so that no other translation unit grows.
// Like its twin it includes no ATen tensor headers, which is what keeps its
// fixed nvcc cost near 3 s rather than about 25 s; the two explicit
// instantiations below are all it contains.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::BinaryKFusedMixedArgs;

    template void CudaBackend::launch_as<double, BinaryKFusedArgs>(
        const GemmShape &, const BinaryKFusedArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, BinaryKFusedMixedArgs>(
        const GemmShape &, const BinaryKFusedMixedArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
