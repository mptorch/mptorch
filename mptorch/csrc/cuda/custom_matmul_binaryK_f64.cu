#include "custom_matmul_kernel.cuh"

// binaryK formats, split mac, in binary64: the float64 kernels of
// custom_matmul_binaryK and its mixed-format twin, whose binary32 kernels are
// custom_matmul_binaryK.cu.
//
// A separate object rather than two more lines in that file, so that no
// existing translation unit grows and a build with MPTORCH_NO_FP64=1 can
// leave the binary64 kernels out by leaving these four files out (setup.py).
// Like its twin it includes no ATen, which is what keeps its fixed cost near
// 3 s; see dev/binary64_carrier_plan.md (phase 4) and cuda/gemm_backend.h.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKSplitMixedArgs;

    template void CudaBackend::launch_as<double, BinaryKSplitArgs>(
        const GemmShape &, const BinaryKSplitArgs &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, BinaryKSplitMixedArgs>(
        const GemmShape &, const BinaryKSplitMixedArgs &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
