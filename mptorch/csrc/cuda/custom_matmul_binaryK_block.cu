#include "custom_matmul_kernel.cuh"

// binaryK formats under AccumulateAlgorithm::BLOCK (two-level block summation;
// common/gemm_accumulate.h), in binary32: the float32, float16 and bfloat16
// kernels of custom_matmul_binaryK and custom_matmul_binaryK_fma.
//
// One of six such files, one per (format family x algorithm). It includes no
// ATen tensor headers, like the eight NAIVE files (custom_matmul_binaryK.cu
// says why). They are separate from the NAIVE files so that those compile
// exactly what they compiled before the three algorithms existed, and from
// each other because the build is throughput-bound: this one holds fourteen
// kernels per mac (seven round modes, with and without an accumulate or fused
// format). There is no *_f64 twin yet: the three algorithms are binary32 only
// until dev/continuation_plan.md's phase G.

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::BinaryKSplitArgs;
    using mptorch::gemm::BinaryKFusedArgs;
    using mptorch::gemm::AccumulateArgs;

    template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::BLOCK, BinaryKSplitArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::BLOCK, BinaryKFusedArgs>(
        const GemmShape &, const AccumulateArgs<BinaryKFusedArgs> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
