// Explicit instantiations of the CUDA GEMM kernel for the counted op
// custom_matmul_fixedpoint_counted (common/fixedpoint_counted.h), every
// Args its twin's plan can run, in the binary32 carrier (float32, float16 and bfloat16 operands).
// Built with MPTORCH_COUNTING=1 only. The entry points are in
// ../custom_matmul_counted_entry.cpp, and ../custom_matmul_kernel.cuh explains how the
// instantiations are divided among the files in this directory.

#include "../../common/fixedpoint_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::FixedPointIntSplitArgs;
    using mptorch::gemm::FixedPointSplitArgs;
    using mptorch::gemm::FixedPointCountingArgs;

    template void CudaBackend::launch_as<float, FixedPointCountingArgs<FixedPointSplitArgs>>(
      const GemmShape &, const FixedPointCountingArgs<FixedPointSplitArgs> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<float, FixedPointCountingArgs<FixedPointIntSplitArgs<false>>>(
      const GemmShape &, const FixedPointCountingArgs<FixedPointIntSplitArgs<false>> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<float, FixedPointCountingArgs<FixedPointIntSplitArgs<true>>>(
      const GemmShape &, const FixedPointCountingArgs<FixedPointIntSplitArgs<true>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
