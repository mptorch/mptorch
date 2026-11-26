// Explicit instantiations of the CUDA GEMM kernel for the counted op
// custom_matmul_fixedpoint_fma_counted (common/fixedpoint_counted.h), every
// Args its twin's plan can run, in the binary64 carrier (float64 operands).
// Built with MPTORCH_COUNTING=1 only. The entry points are in
// ../custom_matmul_counted_entry.cpp, and ../custom_matmul_kernel.cuh explains how the
// instantiations are divided among the files in this directory.

#include "../../common/fixedpoint_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::FixedPointIntFusedArgs;
    using mptorch::gemm::FixedPointCountingArgs;

    template void CudaBackend::launch_as<double, FixedPointCountingArgs<FixedPointIntFusedArgs<false>>>(
      const GemmShape &, const FixedPointCountingArgs<FixedPointIntFusedArgs<false>> &, const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_as<double, FixedPointCountingArgs<FixedPointIntFusedArgs<true>>>(
      const GemmShape &, const FixedPointCountingArgs<FixedPointIntFusedArgs<true>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
