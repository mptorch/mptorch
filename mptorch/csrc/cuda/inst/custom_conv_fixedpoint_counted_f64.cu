// Explicit instantiations of the CUDA GEMM kernel for the counted op
// custom_conv_fixedpoint_counted (common/fixedpoint_counted.h), every
// Args its twin's plan can run, in the binary64 carrier (float64 operands).
// Built with MPTORCH_COUNTING=1 only. The entry points are in
// ../custom_conv_counted_entry.cpp, and ../custom_matmul_kernel.cuh explains how the
// instantiations are divided among the files in this directory.

#include "../../common/fixedpoint_counted.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::FixedPointIntSplitArgs;
    using mptorch::gemm::FixedPointSplitArgs;
    using mptorch::gemm::ConvArgs;
    using mptorch::gemm::FixedPointCountingArgs;

    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, FixedPointCountingArgs<FixedPointSplitArgs>>(
      const GemmShape &, const ConvArgs<FixedPointCountingArgs<FixedPointSplitArgs>> &,
      const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, FixedPointCountingArgs<FixedPointIntSplitArgs<false>>>(
      const GemmShape &, const ConvArgs<FixedPointCountingArgs<FixedPointIntSplitArgs<false>>> &,
      const CudaBackend::LaunchContext &);
    template void CudaBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, FixedPointCountingArgs<FixedPointIntSplitArgs<true>>>(
      const GemmShape &, const ConvArgs<FixedPointCountingArgs<FixedPointIntSplitArgs<true>>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
