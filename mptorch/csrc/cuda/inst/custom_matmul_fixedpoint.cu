// Explicit instantiations of the CUDA GEMM kernel for custom_matmul_fixedpoint
// in the binary32 carrier (float32, float16 and bfloat16 operands): the
// carrier path (SplitMac over the fixed-point float policies) and the
// integer accumulator's split mac, saturating and wrapping around
// (common/fixedpoint_gemm.h). NAIVE only. The entry points are in
// ../custom_matmul_fixedpoint_entry.cpp, and ../custom_matmul_kernel.h
// explains how the instantiations are divided among the files in this
// directory.

#include "../../common/fixedpoint_gemm.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::FixedPointIntSplitArgs;
  using mptorch::gemm::FixedPointSplitArgs;

  template void CudaBackend::launch_as<float, FixedPointSplitArgs>(
      const GemmShape &, const FixedPointSplitArgs &, const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_as<float, FixedPointIntSplitArgs<false>>(
      const GemmShape &, const FixedPointIntSplitArgs<false> &, const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_as<float, FixedPointIntSplitArgs<true>>(
      const GemmShape &, const FixedPointIntSplitArgs<true> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
