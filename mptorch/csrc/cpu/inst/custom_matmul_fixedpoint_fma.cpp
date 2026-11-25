// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_fixedpoint_fma in the binary32 carrier (float32, float16 and
// bfloat16 operands): the integer accumulator's fused mac, saturating and
// wrapping around (common/fixedpoint_gemm.h). NAIVE only. The entry points
// are in ../custom_matmul_fixedpoint_entry.cpp, and ../custom_matmul_kernel.h
// explains how the instantiations are divided among the files in this
// directory.

#include "../../common/fixedpoint_gemm.h"
#include "../custom_matmul_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::FixedPointIntFusedArgs;

  template void CpuBackend::launch_as<float, FixedPointIntFusedArgs<false>>(
      const GemmShape &, const FixedPointIntFusedArgs<false> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, FixedPointIntFusedArgs<true>>(
      const GemmShape &, const FixedPointIntFusedArgs<true> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
