// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_superfp_fma_mixed_odd. They accumulate under
// AccumulateAlgorithm::NAIVE (one running sum) in the binary32 carrier
// (float32, float16 and bfloat16 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::OddPaletteArgs;
  using mptorch::gemm::SuperfpFusedMixedArgs;

  template void CpuBackend::launch_as<float, OddPaletteArgs<SuperfpFusedMixedArgs>>(
      const GemmShape &, const OddPaletteArgs<SuperfpFusedMixedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
