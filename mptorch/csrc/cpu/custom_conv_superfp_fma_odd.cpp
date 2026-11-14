#include "custom_matmul_odd_kernel.h"

// The conv ops over superfp formats, fused mac, carrier step rounded to odd, in
// binary32: the kernels behind custom_conv_superfp_fma_odd and
// custom_conv_superfp_fma_mixed_odd (carrier_rounding=RoundMode.RO) for
// float32, float16 and bfloat16 operands. The kernel template is the NAIVE ops'
// own; only the Args it is instantiated with are new, and no unit that existed
// before these ops includes the headers they come from, so their objects are
// unchanged. Its binary64 twin is custom_conv_superfp_fma_odd_f64.cpp. The
// entry points are in custom_conv_odd_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::OddPaletteArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::SuperfpFusedMixedArgs;

  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddArgs<SuperfpFusedArgs>>(
      const GemmShape &, const ConvArgs<OddArgs<SuperfpFusedArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE, OddPaletteArgs<SuperfpFusedMixedArgs>>(
      const GemmShape &, const ConvArgs<OddPaletteArgs<SuperfpFusedMixedArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
