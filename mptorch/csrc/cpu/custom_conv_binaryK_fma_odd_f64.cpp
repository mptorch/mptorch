#include "custom_matmul_odd_kernel.h"

// The conv ops over binaryK formats, fused mac, carrier step rounded to odd, in
// binary64: the kernels behind custom_conv_binaryK_fma_odd and
// custom_conv_binaryK_fma_mixed_odd (carrier_rounding=RoundMode.RO) for float64
// operands. The kernel template is the NAIVE ops' own; only the Args it is
// instantiated with are new, and no unit that existed before these ops includes
// the headers they come from, so their objects are unchanged. Its binary32 twin
// is custom_conv_binaryK_fma_odd.cpp. The entry points are in
// custom_conv_odd_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKFusedMixedArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::OddPaletteArgs;

  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddArgs<BinaryKFusedArgs>>(
      const GemmShape &, const ConvArgs<OddArgs<BinaryKFusedArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::NAIVE, OddPaletteArgs<BinaryKFusedMixedArgs>>(
      const GemmShape &, const ConvArgs<OddPaletteArgs<BinaryKFusedMixedArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
