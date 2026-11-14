#include "custom_matmul_odd_kernel.h"

// binaryK formats, split mac, a palette, carrier step rounded to odd, in
// binary64: the kernels behind custom_matmul_binaryK_mixed_odd
// (carrier_rounding=RoundMode.RO) for float64 operands. The kernel template is
// the NAIVE ops' own; only the Args it is instantiated with are new, and no
// unit that existed before these ops includes the headers they come from, so
// their objects are unchanged. Its binary32 twin is
// custom_matmul_binaryK_mixed_odd.cpp. The entry points are in
// custom_matmul_odd_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSplitMixedArgs;
  using mptorch::gemm::OddPaletteArgs;

  template void CpuBackend::launch_as<double, OddPaletteArgs<BinaryKSplitMixedArgs>>(
      const GemmShape &, const OddPaletteArgs<BinaryKSplitMixedArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
