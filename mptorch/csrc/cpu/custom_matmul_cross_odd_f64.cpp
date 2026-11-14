#include "custom_matmul_odd_kernel.h"

// The cross-family split macs, carrier step rounded to odd, in binary64: the
// kernels behind custom_matmul_binaryK_superfp_odd and
// custom_matmul_superfp_binaryK_odd (carrier_rounding=RoundMode.RO) for float64
// operands. The kernel template is the NAIVE ops' own; only the Args it is
// instantiated with are new, and no unit that existed before these ops includes
// the headers they come from, so their objects are unchanged. Its binary32 twin
// is custom_matmul_cross_odd.cpp. The entry points are in
// custom_matmul_odd_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_as<double, OddArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const OddArgs<BinaryKSuperfpArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<double, OddArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const OddArgs<SuperfpBinaryKArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
