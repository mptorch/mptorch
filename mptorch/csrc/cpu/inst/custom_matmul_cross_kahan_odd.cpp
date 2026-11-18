// The cross-family GEMMs under AccumulateAlgorithm::KAHAN with the carrier step rounded to odd
// (carrier_rounding=RoundMode.RO; common/carrier_odd_accumulate.h), in binary32.
//
// Instantiation-only, one of the 36 such files per backend that
// MPTORCH_ODD_ACCUMULATED=1 builds (setup.py), each the odd twin of the
// default file without the _odd suffix.
#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, OddArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKSuperfpArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, OddArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpBinaryKArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
