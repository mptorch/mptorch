// binaryK conv passes (split and fused) under AccumulateAlgorithm::BLOCK with the carrier step rounded to odd
// (carrier_rounding=RoundMode.RO; common/carrier_odd_accumulate.h), in binary64.
//
// Instantiation-only, one of the 36 such files per backend that
// MPTORCH_ODD_ACCUMULATED=1 builds (setup.py), each the odd twin of the
// default file without the _odd suffix.
#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<OddArgs<BinaryKSplitArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<BinaryKSplitArgs>>> &,
      const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_conv_as<double, AccumulateAlgorithm::BLOCK, AccumulateArgs<OddArgs<BinaryKFusedArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<BinaryKFusedArgs>>> &,
      const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
