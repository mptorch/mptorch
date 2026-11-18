// superfp conv passes (split) under AccumulateAlgorithm::TREE with the carrier step rounded to odd
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
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  template void CpuBackend::launch_conv_as<float, AccumulateAlgorithm::TREE, AccumulateArgs<OddArgs<SuperfpSplitArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<SuperfpSplitArgs>>> &,
      const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
