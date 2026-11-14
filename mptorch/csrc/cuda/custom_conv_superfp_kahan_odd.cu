// superfp conv passes (split and fused) under AccumulateAlgorithm::KAHAN with the carrier step rounded to odd
// (carrier_rounding=RoundMode.RO; common/carrier_odd_accumulate.h), in binary32.
//
// Instantiation-only, one of the 36 such files per backend that
// MPTORCH_ODD_ACCUMULATED=1 builds (setup.py), each the odd twin of the
// default file without the _odd suffix.
#include "custom_matmul_kernel.cuh"
#include "../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<OddArgs<SuperfpSplitArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<SuperfpSplitArgs>>> &,
      const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::KAHAN, AccumulateArgs<OddArgs<SuperfpFusedArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<SuperfpFusedArgs>>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
