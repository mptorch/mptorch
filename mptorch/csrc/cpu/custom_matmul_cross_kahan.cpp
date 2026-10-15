#include "custom_matmul_kernel.h"

// The cross-family split macs (common/gemm_cross.h) under
// AccumulateAlgorithm::KAHAN (Kahan-compensated summation;
// common/gemm_accumulate.h), in binary32: the kernels of
// custom_matmul_binaryK_superfp_accumulated and
// custom_matmul_superfp_binaryK_accumulated, seven each. One of four
// cross-family GEMM instantiation files (custom_matmul_cross.cpp says why).

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, BinaryKSuperfpArgs>(
    const GemmShape &, const AccumulateArgs<BinaryKSuperfpArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, SuperfpBinaryKArgs>(
    const GemmShape &, const AccumulateArgs<SuperfpBinaryKArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
