// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_binaryK_superfp_odd and custom_matmul_superfp_binaryK_odd. They
// accumulate under AccumulateAlgorithm::NAIVE (one running sum) in the binary32
// carrier (float32, float16 and bfloat16 operands), and round each carrier step
// to odd (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_as<float, OddArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const OddArgs<BinaryKSuperfpArgs> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, OddArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const OddArgs<SuperfpBinaryKArgs> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
