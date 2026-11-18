// Explicit instantiations of the CPU GEMM kernel for
// custom_matmul_superfp_accumulated_odd and
// custom_matmul_superfp_fma_accumulated_odd. They accumulate under
// AccumulateAlgorithm::BLOCK (a sum per block of products, added to the running
// total) in the binary64 carrier (float64 operands), and round each carrier
// step to odd (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.h explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_odd_kernel.h"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, OddArgs<SuperfpSplitArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpSplitArgs>> &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, OddArgs<SuperfpFusedArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpFusedArgs>> &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
