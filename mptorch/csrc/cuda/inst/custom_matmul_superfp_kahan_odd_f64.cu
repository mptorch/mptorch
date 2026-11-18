// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_superfp_accumulated_odd and
// custom_matmul_superfp_fma_accumulated_odd. They accumulate under
// AccumulateAlgorithm::KAHAN (compensated summation) in the binary64 carrier
// (float64 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpFusedArgs;
  using mptorch::gemm::SuperfpSplitArgs;
  template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, OddArgs<SuperfpSplitArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpSplitArgs>> &, const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::KAHAN, OddArgs<SuperfpFusedArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpFusedArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
