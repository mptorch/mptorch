// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_accumulated_odd and
// custom_matmul_binaryK_fma_accumulated_odd. They accumulate under
// AccumulateAlgorithm::BLOCK (a sum per block of products, added to the running
// total) in the binary64 carrier (float64 operands), and round each carrier
// step to odd (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::OddArgs;
  template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, OddArgs<BinaryKSplitArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKSplitArgs>> &, const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_accumulated_as<double, AccumulateAlgorithm::BLOCK, OddArgs<BinaryKFusedArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKFusedArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
