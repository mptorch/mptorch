// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_accumulated_odd. They accumulate under
// AccumulateAlgorithm::TREE (pairwise sums within each block of products) in
// the binary32 carrier (float32, float16 and bfloat16 operands), and round each
// carrier step to odd (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::OddArgs;
  template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::TREE, OddArgs<BinaryKSplitArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKSplitArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
