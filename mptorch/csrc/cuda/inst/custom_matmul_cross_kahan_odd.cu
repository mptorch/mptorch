// Explicit instantiations of the CUDA GEMM kernel for
// custom_matmul_binaryK_superfp_accumulated_odd and
// custom_matmul_superfp_binaryK_accumulated_odd. They accumulate under
// AccumulateAlgorithm::KAHAN (compensated summation) in the binary32 carrier
// (float32, float16 and bfloat16 operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_matmul_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::OddArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;
  template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, OddArgs<BinaryKSuperfpArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<BinaryKSuperfpArgs>> &, const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_accumulated_as<float, AccumulateAlgorithm::KAHAN, OddArgs<SuperfpBinaryKArgs>>(
      const GemmShape &, const AccumulateArgs<OddArgs<SuperfpBinaryKArgs>> &, const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
