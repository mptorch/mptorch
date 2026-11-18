// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_binaryK_odd and custom_conv_binaryK_fma_odd. They accumulate
// under AccumulateAlgorithm::BLOCK (a sum per block of products, added to the
// running total) in the binary32 carrier (float32, float16 and bfloat16
// operands), and round each carrier step to odd
// (carrier_rounding=RoundMode.RO). The entry points are in
// ../custom_conv_odd_entry.cpp, and ../custom_matmul_kernel.cuh explains how
// the instantiations are divided among the files in this directory.

#include "../custom_matmul_kernel.cuh"
#include "../../common/carrier_odd_accumulate.h"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::AccumulateArgs;
  using mptorch::gemm::BinaryKFusedArgs;
  using mptorch::gemm::BinaryKSplitArgs;
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::OddArgs;
  template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::BLOCK, AccumulateArgs<OddArgs<BinaryKSplitArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<BinaryKSplitArgs>>> &,
      const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::BLOCK, AccumulateArgs<OddArgs<BinaryKFusedArgs>>>(
      const GemmShape &, const ConvArgs<AccumulateArgs<OddArgs<BinaryKFusedArgs>>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
