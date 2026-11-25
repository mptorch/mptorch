// Explicit instantiations of the CUDA GEMM kernel's convolution form, which
// gathers its operands from a convolution's tensors, for
// custom_conv_fixedpoint_fma in the binary32 carrier (float32, float16 and
// bfloat16 operands): the integer accumulator's fused mac, saturating and
// wrapping around (common/fixedpoint_gemm.h). NAIVE only. The entry points
// are in ../custom_conv_fixedpoint_entry.cpp, and
// ../custom_matmul_kernel.cuh explains how the instantiations are divided
// among the files in this directory.

#include "../../common/fixedpoint_gemm.h"
#include "../custom_matmul_kernel.cuh"

namespace mptorch::gemm_cuda
{
  using mptorch::gemm::ConvArgs;
  using mptorch::gemm::FixedPointIntFusedArgs;

  template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                            FixedPointIntFusedArgs<false>>(
      const GemmShape &, const ConvArgs<FixedPointIntFusedArgs<false>> &,
      const CudaBackend::LaunchContext &);
  template void CudaBackend::launch_conv_as<float, AccumulateAlgorithm::NAIVE,
                                            FixedPointIntFusedArgs<true>>(
      const GemmShape &, const ConvArgs<FixedPointIntFusedArgs<true>> &,
      const CudaBackend::LaunchContext &);
} // namespace mptorch::gemm_cuda
