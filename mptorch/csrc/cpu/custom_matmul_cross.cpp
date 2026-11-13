#include "custom_matmul_kernel.h"

// The cross-family split macs (common/gemm_cross.h) under
// AccumulateAlgorithm::NAIVE, in binary32: the kernels of
// custom_matmul_binaryK_superfp (binaryK products, superfp sums) and
// custom_matmul_superfp_binaryK (the reverse) for float32, float16 and
// bfloat16 operands, seven each, one per round mode.
//
// One of four such files, one per algorithm, mirroring the single-family
// GEMMs' layout; both directions share each file, since each direction's
// kernels compile both families' casts anyway. They are files of their own so
// that no file that compiled a kernel before these ops existed compiles
// anything new. Its binary64 twin is custom_matmul_cross_f64.cpp. The entry
// points are in custom_matmul_cross_entry.cpp.

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::BinaryKSuperfpArgs;
  using mptorch::gemm::SuperfpBinaryKArgs;

  template void CpuBackend::launch_as<float, BinaryKSuperfpArgs>(
      const GemmShape &, const BinaryKSuperfpArgs &, const CpuBackend::LaunchContext &);
  template void CpuBackend::launch_as<float, SuperfpBinaryKArgs>(
      const GemmShape &, const SuperfpBinaryKArgs &, const CpuBackend::LaunchContext &);
} // namespace mptorch::gemm_cpu
