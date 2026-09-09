// The two CPU GEMM entry points for binaryK formats with a fused mac.
//
// Each is the two things that genuinely differ between the eight ops --
// packing the flat schema into an Args (common/gemm_args.h) and naming the
// backend -- wrapped around the one driver in common/gemm_host.h. The bodies
// are the same text as their CUDA twins in cuda/custom_matmul_entry.cpp,
// which is what finding H1 was for.
//
// Still one translation unit per (format family x mac mode): unlike the CUDA
// side, whose .cu files hold the kernels and whose entry points all fit in
// one .cpp, a CPU entry point instantiates the kernel it calls, so the B2
// split is what keeps these four objects compiling in parallel rather than
// one of them being the pole.

#include "custom_matmul_kernel.h"
#include "../common/gemm_host.h"
#include "../quant_ops.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cpu::CpuBackend;
}

Tensor binaryK_matmul_fma_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               bool fma_quant, int64_t fma_K, int64_t fma_P,
                               int64_t fma_bias, bool fma_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                               int64_t fma_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_binaryK_fma",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// Spatially-varying mixed-format binaryK FMA GEMM: same FusedMac arithmetic
// as binaryK_matmul_fma_cpu (one rounding per K-step), with the FMA format
// per output element taken from the palette. fma_quant=false is rejected:
// FusedMac<IdentityAdder> carries no format, so a palette of it would make
// prec_idx a no-op -- use custom_matmul_binaryK_fma for the unquantized
// fused step.
Tensor binaryK_matmul_fma_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                                     bool trans_a, bool trans_b, bool fma_quant,
                                     c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                     c10::IntArrayRef fma_bias, bool fma_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                     int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_binaryK_fma_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_binaryK_fused_mixed(op, fma_K, fma_P, fma_bias, fma_is_signed,
                                        fma_saturation_mode, fma_subnormals_mode, fma_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_matmul_binaryK_fma");
      });
}
