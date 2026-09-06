// The two CPU GEMM entry points for superfp formats with a fused mac.
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

Tensor superfp_matmul_fma_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                               int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t saturation_mode, int64_t fma_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_superfp_fma",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, saturation_mode, fma_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// superfp analogue of binaryK_matmul_fma_mixed_cpu -- see its comment.
Tensor superfp_matmul_fma_mixed_cpu(Tensor a, Tensor b, Tensor prec_idx,
                                     bool trans_a, bool trans_b, bool fma_quant,
                                     c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                     c10::IntArrayRef fma_normal_binades,
                                     c10::IntArrayRef fma_bias, bool fma_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t saturation_mode, int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_fma_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_fused_mixed(op, fma_man_bits, fma_exp_bits, fma_normal_binades,
                                        fma_bias, fma_is_signed, saturation_mode, fma_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_matmul_superfp_fma");
      });
}
