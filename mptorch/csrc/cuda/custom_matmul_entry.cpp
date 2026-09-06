// The eight CUDA GEMM entry points: everything TORCH_LIBRARY binds, and
// nothing nvcc has to see.
//
// Each is now the two things that genuinely differ between the ops -- packing
// the flat schema into an Args (common/gemm_args.h) and naming the backend --
// wrapped around the one driver in common/gemm_host.h. The ~1,100 lines of
// prologue these used to carry between them are gone; so is the reason the
// GEMM .cu files ever included ATen, which is worth ~22 s of nvcc apiece
// (finding H1, and see cuda/gemm_backend.h).
//
// This file is a .cpp inside csrc/cuda/ on purpose: setup.py globs those only
// when the CUDA build is on, so a CPU-only build skips it exactly as it skips
// the .cu files.

#include "../common/gemm_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_cuda::CudaBackend;
}

Tensor binaryK_matmul_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
                           int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                           bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                           int64_t acc_bias, bool acc_is_signed,
                           int64_t accumulate_algorithm, int64_t round_mode,
                           int64_t saturation_mode, int64_t subnormals_mode,
                           int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_binaryK",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, saturation_mode, subnormals_mode, mul_prng_bits,
                         acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// Spatially-varying (per-output-element) mixed-format binaryK GEMM: same
// SplitMac arithmetic as binaryK_matmul_cuda, but the multiply/accumulate
// format for each output element C[i, j] is picked from a palette of up to
// MAX_GEMM_FORMATS entries by prec_idx[i, j] (see gemm_policy.h's
// FormatPalette). mul_K/mul_P/mul_bias and acc_K/acc_P/acc_bias are
// per-palette-entry lists (all the same length); round_mode/saturation/
// sign/prng_bits are shared across the palette, matching the mm_impl
// prototype where only the format widths are tabulated.
Tensor binaryK_matmul_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
                                 bool trans_a, bool trans_b,
                                 c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                 c10::IntArrayRef mul_bias, bool mul_is_signed,
                                 bool accumulate_quant, c10::IntArrayRef acc_K,
                                 c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
                                 bool acc_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t saturation_mode,
                                 int64_t subnormals_mode, int64_t mul_prng_bits,
                                 int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_matmul_binaryK_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_binaryK_split_mixed(op, mul_K, mul_P, mul_bias, mul_is_signed,
                                        accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
                                        saturation_mode, subnormals_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_fma_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               bool fma_quant, int64_t fma_K, int64_t fma_P,
                               int64_t fma_bias, bool fma_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t saturation_mode, int64_t subnormals_mode,
                               int64_t fma_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_binaryK_fma",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, saturation_mode,
                         subnormals_mode, fma_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// Spatially-varying mixed-format binaryK FMA GEMM: same FusedMac arithmetic
// as binaryK_matmul_fma_cuda (one rounding per K-step), with the FMA format
// per output element taken from the palette. fma_quant=false is rejected:
// FusedMac<IdentityAdder> carries no format, so a palette of it would make
// prec_idx a no-op -- use custom_matmul_binaryK_fma for the unquantized
// fused step.
Tensor binaryK_matmul_fma_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
                                     bool trans_a, bool trans_b, bool fma_quant,
                                     c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                     c10::IntArrayRef fma_bias, bool fma_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t saturation_mode, int64_t subnormals_mode,
                                     int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_binaryK_fma_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_binaryK_fused_mixed(op, fma_K, fma_P, fma_bias, fma_is_signed, saturation_mode,
                                        subnormals_mode, fma_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_matmul_binaryK_fma");
      });
}

Tensor superfp_matmul_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
                           int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                           int64_t mul_bias, bool mul_is_signed,
                           bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                           int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                           int64_t accumulate_algorithm, int64_t round_mode,
                           int64_t saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_superfp",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, saturation_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// superfp analogue of binaryK_matmul_mixed_cuda -- see its comment.
Tensor superfp_matmul_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
                                 bool trans_a, bool trans_b,
                                 c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                                 c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                 bool mul_is_signed,
                                 bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                 c10::IntArrayRef acc_exp_bits,
                                 c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias,
                                 bool acc_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t saturation_mode,
                                 int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_split_mixed(op, mul_man_bits, mul_exp_bits, mul_normal_binades,
                                        mul_bias, mul_is_signed, accumulate_quant, acc_man_bits,
                                        acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed,
                                        saturation_mode, mul_prng_bits, acc_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_fma_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
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

// superfp analogue of binaryK_matmul_fma_mixed_cuda -- see its comment.
Tensor superfp_matmul_fma_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
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
