// The CUDA entry points of the eight NAIVE GEMM ops: four single-format ops
// (binaryK and superfp, each with a split or a fused mac) and their four
// palette twins, in which each output element has its own format, picked
// from a list (the palette). cuda/cuda_ops.cpp registers them for the CUDA
// dispatch key.
//
// An entry point does only the two things that differ between the ops: it
// packs its flat schema arguments into an Args struct (common/gemm_args.h)
// and names the backend. The shared driver in common/gemm_host.h does the
// rest: the input checks, the shape derivation, the output allocation, the
// early return for an empty result, the prec_idx validation, the dtype tag
// and the RNG draw. The bodies are the same text as their CPU, MPS and Meta
// copies, which tests/test_entry_copies.py checks. Keeping at::Tensor here
// rather than in the .cu files is what lets those compile in about 3 s each
// instead of about 25 s (cuda/gemm_backend.h).
//
// This file is a .cpp inside csrc/cuda/ on purpose: setup.py compiles the
// files there only when the CUDA build is on, so a CPU-only build skips it
// as it skips the .cu files.

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
                           int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                           int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                           int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_binaryK",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// The palette twin of binaryK_matmul_cuda: the same split-mac arithmetic,
// but each output element C[i, j] takes its multiply and accumulate formats
// from a palette of up to MAX_GEMM_FORMATS entries, picked by prec_idx[i, j]
// (gemm_policy.h's FormatPalette). mul_K, mul_P and mul_bias, and acc_K,
// acc_P and acc_bias, give one value per palette entry, all lists of one
// length; the round mode, the saturation and subnormals modes, the signs
// and the prng_bits are shared by every entry, since only the format widths
// vary per element.
Tensor binaryK_matmul_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
                                 bool trans_a, bool trans_b,
                                 c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                 c10::IntArrayRef mul_bias, bool mul_is_signed,
                                 bool accumulate_quant, c10::IntArrayRef acc_K,
                                 c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
                                 bool acc_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t mul_saturation_mode,
                                 int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                                 int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                 int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_matmul_binaryK_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_binaryK_split_mixed(op, mul_K, mul_P, mul_bias, mul_is_signed,
                                        accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
                                        mul_saturation_mode, mul_subnormals_mode,
                                        acc_saturation_mode, acc_subnormals_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_fma_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
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

// The palette twin of binaryK_matmul_fma_cuda: the same fused-mac
// arithmetic (one rounding per step), with each output element's fused
// format taken from the palette. fma_quant=false is refused before the
// palette is built: an unrounded fused step has no format, so prec_idx would
// select nothing. custom_matmul_binaryK_fma is the GEMM with unrounded fused
// steps.
Tensor binaryK_matmul_fma_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
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

Tensor superfp_matmul_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
                           int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                           int64_t mul_bias, bool mul_is_signed,
                           bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                           int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                           int64_t accumulate_algorithm, int64_t round_mode,
                           int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                           int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_superfp",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// The superfp twin of binaryK_matmul_mixed_cuda: each output element's
// formats come from the palette, the four width lists (man_bits, exp_bits,
// normal_binades and bias, for each of the two formats) share one length,
// and everything else is shared by every entry.
Tensor superfp_matmul_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
                                 bool trans_a, bool trans_b,
                                 c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                                 c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                 bool mul_is_signed,
                                 bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                 c10::IntArrayRef acc_exp_bits,
                                 c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias,
                                 bool acc_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t mul_saturation_mode,
                                 int64_t acc_saturation_mode, int64_t mul_prng_bits,
                                 int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_split_mixed(op, mul_man_bits, mul_exp_bits, mul_normal_binades,
                                        mul_bias, mul_is_signed, accumulate_quant, acc_man_bits,
                                        acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed,
                                        mul_saturation_mode, acc_saturation_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_fma_cuda(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                               int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  return run_custom_matmul<Backend>(
      "custom_matmul_superfp_fma",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// The superfp twin of binaryK_matmul_fma_mixed_cuda: each output element's
// fused format comes from the palette, and fma_quant=false is refused for
// the same reason: an unrounded fused step has no format to vary.
Tensor superfp_matmul_fma_mixed_cuda(Tensor a, Tensor b, Tensor prec_idx,
                                     bool trans_a, bool trans_b, bool fma_quant,
                                     c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                     c10::IntArrayRef fma_normal_binades,
                                     c10::IntArrayRef fma_bias, bool fma_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_fma_mixed";
  return run_custom_matmul_mixed<Backend>(
      op,
      [&] {
        return pack_superfp_fused_mixed(op, fma_man_bits, fma_exp_bits, fma_normal_binades,
                                        fma_bias, fma_is_signed, fma_saturation_mode,
                                        fma_prng_bits);
      },
      a, b, prec_idx, trans_a, trans_b, accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_matmul_superfp_fma");
      });
}
