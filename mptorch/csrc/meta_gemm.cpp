// The Meta kernels of the 31 GEMM-shaped ops (phase O of
// dev/continuation_plan.md): the NAIVE, accumulated, cross-family and odd
// GEMMs, the conv ops and the block GEMM. Each body is its CPU entry
// point's, with the driver swapped for its twin in common/meta_host.h, which
// holds the rules: the CPU's checks, in its order, and a result of the shape,
// dtype and layout the CPU allocates, with no launch and no RNG draw. The
// registrations are meta_ops.cpp's.

#include "common/meta_host.h"
#include "meta_ops.h"

using at::Tensor;
using namespace mptorch::gemm;
using namespace mptorch::meta;

// From cpu/custom_matmul_entry.cpp.

Tensor binaryK_matmul_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                           int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                           bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                           int64_t acc_bias, bool acc_is_signed,
                           int64_t accumulate_algorithm, int64_t round_mode,
                           int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                           int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                           int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_binaryK",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
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
  return meta_custom_matmul_mixed(
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

Tensor binaryK_matmul_fma_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               bool fma_quant, int64_t fma_K, int64_t fma_P,
                               int64_t fma_bias, bool fma_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                               int64_t fma_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_binaryK_fma",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_fma_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
                                     bool trans_a, bool trans_b, bool fma_quant,
                                     c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                     c10::IntArrayRef fma_bias, bool fma_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                     int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_binaryK_fma_mixed";
  return meta_custom_matmul_mixed(
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

Tensor superfp_matmul_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                           int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                           int64_t mul_bias, bool mul_is_signed,
                           bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                           int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                           int64_t accumulate_algorithm, int64_t round_mode,
                           int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                           int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_superfp",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
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
  return meta_custom_matmul_mixed(
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

Tensor superfp_matmul_fma_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                               int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_superfp_fma",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_fma_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
                                     bool trans_a, bool trans_b, bool fma_quant,
                                     c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                     c10::IntArrayRef fma_normal_binades,
                                     c10::IntArrayRef fma_bias, bool fma_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_fma_mixed";
  return meta_custom_matmul_mixed(
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

// From cpu/custom_matmul_accumulated_entry.cpp.

Tensor binaryK_matmul_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                       int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                       bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                                       int64_t acc_bias, bool acc_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                       int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                       int64_t mul_prng_bits, int64_t acc_prng_bits,
                                      int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                      int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                      int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_binaryK_accumulated",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_fma_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                           bool fma_quant, int64_t fma_K, int64_t fma_P,
                                           int64_t fma_bias, bool fma_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                           int64_t fma_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                          int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                          int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_binaryK_fma_accumulated",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                       int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                                       int64_t mul_bias, bool mul_is_signed,
                                       bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                                       int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                       int64_t mul_prng_bits, int64_t acc_prng_bits,
                                      int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                      int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                      bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_superfp_accumulated",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_fma_accumulated_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                           bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                           int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                          int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                          bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_superfp_fma_accumulated",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// From cpu/custom_matmul_cross_entry.cpp.

Tensor binaryK_superfp_matmul_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                  int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                  int64_t acc_man_bits, int64_t acc_exp_bits,
                                  int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                  int64_t acc_saturation_mode, int64_t mul_prng_bits,
                                  int64_t acc_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_binaryK_superfp",
      pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed, acc_man_bits, acc_exp_bits,
                           acc_normal_binades, acc_bias, acc_is_signed, mul_saturation_mode,
                           mul_subnormals_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_binaryK_matmul_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                  int64_t mul_man_bits, int64_t mul_exp_bits,
                                  int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
                                  int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                  int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                  int64_t acc_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_superfp_binaryK",
      pack_superfp_binaryK(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                           acc_K, acc_P, acc_bias, acc_is_signed, mul_saturation_mode,
                           acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_superfp_matmul_accumulated_meta(
    Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias,
    bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
    int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
    int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades,
    int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
    int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_binaryK_superfp_accumulated",
      pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed, acc_man_bits, acc_exp_bits,
                           acc_normal_binades, acc_bias, acc_is_signed, mul_saturation_mode,
                           mul_subnormals_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_binaryK_matmul_accumulated_meta(
    Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits,
    int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P,
    int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
    int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
    int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
    int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_matmul_accumulated(
      "custom_matmul_superfp_binaryK_accumulated",
      pack_superfp_binaryK(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                           acc_K, acc_P, acc_bias, acc_is_signed, mul_saturation_mode,
                           acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_superfp_conv_meta(
    Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K,
    int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits,
    int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode,
    int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits,
    int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits,
    int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
    bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_conv_single(
      "custom_conv_binaryK_superfp",
      pack_binaryK_superfp(mul_K, mul_P, mul_bias, mul_is_signed, acc_man_bits, acc_exp_bits,
                           acc_normal_binades, acc_bias, acc_is_signed, mul_saturation_mode,
                           mul_subnormals_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups, accumulate_algorithm,
      round_mode);
}

Tensor superfp_binaryK_conv_meta(
    Tensor a, Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride,
    c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits,
    int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode,
    int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits,
    int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K,
    int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
    int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_conv_single(
      "custom_conv_superfp_binaryK",
      pack_superfp_binaryK(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                           acc_K, acc_P, acc_bias, acc_is_signed, mul_saturation_mode,
                           acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups, accumulate_algorithm,
      round_mode);
}

// From cpu/custom_conv_entry.cpp.

Tensor binaryK_conv_meta(Tensor a, Tensor b, int64_t conv_pass,
                        c10::IntArrayRef out_size, c10::IntArrayRef stride,
                        c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                        int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                        bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias,
                        bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
                        int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                        int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                        int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size,
                        bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                        bool outer_is_signed, int64_t outer_saturation_mode,
                        int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return meta_custom_conv_single(
      "custom_conv_binaryK",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor binaryK_conv_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
                              int64_t conv_pass, c10::IntArrayRef out_size,
                              c10::IntArrayRef stride, c10::IntArrayRef padding,
                              c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_K,
                              c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias,
                              bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K,
                              c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
                              bool acc_is_signed, int64_t accumulate_algorithm,
                              int64_t round_mode, int64_t mul_saturation_mode,
                              int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                              int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                              int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_conv_binaryK_mixed";
  return meta_custom_conv_mixed(
      op,
      [&] {
        return pack_binaryK_split_mixed(op, mul_K, mul_P, mul_bias, mul_is_signed,
                                        accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
                                        mul_saturation_mode, mul_subnormals_mode,
                                        acc_saturation_mode, acc_subnormals_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor binaryK_conv_fma_meta(Tensor a, Tensor b, int64_t conv_pass,
                            c10::IntArrayRef out_size, c10::IntArrayRef stride,
                            c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                            bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias,
                            bool fma_is_signed, int64_t accumulate_algorithm,
                            int64_t round_mode, int64_t fma_saturation_mode,
                            int64_t fma_subnormals_mode, int64_t fma_prng_bits,
                            int64_t block_size, bool outer_quant, int64_t outer_K,
                            int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
                            int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                            int64_t outer_prng_bits)
{
  return meta_custom_conv_single(
      "custom_conv_binaryK_fma",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor binaryK_conv_fma_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
                                  int64_t conv_pass, c10::IntArrayRef out_size,
                                  c10::IntArrayRef stride, c10::IntArrayRef padding,
                                  c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                  c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                  c10::IntArrayRef fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                  int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_conv_binaryK_fma_mixed";
  return meta_custom_conv_mixed(
      op,
      [&] {
        return pack_binaryK_fused_mixed(op, fma_K, fma_P, fma_bias, fma_is_signed,
                                        fma_saturation_mode, fma_subnormals_mode, fma_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_conv_binaryK_fma");
      });
}

Tensor superfp_conv_meta(Tensor a, Tensor b, int64_t conv_pass,
                        c10::IntArrayRef out_size, c10::IntArrayRef stride,
                        c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                        int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                        int64_t mul_bias, bool mul_is_signed, bool accumulate_quant,
                        int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
                        int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm,
                        int64_t round_mode, int64_t mul_saturation_mode,
                        int64_t acc_saturation_mode, int64_t mul_prng_bits,
                        int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
                        int64_t outer_man_bits, int64_t outer_exp_bits,
                        int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed,
                        int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_conv_single(
      "custom_conv_superfp",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor superfp_conv_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
                              int64_t conv_pass, c10::IntArrayRef out_size,
                              c10::IntArrayRef stride, c10::IntArrayRef padding,
                              c10::IntArrayRef dilation, int64_t groups,
                              c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                              c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                              bool mul_is_signed, bool accumulate_quant,
                              c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
                              c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias,
                              bool acc_is_signed, int64_t accumulate_algorithm,
                              int64_t round_mode, int64_t mul_saturation_mode,
                              int64_t acc_saturation_mode, int64_t mul_prng_bits,
                              int64_t acc_prng_bits)
{
  constexpr const char *op = "custom_conv_superfp_mixed";
  return meta_custom_conv_mixed(
      op,
      [&] {
        return pack_superfp_split_mixed(op, mul_man_bits, mul_exp_bits, mul_normal_binades,
                                        mul_bias, mul_is_signed, accumulate_quant, acc_man_bits,
                                        acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed,
                                        mul_saturation_mode, acc_saturation_mode, mul_prng_bits,
                                        acc_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor superfp_conv_fma_meta(Tensor a, Tensor b, int64_t conv_pass,
                            c10::IntArrayRef out_size, c10::IntArrayRef stride,
                            c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                            bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                            int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                            int64_t accumulate_algorithm, int64_t round_mode,
                            int64_t fma_saturation_mode, int64_t fma_prng_bits,
                            int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                            int64_t outer_exp_bits, int64_t outer_normal_binades,
                            int64_t outer_bias, bool outer_is_signed,
                            int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return meta_custom_conv_single(
      "custom_conv_superfp_fma",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed, outer_saturation_mode,
                         outer_prng_bits),
      a, b, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode);
}

Tensor superfp_conv_fma_mixed_meta(Tensor a, Tensor b, Tensor prec_idx,
                                  int64_t conv_pass, c10::IntArrayRef out_size,
                                  c10::IntArrayRef stride, c10::IntArrayRef padding,
                                  c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                  c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                  c10::IntArrayRef fma_normal_binades,
                                  c10::IntArrayRef fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_conv_superfp_fma_mixed";
  return meta_custom_conv_mixed(
      op,
      [&] {
        return pack_superfp_fused_mixed(op, fma_man_bits, fma_exp_bits, fma_normal_binades,
                                        fma_bias, fma_is_signed, fma_saturation_mode,
                                        fma_prng_bits);
      },
      a, b, prec_idx, conv_pass, out_size, stride, padding, dilation, groups,
      accumulate_algorithm, round_mode,
      [&] {
        TORCH_CHECK(fma_quant, op, ": fma_quant=false has no per-element format to vary; "
                                   "use custom_conv_superfp_fma");
      });
}

// From cpu/custom_matmul_odd_entry.cpp, through common/carrier_odd_host.h's
// bodies.

Tensor binaryK_matmul_odd_meta(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_K,
                               int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                               bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                               int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm,
                               int64_t round_mode, int64_t mul_saturation_mode,
                               int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                               int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                               int64_t acc_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_binaryK_odd",
      OddArgs<BinaryKSplitArgs>{pack_binaryK_split(
          mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P, acc_bias,
          acc_is_signed, mul_saturation_mode, mul_subnormals_mode, acc_saturation_mode,
          acc_subnormals_mode, mul_prng_bits, acc_prng_bits)},
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_odd_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                               int64_t mul_man_bits, int64_t mul_exp_bits,
                               int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
                               bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                               int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                               int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return meta_custom_matmul(
      "custom_matmul_superfp_odd",
      OddArgs<SuperfpSplitArgs>{pack_superfp_split(
          mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
          accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias,
          acc_is_signed, mul_saturation_mode, acc_saturation_mode, mul_prng_bits,
          acc_prng_bits)},
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor binaryK_matmul_fma_odd_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_K, int64_t fma_P,
                                   int64_t fma_bias, bool fma_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                   int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_binaryK_fma_odd";
  check_fma_quant(fma_quant, op);
  return meta_custom_matmul(
      op,
      OddArgs<BinaryKFusedArgs>{pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias,
                                                   fma_is_signed, fma_saturation_mode,
                                                   fma_subnormals_mode, fma_prng_bits)},
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

Tensor superfp_matmul_fma_odd_meta(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                   int64_t fma_normal_binades, int64_t fma_bias,
                                   bool fma_is_signed, int64_t accumulate_algorithm,
                                   int64_t round_mode, int64_t fma_saturation_mode,
                                   int64_t fma_prng_bits)
{
  constexpr const char *op = "custom_matmul_superfp_fma_odd";
  check_fma_quant(fma_quant, op);
  return meta_custom_matmul(
      op,
      OddArgs<SuperfpFusedArgs>{pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits,
                                                   fma_normal_binades, fma_bias, fma_is_signed,
                                                   fma_saturation_mode, fma_prng_bits)},
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// From cpu/custom_matmul_block_entry.cpp.

Tensor block_matmul_meta(Tensor a_data, Tensor a_scales, int64_t a_cols, bool trans_a,
                         Tensor b_data, Tensor b_scales, int64_t b_cols, bool trans_b,
                         c10::IntArrayRef a_fmt, double a_elem_max, double a_scale_max,
                         double a_tensor_scale, c10::IntArrayRef b_fmt, double b_elem_max,
                         double b_scale_max, double b_tensor_scale, bool fused,
                         bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias,
                         bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
                         int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                         int64_t acc_prng_bits, int64_t block_size, bool outer_quant,
                         int64_t outer_K, int64_t outer_P, int64_t outer_bias,
                         bool outer_is_signed, int64_t outer_saturation_mode,
                         int64_t outer_subnormals_mode, int64_t outer_prng_bits, bool binary64)
{
  const BlockGemmCall call{a_data,     a_scales,    a_cols,         trans_a,  b_data,
                           b_scales,   b_cols,      trans_b,        a_fmt,    a_elem_max,
                           a_scale_max, a_tensor_scale, b_fmt,      b_elem_max, b_scale_max,
                           b_tensor_scale, binary64};
  return meta_custom_matmul_block_entry(
      call, fused, accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed, accumulate_algorithm,
      round_mode, acc_saturation_mode, acc_subnormals_mode, acc_prng_bits,
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits));
}
