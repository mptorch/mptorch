// The MPS entry points of the six counted GEMM ops (common/gemm_counted.h,
// common/fixedpoint_counted.h): the single-format GEMMs under every
// accumulate algorithm, NAIVE included, and the fixed-point GEMMs, counting
// their casts' numeric events.
//
// The bodies are the same text as their CPU twins
// (cpu/custom_matmul_counted_entry.cpp), which tests/test_entry_copies.py
// checks. The drivers (run_custom_matmul_counted,
// run_custom_matmul_fixedpoint_counted) allocate the counts on the operands'
// device and hand the MPS backend their data pointer, the counts' MTLBuffer;
// the kernel is mps/gemm.metal under MPT_COUNTING (mps/gemm_backend.h).

#include "../common/fixedpoint_counted_host.h"
#include "../common/gemm_counted_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;
using namespace mptorch::gemm;

namespace
{
  using Backend = mptorch::gemm_mps::MpsBackend;
}

std::tuple<Tensor, Tensor> binaryK_matmul_counted_mps(Tensor a, Tensor b, bool trans_a, bool trans_b,
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
  return run_custom_matmul_counted<Backend>(
      "custom_matmul_binaryK_counted",
      pack_binaryK_split(mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P,
                         acc_bias, acc_is_signed, mul_saturation_mode, mul_subnormals_mode,
                         acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

std::tuple<Tensor, Tensor> binaryK_matmul_fma_counted_mps(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                           bool fma_quant, int64_t fma_K, int64_t fma_P,
                                           int64_t fma_bias, bool fma_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                           int64_t fma_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                          int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                          int64_t outer_subnormals_mode, int64_t outer_prng_bits)
{
  return run_custom_matmul_counted<Backend>(
      "custom_matmul_binaryK_fma_counted",
      pack_binaryK_fused(fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, fma_saturation_mode,
                         fma_subnormals_mode, fma_prng_bits),
      pack_binaryK_outer(block_size, outer_quant, outer_K, outer_P, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_subnormals_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

std::tuple<Tensor, Tensor> superfp_matmul_counted_mps(Tensor a, Tensor b, bool trans_a, bool trans_b,
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
  return run_custom_matmul_counted<Backend>(
      "custom_matmul_superfp_counted",
      pack_superfp_split(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed,
                         accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades,
                         acc_bias, acc_is_signed, mul_saturation_mode, acc_saturation_mode,
                         mul_prng_bits, acc_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

std::tuple<Tensor, Tensor> superfp_matmul_fma_counted_mps(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                           bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                           int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                          int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                          bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits)
{
  return run_custom_matmul_counted<Backend>(
      "custom_matmul_superfp_fma_counted",
      pack_superfp_fused(fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias,
                         fma_is_signed, fma_saturation_mode, fma_prng_bits),
      pack_superfp_outer(block_size, outer_quant, outer_man_bits, outer_exp_bits,
                         outer_normal_binades, outer_bias, outer_is_signed,
                         outer_saturation_mode, outer_prng_bits),
      a, b, trans_a, trans_b, accumulate_algorithm, round_mode);
}

// The split mac: p = Q_M(RN_c(a * b)), s = Q_A(s + p). `integer` runs the
// integer accumulator where the carrier path would give the same bits, a
// hook for the tests that hold the two paths to each other.
std::tuple<Tensor, Tensor> fixedpoint_matmul_counted_mps(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_wl,
                             int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl,
                             int64_t acc_fl, bool acc_is_signed, bool acc_symmetric,
                             int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits,
                             bool wrap, bool integer)
{
  return run_custom_matmul_fixedpoint_counted<Backend>(
      "custom_matmul_fixedpoint_counted",
      pack_fixedpoint_split(mul_wl, mul_fl, mul_is_signed, mul_symmetric, acc_wl, acc_fl,
                            acc_is_signed, acc_symmetric, mul_prng_bits, acc_prng_bits, wrap,
                            integer),
      a, b, trans_a, trans_b, round_mode);
}

// The fused mac: s = Q_F(RN_c(a * b) + s), one rounding a step.
std::tuple<Tensor, Tensor> fixedpoint_matmul_fma_counted_mps(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t fma_wl,
                                 int64_t fma_fl, bool fma_is_signed, bool fma_symmetric,
                                 int64_t round_mode, int64_t fma_prng_bits, bool wrap)
{
  return run_custom_matmul_fixedpoint_counted<Backend>(
      "custom_matmul_fixedpoint_fma_counted",
      pack_fixedpoint_fused(fma_wl, fma_fl, fma_is_signed, fma_symmetric, fma_prng_bits, wrap), a,
      b, trans_a, trans_b, round_mode);
}
