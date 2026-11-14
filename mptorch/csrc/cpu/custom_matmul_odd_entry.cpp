// The CPU entry points of the four custom_matmul_*_odd ops
// (carrier_rounding=RoundMode.RO): their twins' entry points with the mac's
// carrier steps rounded to odd. The bodies are common/carrier_odd_host.h's,
// shared with the CUDA file; the kernels are instantiated by the
// custom_matmul_{binaryK,superfp}{_fma,}_odd{,_f64} files.

#include "../common/carrier_odd_gemm_host.h"
#include "../quant_ops.h"
#include "gemm_backend.h"

using at::Tensor;

namespace
{
  using Backend = mptorch::gemm_cpu::CpuBackend;
}

Tensor binaryK_matmul_odd_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return mptorch::gemm::run_binaryK_split_odd<Backend>(a, b, trans_a, trans_b, mul_K, mul_P, mul_bias, mul_is_signed, accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed, accumulate_algorithm, round_mode, mul_saturation_mode, mul_subnormals_mode, acc_saturation_mode, acc_subnormals_mode, mul_prng_bits, acc_prng_bits);
}

Tensor superfp_matmul_odd_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return mptorch::gemm::run_superfp_split_odd<Backend>(a, b, trans_a, trans_b, mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias, mul_is_signed, accumulate_quant, acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias, acc_is_signed, accumulate_algorithm, round_mode, mul_saturation_mode, acc_saturation_mode, mul_prng_bits, acc_prng_bits);
}

Tensor binaryK_matmul_fma_odd_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits)
{
  return mptorch::gemm::run_binaryK_fused_odd<Backend>(a, b, trans_a, trans_b, fma_quant, fma_K, fma_P, fma_bias, fma_is_signed, accumulate_algorithm, round_mode, fma_saturation_mode, fma_subnormals_mode, fma_prng_bits);
}

Tensor superfp_matmul_fma_odd_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  return mptorch::gemm::run_superfp_fused_odd<Backend>(a, b, trans_a, trans_b, fma_quant, fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias, fma_is_signed, accumulate_algorithm, round_mode, fma_saturation_mode, fma_prng_bits);
}

Tensor binaryK_matmul_mixed_odd_cpu(Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b,
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
  return mptorch::gemm::run_binaryK_matmul_mixed_odd<Backend>(a, b, prec_idx, trans_a, trans_b,
                                                              mul_K, mul_P, mul_bias, mul_is_signed,
                                                              accumulate_quant, acc_K, acc_P,
                                                              acc_bias, acc_is_signed,
                                                              accumulate_algorithm, round_mode,
                                                              mul_saturation_mode,
                                                              mul_subnormals_mode,
                                                              acc_saturation_mode,
                                                              acc_subnormals_mode, mul_prng_bits,
                                                              acc_prng_bits);
}

Tensor superfp_matmul_mixed_odd_cpu(Tensor a, Tensor b, Tensor prec_idx, bool trans_a, bool trans_b,
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
  return mptorch::gemm::run_superfp_matmul_mixed_odd<Backend>(a, b, prec_idx, trans_a, trans_b,
                                                              mul_man_bits, mul_exp_bits,
                                                              mul_normal_binades, mul_bias,
                                                              mul_is_signed, accumulate_quant,
                                                              acc_man_bits, acc_exp_bits,
                                                              acc_normal_binades, acc_bias,
                                                              acc_is_signed, accumulate_algorithm,
                                                              round_mode, mul_saturation_mode,
                                                              acc_saturation_mode, mul_prng_bits,
                                                              acc_prng_bits);
}

Tensor binaryK_matmul_fma_mixed_odd_cpu(Tensor a, Tensor b, Tensor prec_idx, bool trans_a,
                                        bool trans_b, bool fma_quant, c10::IntArrayRef fma_K,
                                        c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
                                        bool fma_is_signed, int64_t accumulate_algorithm,
                                        int64_t round_mode, int64_t fma_saturation_mode,
                                        int64_t fma_subnormals_mode, int64_t fma_prng_bits)
{
  return mptorch::gemm::run_binaryK_matmul_fma_mixed_odd<Backend>(a, b, prec_idx, trans_a, trans_b,
                                                                  fma_quant, fma_K, fma_P, fma_bias,
                                                                  fma_is_signed,
                                                                  accumulate_algorithm, round_mode,
                                                                  fma_saturation_mode,
                                                                  fma_subnormals_mode,
                                                                  fma_prng_bits);
}

Tensor superfp_matmul_fma_mixed_odd_cpu(Tensor a, Tensor b, Tensor prec_idx, bool trans_a,
                                        bool trans_b, bool fma_quant, c10::IntArrayRef fma_man_bits,
                                        c10::IntArrayRef fma_exp_bits,
                                        c10::IntArrayRef fma_normal_binades,
                                        c10::IntArrayRef fma_bias, bool fma_is_signed,
                                        int64_t accumulate_algorithm, int64_t round_mode,
                                        int64_t fma_saturation_mode, int64_t fma_prng_bits)
{
  return mptorch::gemm::run_superfp_matmul_fma_mixed_odd<Backend>(a, b, prec_idx, trans_a, trans_b,
                                                                  fma_quant, fma_man_bits,
                                                                  fma_exp_bits, fma_normal_binades,
                                                                  fma_bias, fma_is_signed,
                                                                  accumulate_algorithm, round_mode,
                                                                  fma_saturation_mode,
                                                                  fma_prng_bits);
}

Tensor binaryK_superfp_matmul_odd_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b, int64_t mul_K,
                                      int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                      int64_t acc_man_bits, int64_t acc_exp_bits,
                                      int64_t acc_normal_binades, int64_t acc_bias,
                                      bool acc_is_signed, int64_t accumulate_algorithm,
                                      int64_t round_mode, int64_t mul_saturation_mode,
                                      int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                                      int64_t mul_prng_bits, int64_t acc_prng_bits)
{
  return mptorch::gemm::run_binaryK_superfp_matmul_odd<Backend>(a, b, trans_a, trans_b, mul_K,
                                                                mul_P, mul_bias, mul_is_signed,
                                                                acc_man_bits, acc_exp_bits,
                                                                acc_normal_binades, acc_bias,
                                                                acc_is_signed, accumulate_algorithm,
                                                                round_mode, mul_saturation_mode,
                                                                mul_subnormals_mode,
                                                                acc_saturation_mode, mul_prng_bits,
                                                                acc_prng_bits);
}

Tensor superfp_binaryK_matmul_odd_cpu(Tensor a, Tensor b, bool trans_a, bool trans_b,
                                      int64_t mul_man_bits, int64_t mul_exp_bits,
                                      int64_t mul_normal_binades, int64_t mul_bias,
                                      bool mul_is_signed, int64_t acc_K, int64_t acc_P,
                                      int64_t acc_bias, bool acc_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                      int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                      int64_t acc_prng_bits)
{
  return mptorch::gemm::run_superfp_binaryK_matmul_odd<Backend>(a, b, trans_a, trans_b,
                                                                mul_man_bits, mul_exp_bits,
                                                                mul_normal_binades, mul_bias,
                                                                mul_is_signed, acc_K, acc_P,
                                                                acc_bias, acc_is_signed,
                                                                accumulate_algorithm, round_mode,
                                                                mul_saturation_mode,
                                                                acc_saturation_mode,
                                                                acc_subnormals_mode, mul_prng_bits,
                                                                acc_prng_bits);
}
