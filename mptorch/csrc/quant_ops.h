#pragma once

#include <ATen/ATen.h>
#include <cstdint>

at::Tensor binaryK_quantize_cuda(at::Tensor a, int64_t K, int64_t P,
                                 int64_t bias, int64_t prng_bits,
                                 bool is_signed, int64_t round_mode,
                                 int64_t saturation_mode,
                                 int64_t subnormals_mode);

at::Tensor binaryK_quantize_cpu(at::Tensor a, int64_t K, int64_t P,
                                int64_t bias, int64_t prng_bits, bool is_signed,
                                int64_t round_mode, int64_t saturation_mode,
                                int64_t subnormals_mode);

at::Tensor superfp_quantize_cuda(at::Tensor a, int64_t man_bits, int64_t exp_bits,
                                 int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                 bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor superfp_quantize_cpu(at::Tensor a, int64_t man_bits, int64_t exp_bits,
                                int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor binaryK_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                               int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                               bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                               int64_t acc_bias, bool acc_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t saturation_mode, int64_t subnormals_mode);

at::Tensor binaryK_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                              int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                              bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                              int64_t acc_bias, bool acc_is_signed,
                              int64_t accumulate_algorithm, int64_t round_mode,
                              int64_t saturation_mode, int64_t subnormals_mode);

at::Tensor superfp_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                               int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                               int64_t mul_bias, bool mul_is_signed,
                               bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                               int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode);

at::Tensor superfp_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                              int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                              int64_t mul_bias, bool mul_is_signed,
                              bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                              int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                              int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode);

at::Tensor binaryK_matmul_fma_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_K, int64_t fma_P,
                                   int64_t fma_bias, bool fma_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t saturation_mode, int64_t subnormals_mode);

at::Tensor binaryK_matmul_fma_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                  bool fma_quant, int64_t fma_K, int64_t fma_P,
                                  int64_t fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t saturation_mode, int64_t subnormals_mode);

at::Tensor superfp_matmul_fma_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                   int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode);

at::Tensor superfp_matmul_fma_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                  bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                  int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode);