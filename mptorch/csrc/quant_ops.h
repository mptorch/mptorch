#pragma once

// Declarations of every op's per-backend implementations (CPU, CUDA and
// MPS), in the order of the schemas in quant_ops.cpp; the Meta kernels are
// declared in meta/meta_ops.h. The registration files (cpu/cpu_ops.cpp,
// cuda/cuda_ops.cpp, mps/mps_ops.cpp) and the entry point files include
// this header. Every backend is declared in every build; a build without
// CUDA or MPS simply never references those declarations, since only a
// registration file names them. narrow_float64 has no MPS kernel: MPS has no
// float64 tensor to narrow.
//
// <ATen/ATen.h> is deliberately not included: it pulls in ATen/Functions.h,
// the declaration of every operator in ATen, which costs about 22 s per
// translation unit through nvcc before a line of the file's own is
// compiled. <ATen/core/Tensor.h> is all these declarations need.
#include <ATen/core/Tensor.h>
#include <cstdint>
#include <tuple>

at::Tensor binaryK_quantize_cuda(at::Tensor a, int64_t K, int64_t P,
                                 int64_t bias, int64_t prng_bits,
                                 bool is_signed, int64_t round_mode,
                                 int64_t saturation_mode,
                                 int64_t subnormals_mode);

at::Tensor binaryK_quantize_cpu(at::Tensor a, int64_t K, int64_t P,
                                int64_t bias, int64_t prng_bits, bool is_signed,
                                int64_t round_mode, int64_t saturation_mode,
                                int64_t subnormals_mode);

at::Tensor binaryK_quantize_mps(at::Tensor a, int64_t K, int64_t P,
                                int64_t bias, int64_t prng_bits, bool is_signed,
                                int64_t round_mode, int64_t saturation_mode,
                                int64_t subnormals_mode);

at::Tensor superfp_quantize_cuda(at::Tensor a, int64_t man_bits, int64_t exp_bits,
                                 int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                 bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor superfp_quantize_cpu(at::Tensor a, int64_t man_bits, int64_t exp_bits,
                                int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor superfp_quantize_mps(at::Tensor a, int64_t man_bits, int64_t exp_bits,
                                int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor fixedpoint_quantize_cuda(at::Tensor a, int64_t wl, int64_t fl, int64_t prng_bits,
                                    bool is_signed, bool symmetric, int64_t round_mode);

at::Tensor fixedpoint_quantize_cpu(at::Tensor a, int64_t wl, int64_t fl, int64_t prng_bits,
                                   bool is_signed, bool symmetric, int64_t round_mode);

at::Tensor fixedpoint_quantize_mps(at::Tensor a, int64_t wl, int64_t fl, int64_t prng_bits,
                                   bool is_signed, bool symmetric, int64_t round_mode);

// The in-place twins, mptorch::binaryK_quant_, mptorch::superfp_quant_ and
// mptorch::fixedpoint_quant_: the same rounding, written over `a`. They
// return nothing (quant_ops.cpp says why) and copy nothing, so each refuses what its out-of-place op
// would copy (a strided tensor, and on CUDA a view that does not start on a
// 16-byte boundary). On MPS they bind the tensor as both the Metal kernel's
// input and its output.
void binaryK_quantize_cuda_(at::Tensor &a, int64_t K, int64_t P,
                            int64_t bias, int64_t prng_bits,
                            bool is_signed, int64_t round_mode,
                            int64_t saturation_mode,
                            int64_t subnormals_mode);

void binaryK_quantize_cpu_(at::Tensor &a, int64_t K, int64_t P,
                           int64_t bias, int64_t prng_bits, bool is_signed,
                           int64_t round_mode, int64_t saturation_mode,
                           int64_t subnormals_mode);

void binaryK_quantize_mps_(at::Tensor &a, int64_t K, int64_t P,
                           int64_t bias, int64_t prng_bits, bool is_signed,
                           int64_t round_mode, int64_t saturation_mode,
                           int64_t subnormals_mode);

void superfp_quantize_cuda_(at::Tensor &a, int64_t man_bits, int64_t exp_bits,
                            int64_t normal_binades, int64_t bias, int64_t prng_bits,
                            bool is_signed, int64_t round_mode, int64_t saturation_mode);

void superfp_quantize_cpu_(at::Tensor &a, int64_t man_bits, int64_t exp_bits,
                           int64_t normal_binades, int64_t bias, int64_t prng_bits,
                           bool is_signed, int64_t round_mode, int64_t saturation_mode);

void superfp_quantize_mps_(at::Tensor &a, int64_t man_bits, int64_t exp_bits,
                           int64_t normal_binades, int64_t bias, int64_t prng_bits,
                           bool is_signed, int64_t round_mode, int64_t saturation_mode);

void fixedpoint_quantize_cuda_(at::Tensor &a, int64_t wl, int64_t fl, int64_t prng_bits,
                               bool is_signed, bool symmetric, int64_t round_mode);

void fixedpoint_quantize_cpu_(at::Tensor &a, int64_t wl, int64_t fl, int64_t prng_bits,
                              bool is_signed, bool symmetric, int64_t round_mode);

void fixedpoint_quantize_mps_(at::Tensor &a, int64_t wl, int64_t fl, int64_t prng_bits,
                              bool is_signed, bool symmetric, int64_t round_mode);


// A float64 tensor rounded once, to nearest even, onto float32, float16 or
// bfloat16: the store of a result computed in binary64 for a narrower
// tensor. The rounding is common/narrow_binary64.h's.
at::Tensor narrow_float64_cuda(at::Tensor a, c10::ScalarType dtype);

at::Tensor narrow_float64_cpu(at::Tensor a, c10::ScalarType dtype);

at::Tensor binaryK_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                               int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                               bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                               int64_t acc_bias, bool acc_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                               int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                               int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                              int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                              bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                              int64_t acc_bias, bool acc_is_signed,
                              int64_t accumulate_algorithm, int64_t round_mode,
                              int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                              int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                              int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_matmul_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                              int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                              bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                              int64_t acc_bias, bool acc_is_signed,
                              int64_t accumulate_algorithm, int64_t round_mode,
                              int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                              int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                              int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                               int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                               int64_t mul_bias, bool mul_is_signed,
                               bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                               int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                               int64_t accumulate_algorithm, int64_t round_mode,
                               int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                               int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                              int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                              int64_t mul_bias, bool mul_is_signed,
                              bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                              int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                              int64_t accumulate_algorithm, int64_t round_mode,
                              int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                              int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_matmul_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                              int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                              int64_t mul_bias, bool mul_is_signed,
                              bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                              int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                              int64_t accumulate_algorithm, int64_t round_mode,
                              int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                              int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_matmul_fma_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_K, int64_t fma_P,
                                   int64_t fma_bias, bool fma_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                   int64_t fma_prng_bits);

at::Tensor binaryK_matmul_fma_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                  bool fma_quant, int64_t fma_K, int64_t fma_P,
                                  int64_t fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                  int64_t fma_prng_bits);

at::Tensor binaryK_matmul_fma_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                  bool fma_quant, int64_t fma_K, int64_t fma_P,
                                  int64_t fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                  int64_t fma_prng_bits);

at::Tensor superfp_matmul_fma_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                   bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                   int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t fma_saturation_mode, int64_t fma_prng_bits);

at::Tensor superfp_matmul_fma_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                  bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                  int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_prng_bits);

at::Tensor superfp_matmul_fma_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                  bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                  int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                  int64_t accumulate_algorithm, int64_t round_mode,
                                  int64_t fma_saturation_mode, int64_t fma_prng_bits);

// The four single-format GEMMs under an accumulate algorithm other than
// NAIVE (KAHAN, BLOCK or TREE; common/gemm_accumulate.h): their twin's
// arguments, then a block size and an outer format. They are ops of their
// own rather than more arguments on the four above, so that a NAIVE call
// goes through the schema, the argument parsing and the entry point it
// always went through.
at::Tensor binaryK_matmul_accumulated_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                           int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                           bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                                           int64_t acc_bias, bool acc_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                           int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                           int64_t mul_prng_bits, int64_t acc_prng_bits,
                                           int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                           int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                           int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_matmul_accumulated_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                          int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                          bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                                          int64_t acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                          int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                          int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                          int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_matmul_accumulated_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                          int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
                                          bool accumulate_quant, int64_t acc_K, int64_t acc_P,
                                          int64_t acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                          int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                          int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                          int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor superfp_matmul_accumulated_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                           int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                                           int64_t mul_bias, bool mul_is_signed,
                                           bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                                           int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                           int64_t accumulate_algorithm, int64_t round_mode,
                                           int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                           int64_t mul_prng_bits, int64_t acc_prng_bits,
                                           int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                           int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                           bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_matmul_accumulated_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                          int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                                          int64_t mul_bias, bool mul_is_signed,
                                          bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                                          int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                          int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                          bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_matmul_accumulated_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                          int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades,
                                          int64_t mul_bias, bool mul_is_signed,
                                          bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
                                          int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
                                          int64_t accumulate_algorithm, int64_t round_mode,
                                          int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                          int64_t mul_prng_bits, int64_t acc_prng_bits,
                                          int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                          int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                          bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor binaryK_matmul_fma_accumulated_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                               bool fma_quant, int64_t fma_K, int64_t fma_P,
                                               int64_t fma_bias, bool fma_is_signed,
                                               int64_t accumulate_algorithm, int64_t round_mode,
                                               int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                               int64_t fma_prng_bits,
                                               int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                               int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                               int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_matmul_fma_accumulated_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                              bool fma_quant, int64_t fma_K, int64_t fma_P,
                                              int64_t fma_bias, bool fma_is_signed,
                                              int64_t accumulate_algorithm, int64_t round_mode,
                                              int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                              int64_t fma_prng_bits,
                                              int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                              int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                              int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_matmul_fma_accumulated_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                              bool fma_quant, int64_t fma_K, int64_t fma_P,
                                              int64_t fma_bias, bool fma_is_signed,
                                              int64_t accumulate_algorithm, int64_t round_mode,
                                              int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                              int64_t fma_prng_bits,
                                              int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P,
                                              int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode,
                                              int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor superfp_matmul_fma_accumulated_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                               bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                               int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                               int64_t accumulate_algorithm, int64_t round_mode,
                                               int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                               int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                               int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                               bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_matmul_fma_accumulated_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                              bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                              int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                              int64_t accumulate_algorithm, int64_t round_mode,
                                              int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                              int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                              int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                              bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_matmul_fma_accumulated_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b,
                                              bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                              int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                              int64_t accumulate_algorithm, int64_t round_mode,
                                              int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                              int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                              int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias,
                                              bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor binaryK_matmul_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                     bool trans_a, bool trans_b,
                                     c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                     c10::IntArrayRef mul_bias, bool mul_is_signed,
                                     bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P,
                                     c10::IntArrayRef acc_bias, bool acc_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                     int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                     int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_matmul_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                    bool trans_a, bool trans_b,
                                    c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                    c10::IntArrayRef mul_bias, bool mul_is_signed,
                                    bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P,
                                    c10::IntArrayRef acc_bias, bool acc_is_signed,
                                    int64_t accumulate_algorithm, int64_t round_mode,
                                    int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                    int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                    int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_matmul_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                    bool trans_a, bool trans_b,
                                    c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                    c10::IntArrayRef mul_bias, bool mul_is_signed,
                                    bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P,
                                    c10::IntArrayRef acc_bias, bool acc_is_signed,
                                    int64_t accumulate_algorithm, int64_t round_mode,
                                    int64_t mul_saturation_mode, int64_t mul_subnormals_mode,
                                    int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                    int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_matmul_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                     bool trans_a, bool trans_b,
                                     c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                                     c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                     bool mul_is_signed,
                                     bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                     c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades,
                                     c10::IntArrayRef acc_bias, bool acc_is_signed,
                                     int64_t accumulate_algorithm, int64_t round_mode,
                                     int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                     int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_matmul_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                    bool trans_a, bool trans_b,
                                    c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                                    c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                    bool mul_is_signed,
                                    bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                    c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades,
                                    c10::IntArrayRef acc_bias, bool acc_is_signed,
                                    int64_t accumulate_algorithm, int64_t round_mode,
                                    int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                    int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_matmul_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                    bool trans_a, bool trans_b,
                                    c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
                                    c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias,
                                    bool mul_is_signed,
                                    bool accumulate_quant, c10::IntArrayRef acc_man_bits,
                                    c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades,
                                    c10::IntArrayRef acc_bias, bool acc_is_signed,
                                    int64_t accumulate_algorithm, int64_t round_mode,
                                    int64_t mul_saturation_mode, int64_t acc_saturation_mode,
                                    int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_matmul_fma_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                         bool trans_a, bool trans_b, bool fma_quant,
                                         c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                         c10::IntArrayRef fma_bias, bool fma_is_signed,
                                         int64_t accumulate_algorithm, int64_t round_mode,
                                         int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                         int64_t fma_prng_bits);

at::Tensor binaryK_matmul_fma_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                        bool trans_a, bool trans_b, bool fma_quant,
                                        c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                        c10::IntArrayRef fma_bias, bool fma_is_signed,
                                        int64_t accumulate_algorithm, int64_t round_mode,
                                        int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                        int64_t fma_prng_bits);

at::Tensor binaryK_matmul_fma_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                        bool trans_a, bool trans_b, bool fma_quant,
                                        c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                        c10::IntArrayRef fma_bias, bool fma_is_signed,
                                        int64_t accumulate_algorithm, int64_t round_mode,
                                        int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                        int64_t fma_prng_bits);

at::Tensor superfp_matmul_fma_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                         bool trans_a, bool trans_b, bool fma_quant,
                                         c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                         c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias,
                                         bool fma_is_signed, int64_t accumulate_algorithm,
                                         int64_t round_mode, int64_t fma_saturation_mode,
                                         int64_t fma_prng_bits);

at::Tensor superfp_matmul_fma_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                        bool trans_a, bool trans_b, bool fma_quant,
                                        c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                        c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias,
                                        bool fma_is_signed, int64_t accumulate_algorithm,
                                        int64_t round_mode, int64_t fma_saturation_mode,
                                        int64_t fma_prng_bits);

at::Tensor superfp_matmul_fma_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                        bool trans_a, bool trans_b, bool fma_quant,
                                        c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                        c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias,
                                        bool fma_is_signed, int64_t accumulate_algorithm,
                                        int64_t round_mode, int64_t fma_saturation_mode,
                                        int64_t fma_prng_bits);

// The eight conv ops: one pass of a convolution (the forward pass, the
// input gradient or the weight gradient) in the arithmetic of the GEMM op
// each is named after, with its operands gathered from the convolution's
// tensors rather than unfolded first (common/gemm_gather.h). The four
// single-format ops take every AccumulateAlgorithm, the four palette ops
// NAIVE only.
at::Tensor binaryK_conv_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass,
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
                             int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_conv_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass,
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
                            int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_conv_mps(at::Tensor a, at::Tensor b, int64_t conv_pass,
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
                            int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_conv_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                   int64_t conv_pass, c10::IntArrayRef out_size,
                                   c10::IntArrayRef stride, c10::IntArrayRef padding,
                                   c10::IntArrayRef dilation, int64_t groups,
                                   c10::IntArrayRef mul_K, c10::IntArrayRef mul_P,
                                   c10::IntArrayRef mul_bias, bool mul_is_signed,
                                   bool accumulate_quant, c10::IntArrayRef acc_K,
                                   c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
                                   bool acc_is_signed, int64_t accumulate_algorithm,
                                   int64_t round_mode, int64_t mul_saturation_mode,
                                   int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
                                   int64_t acc_subnormals_mode, int64_t mul_prng_bits,
                                   int64_t acc_prng_bits);

at::Tensor binaryK_conv_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
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
                                  int64_t acc_prng_bits);

at::Tensor binaryK_conv_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
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
                                  int64_t acc_prng_bits);

at::Tensor binaryK_conv_fma_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                 c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                 c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                 int64_t groups, bool fma_quant, int64_t fma_K, int64_t fma_P,
                                 int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t fma_saturation_mode,
                                 int64_t fma_subnormals_mode, int64_t fma_prng_bits,
                                 int64_t block_size, bool outer_quant, int64_t outer_K,
                                 int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
                                 int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                                 int64_t outer_prng_bits);

at::Tensor binaryK_conv_fma_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                                bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias,
                                bool fma_is_signed, int64_t accumulate_algorithm,
                                int64_t round_mode, int64_t fma_saturation_mode,
                                int64_t fma_subnormals_mode, int64_t fma_prng_bits,
                                int64_t block_size, bool outer_quant, int64_t outer_K,
                                int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
                                int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                                int64_t outer_prng_bits);

at::Tensor binaryK_conv_fma_mps(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                                bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias,
                                bool fma_is_signed, int64_t accumulate_algorithm,
                                int64_t round_mode, int64_t fma_saturation_mode,
                                int64_t fma_subnormals_mode, int64_t fma_prng_bits,
                                int64_t block_size, bool outer_quant, int64_t outer_K,
                                int64_t outer_P, int64_t outer_bias, bool outer_is_signed,
                                int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
                                int64_t outer_prng_bits);

at::Tensor binaryK_conv_fma_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                       int64_t conv_pass, c10::IntArrayRef out_size,
                                       c10::IntArrayRef stride, c10::IntArrayRef padding,
                                       c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                       c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                       c10::IntArrayRef fma_bias, bool fma_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                       int64_t fma_prng_bits);

at::Tensor binaryK_conv_fma_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                      int64_t conv_pass, c10::IntArrayRef out_size,
                                      c10::IntArrayRef stride, c10::IntArrayRef padding,
                                      c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                      c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                      c10::IntArrayRef fma_bias, bool fma_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                      int64_t fma_prng_bits);

at::Tensor binaryK_conv_fma_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                      int64_t conv_pass, c10::IntArrayRef out_size,
                                      c10::IntArrayRef stride, c10::IntArrayRef padding,
                                      c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                      c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
                                      c10::IntArrayRef fma_bias, bool fma_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t fma_saturation_mode, int64_t fma_subnormals_mode,
                                      int64_t fma_prng_bits);

at::Tensor superfp_conv_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass,
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
                             int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_conv_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass,
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
                            int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_conv_mps(at::Tensor a, at::Tensor b, int64_t conv_pass,
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
                            int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_conv_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
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
                                   int64_t acc_prng_bits);

at::Tensor superfp_conv_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
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
                                  int64_t acc_prng_bits);

at::Tensor superfp_conv_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
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
                                  int64_t acc_prng_bits);

at::Tensor superfp_conv_fma_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                 c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                 c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                 int64_t groups, bool fma_quant, int64_t fma_man_bits,
                                 int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias,
                                 bool fma_is_signed, int64_t accumulate_algorithm,
                                 int64_t round_mode, int64_t fma_saturation_mode,
                                 int64_t fma_prng_bits, int64_t block_size, bool outer_quant,
                                 int64_t outer_man_bits, int64_t outer_exp_bits,
                                 int64_t outer_normal_binades, int64_t outer_bias,
                                 bool outer_is_signed, int64_t outer_saturation_mode,
                                 int64_t outer_prng_bits);

at::Tensor superfp_conv_fma_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                                bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                int64_t accumulate_algorithm, int64_t round_mode,
                                int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                int64_t outer_exp_bits, int64_t outer_normal_binades,
                                int64_t outer_bias, bool outer_is_signed,
                                int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_conv_fma_mps(at::Tensor a, at::Tensor b, int64_t conv_pass,
                                c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                                bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits,
                                int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed,
                                int64_t accumulate_algorithm, int64_t round_mode,
                                int64_t fma_saturation_mode, int64_t fma_prng_bits,
                                int64_t block_size, bool outer_quant, int64_t outer_man_bits,
                                int64_t outer_exp_bits, int64_t outer_normal_binades,
                                int64_t outer_bias, bool outer_is_signed,
                                int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_conv_fma_mixed_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                       int64_t conv_pass, c10::IntArrayRef out_size,
                                       c10::IntArrayRef stride, c10::IntArrayRef padding,
                                       c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                       c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                       c10::IntArrayRef fma_normal_binades,
                                       c10::IntArrayRef fma_bias, bool fma_is_signed,
                                       int64_t accumulate_algorithm, int64_t round_mode,
                                       int64_t fma_saturation_mode, int64_t fma_prng_bits);

at::Tensor superfp_conv_fma_mixed_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                      int64_t conv_pass, c10::IntArrayRef out_size,
                                      c10::IntArrayRef stride, c10::IntArrayRef padding,
                                      c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                      c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                      c10::IntArrayRef fma_normal_binades,
                                      c10::IntArrayRef fma_bias, bool fma_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t fma_saturation_mode, int64_t fma_prng_bits);

at::Tensor superfp_conv_fma_mixed_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx,
                                      int64_t conv_pass, c10::IntArrayRef out_size,
                                      c10::IntArrayRef stride, c10::IntArrayRef padding,
                                      c10::IntArrayRef dilation, int64_t groups, bool fma_quant,
                                      c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
                                      c10::IntArrayRef fma_normal_binades,
                                      c10::IntArrayRef fma_bias, bool fma_is_signed,
                                      int64_t accumulate_algorithm, int64_t round_mode,
                                      int64_t fma_saturation_mode, int64_t fma_prng_bits);

// The block-format ops (common/block_decode.h, common/gemm_block.h): the
// elementwise pack, unpack and quantizers, and the block GEMM. MPS has no
// binary64, so its block ops refuse a float64 result.
std::tuple<at::Tensor, at::Tensor> block_pack_cuda(at::Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
std::tuple<at::Tensor, at::Tensor> block_pack_cpu(at::Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
std::tuple<at::Tensor, at::Tensor> block_pack_mps(at::Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);

at::Tensor block_unpack_cuda(at::Tensor data, at::Tensor scales, int64_t cols, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, at::ScalarType dtype);
at::Tensor block_unpack_cpu(at::Tensor data, at::Tensor scales, int64_t cols, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, at::ScalarType dtype);
at::Tensor block_unpack_mps(at::Tensor data, at::Tensor scales, int64_t cols, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, at::ScalarType dtype);

at::Tensor block_quantize_cuda(at::Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
at::Tensor block_quantize_cpu(at::Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
at::Tensor block_quantize_mps(at::Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);

void block_quantize_cuda_(at::Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
void block_quantize_cpu_(at::Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
void block_quantize_mps_(at::Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);

at::Tensor block_matmul_cuda(at::Tensor a_data, at::Tensor a_scales, int64_t a_cols, bool trans_a, at::Tensor b_data, at::Tensor b_scales, int64_t b_cols, bool trans_b, c10::IntArrayRef a_fmt, double a_elem_max, double a_scale_max, double a_tensor_scale, c10::IntArrayRef b_fmt, double b_elem_max, double b_scale_max, double b_tensor_scale, bool fused, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits, bool binary64);
at::Tensor block_matmul_cpu(at::Tensor a_data, at::Tensor a_scales, int64_t a_cols, bool trans_a, at::Tensor b_data, at::Tensor b_scales, int64_t b_cols, bool trans_b, c10::IntArrayRef a_fmt, double a_elem_max, double a_scale_max, double a_tensor_scale, c10::IntArrayRef b_fmt, double b_elem_max, double b_scale_max, double b_tensor_scale, bool fused, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits, bool binary64);
at::Tensor block_matmul_mps(at::Tensor a_data, at::Tensor a_scales, int64_t a_cols, bool trans_a, at::Tensor b_data, at::Tensor b_scales, int64_t b_cols, bool trans_b, c10::IntArrayRef a_fmt, double a_elem_max, double a_scale_max, double a_tensor_scale, c10::IntArrayRef b_fmt, double b_elem_max, double b_scale_max, double b_tensor_scale, bool fused, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits, bool binary64);

// The six cross-family ops (common/gemm_cross.h): a split mac whose products
// are rounded to a format of one family and whose partial sums to a format
// of the other. They are a NAIVE pair, its *_accumulated twins and the conv
// pair, laid out as the single-family ops are, in both carriers (binary32
// only on MPS).
at::Tensor binaryK_superfp_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_superfp_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_superfp_matmul_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor superfp_binaryK_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_binaryK_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_binaryK_matmul_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);

at::Tensor binaryK_superfp_matmul_accumulated_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_matmul_accumulated_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_matmul_accumulated_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_binaryK_matmul_accumulated_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_matmul_accumulated_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_matmul_accumulated_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);

at::Tensor binaryK_superfp_conv_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_conv_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_conv_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);

at::Tensor superfp_binaryK_conv_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_conv_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_conv_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);

// The four custom_matmul_*_odd ops (common/carrier_odd.h): the NAIVE
// single-family GEMMs with the mac's carrier steps rounded to odd, each on
// its twin's schema; and _carrier_odd, the round-to-odd primitives
// themselves, elementwise, for the tests. MPS runs them in binary32.
at::Tensor binaryK_matmul_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_matmul_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_matmul_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_matmul_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_matmul_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_matmul_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_matmul_fma_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor binaryK_matmul_fma_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor binaryK_matmul_fma_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor superfp_matmul_fma_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor superfp_matmul_fma_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor superfp_matmul_fma_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor carrier_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor c, int64_t op);
at::Tensor carrier_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor c, int64_t op);
at::Tensor carrier_odd_mps(at::Tensor a, at::Tensor b, at::Tensor c, int64_t op);

// _stochastic_cast (common/stochastic_cast_host.h): test-only, with no MPS
// kernel (the MPS entry point raises).
at::Tensor stochastic_cast_cpu(at::Tensor x, at::Tensor draws, int64_t family, c10::IntArrayRef format,
                               int64_t stochastic_mode);
at::Tensor stochastic_cast_cuda(at::Tensor x, at::Tensor draws, int64_t family, c10::IntArrayRef format,
                                int64_t stochastic_mode);
at::Tensor stochastic_cast_mps(at::Tensor x, at::Tensor draws, int64_t family, c10::IntArrayRef format,
                               int64_t stochastic_mode);

// The sixteen odd ops of the palettes, the cross-family macs and the conv
// passes (common/carrier_odd_gemm_host.h, common/carrier_odd_conv_host.h):
// each takes its twin's schema, with the mac's carrier steps rounded to odd.
// MPS runs them in binary32.
at::Tensor binaryK_matmul_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_matmul_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_matmul_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_matmul_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_matmul_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_matmul_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_matmul_fma_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor binaryK_matmul_fma_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor binaryK_matmul_fma_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor superfp_matmul_fma_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits, c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor superfp_matmul_fma_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits, c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor superfp_matmul_fma_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b, bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits, c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor binaryK_superfp_matmul_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_superfp_matmul_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_superfp_matmul_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_binaryK_matmul_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_binaryK_matmul_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_binaryK_matmul_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_conv_odd_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_conv_odd_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_conv_odd_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_conv_odd_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_conv_odd_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_conv_odd_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_conv_fma_odd_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_conv_fma_odd_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_conv_fma_odd_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_conv_fma_odd_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_conv_fma_odd_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_conv_fma_odd_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_conv_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_conv_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_conv_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_conv_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_conv_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor superfp_conv_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed, bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits, c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits);
at::Tensor binaryK_conv_fma_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor binaryK_conv_fma_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor binaryK_conv_fma_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits);
at::Tensor superfp_conv_fma_mixed_odd_cuda(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits, c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor superfp_conv_fma_mixed_odd_cpu(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits, c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor superfp_conv_fma_mixed_odd_mps(at::Tensor a, at::Tensor b, at::Tensor prec_idx, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits, c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits);
at::Tensor binaryK_superfp_conv_odd_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_conv_odd_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_conv_odd_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_conv_odd_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_conv_odd_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_conv_odd_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);

// The odd carrier step under KAHAN, BLOCK and TREE: six ops, each on its
// *_accumulated twin's schema (common/carrier_odd_gemm_host.h). Only a build
// with MPTORCH_ODD_ACCUMULATED=1 has their kernels.
at::Tensor binaryK_matmul_accumulated_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_matmul_accumulated_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_matmul_accumulated_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_matmul_accumulated_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_matmul_accumulated_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_matmul_accumulated_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_matmul_fma_accumulated_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_matmul_fma_accumulated_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor binaryK_matmul_fma_accumulated_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_matmul_fma_accumulated_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_matmul_fma_accumulated_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_matmul_fma_accumulated_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades, int64_t fma_bias, bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t fma_saturation_mode, int64_t fma_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_matmul_accumulated_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_matmul_accumulated_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor binaryK_superfp_matmul_accumulated_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits, int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_matmul_accumulated_odd_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_matmul_accumulated_odd_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);
at::Tensor superfp_binaryK_matmul_accumulated_odd_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias, bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode, int64_t outer_prng_bits);

// The scaled quantizers: each element multiplied by its scale, rounded and
// divided by the scale again, in one pass (common/scaled_quant_host.h). CPU
// in cpu/scaled_quant_kernel.cpp, CUDA in cuda/*_scaled_kernel.cu, MPS in
// mps/quantize.cpp over mps/scaled_quant.metal.
at::Tensor binaryK_quantize_scaled_cpu(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor binaryK_quantize_scaled_cuda(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor superfp_quantize_scaled_cpu(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor superfp_quantize_scaled_cuda(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor fixedpoint_quantize_scaled_cpu(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);
at::Tensor fixedpoint_quantize_scaled_cuda(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);
at::Tensor binaryK_quantize_scaled_mps(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor superfp_quantize_scaled_mps(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor fixedpoint_quantize_scaled_mps(at::Tensor a, at::Tensor scale, int64_t block_size, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);

// P3109's BlockProject: element codes under the caller's scale codes
// (common/block_project.h). CPU in cpu/block_project.cpp, CUDA in
// cuda/block_project.cu, MPS in mps/block_entry.cpp over
// mps/block_project.metal.
at::Tensor block_project_cpu(at::Tensor x, at::Tensor scales, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
at::Tensor block_project_cuda(at::Tensor x, at::Tensor scales, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);
at::Tensor block_project_mps(at::Tensor x, at::Tensor scales, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale, int64_t round_mode);

// The fixed-point GEMM (common/fixedpoint_gemm.h): a split mac and a fused
// one. CPU in cpu/custom_matmul_fixedpoint_entry.cpp, CUDA in
// cuda/custom_matmul_fixedpoint_entry.cpp, MPS in
// mps/custom_matmul_fixedpoint_entry.cpp.
at::Tensor fixedpoint_matmul_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_wl, int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl, int64_t acc_fl, bool acc_is_signed, bool acc_symmetric, int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, bool wrap, bool integer);
at::Tensor fixedpoint_matmul_fma_cpu(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t fma_wl, int64_t fma_fl, bool fma_is_signed, bool fma_symmetric, int64_t round_mode, int64_t fma_prng_bits, bool wrap);
at::Tensor fixedpoint_matmul_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_wl, int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl, int64_t acc_fl, bool acc_is_signed, bool acc_symmetric, int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, bool wrap, bool integer);
at::Tensor fixedpoint_matmul_fma_cuda(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t fma_wl, int64_t fma_fl, bool fma_is_signed, bool fma_symmetric, int64_t round_mode, int64_t fma_prng_bits, bool wrap);
at::Tensor fixedpoint_matmul_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t mul_wl, int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl, int64_t acc_fl, bool acc_is_signed, bool acc_symmetric, int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, bool wrap, bool integer);
at::Tensor fixedpoint_matmul_fma_mps(at::Tensor a, at::Tensor b, bool trans_a, bool trans_b, int64_t fma_wl, int64_t fma_fl, bool fma_is_signed, bool fma_symmetric, int64_t round_mode, int64_t fma_prng_bits, bool wrap);

// The fixed-point conv ops (common/fixedpoint_conv_host.h). CPU in
// cpu/custom_conv_fixedpoint_entry.cpp, CUDA in cuda/custom_conv_fixedpoint_entry.cpp,
// MPS in mps/custom_conv_fixedpoint_entry.cpp.
at::Tensor fixedpoint_conv_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_wl, int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl, int64_t acc_fl, bool acc_is_signed, bool acc_symmetric, int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, bool wrap, bool integer);
at::Tensor fixedpoint_conv_fma_cpu(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t fma_wl, int64_t fma_fl, bool fma_is_signed, bool fma_symmetric, int64_t round_mode, int64_t fma_prng_bits, bool wrap);
at::Tensor fixedpoint_conv_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_wl, int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl, int64_t acc_fl, bool acc_is_signed, bool acc_symmetric, int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, bool wrap, bool integer);
at::Tensor fixedpoint_conv_fma_cuda(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t fma_wl, int64_t fma_fl, bool fma_is_signed, bool fma_symmetric, int64_t round_mode, int64_t fma_prng_bits, bool wrap);
at::Tensor fixedpoint_conv_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t mul_wl, int64_t mul_fl, bool mul_is_signed, bool mul_symmetric, int64_t acc_wl, int64_t acc_fl, bool acc_is_signed, bool acc_symmetric, int64_t round_mode, int64_t mul_prng_bits, int64_t acc_prng_bits, bool wrap, bool integer);
at::Tensor fixedpoint_conv_fma_mps(at::Tensor a, at::Tensor b, int64_t conv_pass, c10::IntArrayRef out_size, c10::IntArrayRef stride, c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups, int64_t fma_wl, int64_t fma_fl, bool fma_is_signed, bool fma_symmetric, int64_t round_mode, int64_t fma_prng_bits, bool wrap);

// The fused pointwise and table ops (common/pointwise.h): one carrier
// operation, or a table's evaluation, and the elementwise quantizer's cast,
// in one pass. CPU in cpu/pointwise_{binaryK,superfp,fixedpoint}.cpp, CUDA
// in cuda/pointwise_entry.cpp.
at::Tensor binaryK_pointwise_cpu(at::Tensor a, std::optional<at::Tensor> b, std::optional<double> scalar, int64_t op, int64_t carrier_rounding, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor superfp_pointwise_cpu(at::Tensor a, std::optional<at::Tensor> b, std::optional<double> scalar, int64_t op, int64_t carrier_rounding, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor fixedpoint_pointwise_cpu(at::Tensor a, std::optional<at::Tensor> b, std::optional<double> scalar, int64_t op, int64_t carrier_rounding, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);
at::Tensor binaryK_pointwise_cuda(at::Tensor a, std::optional<at::Tensor> b, std::optional<double> scalar, int64_t op, int64_t carrier_rounding, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor superfp_pointwise_cuda(at::Tensor a, std::optional<at::Tensor> b, std::optional<double> scalar, int64_t op, int64_t carrier_rounding, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor fixedpoint_pointwise_cuda(at::Tensor a, std::optional<at::Tensor> b, std::optional<double> scalar, int64_t op, int64_t carrier_rounding, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);
at::Tensor binaryK_pwl_cpu(at::Tensor x, at::Tensor entries, double lo, double hi, int64_t tails, bool rsqrt, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor superfp_pwl_cpu(at::Tensor x, at::Tensor entries, double lo, double hi, int64_t tails, bool rsqrt, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor fixedpoint_pwl_cpu(at::Tensor x, at::Tensor entries, double lo, double hi, int64_t tails, bool rsqrt, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);
at::Tensor binaryK_pwl_cuda(at::Tensor x, at::Tensor entries, double lo, double hi, int64_t tails, bool rsqrt, int64_t K, int64_t P, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode);
at::Tensor superfp_pwl_cuda(at::Tensor x, at::Tensor entries, double lo, double hi, int64_t tails, bool rsqrt, int64_t man_bits, int64_t exp_bits, int64_t normal_binades, int64_t bias, int64_t prng_bits, bool is_signed, int64_t round_mode, int64_t saturation_mode);
at::Tensor fixedpoint_pwl_cuda(at::Tensor x, at::Tensor entries, double lo, double hi, int64_t tails, bool rsqrt, int64_t wl, int64_t fl, int64_t prng_bits, bool is_signed, bool symmetric, int64_t round_mode);
