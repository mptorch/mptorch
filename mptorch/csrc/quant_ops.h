#pragma once

// Declarations of every per-backend op implementation, CPU, CUDA and MPS, in
// the same order as the schemas in quant_ops.cpp. The registration files
// (cpu/cpu_ops.cpp, cuda/cuda_ops.cpp, mps/mps_ops.cpp) and the entry point
// files include it. Every backend is declared in every build; a build
// without CUDA or MPS simply never references those, since only a
// registration file names them. narrow_float64 has no MPS kernel: MPS has
// no float64 tensor to narrow.
//
// <ATen/ATen.h> is deliberately not included: it pulls in ATen/Functions.h,
// the declaration of every operator in ATen, which costs about 22 s per
// translation unit through nvcc before a line of the file's own is compiled.
// <ATen/core/Tensor.h> is all these declarations need.
#include <ATen/core/Tensor.h>
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

// The in-place twins, mptorch::binaryK_quant_ and mptorch::superfp_quant_:
// the same rounding written over `a`, which is returned. They copy nothing,
// so each refuses what its out-of-place op launders (a strided tensor, and
// on CUDA a view that starts off a 16-byte boundary). The MPS pair raises:
// the Metal kernel is dev/continuation_plan.md's phase H.
at::Tensor &binaryK_quantize_cuda_(at::Tensor &a, int64_t K, int64_t P,
                                   int64_t bias, int64_t prng_bits,
                                   bool is_signed, int64_t round_mode,
                                   int64_t saturation_mode,
                                   int64_t subnormals_mode);

at::Tensor &binaryK_quantize_cpu_(at::Tensor &a, int64_t K, int64_t P,
                                  int64_t bias, int64_t prng_bits, bool is_signed,
                                  int64_t round_mode, int64_t saturation_mode,
                                  int64_t subnormals_mode);

at::Tensor &binaryK_quantize_mps_(at::Tensor &a, int64_t K, int64_t P,
                                  int64_t bias, int64_t prng_bits, bool is_signed,
                                  int64_t round_mode, int64_t saturation_mode,
                                  int64_t subnormals_mode);

at::Tensor &superfp_quantize_cuda_(at::Tensor &a, int64_t man_bits, int64_t exp_bits,
                                   int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                   bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor &superfp_quantize_cpu_(at::Tensor &a, int64_t man_bits, int64_t exp_bits,
                                  int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                  bool is_signed, int64_t round_mode, int64_t saturation_mode);

at::Tensor &superfp_quantize_mps_(at::Tensor &a, int64_t man_bits, int64_t exp_bits,
                                  int64_t normal_binades, int64_t bias, int64_t prng_bits,
                                  bool is_signed, int64_t round_mode, int64_t saturation_mode);

// A float64 tensor rounded once, to nearest even, onto float32, float16 or
// bfloat16: the store of a result computed in binary64 for a narrower
// tensor. The rounding is common/narrow_binary64.h.
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

// The four single-format GEMMs under an accumulate algorithm other than NAIVE
// (KAHAN, BLOCK, TREE; common/gemm_accumulate.h): their twin's arguments, then
// a block size and an outer format. Ops of their own rather than more
// arguments on the four above, so that a NAIVE call goes through the schema,
// the argument parsing and the entry point it always went through. On MPS
// they are bound to functions that raise (dev/continuation_plan.md, phase H).
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
