#pragma once

#include "binary8_kernel.h"
#include <ATen/ATen.h>
#include <tuple>
#include <cublas_v2.h>

using namespace at;

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * having a symmeric number range.
 * Stochastic Rounding.
 **/
Tensor fixed_point_quantize_stochastic_cuda(Tensor a,
                                            int wl, int fl,
                                            bool use_clamp, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * having a symmeric number range.
 * Nearest Rounding.
 **/
Tensor fixed_point_quantize_nearest_cuda(Tensor a,
                                         int wl, int fl,
                                         bool use_clamp, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the
 * clamping into a mask, with the option of having a symmetric number range
 * Stochastic Rounding.
 **/
std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask_cuda(Tensor a,
                                          int wl, int fl,
                                          bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the
 * clamping into a mask, with the option of having a symmetric number range
 * Nearest Rounding.
 **/
std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask_cuda(Tensor a,
                                       int wl, int fl,
                                       bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Stochastic Rounding.
 **/
Tensor block_quantize_stochastic_cuda(Tensor a, int wl, int dim);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Nearest Rounding.
 **/
Tensor block_quantize_nearest_cuda(Tensor a, int wl, int dim);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Stochastic Rounding.
 **/
Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Nearest Rounding.
 **/
Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Stochastic Rounding.
 **/
Tensor float_quantize_stochastic_cuda(Tensor a,
                                      int man_bits, int exp_bits,
                                      bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits using
 * [prng_bits] random bits.
 * Stochastic Rounding.
 **/
Tensor float_quantize_stochastic_cuda(Tensor a,
                                      int man_bits, int exp_bits, int prng_bits,
                                      bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Nearest Rounding.
 **/
Tensor float_quantize_nearest_cuda(Tensor a,
                                   int man_bits, int exp_bits,
                                   bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point SuperFloat Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Nearest Rounding.
 **/
Tensor superfp_quantize_nearest_cuda(Tensor a,
                                     int man_bits, int exp_bits,
                                     int binades_l, int binades_u,
                                     bool saturate);

/**
 * quantize a FloatTensor into a P3109-compliant floating point
 * Tensor (signed or unsigned version, with or without subnormal support)
 * with [P] precision bits.
 * Nearest Rounding.
 */
Tensor binary8_quantize_nearest_cuda(Tensor a,
                                     int P, bool is_signed, OverflowPolicy overflow_policy,
                                     bool subnormals);

/**
 * quantize a FloatTensor into a P3109-compliant floating point
 * Tensor (signed or unsigned version, with or without subnormal support)
 * with [P] precision bits.
 * Stochastic Rounding (with user-given PRNG resulution [prng_bits]).
 */
Tensor binary8_quantize_stochastic_cuda(Tensor a,
                                        int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy,
                                        bool subnormals);

/**
 * quantize a FloatTensor into a P3109-compliant floating point
 * Tensor (signed or unsigned version, with or without subnormal support)
 * with [P] precision bits.
 * Troncate Rounding (no rounding, just truncate the number).
 */
Tensor binary8_quantize_truncate_cuda(Tensor a,
                                      int P, bool is_signed, OverflowPolicy overflow_policy,
                                      bool subnormals);

/**
 * perform matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision floating-point compute; input tensors a
 * (size M x K) and b (size K x N) are multiplied with the result stored
 * in the output tensor c (size M x N).
 * Nearest Rounding.
 **/
void float_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                    int M, int N, int K,
                                    int man_add, int exp_add,
                                    int man_mul, int exp_mul,
                                    bool subnormals, bool saturate);

/**
 * perform batch matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision floating-point compute; input tensors a
 * (size B x M x K) and b (size B x K x N) are multiplied with the result stored
 * in the output tensor c (size B x M x N); B is inferred from the first dimension(s)
 * of the a and b tensors.
 * Nearest Rounding.
 **/
void float_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                     int M, int N, int K,
                                     int man_add, int exp_add,
                                     int man_mul, int exp_mul,
                                     bool subnormals, bool saturate);

/**
 * perform matrix multiplication with quantized FMA operations that simulate
 * low-precision floating-point compute; input tensors a (size M x K) and
 * b (size K x N) are multiplied with the result stored in the output tensor
 * c (size M x N).
 * Nearest Rounding.
 **/
void float_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_fma, int exp_fma,
                                        bool subnormals, bool saturate);

/**
 * perform batch matrix multiplication with quantized FMA operations that simulate
 * low-precision floating-point compute; input tensors a (size B x M x K) and
 * b (size B x K x N) are multiplied with the result stored in the output tensor
 * c (size B x M x N); B is inferred from the first dimension(s) of the a and b tensors.
 * Nearest Rounding.
 **/
void float_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                         int M, int N, int K,
                                         int man_fma, int exp_fma,
                                         bool subnormals, bool saturate);

/**
 * perform matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision super floating-point compute; input tensors a
 * (size M x K) and b (size K x N) are multiplied with the result stored
 * in the output tensor c (size M x N).
 * Nearest Rounding.
 **/
void superfp_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                      int M, int N, int K,
                                      int man_add, int exp_add,
                                      int man_mul, int exp_mul,
                                      int binades_add_l, int binades_add_u,
                                      int binades_mul_l, int binades_mul_u,
                                      bool saturate);

/**
 * perform batch matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision super floating-point compute; input tensors a
 * (size B x M x K) and b (size B x K x N) are multiplied with the result stored
 * in the output tensor c (size B x M x N); B is inferred from the first dimension(s)
 * of the a and b tensors.
 * Nearest Rounding.
 **/
void superfp_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_add, int exp_add,
                                       int man_mul, int exp_mul,
                                       int binades_add_l, int binades_add_u,
                                       int binades_mul_l, int binades_mul_u,
                                       bool saturate);

/**
 * perform matrix multiplication with quantized FMA operations that simulate
 * low-precision super floating-point compute; input tensors a (size M x K) and
 * b (size K x N) are multiplied with the result stored in the output tensor
 * c (size M x N).
 * Nearest Rounding.
 **/
void superfp_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                          int M, int N, int K,
                                          int man_fma, int exp_fma,
                                          int binades_fma_l, int binades_fma_u,
                                          bool saturate);

/**
 * perform batch matrix multiplication with quantized FMA operations that simulate
 * low-precision super floating-point compute; input tensors a (size B x M x K) and
 * b (size B x K x N) are multiplied with the result stored in the output tensor
 * c (size B x M x N); B is inferred from the first dimension(s) of the a and b tensors.
 * Nearest Rounding.
 **/
void superfp_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma,
                                           int binades_fma_l, int binades_fma_u,
                                           bool saturate);

/**
 * perform matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision floating-point compute; input tensors a
 * (size M x K) and b (size K x N) are multiplied with the result stored
 * in the output tensor c (size M x N).
 * Stochastic Rounding.
 **/
void float_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_add, int exp_add, int rbits_add,
                                       int man_mul, int exp_mul, int rbits_mul,
                                       bool subnormals, bool saturate);

/**
 * perform batch matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision floating-point compute; input tensors a
 * (size B x M x K) and b (size B x K x N) are multiplied with the result stored
 * in the output tensor c (size B x M x N); B is inferred from the first dimension(s)
 * of the a and b tensors.
 * Stochastic Rounding.
 **/
void float_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_add, int exp_add, int rbits_add,
                                        int man_mul, int exp_mul, int rbits_mul,
                                        bool subnormals, bool saturate);

/**
 * perform matrix multiplication with quantized FMA operations that simulate
 * low-precision floating-point compute; input tensors a (size M x K) and
 * b (size K x N) are multiplied with the result stored in the output tensor
 * c (size M x N).
 * Stochastic Rounding.
 **/
void float_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma, int rbits_fma,
                                           bool subnormals, bool saturate);

/**
 * perform batch matrix multiplication with quantized FMA operations that simulate
 * low-precision floating-point compute; input tensors a (size B x M x K) and
 * b (size B x K x N) are multiplied with the result stored in the output tensor
 * c (size B x M x N); B is inferred from the first dimension(s) of the a and b tensors.
 * Stochastic Rounding.
 **/
void float_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                            int M, int N, int K,
                                            int man_fma, int exp_fma, int rbits_fma,
                                            bool subnormals, bool saturate);

/**
 * perform matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision fixed-point compute; input tensors a
 * (size M x K) and b (size K x N) are multiplied with the result stored
 * in the output tensor c (size M x N).
 * Nearest Rounding.
 **/
void fixed_point_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                          int M, int N, int K,
                                          int wl_add, int fl_add,
                                          int wl_mul, int fl_mul,
                                          bool symmetric);

/**
 * perform batch matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision fixed-point compute; input tensors a
 * (size B x M x K) and b (size B x K x N) are multiplied with the result stored
 * in the output tensor c (size B x M x N); B is inferred from the first dimension(s)
 * of the a and b tensors.
 * Nearest Rounding.
 **/
void fixed_point_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int wl_add, int fl_add,
                                           int wl_mul, int fl_mul,
                                           bool symmetric);

/**
 * perform matrix multiplication with quantized FMA operations that simulate
 * low-precision fixed-point compute; input tensors a (size M x K) and
 * b (size K x N) are multiplied with the result stored in the output tensor
 * c (size M x N).
 * Nearest Rounding.
 **/
void fixed_point_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K,
                                              int wl_fma, int fl_fma,
                                              bool symmetric);

/**
 * perform batch matrix multiplication with quantized FMA operations that simulate
 * low-precision fixed-point compute; input tensors a (size B x M x K) and
 * b (size B x K x N) are multiplied with the result stored in the output tensor
 * c (size B x M x N); B is inferred from the first dimension(s) of the a and b tensors.
 * Nearest Rounding.
 **/
void fixed_point_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                               int M, int N, int K,
                                               int wl_fma, int fl_fma,
                                               bool symmetric);

/**
 * perform matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision fixed-point compute; input tensors a
 * (size M x K) and b (size K x N) are multiplied with the result stored
 * in the output tensor c (size M x N).
 * Stochastic Rounding.
 **/
void fixed_point_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                             int M, int N, int K,
                                             int wl_add, int fl_add,
                                             int wl_mul, int fl_mul,
                                             bool symmetric);

/**
 * perform batch matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision fixed-point compute; input tensors a
 * (size B x M x K) and b (size B x K x N) are multiplied with the result stored
 * in the output tensor c (size B x M x N); B is inferred from the first dimension(s)
 * of the a and b tensors.
 * Stochastic Rounding.
 **/
void fixed_point_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K,
                                              int wl_add, int fl_add,
                                              int wl_mul, int fl_mul,
                                              bool symmetric);

/**
 * perform matrix multiplication with quantized FMA operations that simulate
 * low-precision fixed-point compute; input tensors a (size M x K) and
 * b (size K x N) are multiplied with the result stored in the output tensor
 * c (size M x N).
 * Stochastic Rounding.
 **/
void fixed_point_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                 int M, int N, int K,
                                                 int wl_fma, int fl_fma,
                                                 bool symmetric);

/**
 * perform batch matrix multiplication with quantized FMA operations that simulate
 * low-precision fixed-point compute; input tensors a (size B x M x K) and
 * b (size B x K x N) are multiplied with the result stored in the output tensor
 * c (size B x M x N); B is inferred from the first dimension(s) of the a and b tensors.
 * Stochastic Rounding.
 **/
void fixed_point_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                  int M, int N, int K,
                                                  int wl_fma, int fl_fma,
                                                  bool symmetric);

/**
 * Precision configuration structure. Holds information
 * about the I/O matrix datatypes and compute precision
 * used during the CUBLAS (B)MM calls.
 */
struct CUBLASGemmConfig
{
    cudaDataType matrix_a;
    cudaDataType matrix_b;
    cudaDataType matrix_c;
    cudaDataType scalar;
    cublasComputeType_t compute;

    void summary() const;
};

/**
 * Possible I/O matrix datatypes (binary32, binary16 and bfloat16).
 */
enum class CUBLASMatrixType
{
    kF32,
    kF16,
    kBF16
};

/**
 * Compute precision/reduction configuration for CUBLAS computations.
 * The **fast** variations allow the use of tensor core with automatic
 * downconversion and binary16/bfloat16/tfloat32 compute for binary32
 * I/O matrices.
 */
enum class CUBLASComputeType
{
    kF32,
    kF16,
    kF32FastF16,
    kF32FastBF16,
    kF32FastTF32
};

/**
 * Creates a new cuBLAS handle, if none hasn't been created yet. If it
 * already exists, nothing happens.
 */
void create_cublas_handle();

/**
 * Deletes the current cuBLAS handle, if has already been created. If not,
 * throws an error.
 */
void delete_cublas_handle();

/**
 * Performs a matrix multiplication using cuBLAS API with the precision
 * configuration defined by the user.
 */
void float_mm_cublas(Tensor a, Tensor b, Tensor c,
                     int M, int N, int K,
                     CUBLASMatrixType AB_type, CUBLASMatrixType C_type,
                     CUBLASComputeType compute_type, bool pedantic);

/**
 * Performs a batched matrix multiplication using cuBLAS API with the
 * precision configuration defined by the user.
 */
void float_bmm_cublas(Tensor a, Tensor b, Tensor c,
                      int M, int N, int K,
                      CUBLASMatrixType AB_type, CUBLASMatrixType C_type,
                      CUBLASComputeType compute_type, bool pedantic);

/**
 * Performs layer normalization on a specified normalized shape with the
 * percision configuration defined by the user.
 */
void float_quantize_nearest_layernorm_forward_cuda(Tensor input, Tensor weight, Tensor bias,
                                                   Tensor output, Tensor mean, Tensor rstd,
                                                   float eps, std::vector<int> &dims,
                                                   int man_acc, int exp_acc,
                                                   int man_mul, int exp_mul,
                                                   int man_div, int exp_div,
                                                   int man_sqrt, int exp_sqrt,
                                                   bool subnormals, bool saturate);

/**
 * Performs layer normalization on a specified normalized shape with the
 * percision configuration defined by the user.
 */
void float_quantize_nearest_layernorm_backward_cuda(Tensor input, Tensor grad_output,
                                                    Tensor weight, Tensor bias,
                                                    Tensor mean, Tensor rstd,
                                                    Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                                    std::vector<int> &dims,
                                                    int man_acc, int exp_acc,
                                                    int man_mul, int exp_mul,
                                                    int man_div, int exp_div,
                                                    bool subnormals, bool saturate);

/**
 * Performs layer normalization on a specified normalized shape with the
 * percision configuration defined by the user.
 * Using super floating point for intermediate calculations.
 */
void superfp_quantize_nearest_layernorm_forward_cuda(Tensor input, Tensor weight, Tensor bias,
                                                     Tensor output, Tensor mean, Tensor rstd,
                                                     float eps, std::vector<int> &dims,
                                                     int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                                     int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                                     int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                                     int man_sqrt, int exp_sqrt, int binades_sqrt_l, int binades_sqrt_u,
                                                     bool saturate);

/**
 * Performs layer normalization on a specified normalized shape with the
 * percision configuration defined by the user.
 * Using super floating point for intermediate calculations.
 */
void superfp_quantize_nearest_layernorm_backward_cuda(Tensor input, Tensor grad_output,
                                                      Tensor weight, Tensor bias,
                                                      Tensor mean, Tensor rstd,
                                                      Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                                      std::vector<int> &dims,
                                                      int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                                      int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                                      int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                                      bool saturate);

/**
 * Performs layer normalization on a specified normalized shape with the
 * percision configuration defined by the user.
 * Using binary8 floating point for intermediate calculations.
 */
void binary8_quantize_nearest_layernorm_forward_cuda(Tensor input, Tensor weight, Tensor bias,
                                                     Tensor output, Tensor mean, Tensor rstd,
                                                     float eps, std::vector<int> &dims,
                                                     int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                                     int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                                     int P_div, OverflowPolicy op_div, bool signed_div,
                                                     int P_sqrt, OverflowPolicy op_sqrt, bool signed_sqrt,
                                                     bool subnormals);

/**
 * Performs layer normalization on a specified normalized shape with the
 * percision configuration defined by the user.
 * Using binary8 floating point for intermediate calculations.
 */
void binary8_quantize_nearest_layernorm_backward_cuda(Tensor input, Tensor grad_output,
                                                      Tensor weight, Tensor bias,
                                                      Tensor mean, Tensor rstd,
                                                      Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                                      std::vector<int> &dims,
                                                      int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                                      int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                                      int P_div, OverflowPolicy op_div, bool signed_div,
                                                      bool subnormals);

/**
 * Performs a softmax along the specified dimension, using custom floating
 * point formats for intermediate computations. This version implements
 * the regular accumulation of exponentials.
 */
void float_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                 int man_exp, int exp_exp,
                                                 int man_off, int exp_off,
                                                 int man_acc, int exp_acc,
                                                 bool subnormals, bool saturate);

/**
 * Performs a softmax along the specified dimension, using custom floating
 * point formats for intermediate computations. This version computes the
 * sum of exponentials via LogSumExp iterations, and does not use divisons.
 */
void float_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                     int man_off, int exp_off,
                                                     int man_lse, int exp_lse,
                                                     bool subnormals, bool saturate);

/**
 * Performs a regular softmax backward along the specified dimension, using custom
 * floating point formats for the intermediate computations.
 */
void float_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                  int man_add, int exp_add,
                                                  int man_mul, int exp_mul,
                                                  bool subnormals, bool saturate);

/**
 * Performs a softmax along the specified dimension, using custom super floating-point
 * formats for intermediate computations. This version implements
 * the regular accumulation of exponentials.
 */
void superfp_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                   int man_exp, int exp_exp, int binades_exp_l, int binades_exp_u,
                                                   int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                                   int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                                   bool saturate);

/**
 * Performs a softmax along the specified dimension, using custom super floating-point
 * formats for intermediate computations. This version computes the
 * sum of exponentials via LogSumExp iterations, and does not use divisons.
 */
void superfp_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                       int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                                       int man_lse, int exp_lse, int binades_lse_l, int binades_lse_u,
                                                       bool saturate);

/**
 * Performs a regular softmax backward along the specified dimension, using custom
 * super floating-point formats for the intermediate computations.
 */
void superfp_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                    int man_add, int exp_add, int binades_add_l, int binades_add_u,
                                                    int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                                    bool saturate);

/**
 * Performs a softmax along the specified dimension, using custom binary8
 * formats for intermediate computations. This version implements
 * the regular accumulation of exponentials.
 */
void binary8_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                   int P_exp, OverflowPolicy op_exp, bool signed_exp,
                                                   int P_off, OverflowPolicy op_off, bool signed_off,
                                                   int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                                   bool subnormals);

/**
 * Performs a softmax along the specified dimension, using custom binary8
 * formats for intermediate computations. This version computes the
 * sum of exponentials via LogSumExp iterations, and does not use divisons.
 */
void binary8_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                       int P_off, OverflowPolicy op_off, bool signed_off,
                                                       int P_lse, OverflowPolicy op_lse, bool signed_lse,
                                                       bool subnormals);

/**
 * Performs a regular softmax backward along the specified dimension, using custom
 * binary8 formats for the intermediate computations.
 */
void binary8_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                    int P_add, OverflowPolicy op_add, bool signed_add,
                                                    int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                                    bool subnormals);
