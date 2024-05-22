#include <ATen/ATen.h>
#include <tuple>

using namespace at;

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * having a symmeric number range.
 * Stochastic Rounding.
 **/
Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl,
                                            bool use_clamp, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * having a symmeric number range.
 * Nearest Rounding.
 **/
Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl,
                                         bool use_clamp, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the
 *clamping into a mask, with the option of having a symmetric number range
 * Stochastic Rounding.
 **/
std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask_cuda(Tensor a, int wl, int fl,
                                          bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the
 *clamping into a mask, with the option of having a symmetric number range
 * Nearest Rounding.
 **/
std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask_cuda(Tensor a, int wl, int fl,
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
 * Does not handle NaN, Inf, and denormal.
 * Stochastic Rounding.
 **/
Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits,
                                      bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Does not handle NaN, Inf, and denormal.
 * Nearest Rounding.
 **/
Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits,
                                   bool subnormals, bool saturate);

void float_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c, int M, int N,
                                    int K, int man_add, int exp_add,
                                    int man_mul, int exp_mul, bool subnormals,
                                    bool saturate);

void float_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c, int M, int N,
                                     int K, int man_add, int exp_add,
                                     int man_mul, int exp_mul, bool subnormals,
                                     bool saturate);

void float_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int man_fma, int exp_fma,
                                        bool subnormals, bool saturate);

void float_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int man_fma, int exp_fma,
                                         bool subnormals, bool saturate);

void float_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                       int N, int K, int man_add, int exp_add,
                                       int man_mul, int exp_mul,
                                       bool subnormals, bool saturate);

void float_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int man_add, int exp_add,
                                        int man_mul, int exp_mul,
                                        bool subnormals, bool saturate);

void float_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                           int N, int K, int man_fma,
                                           int exp_fma, bool subnormals,
                                           bool saturate);

void float_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                            int N, int K, int man_fma,
                                            int exp_fma, bool subnormals,
                                            bool saturate);

void fixed_point_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                          int N, int K, int wl_add, int fl_add,
                                          int wl_mul, int fl_mul,
                                          bool symmetric);

void fixed_point_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                           int N, int K, int wl_add, int fl_add,
                                           int wl_mul, int fl_mul,
                                           bool symmetric);

void fixed_point_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K, int wl_fma,
                                              int fl_fma, bool symmetric);

void fixed_point_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                               int M, int N, int K, int wl_fma,
                                               int fl_fma, bool symmetric);

void fixed_point_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                             int M, int N, int K, int wl_add,
                                             int fl_add, int wl_mul, int fl_mul,
                                             bool symmetric);

void fixed_point_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K, int wl_add,
                                              int fl_add, int wl_mul,
                                              int fl_mul, bool symmetric);

void fixed_point_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                 int M, int N, int K,
                                                 int wl_fma, int fl_fma,
                                                 bool symmetric);

void fixed_point_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                  int M, int N, int K,
                                                  int wl_fma, int fl_fma,
                                                  bool symmetric);