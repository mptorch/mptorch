#include <ATen/ATen.h>
#include <tuple>
#include "binary8.h"

using namespace at;

enum Mode { rNearest, rStochastic };

float round(float a, float r, int sigma);

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * having a symmeric number range.
 * Stochastic Rounding.
 **/
Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool clamp,
                                       bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], with option of clamping the over/underflow numbers
 * having a symmeric number range.
 * Nearest Rounding.
 **/
Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool clamp,
                                    bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the
 * clamping into a mask, with the option of having a symmetric number range
 * Stochastic Rounding.
 **/
std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl], clamp the over/underflow number and recording the
 * clamping into a mask, with the option of having a symmetric number range
 * Nearest Rounding.
 **/
std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl, bool symmetric);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Nearest Rounding.
 **/
Tensor block_quantize_nearest(Tensor a, int wl, int dim);

/**
 * quantize a FloatTensor into fixed point number with word length [wl]
 * and fractional bits [fl]
 * Stochastic Rounding.
 **/
Tensor block_quantize_stochastic(Tensor a, int wl, int dim);


Tensor float_quantize(Tensor a, int man_bits, int exp_bits, Mode rounding,
                      bool subnormal_support, bool saturate);

Tensor superfp_quantize(Tensor a, int man_bits, int exp_bits, int binades, bool saturate); 

/**
 * perform matrix multiplication with quantized addition and multiplication
 * operations that simulate low-precision floating-point compute; input tensors a
 * (size M x K) and b (size K x N) are multiplied with the result stored
 * in the output tensor c (size M x N).
 * Nearest Rounding.
 **/
void float_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                               int K, int man_add, int exp_add, int man_mul,
                               int exp_mul, bool subnormals, bool saturate);

/**
 * perform matrix multiplication with quantized FMA operations that simulate
 * low-precision floating-point compute; input tensors a (size M x K) and
 * b (size K x N) are multiplied with the result stored in the output tensor
 * c (size M x N).
 * Nearest Rounding.
 **/
void float_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M, int N, int K, int man_fma, int exp_fma, bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Stochastic Rounding.
 **/
Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits, bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Nearest Rounding.
 **/
Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits, bool subnormals, bool saturate);

/**
 * quantize a FloatTensor into a low bit-width floating point SuperFloat Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Nearest Rounding.
 **/
Tensor superfp_quantize_nearest(Tensor a, int man_bits, int exp_bits, int binades, bool saturate);

Tensor binary8_quantize_nearest_cuda(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals);
Tensor binary8_quantize_stochastic_cuda(Tensor a, int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy, bool subnormals);
Tensor binary8_quantize_truncate_cuda(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals);