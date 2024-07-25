#include <ATen/ATen.h>
#include <tuple>
#include "binary8.h"

using namespace at;

enum Mode { rNearest, rStochastic };

struct DimStrides
{
    int outer_size;
    int inner_size;
    int outer_stride;
    int dim_size;
    int dim_stride;
};

float round(float a, float r, int sigma);

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max);

void dim_striding(Tensor a, int dim, DimStrides &strides);

uint32_t clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
                           uint32_t quantized_num, bool saturate);

uint32_t clip_max_exponent(int man_bits, uint32_t max_exponent,
                               uint32_t quantized_num);

std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl, bool symmetric);

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl, bool symmetric);

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

void float_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                            int man_exp, int exp_exp,
                                            int man_off, int exp_off,
                                            int man_acc, int exp_acc,
                                            bool subnormals, bool saturate);

void float_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                            int man_off, int exp_off,
                                            int man_lse, int exp_lse,
                                            bool subnormals, bool saturate);

void float_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                            int man_add, int exp_add,
                                            int man_mul, int exp_mul,
                                            bool subnormals, bool saturate);
/**
 * quantize a FloatTensor into a low bit-width floating point SuperFloat Tensor
 * with [man_bits] mantissa bits and [exp_bits] exponent bits.
 * Nearest Rounding.
 **/
Tensor superfp_quantize_nearest(Tensor a, int man_bits, int exp_bits, int binades, bool saturate);


/**
 * Quantizes a tensor to binary8 format using nearest rounding.
 * 
 * @param a                Input tensor.
 * @param P                The precision parameter for binary8 format.
 * @param is_signed        Flag indicating whether the values are signed or unsigned.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 * @return                 Quantized tensor.
 */
Tensor binary8_quantize_nearest_cpu(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Quantizes a tensor to binary8 format using stochastic rounding.
 * 
 * @param a                Input tensor.
 * @param P                The precision parameter for binary8 format.
 * @param prng_bits        The number of bits used for the pseudo-random number generator.
 * @param is_signed        Flag indicating whether the values are signed or unsigned.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 * @return                 Quantized tensor.
 */
Tensor binary8_quantize_stochastic_cpu(Tensor a, int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy, bool subnormals);

/**
 * Quantizes a tensor to binary8 format using truncation.
 * 
 * @param a                Input tensor.
 * @param P                The precision parameter for binary8 format.
 * @param is_signed        Flag indicating whether the values are signed or unsigned.
 * @param overflow_policy  Policy to handle overflow scenarios.
 * @param subnormals       Flag to enable or disable subnormal numbers.
 * @return                 Quantized tensor.
 */
Tensor binary8_quantize_truncate_cpu(Tensor a, int P, bool is_signed, OverflowPolicy overflow_policy, bool subnormals);
