#include <ATen/ATen.h>
#include <tuple>

using namespace at;

enum Mode { rNearest, rStochastic };

float round(float a, float r, int sigma);

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max);

uint32_t clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
                           uint32_t quantized_num, bool saturate);

uint32_t clip_max_exponent(int man_bits, uint32_t max_exponent,
                               uint32_t quantized_num);

std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask(Tensor a, int wl, int fl, bool symmetric);

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask(Tensor a, int wl, int fl, bool symmetric);

Tensor fixed_point_quantize_stochastic(Tensor a, int wl, int fl, bool clamp,
                                       bool symmetric);

Tensor fixed_point_quantize_nearest(Tensor a, int wl, int fl, bool clamp,
                                    bool symmetric);

Tensor block_quantize_nearest(Tensor a, int wl, int dim);

Tensor block_quantize_stochastic(Tensor a, int wl, int dim);

Tensor float_quantize(Tensor a, int man_bits, int exp_bits, Mode rounding,
                      bool subnormal_support, bool saturate);

Tensor superfp_quantize(Tensor a, int man_bits, int exp_bits, bool saturate); 

void float_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                               int K, int man_add, int exp_add, int man_mul,
                               int exp_mul, bool subnormals, bool saturate);

void float_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M, int N, int K, int man_fma, int exp_fma, bool subnormals, bool saturate);

Tensor float_quantize_stochastic(Tensor a, int man_bits, int exp_bits, bool subnormals, bool saturate);

Tensor float_quantize_nearest(Tensor a, int man_bits, int exp_bits, bool subnormals, bool saturate);

Tensor superfp_quantize_nearest(Tensor a, int man_bits, int exp_bits, bool saturate);