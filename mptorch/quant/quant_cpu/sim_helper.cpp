#include <cmath>
#include <cstdint>
#include "quant.h"

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max)
{
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

float round_helper(float a, float r) {
  // return floor(a+r);
  return nearbyint(a + r - 0.5);
}

float round(float a, float r, int sigma) {
  a = ldexp(a, -sigma);
  a = round_helper(a, r);
  a = ldexp(a, sigma);
  return a;
}

void dim_striding(Tensor a, int dim, DimStrides &strides) {
  int real_dim = (a.dim() + (dim % a.dim())) % a.dim();
  strides.outer_size = 1;
  strides.dim_size = a.size(real_dim);
  strides.inner_size = 1;
  for (int i = 0; i < real_dim; ++i) {
    strides.outer_size *= a.size(i);
  }
  for (int i = real_dim + 1; i < a.dim(); ++i) {
    strides.inner_size *= a.size(i);
  }
  strides.dim_stride = strides.inner_size;
  strides.outer_stride = strides.dim_size * strides.dim_stride;
}