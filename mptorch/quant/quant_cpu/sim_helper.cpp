#include "quant.h"
#include <cmath>
#include <cstdint>

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max)
{
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

float round(float a, float r, int sigma)
{
  a = ldexp(a, -sigma);
  a = nearbyint(a + r - 0.5);
  // a = floor(a + r);
  a = ldexp(a, sigma);
  return a;
}

void tensor_strides(Tensor a, int dim, TensorStrides &strides) {
  // dimension we are iterating across, needed for negative dims
  int real_dim = (a.dim() + (dim % a.dim())) % a.dim();
  
  // columns to iterate over
  strides.outer_size = 1;
  for (int i = 0; i < real_dim; ++i) {
    strides.outer_size *= a.size(i);
  }

  // rows to iterate over
  strides.inner_size = 1;
  for (int i = real_dim + 1; i < a.dim(); ++i) {
    strides.inner_size *= a.size(i);
  }

  // size of the dimesnion we are iterating accross
  strides.dim_stride = strides.inner_size;

  // how much to iterate to get to next set of elements in dim
  int dim_size = a.size(real_dim);
  strides.dim_size = dim_size;
  strides.outer_stride = dim_size * strides.dim_stride;
}