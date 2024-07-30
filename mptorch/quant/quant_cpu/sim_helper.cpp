#include "quant.h"
#include <cmath>
#include <cstdint>
#include <ATen/ATen.h>

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

DimSizes partition_tensor(Tensor a, int dim) {
  DimSizes sizes;
  int real_dim = (a.dim() + (dim % a.dim())) % a.dim();
  sizes.outer = 1;
  sizes.channel = a.size(real_dim);
  sizes.inner = 1;
  for (int i = 0; i < real_dim; ++i) {
    sizes.outer *= a.size(i);
  }
  for (int i = real_dim + 1; i < a.dim(); ++i) {
    sizes.inner *= a.size(i);
  }
  return sizes;
}