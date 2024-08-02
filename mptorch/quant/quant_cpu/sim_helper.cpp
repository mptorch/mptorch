#include <cmath>
#include <cstdint>
#include <vector>
#include <ATen/ATen.h>
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

DimSizes dim_striding(Tensor input, std::vector<int> &dims){
  DimSizes sizes;
  std::vector<int> real_dims(dims.size());
  for (int i = 0; i < dims.size(); i++){
    real_dims[i] = (input.dim() + (dims[i] % input.dim())) % input.dim();
  }

  sizes.channel = 1;
  for (int dim : real_dims){
    sizes.channel *= input.size(dim);
  }

  int min_dim = real_dims.back();
  int max_dim = real_dims.front();

  sizes.outer = 1;
  for (int i = 0; i < min_dim; i++){
    sizes.outer *= input.size(i);
  }

  sizes.inner = 1;
  for (int i = max_dim + 1; i < input.dim(); i++){
    sizes.inner *= input.size(i);
  }
  return sizes;
}