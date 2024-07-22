#include "quant_kernel.h"
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

float nearest_round(float a, int sigma) {
  a = ldexp(a, -sigma);
  // a = nearbyint(a);
  a = round(a);
  // a = floor(a+0.5);
  // a = ceil(a-0.5);
  a = ldexp(a, sigma);
  return a;
}