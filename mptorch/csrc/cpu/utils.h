#pragma once

#include <cstdint>

template <class Quant>
void quant_kernel(float *a, float *o, int size, Quant quant)
{
  for (int i{0}; i < size; ++i)
  {
    o[i] = quant(a[i]);
  }
}

template <class Quant>
void quant_kernel(float *a, int *r, float *o, int size, Quant quant)
{
  for (int i{0}; i < size; ++i)
  {
    o[i] = quant(a[i], (uint32_t)r[i]);
  }
}
