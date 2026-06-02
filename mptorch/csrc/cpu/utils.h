#pragma once

#include <cstdint>

template <typename scalar_t, class Quant>
void quant_kernel(scalar_t *a, scalar_t *o, int size, Quant quant)
{
  for (int i{0}; i < size; ++i)
  {
    o[i] = quant(a[i]);
  }
}

template <typename scalar_t, class Quant>
void quant_kernel(scalar_t *a, int *r, scalar_t *o, int size, Quant quant)
{
  for (int i{0}; i < size; ++i)
  {
    o[i] = quant(a[i], (uint32_t)r[i]);
  }
}
