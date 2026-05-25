#pragma once

#include "mm_kernel.h"

template <class Qadd, class Qmul>
void launch_mm_addmul(float *a, float *b, float *c, int M, int K, int N, int batch,
                      bool compensated, Qadd quant_add, Qmul quant_mul)
{
  if (batch > 1)
  {
    if (compensated)
    {
      bmm_kahan_kernel(a, b, c, batch, M, K, N, quant_add, quant_mul);
    }
    else
    {
      bmm_kernel(a, b, c, batch, M, K, N, quant_add, quant_mul);
    }
  }
  else if (compensated)
  {
    mm_kahan_kernel(a, b, c, M, K, N, quant_add, quant_mul);
  }
  else
  {
    mm_kernel(a, b, c, M, K, N, quant_add, quant_mul);
  }
}

template <class Qfma>
void launch_mm_fma(float *a, float *b, float *c, int M, int K, int N, int batch,
                   bool compensated, Qfma quant_fma)
{
  if (batch > 1)
  {
    if (compensated)
    {
      bmm_kahan_fma_kernel(a, b, c, batch, M, K, N, quant_fma);
    }
    else
    {
      bmm_fma_kernel(a, b, c, batch, M, K, N, quant_fma);
    }
  }
  else if (compensated)
  {
    mm_kahan_fma_kernel(a, b, c, M, K, N, quant_fma);
  }
  else
  {
    mm_fma_kernel(a, b, c, M, K, N, quant_fma);
  }
}
