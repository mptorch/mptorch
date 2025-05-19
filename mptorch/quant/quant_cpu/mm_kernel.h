#pragma once

#include "quant.h"
#include <torch/torch.h>
#include <cmath>

using namespace at;

template <class Qadd, class Qmul>
void mm_kernel(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
            {
                c[i * N + j] = quant_add(c[i * N + j] + quant_mul(a[i * K + k] * b[k * N + j]));
            }
}

template<class Qadd, class Qmul>
void bmm_kernel(float *a, float *b, float *c, int B, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    for (int i = 0; i < B; ++i) {
        float* ia = a + i * M * K;
        float* ib = b + i * K * N;
        float* ic = c + i * K * M;
        mm_kernel(ia, ib, ic, M, K, N, quant_add, quant_mul);
    }
}

template <class Qfma>
void mm_fma_kernel(float *a, float *b, float *c, int M, int K, int N, Qfma quant_fma)
{
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                c[i * N + j] = quant_fma(fmaf(a[i * K + k], b[k * N + j], c[i * N + j]));
}

template<class Qfma>
void bmm_fma_kernel(float *a, float *b, float *c, int B, int M, int K, int N, Qfma quant_fma)
{
    for (int i = 0; i < B; ++i) {
        float* ia = a + i * M * K;
        float* ib = b + i * K * N;
        float* ic = c + i * K * M;
        mm_fma_kernel(ia, ib, ic, M, K, N, quant_fma);
    }
}

template <class Qadd, class Qmul>
void mm_kahan_kernel(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    float comp_term = 0.0f;
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
            {
                float update = quant_mul(a[i * K + k] * b[k * N + j]);
                float y = quant_add(update - comp_term);
                float t = quant_add(quant_add(t - c[i * N + j]) + y);
                c[i * N + j] = t;
            }
}

template <class Qadd, class Qmul>
void bmm_kahan_kernel(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul) {
    for (int i = 0; i < B; ++i) {
        float* ia = a + i * M * K;
        float* ib = b + i * K * N;
        float* ic = c + i * K * M;
        mm_kahan_kernel(ia, ib, ic, M, K, N, quant_add, quant_mul);
    }    
}

template <class Qfma>
void mm_kahan_fma_kernel(float *a, float *b, float *c, int M, int K, int N, Qfma quant_fma)
{
    float comp_term = 0.0f;
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
            {
                float y = quant_fma(fmaf(a[i * K + k], b[k * N + j], -comp_term));
                float t = quant_fma(quant_fma(t - c[i * N + j]) + y);
                c[i * N + j] = t;
            }
}

template <class Qfma>
void bmm_kahan_fma_kernel(float *a, float *b, float *c, int M, int K, int N, Qfma quant_fma)
{
    for (int i = 0; i < B; ++i) {
        float* ia = a + i * M * K;
        float* ib = b + i * K * N;
        float* ic = c + i * K * M;
        mm_kahan_fma_kernel(ia, ib, ic, M, K, N, quant_fma);
    }
}