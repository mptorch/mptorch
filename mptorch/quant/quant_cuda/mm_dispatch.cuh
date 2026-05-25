#pragma once

#include "mm_kernel.h"
#include <cuda_runtime.h>

struct MmLaunchConfig
{
  static constexpr size_t THREADS_X{8U};
  static constexpr size_t THREADS_Y{8U};
  static constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};

  dim3 thread_dim;
  dim3 block_dim;

  MmLaunchConfig(int M, int N, int batch = 1)
      : thread_dim{THREADS_X, THREADS_Y, 1U},
        block_dim{
            static_cast<unsigned int>(
                (static_cast<uint32_t>(N) + THREADS_X - 1U) / THREADS_X),
            static_cast<unsigned int>(
                (static_cast<uint32_t>(M) + THREADS_Y - 1U) / THREADS_Y),
            static_cast<unsigned int>(batch)}
  {
  }
};

inline void init_curand_state(curandState_t **state, const dim3 &block_dim)
{
  cudaMalloc(reinterpret_cast<void **>(state),
             block_dim.x * block_dim.y * sizeof(curandState_t));
  seed_init<<<block_dim, 1>>>(*state);
}

inline void free_curand_state(curandState_t *state)
{
  cudaFree(state);
}

template <class Qadd, class Qmul>
void launch_mm_addmul(float *a, float *b, float *c, int M, int K, int N, int batch,
                      bool compensated, Qadd quant_add, Qmul quant_mul)
{
  MmLaunchConfig cfg(M, N, batch);
  if (batch > 1)
  {
    if (compensated)
    {
      bmm_kahan_impl<MmLaunchConfig::SHMEM_SIZE>
          <<<cfg.block_dim, cfg.thread_dim>>>(a, b, c, M, K, N, quant_add, quant_mul);
    }
    else
    {
      bmm_impl<1u, MmLaunchConfig::SHMEM_SIZE><<<cfg.block_dim, cfg.thread_dim>>>(
          a, b, c, M, K, N, quant_add, quant_mul);
    }
  }
  else if (compensated)
  {
    mm_kahan_impl<MmLaunchConfig::SHMEM_SIZE><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, M, K, N, quant_add, quant_mul);
  }
  else
  {
    mm_impl<1u, MmLaunchConfig::SHMEM_SIZE><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, M, K, N, quant_add, quant_mul);
  }
}

template <class Qfma>
void launch_mm_fma(float *a, float *b, float *c, int M, int K, int N, int batch,
                   bool compensated, Qfma quant_fma)
{
  MmLaunchConfig cfg(M, N, batch);
  if (batch > 1)
  {
    if (compensated)
    {
      bmm_kahan_fma_impl<MmLaunchConfig::SHMEM_SIZE>
          <<<cfg.block_dim, cfg.thread_dim>>>(a, b, c, M, K, N, quant_fma);
    }
    else
    {
      bmm_fma_impl<1u, MmLaunchConfig::SHMEM_SIZE><<<cfg.block_dim, cfg.thread_dim>>>(
          a, b, c, M, K, N, quant_fma);
    }
  }
  else if (compensated)
  {
    mm_kahan_fma_impl<MmLaunchConfig::SHMEM_SIZE><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, M, K, N, quant_fma);
  }
  else
  {
    mm_fma_impl<1u, MmLaunchConfig::SHMEM_SIZE><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, M, K, N, quant_fma);
  }
}

template <class RandType, class Qadd, class Qmul>
void launch_mm_stochastic(float *a, float *b, float *c, int M, int K, int N, int batch,
                            Qadd quant_add, Qmul quant_mul)
{
  MmLaunchConfig cfg(M, N, batch);
  curandState_t *state;
  init_curand_state(&state, cfg.block_dim);
  if (batch > 1)
  {
    bmm_sr_impl<MmLaunchConfig::SHMEM_SIZE, RandType><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, state, M, K, N, quant_add, quant_mul);
  }
  else
  {
    mm_sr_impl<MmLaunchConfig::SHMEM_SIZE, RandType><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, state, M, K, N, quant_add, quant_mul);
  }
  free_curand_state(state);
}

template <class RandType, class Qfma>
void launch_mm_fma_stochastic(float *a, float *b, float *c, int M, int K, int N, int batch,
                              Qfma quant_fma)
{
  MmLaunchConfig cfg(M, N, batch);
  curandState_t *state;
  init_curand_state(&state, cfg.block_dim);
  if (batch > 1)
  {
    bmm_sr_fma_impl<MmLaunchConfig::SHMEM_SIZE, RandType>
        <<<cfg.block_dim, cfg.thread_dim>>>(a, b, c, state, M, K, N, quant_fma);
  }
  else
  {
    mm_sr_fma_impl<MmLaunchConfig::SHMEM_SIZE, RandType><<<cfg.block_dim, cfg.thread_dim>>>(
        a, b, c, state, M, K, N, quant_fma);
  }
  free_curand_state(state);
}
