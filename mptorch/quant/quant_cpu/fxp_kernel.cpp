#include "quant_kernel.h"
#include "sim_helper.cpp" // Assuming this contains necessary helper functions
#include <cmath>
#include <iostream>
#include <random>
#include <ctime>

template <typename T>
inline T clamp_helper(T a, T min, T max) {
  if (a > max)
    return max;
  else if (a < min)
    return min;
  else
    return a;
}

template <typename T>
inline T clamp_mask_helper(T a, T min, T max, uint8_t *mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  }
  *mask = 0;
  return a;
}

float cast_fxp_nearest(float origin_float, int sigma, float t_min, float t_max) {
  origin_float = nearest_round(origin_float, sigma);
  origin_float = clamp_helper(origin_float, t_min, t_max);
  return origin_float;
}

float cast_fxp_stochastic(float origin_float, float rand_prob, int sigma, float t_min, float t_max) {
  origin_float = round(origin_float, rand_prob, sigma);
  origin_float = clamp_helper(origin_float, t_min, t_max);
  return origin_float;
}

// quantize an array of real numbers into fixed point with word length [wl] and
// [fl] fractional bits 2**-[sigma] is the smallest unit of the fixed point
// representation. Stochastic Rounding with r.
void fixed_point_quantize_kernel_stochastic(float *a, float *r, float *o, int size, int sigma, bool use_clamp, float t_min, float t_max) {
  for (int index = 0; index < size; ++index) {
    o[index] = round(a[index], r[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

// quantize an array of real numbers into fixed point with word length [wl] and
// [fl] fractional bits 2**-[sigma] is the smallest unit of the fixed point
// representation. Nearest Neighbor Rounding.
void fixed_point_quantize_kernel_nearest(float *a, float *o, int size, int sigma, bool use_clamp, float t_min, float t_max) {
  for (int index = 0; index < size; ++index) {
    o[index] = nearest_round(a[index], sigma);
    if (use_clamp) {
      o[index] = clamp_helper(o[index], t_min, t_max);
    }
  }
}

void fixed_point_quantize_kernel_mask_stochastic(float *a, float *r, float *o, uint8_t *m, int size, int sigma, float t_min, float t_max) {
  for (int index = 0; index < size; ++index) {
    o[index] = round(a[index], r[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m + index);
  }
}

void fixed_point_quantize_kernel_mask_nearest(float *a, float *o, uint8_t *m, int size, int sigma, float t_min, float t_max) {
  for (int index = 0; index < size; ++index) {
    o[index] = nearest_round(a[index], sigma);
    o[index] = clamp_mask_helper(o[index], t_min, t_max, m + index);
  }
}

template <size_t SHMEM_SIZE>
void mm_fxp_nearest_impl(float *a, float *b, float *c, int M, int K, int N, int sigma_add, int t_min_add, int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul) {
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) {
        s_a[i] = a[row * K + i];
        s_b[i] = b[i * N + col];
      }
      for (int j = 0; j < K; ++j) {
        tmp = cast_fxp_nearest(tmp + cast_fxp_nearest(s_a[j] * s_b[j], sigma_mul, t_min_mul, t_max_mul), sigma_add, t_min_add, t_max_add);
      }
      c[row * N + col] = tmp;
    }
  }
}

template <size_t SHMEM_SIZE>
void bmm_fxp_nearest_impl(float *a, float *b, float *c, int B, int M, int K, int N, int sigma_add, int t_min_add, int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul) {
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int batch = 0; batch < B; ++batch) {
    int batch_a = batch * M * K;
    int batch_b = batch * K * N;
    int batch_c = batch * M * N;
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
          s_a[i] = a[batch_a + row * K + i];
          s_b[i] = b[batch_b + i * N + col];
        }
        for (int j = 0; j < K; ++j) {
          tmp = cast_fxp_nearest(tmp + cast_fxp_nearest(s_a[j] * s_b[j], sigma_mul, t_min_mul, t_max_mul), sigma_add, t_min_add, t_max_add);
        }
        c[batch_c + row * N + col] = tmp;
      }
    }
  }
}

template <size_t SHMEM_SIZE>
void mm_fxp_fma_nearest_impl(float *a, float *b, float *c, int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) {
        s_a[i] = a[row * K + i];
        s_b[i] = b[i * N + col];
      }
      for (int j = 0; j < K; ++j) {
        tmp = cast_fxp_nearest(std::fmaf(s_a[j], s_b[j], tmp), sigma_fma, t_min_fma, t_max_fma);
      }
      c[row * N + col] = tmp;
    }
  }
}

template <size_t SHMEM_SIZE>
void bmm_fxp_fma_nearest_impl(float *a, float *b, float *c, int B, int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int batch = 0; batch < B; ++batch) {
    int batch_a = batch * M * K;
    int batch_b = batch * K * N;
    int batch_c = batch * M * N;
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
          s_a[i] = a[batch_a + row * K + i];
          s_b[i] = b[batch_b + i * N + col];
        }
        for (int j = 0; j < K; ++j) {
          tmp = cast_fxp_nearest(std::fmaf(s_a[j], s_b[j], tmp), sigma_fma, t_min_fma, t_max_fma);
        }
        c[batch_c + row * N + col] = tmp;
      }
    }
  }
}

void mm_fxp_nearest(float *a, float *b, float *c, int M, int K, int N, int sigma_add, int t_min_add, int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  mm_fxp_nearest_impl<SHMEM_SIZE>(a, b, c, M, K, N, sigma_add, t_min_add, t_max_add, sigma_mul, t_min_mul, t_max_mul);
}

void bmm_fxp_nearest(float *a, float *b, float *c, int B, int M, int K, int N, int sigma_add, int t_min_add, int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  bmm_fxp_nearest_impl<SHMEM_SIZE>(a, b, c, B, M, K, N, sigma_add, t_min_add, t_max_add, sigma_mul, t_min_mul, t_max_mul);
}

void mm_fxp_fma_nearest(float *a, float *b, float *c, int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  mm_fxp_fma_nearest_impl<SHMEM_SIZE>(a, b, c, M, K, N, sigma_fma, t_min_fma, t_max_fma);
}

void bmm_fxp_fma_nearest(float *a, float *b, float *c, int B, int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  bmm_fxp_fma_nearest_impl<SHMEM_SIZE>(a, b, c, B, M, K, N, sigma_fma, t_min_fma, t_max_fma);
}

std::mt19937 gen(static_cast<unsigned int>(std::time(nullptr))); // Random number generator
std::uniform_real_distribution<float> dist(0.0, 1.0);

float get_random_prob() {
  return dist(gen);
}

void mm_fxp_stochastic(float *a, float *b, float *c, int M, int K, int N, int sigma_add, int t_min_add, int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) {
        s_a[i] = a[row * K + i];
        s_b[i] = b[i * N + col];
      }
      for (int j = 0; j < K; ++j) {
        float radd = get_random_prob();
        float rmul = get_random_prob();
        tmp = cast_fxp_stochastic(tmp + cast_fxp_stochastic(s_a[j] * s_b[j], rmul, sigma_mul, t_min_mul, t_max_mul), radd, sigma_add, t_min_add, t_max_add);
      }
      c[row * N + col] = tmp;
    }
  }
}

void bmm_fxp_stochastic(float *a, float *b, float *c, int B, int M, int K, int N, int sigma_add, int t_min_add, int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int batch = 0; batch < B; ++batch) {
    int batch_a = batch * M * K;
    int batch_b = batch * K * N;
    int batch_c = batch * M * N;
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
          s_a[i] = a[batch_a + row * K + i];
          s_b[i] = b[batch_b + i * N + col];
        }
        for (int j = 0; j < K; ++j) {
          float radd = get_random_prob();
          float rmul = get_random_prob();
          tmp = cast_fxp_stochastic(tmp + cast_fxp_stochastic(s_a[j] * s_b[j], rmul, sigma_mul, t_min_mul, t_max_mul), radd, sigma_add, t_min_add, t_max_add);
        }
        c[batch_c + row * N + col] = tmp;
      }
    }
  }
}

void mm_fxp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int row = 0; row < M; ++row) {
    for (int col = 0; col < N; ++col) {
      float tmp = 0.0f;
      for (int i = 0; i < K; ++i) {
        s_a[i] = a[row * K + i];
        s_b[i] = b[i * N + col];
      }
      for (int j = 0; j < K; ++j) {
        float rfma = get_random_prob();
        tmp = cast_fxp_stochastic(std::fmaf(s_a[j], s_b[j], tmp), rfma, sigma_fma, t_min_fma, t_max_fma);
      }
      c[row * N + col] = tmp;
    }
  }
}

void bmm_fxp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma) {
  constexpr size_t SHMEM_SIZE{8U * 8U};
  float s_a[SHMEM_SIZE];
  float s_b[SHMEM_SIZE];

  for (int batch = 0; batch < B; ++batch) {
    int batch_a = batch * M * K;
    int batch_b = batch * K * N;
    int batch_c = batch * M * N;
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
          s_a[i] = a[batch_a + row * K + i];
          s_b[i] = b[batch_b + i * N + col];
        }
        for (int j = 0; j < K; ++j) {
          float rfma = get_random_prob();
          tmp = cast_fxp_stochastic(std::fmaf(s_a[j], s_b[j], tmp), rfma, sigma_fma, t_min_fma, t_max_fma);
        }
        c[batch_c + row * N + col] = tmp;
      }
    }
  }
}

int main() {
    // Example testing code here
    const int size = 10;
    float a[size] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
    float r[size] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    float o[size];

    // Test a function from fxp_kernel.cpp
    fixed_point_quantize_kernel_stochastic(a, r, o, size, 2, true, -1.0f, 1.0f);

    // Print results
    for (int i = 0; i < size; ++i) {
        std::cout << o[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
