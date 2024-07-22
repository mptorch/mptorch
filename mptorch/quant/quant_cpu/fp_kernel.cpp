#include "bit_helper.cpp"
#include "quant_kernel.h"
#include <cmath>
#include <cstdint>
#include <random>

inline float cast_fp_nearest(float origin_float, int man_bits, int exp_bits, bool subnormal_support = true, bool saturate = false) {
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

    if (noquantize) {
        quantized = origin_float;
    } else {
        if (subnormal && subnormal_support) {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest(target, exp_diff);
            quantize_bits = clip_exponent_with_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        } else if (target_exp == 128) {
            quantized = origin_float;
        } else {
            quantize_bits = round_bitwise_nearest(target, man_bits);
            quantize_bits = clip_exponent_without_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

inline float cast_fp_stochastic(float origin_float, uint32_t rand_prob, int man_bits, int exp_bits, bool subnormal_support = true, bool saturate = false) {
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);

    if (subnormal && subnormal_support) {
        float shift_float, val;
        int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
        shift_float = BITS_TO_FLOAT(&shift_bits);
        val = origin_float + shift_float;
        target = FLOAT_TO_BITS(&val);
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    } else {
        quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
        quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
        quantized = BITS_TO_FLOAT(&quantize_bits);
    }

    return quantized;
}

// Function to initialize random number generator (replace curandState_t with std::mt19937)
void seed_init(std::mt19937 &gen, uint32_t seed) {
    gen.seed(seed);
}

// Function to quantize a float into a floating point with [exp_bits] exponent and [man_bits] mantissa using stochastic rounding
void float_kernel_stochastic(float *__restrict__ a, int *__restrict__ r, float *o, int size, int man_bits, int exp_bits, bool subnormal_support, bool saturate) {
    for (int index = 0; index < size; ++index) {
        o[index] = cast_fp_stochastic(a[index], (uint32_t)r[index], man_bits, exp_bits, subnormal_support, saturate);
    }
}

// Function to quantize a float into a floating point with [exp_bits] exponent and [man_bits] mantissa using nearest rounding
void float_kernel_nearest(float *__restrict__ a, float *o, int size, int man_bits, int exp_bits, bool subnormal_support, bool saturate) {
    for (int index = 0; index < size; ++index) {
        o[index] = cast_fp_nearest(a[index], man_bits, exp_bits, subnormal_support, saturate);
    }
}

// Matrix multiplication with nearest rounding
template <size_t SHMEM_SIZE>
void mm_fp_nearest_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int M, int K, int N, int man_add, int exp_add, int man_mul, int exp_mul, bool subnormals, bool saturate) {
    std::vector<float> s_a(SHMEM_SIZE);
    std::vector<float> s_b(SHMEM_SIZE);

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float tmp = 0.0f;

            for (int i = 0; i < K; ++i) {
                float a_val = (row < M && i < K) ? a[row * K + i] : 0.0f;
                float b_val = (col < N && i < K) ? b[i * N + col] : 0.0f;
                tmp += cast_fp_nearest(a_val * b_val, man_mul, exp_mul, subnormals, saturate);
            }
            c[row * N + col] = cast_fp_nearest(tmp, man_add, exp_add, subnormals, saturate);
        }
    }
}

// Batched matrix multiplication with nearest rounding
template <size_t SHMEM_SIZE>
void bmm_fp_nearest_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int B, int M, int K, int N, int man_add, int exp_add, int man_mul, int exp_mul, bool subnormals, bool saturate) {
    std::vector<float> s_a(SHMEM_SIZE);
    std::vector<float> s_b(SHMEM_SIZE);

    for (int batch_idx = 0; batch_idx < B; ++batch_idx) {
        int batch_a = batch_idx * M * K;
        int batch_b = batch_idx * K * N;
        int batch_c = batch_idx * M * N;

        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                float tmp = 0.0f;

                for (int i = 0; i < K; ++i) {
                    float a_val = (row < M && i < K) ? a[batch_a + row * K + i] : 0.0f;
                    float b_val = (col < N && i < K) ? b[batch_b + i * N + col] : 0.0f;
                    tmp += cast_fp_nearest(a_val * b_val, man_mul, exp_mul, subnormals, saturate);
                }
                c[batch_c + row * N + col] = cast_fp_nearest(tmp, man_add, exp_add, subnormals, saturate);
            }
        }
    }
}

// Matrix multiplication with FMA and nearest rounding
template <size_t SHMEM_SIZE>
void mm_fp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int M, int K, int N, int man_fma, int exp_fma, bool subnormal_support, bool saturate) {
    std::vector<float> s_a(SHMEM_SIZE);
    std::vector<float> s_b(SHMEM_SIZE);

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float tmp = 0.0f;

            for (int i = 0; i < K; ++i) {
                tmp = cast_fp_nearest(std::fmaf(a[row * K + i], b[i * N + col], tmp), man_fma, exp_fma, subnormal_support, saturate);
            }
            c[row * N + col] = tmp;
        }
    }
}

// Batched matrix multiplication with FMA and nearest rounding
template <size_t SHMEM_SIZE>
void bmm_fp_fma_nearest_impl(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, int B, int M, int K, int N, int man_fma, int exp_fma, bool subnormal_support, bool saturate) {
    std::vector<float> s_a(SHMEM_SIZE);
    std::vector<float> s_b(SHMEM_SIZE);

    for (int batch_idx = 0; batch_idx < B; ++batch_idx) {
        int batch_a = batch_idx * M * K;
        int batch_b = batch_idx * K * N;
        int batch_c = batch_idx * M * N;

        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                float tmp = 0.0f;

                for (int i = 0; i < K; ++i) {
                    tmp = cast_fp_nearest(std::fmaf(a[batch_a + row * K + i], b[batch_b + i * N + col], tmp), man_fma, exp_fma, subnormal_support, saturate);
                }
                c[batch_c + row * N + col] = tmp;
            }
        }
    }
}

// Function to perform matrix multiplication using nearest rounding
void mm_fp_nearest(float *a, float *b, float *c, int M, int K, int N, int man_add, int exp_add, int man_mul, int exp_mul, bool subnormals, bool saturate) {
    constexpr size_t SHMEM_SIZE = 1024;
    mm_fp_nearest_impl<SHMEM_SIZE>(a, b, c, M, K, N, man_add, exp_add, man_mul, exp_mul, subnormals, saturate);
}

// Function to perform batched matrix multiplication using nearest rounding
void bmm_fp_nearest(float *a, float *b, float *c, int B, int M, int K, int N, int man_add, int exp_add, int man_mul, int exp_mul, bool subnormals, bool saturate) {
    constexpr size_t SHMEM_SIZE = 1024;
    bmm_fp_nearest_impl<SHMEM_SIZE>(a, b, c, B, M, K, N, man_add, exp_add, man_mul, exp_mul, subnormals, saturate);
}

// Function to perform matrix multiplication with FMA and nearest rounding
void mm_fp_fma_nearest(float *a, float *b, float *c, int M, int K, int N, int man_fma, int exp_fma, bool subnormals, bool saturate) {
    constexpr size_t SHMEM_SIZE = 1024;
    mm_fp_fma_nearest_impl<SHMEM_SIZE>(a, b, c, M, K, N, man_fma, exp_fma, subnormals, saturate);
}

// Function to perform batched matrix multiplication with FMA and nearest rounding
void bmm_fp_fma_nearest(float *a, float *b, float *c, int B, int M, int K, int N, int man_fma, int exp_fma, bool subnormals, bool saturate) {
    constexpr size_t SHMEM_SIZE = 1024;
    bmm_fp_fma_nearest_impl<SHMEM_SIZE>(a, b, c, B, M, K, N, man_fma, exp_fma, subnormals, saturate);
}

