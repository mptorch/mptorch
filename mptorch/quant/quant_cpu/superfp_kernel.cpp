#include "bit_helper.cpp"
#include "quant_kernel.h"
#include <cmath>
#include <cstdint>

// Helper function to replace __device__
inline float cast_superfp_nearest(float origin, int man_bits, int exp_bits, int binades = 1, bool saturate = false) {
    int32_t sat = saturate;
    uint32_t target;
    target = FLOAT_TO_BITS(&origin);
    float ftarget{0u};

    int32_t target_exp = (target << 1 >> 24) - 127;
    int32_t min_exp = 1 - ((1 << (exp_bits - 1))) + (binades - 1);
    int32_t max_exp = ((1 << (exp_bits - 1)) - 2) - (binades - 1);
    bool subnormal = (target_exp < min_exp);
    bool supnormal = (target_exp > max_exp);
    if(subnormal) {
        if (target_exp < min_exp - binades * (1 << man_bits) + 1) // underflow
            return 0.0f;
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    } else if (supnormal) {
        if (target_exp == 128) { // NaN/inf
            if (saturate) {
                if ((target & 0x7FFFFFFF) == 0x7F800000) { // inf
                    uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                    return BITS_TO_FLOAT(&qtarget);
                } else { // NaN
                    return origin;
                }
            } else {
                return origin;
            }
        } else if (target_exp >= max_exp + binades * (1 << man_bits) - 1 + sat) { // overflow
            if (saturate) {
                uint32_t qtarget = (target >> 31 << 31) | ((max_exp + binades * (1 << man_bits) + 127u) << 23);
                return BITS_TO_FLOAT(&qtarget);
            } else {
                if (((target << 9) == 0u) && (target_exp == max_exp + binades * (1 << man_bits) - 1))
                    return origin;
                else {
                    float infty = INFINITY;
                    uint32_t qtarget = (target >> 31 << 31) | FLOAT_TO_BITS(&infty);
                    return BITS_TO_FLOAT(&qtarget);
                }
            }
        }
        uint32_t mask = (1 << 23) - 1;
        uint32_t eps = 1 << 22;
        uint32_t add_r = target + eps;
        uint32_t qtarget = add_r & ~mask;
        ftarget = BITS_TO_FLOAT(&qtarget);
    } else {
        uint32_t qtarget = round_bitwise_nearest(target, man_bits);
        ftarget = BITS_TO_FLOAT(&qtarget);
    }

    return ftarget;
}

void superfp_kernel_nearest(float *a, float *o, int size, 
                            int man_bits, int exp_bits, int binades,
                            bool saturate) {
    for (int index = 0; index < size; ++index) {
        o[index] = cast_superfp_nearest(a[index], man_bits, exp_bits, binades, saturate);
    }
}

void mm_superfp_nearest(float *a, float *b, float *c, int M, int K, int N,
                        int man_add, int exp_add, int man_mul, int exp_mul,
                        int binades_add, int binades_mul, bool saturate) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k) {
                tmp = cast_superfp_nearest(tmp + cast_superfp_nearest(a[row * K + k] * b[k * N + col],
                                           man_mul, exp_mul, binades_mul, saturate),
                                           man_add, exp_add, binades_add, saturate);
            }
            c[row * N + col] = tmp;
        }
    }
}

void bmm_superfp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                         int man_add, int exp_add, int man_mul, int exp_mul,
                         int binades_add, int binades_mul, bool saturate) {
    for (int batch = 0; batch < B; ++batch) {
        int batch_a = batch * M * K;
        int batch_b = batch * K * N;
        int batch_c = batch * M * N;
        
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                float tmp = 0.0f;
                for (int k = 0; k < K; ++k) {
                    tmp = cast_superfp_nearest(tmp + cast_superfp_nearest(a[batch_a + row * K + k] * b[batch_b + k * N + col],
                                               man_mul, exp_mul, binades_mul, saturate),
                                               man_add, exp_add, binades_add, saturate);
                }
                c[batch_c + row * N + col] = tmp;
            }
        }
    }
}

void mm_superfp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                            int man_fma, int exp_fma, int binades_fma, bool saturate) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float tmp = 0.0f;
            for (int k = 0; k < K; ++k) {
                tmp = cast_superfp_nearest(std::fmaf(a[row * K + k], b[k * N + col], tmp),
                                           man_fma, exp_fma, binades_fma, saturate);
            }
            c[row * N + col] = tmp;
        }
    }
}

void bmm_superfp_fma_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                             int man_fma, int exp_fma, int binades_fma, bool saturate) {
    for (int batch = 0; batch < B; ++batch) {
        int batch_a = batch * M * K;
        int batch_b = batch * K * N;
        int batch_c = batch * M * N;
        
        for (int row = 0; row < M; ++row) {
            for (int col = 0; col < N; ++col) {
                float tmp = 0.0f;
                for (int k = 0; k < K; ++k) {
                    tmp = cast_superfp_nearest(std::fmaf(a[batch_a + row * K + k], b[batch_b + k * N + col], tmp),
                                               man_fma, exp_fma, binades_fma, saturate);
                }
                c[batch_c + row * N + col] = tmp;
            }
        }
    }
}