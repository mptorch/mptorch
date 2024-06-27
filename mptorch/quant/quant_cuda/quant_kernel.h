#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include "p3109_kernel.h"

__global__ void seed_init(uint32_t seed, curandState_t *state);

__global__ void fixed_point_quantize_kernel_stochastic(
    float *__restrict__ a, float *__restrict__ r, float *o, int size, int sigma,
    bool clamp, float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_nearest(float *__restrict__ a,
                                                    float *o, int size,
                                                    int sigma, bool clamp,
                                                    float t_min, float t_max);

__global__ void fixed_point_quantize_kernel_mask_stochastic(
    float *__restrict__ a, float *__restrict__ r, float *o, uint8_t *mask,
    int size, int sigma, float t_min, float t_max);

__global__ void
fixed_point_quantize_kernel_mask_nearest(float *__restrict__ a, float *o,
                                         uint8_t *mask, int size, int sigma,
                                         float t_min, float t_max);

__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        bool subnormals, bool saturate);

__global__ void float_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     int man_bits, int exp_bits,
                                     bool subnormals, bool saturate);

__global__ void superfp_kernel_nearest(float *__restrict__ a, float *o, int size, 
                                        int man_bits, int exp_bits,
                                        bool saturate);

__global__ void p3109_signed_kernel_nearest(float *__restrict__ a, float *o, int size,
                                            int P, SaturateMode saturation_mode, bool subnormals);

__global__ void p3109_unsigned_kernel_nearest(float *__restrict__ a, float *o, int size,
                                              int P, SaturateMode saturation_mode, bool subnormals);

__global__ void p3109_signed_kernel_stochastic(float *__restrict__ a, int *__restrict__ r, float *o, int size,
                                               int P, int prng_bits, SaturateMode saturation_mode, bool subnormals);

__global__ void p3109_unsigned_kernel_stochastic(float *__restrict__ a, int *__restrict__ r, float *o, int size,
                                                 int P, int prng_bits, SaturateMode saturation_mode, bool subnormals);

__global__ void block_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        float *__restrict__ max_entry,
                                        int man_bits);

__global__ void block_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     float *__restrict__ max_entry,
                                     int man_bits);

__global__ void block_kernel_sim_stochastic(float *__restrict__ a,
                                            float *__restrict__ r, float *o,
                                            int size, float *max_entry, int wl);

__global__ void block_kernel_sim_nearest(float *__restrict__ a, float *o,
                                         int size, float *max_entry, int wl);

void mm_fp_nearest(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   bool subnormals, bool saturate);

void mm_fp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                       int man_fma, int exp_fma, bool subnormals,
                       bool saturate);

void mm_fp_stochastic(float *a, float *b, float *c, int M, int K, int N,
                      int man_add, int exp_add, int man_mul, int exp_mul,
                      bool subnormals, bool saturate);

void mm_fp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N,
                          int man_fma, int exp_fma, bool subnormals,
                          bool saturate);

void bmm_fp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    bool subnormals, bool saturate);

void bmm_fp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma, bool subnormals,
                        bool saturate);

void bmm_fp_stochastic(float *a, float *b, float *c, int B, int M, int K, int N,
                       int man_add, int exp_add, int man_mul, int exp_mul,
                       bool subnormals, bool saturate);

void bmm_fp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K,
                           int N, int man_fma, int exp_fma, bool subnormals,
                           bool saturate);

void mm_superfp_nearest(float *a, float *b, float *c, int M, int K, int N,
                   int man_add, int exp_add, int man_mul, int exp_mul,
                   bool saturate);

void bmm_superfp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    bool saturate);

void mm_superfp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                       int man_fma, int exp_fma,
                       bool saturate);

void bmm_superfp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma,
                        bool saturate);

void mm_fxp_nearest(float *a, float *b, float *c, int M, int K, int N,
                    int sigma_add, int t_min_add, int t_max_add, int sigma_mul,
                    int t_min_mul, int t_max_mul);

void mm_fxp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                        int sigma_fma, int t_min_fma, int t_max_fma);

void mm_fxp_stochastic(float *a, float *b, float *c, int M, int K, int N,
                       int sigma_add, int t_min_add, int t_max_add,
                       int sigma_mul, int t_min_mul, int t_max_mul);

void mm_fxp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N,
                           int sigma_fma, int t_min_fma, int t_max_fma);

void bmm_fxp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                     int sigma_add, int t_min_add, int t_max_add, int sigma_mul,
                     int t_min_mul, int t_max_mul);

void bmm_fxp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                         int N, int sigma_fma, int t_min_fma, int t_max_fma);

void bmm_fxp_stochastic(float *a, float *b, float *c, int B, int M, int K,
                        int N, int sigma_add, int t_min_add, int t_max_add,
                        int sigma_mul, int t_min_mul, int t_max_mul);

void bmm_fxp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K,
                            int N, int sigma_fma, int t_min_fma, int t_max_fma);