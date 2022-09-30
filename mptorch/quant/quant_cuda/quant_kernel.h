#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>

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

__global__ void block_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        float *max_entry, int man_bits);

__global__ void block_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     float *max_entry, int man_bits);

__global__ void block_kernel_sim_stochastic(float *__restrict__ a,
                                            float *__restrict__ r, float *o,
                                            int size, float *max_entry, int wl);

__global__ void block_kernel_sim_nearest(float *__restrict__ a, float *o,
                                         int size, float *max_entry, int wl);

__global__ void gemm_fp_nearest(float *__restrict__ a, float *__restrict__ b,
                                float *__restrict__ c, int M, int K, int N,
                                int man_add, int exp_add, int man_mul,
                                int exp_mul, bool subnormals, bool saturate);

__global__ void gemm_fp_fma_nearest(float *__restrict__ a,
                                    float *__restrict__ b,
                                    float *__restrict__ c, int M, int K, int N,
                                    int man_fma, int exp_fma, bool subnormals,
                                    bool saturate);

__global__ void gemm_fp_stochastic(float *__restrict__ a, float *__restrict__ b,
                                   float *__restrict__ c,
                                   curandState_t *state, // int *__restrict__ r,
                                   int M, int K, int N, int man_add,
                                   int exp_add, int man_mul, int exp_mul,
                                   bool subnormals, bool saturate);

__global__ void
gemm_fp_fma_stochastic(float *__restrict__ a, float *__restrict__ b,
                       float *__restrict__ c,
                       curandState_t *state, // int *__restrict__ r,
                       int M, int K, int N, int man_fma, int exp_fma,
                       bool subnormals, bool saturate);

__global__ void gemm_fxp_nearest(float *__restrict__ a, float *__restrict__ b,
                                 float *__restrict__ c, int M, int K, int N,
                                 int sigma_add, int t_min_add, int t_max_add,
                                 int sigma_mul, int t_min_mul, int t_max_mul);

__global__ void gemm_fxp_fma_nearest(float *__restrict__ a,
                                     float *__restrict__ b,
                                     float *__restrict__ c, int M, int K, int N,
                                     int sigma_fma, int t_min_fma,
                                     int t_max_fma);

__global__ void
gemm_fxp_stochastic(float *__restrict__ a, float *__restrict__ b,
                    float *__restrict__ c,
                    curandState_t *state, // float *__restrict__ r,
                    int M, int K, int N, int sigma_add, int t_min_add,
                    int t_max_add, int sigma_mul, int t_min_mul, int t_max_mul);

__global__ void gemm_fxp_fma_stochastic(
    float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
    curandState_t *state, // float *__restrict__ r,
    int M, int K, int N, int sigma_fma, int t_min_fma, int t_max_fma);