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
                                        int man_bits, int exp_bits);

__global__ void float_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     int man_bits, int exp_bits);

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

__global__ void gemm_fp_algo0(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits);

__global__ void gemm_fp_algo1(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits);

__global__ void gemm_fp_algo2(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits);

__global__ void gemm_fp_algo3(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits);

__global__ void gemm_fp_algo4(float *__restrict__ a, float *__restrict__ b,
                                 float *__restrict__ c, int M, int K, int N,
                                 int man_bits, int exp_bits);