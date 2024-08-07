#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include "binary8_kernel.h"

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
                                        int man_bits, int exp_bits, int binades,
                                        bool saturate);

__global__ void binary8_signed_kernel_nearest(float *__restrict__ a, float *o, int size,
                                            int P, OverflowPolicy overflow_policy, bool subnormals);

__global__ void binary8_unsigned_kernel_nearest(float *__restrict__ a, float *o, int size,
                                              int P, OverflowPolicy overflow_policy, bool subnormals);

__global__ void binary8_signed_kernel_stochastic(float *__restrict__ a, int *__restrict__ r, float *o, int size,
                                               int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals);

__global__ void binary8_unsigned_kernel_stochastic(float *__restrict__ a, int *__restrict__ r, float *o, int size,
                                                 int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals);

__global__ void binary8_signed_kernel_truncate(float *__restrict__ a, float *o, int size,
                                               int P, OverflowPolicy overflow_policy, bool subnormals);

__global__ void binary8_unsigned_kernel_truncate(float *__restrict__ a, float *o, int size,
                                                 int P, OverflowPolicy overflow_policy, bool subnormals);

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
                   int binades_add, int binades_mul, bool saturate);

void bmm_superfp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    int binades_add, int binades_mul, bool saturate);

void mm_superfp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                       int man_fma, int exp_fma, int binades_fma, bool saturate);

void bmm_superfp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma, int binades_fma, bool saturate);

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

struct DimSizes
{
    int outer;
    int inner;
    int channel;
};

void layernorm_forward_fp_nearest(float *input, float *weight, float *bias,
                              float *output, float *mean, float *rstd,
                              float eps, const DimSizes& sizes,
                              int man_acc, int exp_acc,
                              int man_mul, int exp_mul,
                              int man_div, int exp_div,
                              int man_sqrt, int exp_sqrt,
                              bool subnormals, bool saturate);

void layernorm_backward_fp_nearest(float *input, float *grad_output,
                              float *weight, float *bias,
                              float *mean, float *rstd,
                              float *grad_input, float *grad_gamma, float *grad_beta, 
                              const DimSizes& sizes,
                              int man_acc, int exp_acc,
                              int man_mul, int exp_mul,
                              int man_div, int exp_div,
                              bool subnormals, bool saturate);
void softmax_forward_fp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_exp, int exp_exp,
                                int man_off, int exp_off,
                                int man_acc, int exp_acc,
                                bool subnormals, bool saturate);

void softmax_lse_forward_fp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_off, int exp_off,
                                int man_lse, int exp_lse,
                                bool subnormals, bool saturate);

void softmax_backward_fp_nearest(float *a, float *g, float *o,
                                const DimSizes& sizes,
                                int man_add, int exp_add,
                                int man_mul, int exp_mul,
                                bool subnormals, bool saturate);

void softmax_forward_superfp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_exp, int exp_exp, int binades_exp,
                                int man_off, int exp_off, int binades_off,
                                int man_acc, int exp_acc, int binades_acc,
                                bool saturate);

void softmax_lse_forward_superfp_nearest(float *a, float *o,
                                const DimSizes& sizes,
                                int man_off, int exp_off, int binades_off,
                                int man_lse, int exp_lse, int binades_lse,
                                bool saturate);

void softmax_backward_superfp_nearest(float *a, float *g, float *o,
                                const DimSizes& sizes,
                                int man_add, int exp_add, int binades_add,
                                int man_mul, int exp_mul, int binades_mul,
                                bool saturate);

void softmax_forward_binary8_nearest(float* a, float* o,
                                const DimSizes& sizes,
                                int P_exp, OverflowPolicy op_exp, bool signed_exp,
                                int P_off, OverflowPolicy op_off, bool signed_off,
                                int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                bool subnormals);

void softmax_lse_forward_binary8_nearest(float* a, float* o,
                                const DimSizes& sizes,
                                int P_off, OverflowPolicy op_off, bool signed_off,
                                int P_lse, OverflowPolicy op_lse, bool signed_lse,
                                bool subnormals);

void softmax_backward_binary8_nearest(float* a, float* g, float* o,
                                const DimSizes& sizes,
                                int P_add, OverflowPolicy op_add, bool signed_add,
                                int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                bool subnormals);
