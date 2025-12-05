#pragma once

#include "modes.h"
#include <cstdint>
#include <curand_kernel.h>

__global__ void seed_init(curandState_t *state);

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

__global__ void float_kernel_nearest_even(float *__restrict__ a, float *o,
                                          int size, int man_bits, int exp_bits,
                                          bool saturate,
                                          SubnormalsMode subnormals);

__global__ void float_kernel_nearest_away(float *__restrict__ a, float *o,
                                          int size, int man_bits, int exp_bits,
                                          bool saturate,
                                          SubnormalsMode subnormals);

__global__ void float_kernel_up(float *__restrict__ a, float *o, int size,
                                int man_bits, int exp_bits, bool saturate,
                                SubnormalsMode subnormals);

__global__ void float_kernel_down(float *__restrict__ a, float *o, int size,
                                  int man_bits, int exp_bits, bool saturate,
                                  SubnormalsMode subnormals);

__global__ void float_kernel_zero(float *__restrict__ a, float *o, int size,
                                  int man_bits, int exp_bits, bool saturate,
                                  SubnormalsMode subnormals);

__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        bool saturate,
                                        SubnormalsMode subnormals);

__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        int prng_bits, bool saturate,
                                        SubnormalsMode subnormals);

__global__ void superfp_kernel_nearest(float *__restrict__ a, float *o,
                                       int size, int man_bits, int exp_bits,
                                       int binades_l, int binades_u,
                                       bool saturate);

void binaryK_kernel(float *__restrict__ a, float *o, int size,
                    int K, int P, int bias, bool is_signed,
                    RoundMode round_mode, SaturationMode saturation_mode,
                    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void binaryK_kernel(float *__restrict__ a,
                    int *__restrict__ r, float *o, int size,
                    int K, int P, int bias, int prng_bits, bool is_signed,
                    RoundMode round_mode, SaturationMode saturation_mode,
                    SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void fp_kernel(float *__restrict__ a, float *o, int size,
               int man_bits, int exp_bits, int bias,
               bool saturate,
               RoundMode round_mode,
               SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void fp_kernel(float *__restrict__ a, int *__restrict__ r, float *o, int size,
               int man_bits, int exp_bits, int bias, int prng_bits,
               bool saturate,
               RoundMode round_mode,
               SubnormalsMode subnormals = SubnormalsMode::SUBNORMALS);

void superfp_kernel(float *__restrict__ a, float *o, int size,
                    int man_bits, int exp_bits, int bias,
                    int binades_l, int binades_h,
                    bool saturate,
                    RoundMode round_mode);

void superfp_kernel(float *__restrict__ a, int *__restrict__ r, float *o, int size,
                    int man_bits, int exp_bits, int prng_bits, int bias,
                    int binades_l, int binades_h,
                    bool saturate,
                    RoundMode round_mode);

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
                   bool saturate, SubnormalsMode subnormals, bool compensated);

void mm_fp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                       int man_fma, int exp_fma, bool saturate,
                       SubnormalsMode subnormals, bool compensated);

void mm_fp_stochastic(float *a, float *b, float *c, int M, int K, int N,
                      int man_add, int exp_add, int rbits_add, int man_mul,
                      int exp_mul, int rbits_mul, bool saturate,
                      SubnormalsMode subnormals);

void mm_fp_fma_stochastic(float *a, float *b, float *c, int M, int K, int N,
                          int man_fma, int exp_fma, int rbits_fma,
                          bool saturate, SubnormalsMode subnormals);

void bmm_fp_nearest(float *a, float *b, float *c, int B, int M, int K, int N,
                    int man_add, int exp_add, int man_mul, int exp_mul,
                    bool saturate, SubnormalsMode subnormals, bool compensated);

void bmm_fp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                        int N, int man_fma, int exp_fma, bool saturate,
                        SubnormalsMode subnormals, bool compensated);

void bmm_fp_stochastic(float *a, float *b, float *c, int B, int M, int K, int N,
                       int man_add, int exp_add, int rbits_add, int man_mul,
                       int exp_mul, int rbits_mul, bool saturate,
                       SubnormalsMode subnormals);

void bmm_fp_fma_stochastic(float *a, float *b, float *c, int B, int M, int K,
                           int N, int man_fma, int exp_fma, int rbits_fma,
                           bool saturate, SubnormalsMode subnormals);

void mm_superfp_nearest(float *a, float *b, float *c, int M, int K, int N,
                        int man_add, int exp_add, int man_mul, int exp_mul,
                        int binades_add_l, int binades_add_u, int binades_mul_l,
                        int binades_mul_u, bool saturate);

void bmm_superfp_nearest(float *a, float *b, float *c, int B, int M, int K,
                         int N, int man_add, int exp_add, int man_mul,
                         int exp_mul, int binades_add_l, int binades_add_u,
                         int binades_mul_l, int binades_mul_u, bool saturate);

void mm_superfp_fma_nearest(float *a, float *b, float *c, int M, int K, int N,
                            int man_fma, int exp_fma, int binades_fma_l,
                            int binades_fma_u, bool saturate);

void bmm_superfp_fma_nearest(float *a, float *b, float *c, int B, int M, int K,
                             int N, int man_fma, int exp_fma, int binades_fma_l,
                             int binades_fma_u, bool saturate);

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
                                  float eps, const DimSizes &sizes, int man_acc,
                                  int exp_acc, int man_mul, int exp_mul,
                                  int man_div, int exp_div, int man_sqrt,
                                  int exp_sqrt, bool saturate,
                                  SubnormalsMode subnormals);

void layernorm_backward_fp_nearest(
    float *input, float *grad_output, float *weight, float *bias, float *mean,
    float *rstd, float *grad_input, float *grad_gamma, float *grad_beta,
    const DimSizes &sizes, int man_acc, int exp_acc, int man_mul, int exp_mul,
    int man_div, int exp_div, bool saturate, SubnormalsMode subnormals);

void layernorm_forward_superfp_nearest(
    float *input, float *weight, float *bias, float *output, float *mean,
    float *rstd, float eps, const DimSizes &sizes, int man_acc, int exp_acc,
    int binades_acc_l, int binades_acc_u, int man_mul, int exp_mul,
    int binades_mul_l, int binades_mul_u, int man_div, int exp_div,
    int binades_div_l, int binades_div_u, int man_sqrt, int exp_sqrt,
    int binades_sqrt_l, int binades_sqrt_u, bool saturate);

void layernorm_backward_superfp_nearest(
    float *input, float *grad_output, float *weight, float *bias, float *mean,
    float *rstd, float *grad_input, float *grad_gamma, float *grad_beta,
    const DimSizes &sizes, int man_acc, int exp_acc, int binades_acc_l,
    int binades_acc_u, int man_mul, int exp_mul, int binades_mul_l,
    int binades_mul_u, int man_div, int exp_div, int binades_div_l,
    int binades_div_u, bool saturate);

void softmax_forward_fp_nearest(float *a, float *o, const DimSizes &sizes,
                                int man_exp, int exp_exp, int man_off,
                                int exp_off, int man_acc, int exp_acc,
                                bool saturate, SubnormalsMode subnormals);

void softmax_lse_forward_fp_nearest(float *a, float *o, const DimSizes &sizes,
                                    int man_off, int exp_off, int man_lse,
                                    int exp_lse, bool saturate,
                                    SubnormalsMode subnormals);

void softmax_backward_fp_nearest(float *a, float *g, float *o,
                                 const DimSizes &sizes, int man_add,
                                 int exp_add, int man_mul, int exp_mul,
                                 bool saturate, SubnormalsMode subnormals);

void softmax_forward_superfp_nearest(
    float *a, float *o, const DimSizes &sizes, int man_exp, int exp_exp,
    int binades_exp_l, int binades_exp_u, int man_off, int exp_off,
    int binades_off_l, int binades_off_u, int man_acc, int exp_acc,
    int binades_acc_l, int binades_acc_u, bool saturate);

void softmax_lse_forward_superfp_nearest(float *a, float *o,
                                         const DimSizes &sizes, int man_off,
                                         int exp_off, int binades_off_l,
                                         int binades_off_u, int man_lse,
                                         int exp_lse, int binades_lse_l,
                                         int binades_lse_u, bool saturate);

void softmax_backward_superfp_nearest(float *a, float *g, float *o,
                                      const DimSizes &sizes, int man_add,
                                      int exp_add, int binades_add_l,
                                      int binades_add_u, int man_mul,
                                      int exp_mul, int binades_mul_l,
                                      int binades_mul_u, bool saturate);