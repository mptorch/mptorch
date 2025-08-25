#include "quant.h"
#include <torch/torch.h>
#include <tuple>
#include "binary8_kernel.h"

using namespace at;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
      TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
      CHECK_CUDA(x);   \
      CHECK_CONTIGUOUS(x)

Tensor fixed_point_quantize_nearest(Tensor a,
                                    int wl, int fl,
                                    bool use_clamp, bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_nearest_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask(Tensor a,
                                  int wl, int fl,
                                  bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_nearest_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_nearest(Tensor a, int wl, int dim)
{
      CHECK_INPUT(a);
      return block_quantize_nearest_cuda(a, wl, dim);
}

Tensor block_quantize_sim_nearest(Tensor a, int wl)
{
      CHECK_INPUT(a);
      return block_quantize_sim_nearest_cuda(a, wl);
}

Tensor float_quantize_nearest(Tensor a,
                              int man_bits, int exp_bits,
                              bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      return float_quantize_nearest_cuda(a, man_bits, exp_bits, subnormals,
                                         saturate);
}

Tensor float_quantize_nearest_mp(Tensor a, Tensor s, Tensor mans, Tensor exps, bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(s);
      CHECK_INPUT(mans);
      CHECK_INPUT(exps);
      return float_quantize_nearest_mp_cuda(a, s, mans, exps, subnormals, saturate);
}

Tensor superfp_quantize_nearest(Tensor a,
                                int man_bits, int exp_bits, int binades_l, int binades_u,
                                bool saturate)
{
      CHECK_INPUT(a);
      return superfp_quantize_nearest_cuda(a, man_bits, exp_bits, binades_l, binades_u, saturate);
}

Tensor binary8_quantize_nearest(Tensor a,
                                int P, bool is_signed, OverflowPolicy overflow_policy,
                                bool subnormals)
{
      CHECK_INPUT(a);
      return binary8_quantize_nearest_cuda(a, P, is_signed, overflow_policy, subnormals);
}

Tensor fixed_point_quantize_stochastic(Tensor a,
                                       int wl, int fl,
                                       bool use_clamp, bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_stochastic_cuda(a, wl, fl, use_clamp, symmetric);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask(Tensor a,
                                     int wl, int fl,
                                     bool symmetric)
{
      CHECK_INPUT(a);
      return fixed_point_quantize_stochastic_mask_cuda(a, wl, fl, symmetric);
}

Tensor block_quantize_stochastic(Tensor a, int wl, int dim)
{
      CHECK_INPUT(a);
      return block_quantize_stochastic_cuda(a, wl, dim);
}

Tensor block_quantize_sim_stochastic(Tensor a, int wl)
{
      CHECK_INPUT(a);
      return block_quantize_sim_stochastic_cuda(a, wl);
}

Tensor float_quantize_stochastic(Tensor a,
                                 int man_bits, int exp_bits, int prng_bits,
                                 bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      return float_quantize_stochastic_cuda(a, man_bits, exp_bits, prng_bits, subnormals,
                                            saturate);
}

Tensor binary8_quantize_stochastic(Tensor a,
                                   int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy,
                                   bool subnormals)
{
      CHECK_INPUT(a);
      return binary8_quantize_stochastic_cuda(a, P, prng_bits, is_signed, overflow_policy, subnormals);
}

Tensor binary8_quantize_truncate(Tensor a,
                                 int P, bool is_signed, OverflowPolicy overflow_policy,
                                 bool subnormals)
{
      CHECK_INPUT(a);
      return binary8_quantize_truncate_cuda(a, P, is_signed, overflow_policy, subnormals);
}

void float_quantize_nearest_mm(Tensor a, Tensor b, Tensor c,
                               int M, int N, int K,
                               int man_add, int exp_add,
                               int man_mul, int exp_mul,
                               bool subnormals,
                               bool saturate,
                               bool compensated)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_mm_cuda(a, b, c, M, N, K,
                                     man_add, exp_add, man_mul, exp_mul,
                                     subnormals, saturate, compensated);
}

void float_quantize_nearest_mm_mp(Tensor a, Tensor b, Tensor c, Tensor s,
                                  Tensor mans, Tensor exps,
                                  int M, int N, int K,
                                  bool subnormals,
                                  bool saturate,
                                  bool compensated)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      CHECK_INPUT(s);
      CHECK_INPUT(mans);
      CHECK_INPUT(exps);
      float_quantize_nearest_mm_cuda(a, b, c, s, mans, exps, M, N, K,
                                     subnormals, saturate, compensated);
}

void float_quantize_nearest_bmm(Tensor a, Tensor b, Tensor c,
                                int M, int N, int K,
                                int man_add, int exp_add,
                                int man_mul, int exp_mul,
                                bool subnormals,
                                bool saturate,
                                bool compensated)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_bmm_cuda(a, b, c, M, N, K,
                                      man_add, exp_add, man_mul, exp_mul,
                                      subnormals, saturate, compensated);
      return;
}

void float_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c,
                                   int M, int N, int K,
                                   int man_fma, int exp_fma,
                                   bool subnormals,
                                   bool saturate,
                                   bool compensated)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K,
                                         man_fma, exp_fma,
                                         subnormals, saturate, compensated);
      return;
}

void float_quantize_nearest_mm_fma_mp(Tensor a, Tensor b, Tensor c, Tensor s,
                                      Tensor mans, Tensor exps,
                                      int M, int N, int K,
                                      bool subnormals,
                                      bool saturate,
                                      bool compensated)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      CHECK_INPUT(s);
      CHECK_INPUT(mans);
      CHECK_INPUT(exps);
      float_quantize_nearest_mm_fma_cuda(a, b, c, s, mans, exps, M, N, K,
                                         subnormals, saturate, compensated);
}

void float_quantize_nearest_bmm_fma(Tensor a, Tensor b, Tensor c,
                                    int M, int N, int K,
                                    int man_fma, int exp_fma,
                                    bool subnormals,
                                    bool saturate,
                                    bool compensated)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_nearest_bmm_fma_cuda(a, b, c, M, N, K,
                                          man_fma, exp_fma,
                                          subnormals, saturate, compensated);
      return;
}

void superfp_quantize_nearest_mm(Tensor a, Tensor b, Tensor c,
                                 int M, int N, int K,
                                 int man_add, int exp_add,
                                 int man_mul, int exp_mul,
                                 int binades_add_l, int binades_add_u,
                                 int binades_mul_l, int binades_mul_u,
                                 bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_mm_cuda(a, b, c, M, N, K, man_add, exp_add, man_mul, exp_mul,
                                       binades_add_l, binades_add_u,
                                       binades_mul_l, binades_mul_u,
                                       saturate);
      return;
}

void superfp_quantize_nearest_bmm(Tensor a, Tensor b, Tensor c,
                                  int M, int N, int K,
                                  int man_add, int exp_add,
                                  int man_mul, int exp_mul,
                                  int binades_add_l, int binades_add_u,
                                  int binades_mul_l, int binades_mul_u,
                                  bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_bmm_cuda(a, b, c, M, N, K, man_add, exp_add, man_mul, exp_mul,
                                        binades_add_l, binades_add_u,
                                        binades_mul_l, binades_mul_u, saturate);
      return;
}

void superfp_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c,
                                     int M, int N, int K,
                                     int man_fma, int exp_fma,
                                     int binades_fma_l, int binades_fma_u,
                                     bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                           binades_fma_l, binades_fma_u, saturate);
      return;
}

void superfp_quantize_nearest_bmm_fma(Tensor a, Tensor b, Tensor c,
                                      int M, int N, int K,
                                      int man_fma, int exp_fma,
                                      int binades_fma_l, int binades_fma_u,
                                      bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      superfp_quantize_nearest_bmm_fma_cuda(a, b, c, M, N, K,
                                            man_fma, exp_fma,
                                            binades_fma_l, binades_fma_u,
                                            saturate);
      return;
}

void float_quantize_stochastic_mm(Tensor a, Tensor b, Tensor c,
                                  int M, int N, int K,
                                  int man_add, int exp_add, int rbits_add,
                                  int man_mul, int exp_mul, int rbits_mul,
                                  bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_mm_cuda(a, b, c, M, N, K,
                                        man_add, exp_add, rbits_add,
                                        man_mul, exp_mul, rbits_mul,
                                        subnormals, saturate);
      return;
}

void float_quantize_stochastic_bmm(Tensor a, Tensor b, Tensor c,
                                   int M, int N, int K,
                                   int man_add, int exp_add, int rbits_add,
                                   int man_mul, int exp_mul, int rbits_mul,
                                   bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_bmm_cuda(a, b, c, M, N, K,
                                         man_add, exp_add, rbits_add,
                                         man_mul, exp_mul, rbits_mul,
                                         subnormals, saturate);
      return;
}

void float_quantize_stochastic_mm_fma(Tensor a, Tensor b, Tensor c,
                                      int M, int N, int K,
                                      int man_fma, int exp_fma, int rbits_fma,
                                      bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_mm_fma_cuda(a, b, c, M, N, K,
                                            man_fma, exp_fma, rbits_fma,
                                            subnormals, saturate);
      return;
}

void float_quantize_stochastic_bmm_fma(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_fma, int exp_fma, int rbits_fma,
                                       bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_quantize_stochastic_bmm_fma_cuda(a, b, c, M, N, K,
                                             man_fma, exp_fma, rbits_fma,
                                             subnormals, saturate);
      return;
}

void fixed_point_quantize_nearest_mm(Tensor a, Tensor b, Tensor c, int M, int N,
                                     int K, int wl_add, int fl_add, int wl_mul,
                                     int fl_mul, bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_nearest_mm_cuda(a, b, c, M, N, K, wl_add, fl_add, wl_mul,
                                           fl_mul, symmetric);
      return;
}

void fixed_point_quantize_nearest_mm_fma(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int wl_fma, int fl_fma,
                                         bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                               symmetric);
      return;
}

void fixed_point_quantize_nearest_bmm(Tensor a, Tensor b, Tensor c, int M, int N,
                                      int K, int wl_add, int fl_add, int wl_mul,
                                      int fl_mul, bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_nearest_bmm_cuda(a, b, c, M, N, K, wl_add, fl_add, wl_mul,
                                            fl_mul, symmetric);
      return;
}

void fixed_point_quantize_nearest_bmm_fma(Tensor a, Tensor b, Tensor c, int M,
                                          int N, int K, int wl_fma, int fl_fma,
                                          bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_nearest_bmm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                                symmetric);
      return;
}

void fixed_point_quantize_stochastic_mm(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int wl_add, int fl_add,
                                        int wl_mul, int fl_mul,
                                        bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_stochastic_mm_cuda(a, b, c, M, N, K, wl_add, fl_add,
                                              wl_mul, fl_mul, symmetric);
      return;
}

void fixed_point_quantize_stochastic_mm_fma(Tensor a, Tensor b, Tensor c, int M,
                                            int N, int K, int wl_fma,
                                            int fl_fma, bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_stochastic_mm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                                  symmetric);
      return;
}

void fixed_point_quantize_stochastic_bmm(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int wl_add, int fl_add,
                                         int wl_mul, int fl_mul,
                                         bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_stochastic_bmm_cuda(a, b, c, M, N, K, wl_add, fl_add,
                                               wl_mul, fl_mul, symmetric);
      return;
}

void fixed_point_quantize_stochastic_bmm_fma(Tensor a, Tensor b, Tensor c, int M,
                                             int N, int K, int wl_fma,
                                             int fl_fma, bool symmetric)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      fixed_point_quantize_stochastic_bmm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                                   symmetric);
      return;
}

void floating_point_mm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                              CUBLASMatrixType AB_type, CUBLASMatrixType C_type,
                              CUBLASComputeType compute_type, bool pedantic)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_mm_cublas(a, b, c, M, N, K, AB_type, C_type, compute_type, pedantic);
      return;
}

void floating_point_bmm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                               CUBLASMatrixType AB_type, CUBLASMatrixType C_type,
                               CUBLASComputeType compute_type, bool pedantic)
{
      CHECK_INPUT(a);
      CHECK_INPUT(b);
      CHECK_INPUT(c);
      float_bmm_cublas(a, b, c, M, N, K, AB_type, C_type, compute_type, pedantic);
      return;
}

void float_quantize_layernorm_forward(Tensor input, Tensor weight, Tensor bias,
                                      Tensor output, Tensor mean, Tensor rstd,
                                      float eps, std::vector<int> &dims,
                                      int man_acc, int exp_acc,
                                      int man_mul, int exp_mul,
                                      int man_div, int exp_div,
                                      int man_sqrt, int exp_sqrt,
                                      bool subnormals, bool saturate)
{
      CHECK_INPUT(input);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      CHECK_INPUT(output);
      CHECK_INPUT(mean);
      CHECK_INPUT(rstd);
      float_quantize_nearest_layernorm_forward_cuda(input, weight, bias,
                                                    output, mean, rstd,
                                                    eps, dims,
                                                    man_acc, exp_acc,
                                                    man_mul, exp_mul,
                                                    man_div, exp_div,
                                                    man_sqrt, exp_sqrt,
                                                    subnormals, saturate);
}

void float_quantize_layernorm_backward(Tensor input, Tensor grad_output,
                                       Tensor weight, Tensor bias,
                                       Tensor mean, Tensor rstd,
                                       Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                       std::vector<int> &dims,
                                       int man_acc, int exp_acc,
                                       int man_mul, int exp_mul,
                                       int man_div, int exp_div,
                                       bool subnormals, bool saturate)
{
      CHECK_INPUT(input);
      CHECK_INPUT(grad_output);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      CHECK_INPUT(mean);
      CHECK_INPUT(rstd);
      CHECK_INPUT(grad_input);
      CHECK_INPUT(grad_gamma);
      CHECK_INPUT(grad_beta);
      float_quantize_nearest_layernorm_backward_cuda(input, grad_output,
                                                     weight, bias,
                                                     mean, rstd,
                                                     grad_input, grad_gamma, grad_beta,
                                                     dims,
                                                     man_acc, exp_acc,
                                                     man_mul, exp_mul,
                                                     man_div, exp_div,
                                                     subnormals, saturate);
}

void superfp_quantize_layernorm_forward(Tensor input, Tensor weight, Tensor bias,
                                        Tensor output, Tensor mean, Tensor rstd,
                                        float eps, std::vector<int> &dims,
                                        int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                        int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                        int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                        int man_sqrt, int exp_sqrt, int binades_sqrt_l, int binades_sqrt_u,
                                        bool saturate)
{
      CHECK_INPUT(input);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      CHECK_INPUT(output);
      CHECK_INPUT(mean);
      CHECK_INPUT(rstd);
      superfp_quantize_nearest_layernorm_forward_cuda(input, weight, bias,
                                                      output, mean, rstd,
                                                      eps, dims,
                                                      man_acc, exp_acc, binades_acc_l, binades_acc_u,
                                                      man_mul, exp_mul, binades_mul_l, binades_mul_u,
                                                      man_div, exp_div, binades_div_l, binades_div_u,
                                                      man_sqrt, exp_sqrt, binades_sqrt_l, binades_sqrt_u,
                                                      saturate);
}

void superfp_quantize_layernorm_backward(Tensor input, Tensor grad_output,
                                         Tensor weight, Tensor bias,
                                         Tensor mean, Tensor rstd,
                                         Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                         std::vector<int> &dims,
                                         int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                         int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                         int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                         bool saturate)
{
      CHECK_INPUT(input);
      CHECK_INPUT(grad_output);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      CHECK_INPUT(mean);
      CHECK_INPUT(rstd);
      CHECK_INPUT(grad_input);
      CHECK_INPUT(grad_gamma);
      CHECK_INPUT(grad_beta);
      superfp_quantize_nearest_layernorm_backward_cuda(input, grad_output,
                                                       weight, bias,
                                                       mean, rstd,
                                                       grad_input, grad_gamma, grad_beta,
                                                       dims,
                                                       man_acc, exp_acc, binades_acc_l, binades_acc_u,
                                                       man_mul, exp_mul, binades_mul_l, binades_mul_u,
                                                       man_div, exp_div, binades_div_l, binades_div_u,
                                                       saturate);
}

void binary8_quantize_layernorm_forward(Tensor input, Tensor weight, Tensor bias,
                                        Tensor output, Tensor mean, Tensor rstd,
                                        float eps, std::vector<int> &dims,
                                        int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                        int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                        int P_div, OverflowPolicy op_div, bool signed_div,
                                        int P_sqrt, OverflowPolicy op_sqrt, bool signed_sqrt,
                                        bool subnormals)
{
      CHECK_INPUT(input);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      CHECK_INPUT(output);
      CHECK_INPUT(mean);
      CHECK_INPUT(rstd);
      binary8_quantize_nearest_layernorm_forward_cuda(input, weight, bias,
                                                      output, mean, rstd,
                                                      eps, dims,
                                                      P_acc, op_acc, signed_acc,
                                                      P_mul, op_mul, signed_mul,
                                                      P_div, op_div, signed_div,
                                                      P_sqrt, op_sqrt, signed_sqrt,
                                                      subnormals);
}

void binary8_quantize_layernorm_backward(Tensor input, Tensor grad_output,
                                         Tensor weight, Tensor bias,
                                         Tensor mean, Tensor rstd,
                                         Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                         std::vector<int> &dims,
                                         int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                         int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                         int P_div, OverflowPolicy op_div, bool signed_div,
                                         bool subnormals)
{
      CHECK_INPUT(input);
      CHECK_INPUT(grad_output);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      CHECK_INPUT(mean);
      CHECK_INPUT(rstd);
      CHECK_INPUT(grad_input);
      CHECK_INPUT(grad_gamma);
      CHECK_INPUT(grad_beta);
      binary8_quantize_nearest_layernorm_backward_cuda(input, grad_output,
                                                       weight, bias,
                                                       mean, rstd,
                                                       grad_input, grad_gamma, grad_beta,
                                                       dims,
                                                       P_acc, op_acc, signed_acc,
                                                       P_mul, op_mul, signed_mul,
                                                       P_div, op_div, signed_div,
                                                       subnormals);
}

void float_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                            int man_exp, int exp_exp,
                                            int man_off, int exp_off,
                                            int man_acc, int exp_acc,
                                            bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(o);
      float_quantize_nearest_softmax_forward_cuda(
          a, o, dim,
          man_exp, exp_exp,
          man_off, exp_off,
          man_acc, exp_acc,
          subnormals, saturate);
}

void float_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                                int man_off, int exp_off,
                                                int man_lse, int exp_lse,
                                                bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(o);
      float_quantize_nearest_softmax_lse_forward_cuda(
          a, o, dim,
          man_off, exp_off,
          man_lse, exp_lse,
          subnormals, saturate);
}

void float_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                             int man_add, int exp_add,
                                             int man_mul, int exp_mul,
                                             bool subnormals, bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(g);
      CHECK_INPUT(o);
      float_quantize_nearest_softmax_backward_cuda(
          a, g, o, dim,
          man_add, exp_add,
          man_mul, exp_mul,
          subnormals, saturate);
}

void superfp_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                              int man_exp, int exp_exp, int binades_exp_l, int binades_exp_u,
                                              int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                              int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                              bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(o);
      superfp_quantize_nearest_softmax_forward_cuda(
          a, o, dim,
          man_exp, exp_exp, binades_exp_l, binades_exp_u,
          man_off, exp_off, binades_off_l, binades_off_u,
          man_acc, exp_acc, binades_acc_l, binades_acc_u,
          saturate);
}

void superfp_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                                  int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                                  int man_lse, int exp_lse, int binades_lse_l, int binades_lse_u,
                                                  bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(o);
      superfp_quantize_nearest_softmax_lse_forward_cuda(
          a, o, dim,
          man_off, exp_off, binades_off_l, binades_off_u,
          man_lse, exp_lse, binades_lse_l, binades_lse_u,
          saturate);
}

void superfp_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                               int man_add, int exp_add, int binades_add_l, int binades_add_u,
                                               int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                               bool saturate)
{
      CHECK_INPUT(a);
      CHECK_INPUT(g);
      CHECK_INPUT(o);
      superfp_quantize_nearest_softmax_backward_cuda(
          a, g, o, dim,
          man_add, exp_add, binades_add_l, binades_add_u,
          man_mul, exp_mul, binades_mul_l, binades_mul_u,
          saturate);
}

void binary8_quantize_nearest_softmax_forward(Tensor a, Tensor o, int dim,
                                              int P_exp, OverflowPolicy op_exp, bool signed_exp,
                                              int P_off, OverflowPolicy op_off, bool signed_off,
                                              int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                              bool subnormals)
{
      CHECK_INPUT(a);
      CHECK_INPUT(o);
      binary8_quantize_nearest_softmax_forward_cuda(
          a, o, dim,
          P_exp, op_exp, signed_exp,
          P_off, op_off, signed_off,
          P_acc, op_acc, signed_acc,
          subnormals);
}

void binary8_quantize_nearest_softmax_lse_forward(Tensor a, Tensor o, int dim,
                                                  int P_off, OverflowPolicy op_off, bool signed_off,
                                                  int P_lse, OverflowPolicy op_lse, bool signed_lse,
                                                  bool subnormals)
{
      CHECK_INPUT(a);
      CHECK_INPUT(o);
      binary8_quantize_nearest_softmax_lse_forward_cuda(
          a, o, dim,
          P_off, op_off, signed_off,
          P_lse, op_lse, signed_lse,
          subnormals);
}

void binary8_quantize_nearest_softmax_backward(Tensor a, Tensor g, Tensor o, int dim,
                                               int P_add, OverflowPolicy op_add, bool signed_add,
                                               int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                               bool subnormals)
{
      CHECK_INPUT(a);
      CHECK_INPUT(g);
      CHECK_INPUT(o);
      binary8_quantize_nearest_softmax_backward_cuda(
          a, g, o, dim,
          P_add, op_add, signed_add,
          P_mul, op_mul, signed_mul,
          subnormals);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("float_quantize_nearest",
            &float_quantize_nearest,
            "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("float_quantize_nearest_mp",
            &float_quantize_nearest_mp,
            "Low-Bitwidth Floating Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("float_quantize_stochastic",
            &float_quantize_stochastic,
            "Low-Bitwidth Floating Point Number Stochastic Quantization (CUDA)");

      m.def("fixed_point_quantize_nearest",
            &fixed_point_quantize_nearest,
            "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_mask",
            &fixed_point_quantize_nearest_mask,
            "Fixed Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic",
            &fixed_point_quantize_stochastic,
            "Fixed Point Number Stochastic Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic_mask",
            &fixed_point_quantize_stochastic_mask,
            "Fixed Point Number Stochastic Quantization (CUDA)");

      m.def("block_quantize_nearest",
            &block_quantize_nearest,
            "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("block_quantize_sim_nearest",
            &block_quantize_sim_nearest,
            "Block Floating Point Number Nearest Neighbor Quantization (CUDA)");
      m.def("block_quantize_stochastic",
            &block_quantize_stochastic,
            "Block Floating Point Number Stochastic Quantization (CUDA)");
      m.def("block_quantize_sim_stochastic",
            &block_quantize_sim_stochastic,
            "Block Floating Point Number Stochastic Quantization (CUDA)");

      m.def("superfp_quantize_nearest",
            &superfp_quantize_nearest,
            "Low-Bitwidth Super Normal Floating Point Number Nearest Neighbor Quantization "
            "(CUDA)");

      m.def("binary8_quantize_nearest",
            &binary8_quantize_nearest,
            "Low-Bitwidth P3109 Floating-Point Number Nearest Quantization (CUDA)");
      m.def("binary8_quantize_truncate",
            &binary8_quantize_truncate,
            "Low-Bitwidth P3109 Floating-Point Number truncate Quantization (CUDA)");
      m.def("binary8_quantize_stochastic",
            &binary8_quantize_stochastic,
            "Low-Bitwidth P3109 Floating-Point Number Stochastic Quantization (CUDA)");

      py::enum_<OverflowPolicy>(m, "OverflowPolicy", py::arithmetic(), py::module_local())
          .value("SATURATE_INFTY", OverflowPolicy::SATURATE_INFTY)
          .value("SATURATE_MAXFLOAT", OverflowPolicy::SATURATE_MAXFLOAT)
          .value("SATURATE_MAXFLOAT2", OverflowPolicy::SATURATE_MAXFLOAT2);

      m.def("float_quantize_nearest_mm",
            &float_quantize_nearest_mm,
            "Low-Bitwidth Floating Point Number GEMM Quantization (CUDA)");
      m.def("float_quantize_nearest_mm_mp",
            &float_quantize_nearest_mm_mp,
            "Low-Bitwidth Floating Point Number GEMM Quantization (CUDA)");
      m.def("float_quantize_nearest_bmm",
            &float_quantize_nearest_bmm,
            "Low-Bitwidth Floating Point Number BGEMM Quantization (CUDA)");
      m.def("float_quantize_nearest_mm_fma",
            &float_quantize_nearest_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM Quantization (CUDA)");
      m.def("float_quantize_nearest_mm_fma_mp",
            &float_quantize_nearest_mm_fma_mp,
            "Low-Bitwidth Floating Point Number FMA-based GEMM Quantization (CUDA)");
      m.def("float_quantize_nearest_bmm_fma",
            &float_quantize_nearest_bmm_fma,
            "Low-Bitwidth Floating Point Number FMA-based BGEMM Quantization (CUDA)");

      m.def("float_quantize_stochastic_mm",
            &float_quantize_stochastic_mm,
            "Low-Bitwidth Floating Point Number GEMM with Stochastic Quantization (CUDA)");
      m.def("float_quantize_stochastic_bmm",
            &float_quantize_stochastic_bmm,
            "Low-Bitwidth Floating Point Number BGEMM with Stochastic Quantization (CUDA)");
      m.def("float_quantize_stochastic_mm_fma",
            &float_quantize_stochastic_mm_fma,
            "Low-Bitwidth Floating Point Number FMA-based GEMM with Stochastic Quantization (CUDA)");
      m.def("float_quantize_stochastic_bmm_fma",
            &float_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Floating Point Number FMA-based BGEMM with Stochastic Quantization (CUDA)");

      m.def("superfp_quantize_nearest_mm",
            &superfp_quantize_nearest_mm,
            "Low-Bitwidth SuperNormal Floating Point Number GEMM Quantization (CUDA)");
      m.def("superfp_quantize_nearest_bmm",
            &superfp_quantize_nearest_bmm,
            "Low-Bitwidth SuperNormal Floating Point Number BGEMM Quantization (CUDA)");
      m.def("superfp_quantize_nearest_mm_fma",
            &superfp_quantize_nearest_mm_fma,
            "Low-Bitwidth SuperNormal Floating Point Number FMA-based GEMM Quantization (CUDA)");
      m.def("superfp_quantize_nearest_bmm_fma",
            &superfp_quantize_nearest_bmm_fma,
            "Low-Bitwidth SuperNormal Floating Point Number FMA-based BGEMM Quantization (CUDA)");

      m.def("fixed_point_quantize_nearest_mm",
            &fixed_point_quantize_nearest_mm,
            "Low-Bitwidth Fixed Point Number GEMM Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_bmm",
            &fixed_point_quantize_nearest_bmm,
            "Low-Bitwidth Fixed Point Number BGEMM Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_mm_fma",
            &fixed_point_quantize_nearest_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM Quantization (CUDA)");
      m.def("fixed_point_quantize_nearest_bmm_fma",
            &fixed_point_quantize_nearest_bmm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based BGEMM Quantization (CUDA)");

      m.def("fixed_point_quantize_stochastic_mm",
            &fixed_point_quantize_stochastic_mm,
            "Low-Bitwidth Fixed Point Number GEMM with Stochastic Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic_bmm",
            &fixed_point_quantize_stochastic_bmm,
            "Low-Bitwidth Fixed Point Number BGEMM with Stochastic Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic_mm_fma",
            &fixed_point_quantize_stochastic_mm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic Quantization (CUDA)");
      m.def("fixed_point_quantize_stochastic_bmm_fma",
            &fixed_point_quantize_stochastic_bmm_fma,
            "Low-Bitwidth Fixed Point Number FMA-based GEMM with Stochastic Quantization (CUDA)");

      py::enum_<CUBLASMatrixType>(m, "CUBLASMatrixType", py::arithmetic())
          .value("F32", CUBLASMatrixType::kF32)
          .value("F16", CUBLASMatrixType::kF16)
          .value("BF16", CUBLASMatrixType::kBF16);

      py::enum_<CUBLASComputeType>(m, "CUBLASComputeType", py::arithmetic())
          .value("F32", CUBLASComputeType::kF32)
          .value("F16", CUBLASComputeType::kF16)
          .value("F32_FAST_F16", CUBLASComputeType::kF32FastF16)
          .value("F32_FAST_BF16", CUBLASComputeType::kF32FastBF16)
          .value("F32_FAST_TF32", CUBLASComputeType::kF32FastTF32);

      m.def("create_cublas_handle", &create_cublas_handle, "Creates a new cuBLAS handle");
      m.def("delete_cublas_handle", &delete_cublas_handle, "Deletes the current cuBLAS handle");
      m.def("floating_point_mm_cublas",
            &floating_point_mm_cublas,
            "cuBLAS accelerated matrix multiply, using the specified precision and "
            "compute mode (CUDA)");
      m.def("floating_point_bmm_cublas",
            &floating_point_bmm_cublas,
            "cuBLAS accelerated batched matrix multiply, using the specified precision "
            "and compute mode (CUDA)");

      m.def("float_quantize_nearest_softmax_forward",
            &float_quantize_nearest_softmax_forward,
            "Low-Bitwidth Floating Point Softmax Forward using division. (CUDA)");
      m.def("float_quantize_nearest_softmax_lse_forward",
            &float_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Floating Point Softmax Forward using LogSumExp. (CUDA)");
      m.def("float_quantize_nearest_softmax_backward",
            &float_quantize_nearest_softmax_backward,
            "Low-Bitwidth Floating Point Softmax Backward. (CUDA)");

      m.def("superfp_quantize_nearest_softmax_forward",
            &superfp_quantize_nearest_softmax_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using division. (CUDA)");
      m.def("superfp_quantize_nearest_softmax_lse_forward",
            &superfp_quantize_nearest_softmax_lse_forward,
            "Low-Bitwidth Super Floating Point Softmax Forward using LogSumExp. (CUDA)");
      m.def("superfp_quantize_nearest_softmax_backward",
            &superfp_quantize_nearest_softmax_backward,
            "Low-Bitwidth Super Floating Point Softmax Backward. (CUDA)");

      m.def("binary8_quantize_nearest_softmax_forward",
            &binary8_quantize_nearest_softmax_forward,
            "Binary8 Softmax Forward using division. (CUDA)");
      m.def("binary8_quantize_nearest_softmax_lse_forward",
            &binary8_quantize_nearest_softmax_lse_forward,
            "Binary8 Softmax Forward using LogSumExp. (CUDA)");
      m.def("binary8_quantize_nearest_softmax_backward",
            &binary8_quantize_nearest_softmax_backward,
            "Binary8 Softmax Backward. (CUDA)");

      m.def("float_quantize_layernorm_forward",
            &float_quantize_layernorm_forward,
            "Low-Bitwidth Floating Point Layer Normalization (CUDA)");
      m.def("float_quantize_layernorm_backward",
            &float_quantize_layernorm_backward,
            "Low-Bitwidth Floating Point Layer Normalization Backward (CUDA)");

      m.def("superfp_quantize_layernorm_forward",
            &superfp_quantize_layernorm_forward,
            "Low-Bitwidth Super Floating Point Layer Normalization (CUDA)");
      m.def("superfp_quantize_layernorm_backward",
            &superfp_quantize_layernorm_backward,
            "Low-Bitwidth Super Floating Point Layer Normalization Backward (CUDA)");

      m.def("binary8_quantize_layernorm_forward",
            &binary8_quantize_layernorm_forward,
            "Low-Bitwidth binary8 Layer Normalization (CUDA)");
      m.def("binary8_quantize_layernorm_backward",
            &binary8_quantize_layernorm_backward,
            "Low-Bitwidth binary8 Layer Normalization Backward (CUDA)");
}
