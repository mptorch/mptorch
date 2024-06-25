#include "quant.h"
#include "quant_kernel.h"
#include "cublas_helper.h"
#include <ATen/ATen.h>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <cassert>

using namespace at;

Tensor get_max_entry(Tensor a, int dim)
{
  Tensor max_entry;
  if (dim == -1)
  {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  }
  else if (dim == 0)
  {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true))
                    .expand_as(input_view)
                    .view_as(a)
                    .contiguous();
  }
  else
  {
    Tensor input_transpose = a.transpose(0, dim);
    Tensor input_view =
        input_transpose.contiguous().view({input_transpose.size(0), -1});
    Tensor max_transpose = std::get<0>(input_view.abs().max(1, true))
                               .expand_as(input_view)
                               .view_as(input_transpose);
    max_entry = max_transpose.transpose(dim, 0).contiguous();
  }
  return max_entry;
}

Tensor block_quantize_stochastic_cuda(Tensor a, int wl, int dim)
{
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = get_max_entry(a, dim);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
      max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor block_quantize_nearest_cuda(Tensor a, int wl, int dim)
{
  auto o = at::zeros_like(a);
  int64_t size = a.numel();

  Tensor max_entry = get_max_entry(a, dim);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size,
      max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl)
{
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_probs.data_ptr<float>(), o.data_ptr<float>(),
      size, max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl)
{
  auto o = at::zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int64_t size = a.numel();

  Tensor max_entry = at::max(at::abs(a));
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  block_kernel_sim_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size,
      max_entry.data_ptr<float>(), wl);
  return o;
}

Tensor float_quantize_stochastic_cuda(Tensor a, int man_bits, int exp_bits,
                                      bool subnormals, bool saturate)
{
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
      man_bits, exp_bits, subnormals, saturate);
  return o;
}

Tensor float_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits,
                                   bool subnormals, bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
      subnormals, saturate);
  return o;
}

Tensor superfp_quantize_nearest_cuda(Tensor a, int man_bits, int exp_bits,
                                    bool saturate) 
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  superfp_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits, saturate);
  return o;

}

Tensor p3109_quantize_nearest_cuda(Tensor a, int P, bool is_signed, bool subnormals)
{
  auto o = zeros_like(a);
  // TODO
  return o;
}

Tensor p3109_quantize_stochastic_cuda(Tensor a, int P, int prng_bits, bool is_signed, bool subnormals)
{
  auto o = zeros_like(a);
  // TODO
  return o;
}

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max)
{
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl,
                                            bool use_clamp, bool symmetric)
{
  // use external random number right now
  auto o = at::zeros_like(a);
  auto rand_probs = rand_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_probs.data_ptr<float>(), o.data_ptr<float>(),
      size, sigma, use_clamp, t_min, t_max);
  return o;
}

Tensor fixed_point_quantize_nearest_cuda(Tensor a, int wl, int fl,
                                         bool use_clamp, bool symmetric)
{
  // use external random number right now
  auto o = at::zeros_like(a);
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, sigma, use_clamp, t_min,
      t_max);
  return o;
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_stochastic_mask_cuda(Tensor a, int wl, int fl,
                                          bool symmetric)
{
  // use external random number right now
  auto o = zeros_like(a);
  auto rand_probs = rand_like(a);
  auto m = zeros_like(a, a.options().dtype(kByte));
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_mask_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_probs.data_ptr<float>(), o.data_ptr<float>(),
      m.data_ptr<uint8_t>(), size, sigma, t_min, t_max);
  return std::make_tuple(o, m);
}

std::tuple<Tensor, Tensor>
fixed_point_quantize_nearest_mask_cuda(Tensor a, int wl, int fl,
                                       bool symmetric)
{
  // use external random number right now
  auto o = at::zeros_like(a);
  auto m = zeros_like(a, a.options().dtype(kByte));
  int64_t size = a.numel();
  int sigma = -fl;
  float t_min, t_max;
  fixed_min_max(wl, fl, symmetric, &t_min, &t_max);
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  fixed_point_quantize_kernel_mask_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), m.data_ptr<uint8_t>(), size,
      sigma, t_min, t_max);
  return std::make_tuple(o, m);
}

void float_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c, int M, int N,
                                    int K, int man_add, int exp_add,
                                    int man_mul, int exp_mul, bool subnormals,
                                    bool saturate)
{
  mm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                M, K, N, man_add, exp_add, man_mul, exp_mul, subnormals,
                saturate);
  return;
}

void float_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int man_fma, int exp_fma,
                                        bool subnormals, bool saturate)
{
  mm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                    c.data_ptr<float>(), M, K, N, man_fma, exp_fma, subnormals,
                    saturate);
  return;
}

void float_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c, int M, int N,
                                     int K, int man_add, int exp_add,
                                     int man_mul, int exp_mul, bool subnormals,
                                     bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                   c.data_ptr<float>(), a.sizes()[0], M, K, N, man_add, exp_add,
                   man_mul, man_add, subnormals, saturate);
  else
    bmm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                   c.data_ptr<float>(), 1, M, K, N, man_add, exp_add, man_mul,
                   man_add, subnormals, saturate);
  return;
}

void float_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int man_fma, int exp_fma,
                                         bool subnormals, bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), a.sizes()[0], M, K, N, man_fma,
                       exp_fma, subnormals, saturate);
  else
    bmm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), 1, M, K, N, man_fma, exp_fma,
                       subnormals, saturate);
  return;
}

void superfp_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c, int M, int N,
                                    int K, int man_add, int exp_add,
                                    int man_mul, int exp_mul,
                                    bool saturate) 
{
  mm_superfp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                M, K, N, man_add, exp_add, man_mul, exp_mul, saturate);  
}

void superfp_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int man_fma, int exp_fma,
                                        bool saturate)
{
  mm_superfp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                    c.data_ptr<float>(), M, K, N, man_fma, exp_fma,
                    saturate);
  return;
}

void superfp_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c, int M, int N,
                                     int K, int man_add, int exp_add,
                                     int man_mul, int exp_mul, bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_superfp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                   c.data_ptr<float>(), a.sizes()[0], M, K, N, man_add, exp_add,
                   man_mul, man_add, saturate);
  else
    bmm_superfp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                   c.data_ptr<float>(), 1, M, K, N, man_add, exp_add, man_mul,
                   man_add, saturate);
  return;
}

void superfp_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int man_fma, int exp_fma, bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_superfp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), a.sizes()[0], M, K, N, man_fma,
                       exp_fma, saturate);
  else
    bmm_superfp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), 1, M, K, N, man_fma, exp_fma, saturate);
  return;
}

void float_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                       int N, int K, int man_add, int exp_add,
                                       int man_mul, int exp_mul,
                                       bool subnormals, bool saturate)
{
  mm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                   c.data_ptr<float>(), M, K, N, man_add, exp_add, man_mul,
                   exp_mul, subnormals, saturate);
  return;
}

void float_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                           int N, int K, int man_fma,
                                           int exp_fma, bool subnormals,
                                           bool saturate)
{
  mm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), M, K, N, man_fma, exp_fma,
                       subnormals, saturate);
  return;
}

void float_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                        int N, int K, int man_add, int exp_add,
                                        int man_mul, int exp_mul,
                                        bool subnormals, bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                      c.data_ptr<float>(), a.sizes()[0], M, K, N, man_add,
                      exp_add, man_mul, man_add, subnormals, saturate);
  else
    bmm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                      c.data_ptr<float>(), 1, M, K, N, man_add, exp_add,
                      man_mul, man_add, subnormals, saturate);
}

void float_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                            int N, int K, int man_fma,
                                            int exp_fma, bool subnormals,
                                            bool saturate)
{

  if (a.sizes().size() > 2)
    bmm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                          c.data_ptr<float>(), a.sizes()[0], M, K, N, man_fma,
                          exp_fma, subnormals, saturate);
  else
    bmm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                          c.data_ptr<float>(), 1, M, K, N, man_fma, exp_fma,
                          subnormals, saturate);
}

void fixed_point_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                          int N, int K, int wl_add, int fl_add,
                                          int wl_mul, int fl_mul,
                                          bool symmetric)
{
  int sigma_add = -fl_add;
  int sigma_mul = -fl_mul;
  float t_min_add, t_max_add, t_min_mul, t_max_mul;
  fixed_min_max(wl_add, fl_add, symmetric, &t_min_add, &t_max_add);
  fixed_min_max(wl_mul, fl_mul, symmetric, &t_min_mul, &t_max_mul);
  mm_fxp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                 M, K, N, sigma_add, t_min_add, t_max_add, sigma_mul, t_min_mul,
                 t_max_mul);
  return;
}

void fixed_point_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                           int N, int K, int wl_add, int fl_add,
                                           int wl_mul, int fl_mul,
                                           bool symmetric)
{

  int sigma_add = -fl_add;
  int sigma_mul = -fl_mul;
  float t_min_add, t_max_add, t_min_mul, t_max_mul;
  fixed_min_max(wl_add, fl_add, symmetric, &t_min_add, &t_max_add);
  fixed_min_max(wl_mul, fl_mul, symmetric, &t_min_mul, &t_max_mul);
  if (a.sizes().size() > 2)
    bmm_fxp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                    c.data_ptr<float>(), a.sizes()[0], M, K, N, sigma_add,
                    t_min_add, t_max_add, sigma_mul, t_min_mul, t_max_mul);
  else
    bmm_fxp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                    c.data_ptr<float>(), 1, M, K, N, sigma_add, t_min_add,
                    t_max_add, sigma_mul, t_min_mul, t_max_mul);
  return;
}

void fixed_point_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K, int wl_fma,
                                              int fl_fma, bool symmetric)
{
  int sigma_fma = -fl_fma;
  float t_min_fma, t_max_fma;
  fixed_min_max(wl_fma, fl_fma, symmetric, &t_min_fma, &t_max_fma);
  mm_fxp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                     c.data_ptr<float>(), M, K, N, sigma_fma, t_min_fma,
                     t_max_fma);
  return;
}

void fixed_point_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                               int M, int N, int K, int wl_fma,
                                               int fl_fma, bool symmetric)
{
  int sigma_fma = -fl_fma;
  float t_min_fma, t_max_fma;
  fixed_min_max(wl_fma, fl_fma, symmetric, &t_min_fma, &t_max_fma);
  if (a.sizes().size() > 2)
    bmm_fxp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                        c.data_ptr<float>(), a.sizes()[0], M, K, N, sigma_fma,
                        t_min_fma, t_max_fma);
  else
    bmm_fxp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                        c.data_ptr<float>(), 1, M, K, N, sigma_fma, t_min_fma,
                        t_max_fma);
  return;
}

void fixed_point_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                             int M, int N, int K, int wl_add,
                                             int fl_add, int wl_mul, int fl_mul,
                                             bool symmetric)
{
  int sigma_add = -fl_add;
  int sigma_mul = -fl_mul;
  float t_min_add, t_max_add, t_min_mul, t_max_mul;
  fixed_min_max(wl_add, fl_add, symmetric, &t_min_add, &t_max_add);
  fixed_min_max(wl_mul, fl_mul, symmetric, &t_min_mul, &t_max_mul);
  mm_fxp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                    c.data_ptr<float>(), M, K, N, sigma_add, t_min_add,
                    t_max_add, sigma_mul, t_min_mul, t_max_mul);
  return;
}

void fixed_point_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K, int wl_add,
                                              int fl_add, int wl_mul,
                                              int fl_mul, bool symmetric)
{
  int sigma_add = -fl_add;
  int sigma_mul = -fl_mul;
  float t_min_add, t_max_add, t_min_mul, t_max_mul;
  fixed_min_max(wl_add, fl_add, symmetric, &t_min_add, &t_max_add);
  fixed_min_max(wl_mul, fl_mul, symmetric, &t_min_mul, &t_max_mul);
  if (a.sizes().size() > 2)
    bmm_fxp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), a.sizes()[0], M, K, N, sigma_add,
                       t_min_add, t_max_add, sigma_mul, t_min_mul, t_max_mul);
  else
    bmm_fxp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                       c.data_ptr<float>(), 1, M, K, N, sigma_add, t_min_add,
                       t_max_add, sigma_mul, t_min_mul, t_max_mul);
  return;
}

void fixed_point_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                 int M, int N, int K,
                                                 int wl_fma, int fl_fma,
                                                 bool symmetric)
{
  int sigma_fma = -fl_fma;
  float t_min_fma, t_max_fma;
  fixed_min_max(wl_fma, fl_fma, symmetric, &t_min_fma, &t_max_fma);
  mm_fxp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                        c.data_ptr<float>(), M, K, N, sigma_fma, t_min_fma,
                        t_max_fma);
  return;
}

void fixed_point_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                  int M, int N, int K,
                                                  int wl_fma, int fl_fma,
                                                  bool symmetric)
{
  int sigma_fma = -fl_fma;
  float t_min_fma, t_max_fma;
  fixed_min_max(wl_fma, fl_fma, symmetric, &t_min_fma, &t_max_fma);
  if (a.sizes().size() > 2)
    bmm_fxp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                           c.data_ptr<float>(), a.sizes()[0], M, K, N,
                           sigma_fma, t_min_fma, t_max_fma);
  else
    bmm_fxp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(),
                           c.data_ptr<float>(), 1, M, K, N, sigma_fma,
                           t_min_fma, t_max_fma);
  return;
}



void floating_point_mm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                              cublas_matrix_dt AB_type, cublas_matrix_dt C_type,
                              cublas_compute_dt compute_type, bool pedantic)
{
  // Tensors a, b, and c are assumed to have the right datatype and transposed.
  cublas_config config;
  get_cublas_configuration(AB_type, C_type, compute_type, pedantic, config);

  cublasMath_t math = pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH;
  math = (cublasMath_t)(math | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  cublasSetMathMode(get_cublas_handle(), math);

  // special case for scalar types: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
  switch (config.scalar) {
  case CUDA_R_16F:
    {
    half alpha = __float2half(1.f);
    half beta = __float2half(0.f);
    cublasGemmEx(get_cublas_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                  a.data_ptr(), config.matrix_a, M,
                  b.data_ptr(), config.matrix_b, K, &beta,
                  c.data_ptr(), config.matrix_c, M,
                  config.compute,
                  CUBLAS_GEMM_DEFAULT);
    }
    break;
  default:
    {
    float alpha = 1.f;
    float beta = 0.f;
    cublasGemmEx(get_cublas_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                  a.data_ptr(), config.matrix_a, M,
                  b.data_ptr(), config.matrix_b, K, &beta,
                  c.data_ptr(), config.matrix_c, M,
                  config.compute,
                  CUBLAS_GEMM_DEFAULT);
    }
    break;
  }
}


void floating_point_bmm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                               cublas_matrix_dt AB_type, cublas_matrix_dt C_type,
                               cublas_compute_dt compute_type, bool pedantic)
{
  // Tensors a, b, and c are assumed to have the right datatype and transposed.
  cublas_config config;
  get_cublas_configuration(AB_type, C_type, compute_type, pedantic, config);

  cublasMath_t math = pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH;
  math = (cublasMath_t)(math | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  cublasSetMathMode(get_cublas_handle(), math);

  int B = a.sizes().size() > 2 ? a.size(0) : 1; // batch count

  // Allocate the array of pointers to each matrices  
  auto copy_ptrs = [B](void** arr, Tensor a, cudaDataType t, int stride) {
    switch (t) {
    case CUDA_R_32F:
      {
      float *p = a.data_ptr<float>();
      for (int i = 0; i < B; i++) {
        arr[i] = p + i * stride;
      }
      }
      break;
    case CUDA_R_16F:
    case CUDA_R_16BF:
      {
      at::Half *p = a.data_ptr<at::Half>();
      for (int i = 0; i < B; i++) {
        arr[i] = p + i * stride;
      }
      }
      break;
    default:
      throw std::invalid_argument("Invalid datatype.");
    }
  };
  void *a_array[B];
  void *b_array[B];
  void *c_array[B];
  copy_ptrs(a_array, a, config.matrix_a, M*K);
  copy_ptrs(b_array, b, config.matrix_b, K*N);
  copy_ptrs(c_array, c, config.matrix_c, M*N);

  // special case for scalar types: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex
  // TODO: Fix, it crashes, memory alignement issues apparently
  switch (config.scalar) {
  case CUDA_R_16F:
    // {
    // half alpha = __float2half(1.f);
    // half beta = __float2half(0.f);
    // cublasGemmBatchedEx(get_cublas_handle(),
    //               CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
    //               a_array, config.matrix_a, M,
    //               b_array, config.matrix_b, K, &beta,
    //               c_array, config.matrix_c, M, B,
    //               config.compute,
    //               config.algo);
    // }
    break;
  default:
    // {
    // float alpha = 1.f;
    // float beta = 0.f;
    // cublasGemmBatchedEx(get_cublas_handle(),
    //               CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
    //               (void**)a_array, config.matrix_a, M,
    //               (void**)b_array, config.matrix_b, K, &beta,
    //               (void**)c_array, config.matrix_c, M, B,
    //               config.compute,
    //               config.algo);
    // }
    break;
  }
}