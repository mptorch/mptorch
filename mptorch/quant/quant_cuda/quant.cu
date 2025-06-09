#include "quant.h"
#include "quant_kernel.h"
#include "binary8_kernel.h"
#include <ATen/ATen.h>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <cassert>
#include <vector>

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

Tensor float_quantize_stochastic_cuda(Tensor a,
                                      int man_bits, int exp_bits,
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

Tensor float_quantize_stochastic_cuda(Tensor a,
                                      int man_bits, int exp_bits, int prng_bits,
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
      man_bits, exp_bits, prng_bits, subnormals, saturate);
  return o;
}

Tensor float_quantize_nearest_cuda(Tensor a,
                                   int man_bits, int exp_bits,
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

Tensor superfp_quantize_nearest_cuda(Tensor a,
                                     int man_bits, int exp_bits,
                                     int binades_l, int binades_u,
                                     bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  superfp_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits, binades_l, binades_u, saturate);
  return o;
}

Tensor binary8_quantize_nearest_cuda(Tensor a,
                                     int P, bool is_signed, OverflowPolicy overflow_policy,
                                     bool subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel(); // gets number of elements in tensor a
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  if (is_signed == true)
  { // signed
    binary8_signed_kernel_nearest<<<blockNums, blockSize>>>(
        a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  }
  else
  { // unsigned
    binary8_unsigned_kernel_nearest<<<blockNums, blockSize>>>(
        a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  }

  return o;
}

Tensor binary8_quantize_stochastic_cuda(Tensor a,
                                        int P, int prng_bits, bool is_signed, OverflowPolicy overflow_policy,
                                        bool subnormals)
{
  auto o = zeros_like(a);
  // generate random number on the GPU for the SR operation
  auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
  int size = a.numel(); // gets number of elements in tensor a
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  if (is_signed == true)
  { // signed
    binary8_signed_kernel_stochastic<<<blockNums, blockSize>>>(
        a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size, P, prng_bits, overflow_policy, subnormals);
  }
  else
  { // unsigned
    binary8_unsigned_kernel_stochastic<<<blockNums, blockSize>>>(
        a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size, P, prng_bits, overflow_policy, subnormals);
  }

  return o;
}

Tensor binary8_quantize_truncate_cuda(Tensor a,
                                      int P, bool is_signed, OverflowPolicy overflow_policy,
                                      bool subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel(); // gets number of elements in tensor a
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  if (is_signed == true)
  { // signed
    binary8_signed_kernel_truncate<<<blockNums, blockSize>>>(
        a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  }
  else
  { // unsigned
    binary8_unsigned_kernel_truncate<<<blockNums, blockSize>>>(
        a.data_ptr<float>(), o.data_ptr<float>(), size, P, overflow_policy, subnormals);
  }

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

Tensor fixed_point_quantize_stochastic_cuda(Tensor a,
                                            int wl, int fl,
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

Tensor fixed_point_quantize_nearest_cuda(Tensor a,
                                         int wl, int fl,
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
fixed_point_quantize_stochastic_mask_cuda(Tensor a,
                                          int wl, int fl,
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
fixed_point_quantize_nearest_mask_cuda(Tensor a,
                                       int wl, int fl,
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

void float_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                    int M, int N, int K,
                                    int man_add, int exp_add,
                                    int man_mul, int exp_mul,
                                    bool subnormals,
                                    bool saturate,
                                    bool compensated)
{
  mm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                M, K, N, man_add, exp_add, man_mul, exp_mul,
                subnormals, saturate, compensated);
  return;
}

void float_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_fma, int exp_fma,
                                        bool subnormals,
                                        bool saturate,
                                        bool compensated)
{
  mm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                    M, K, N, man_fma, exp_fma,
                    subnormals, saturate, compensated);
  return;
}

void float_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                     int M, int N, int K,
                                     int man_add, int exp_add,
                                     int man_mul, int exp_mul,
                                     bool subnormals,
                                     bool saturate,
                                     bool compensated)
{
  if (a.sizes().size() > 2)
    bmm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                   a.sizes()[0], M, K, N,
                   man_add, exp_add, man_mul, exp_mul,
                   subnormals, saturate, compensated);
  else
    bmm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                   1, M, K, N,
                   man_add, exp_add, man_mul, exp_mul,
                   subnormals, saturate, compensated);
  return;
}

void float_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                         int M, int N, int K,
                                         int man_fma, int exp_fma,
                                         bool subnormals,
                                         bool saturate,
                                         bool compensated)
{
  if (a.sizes().size() > 2)
    bmm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                       a.sizes()[0], M, K, N,
                       man_fma, exp_fma,
                       subnormals, saturate, compensated);
  else
    bmm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                       1, M, K, N,
                       man_fma, exp_fma,
                       subnormals, saturate, compensated);
  return;
}

void superfp_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                      int M, int N, int K,
                                      int man_add, int exp_add,
                                      int man_mul, int exp_mul,
                                      int binades_add_l, int binades_add_u,
                                      int binades_mul_l, int binades_mul_u,
                                      bool saturate)
{
  mm_superfp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                     M, K, N, man_add, exp_add, man_mul, exp_mul,
                     binades_add_l, binades_add_u,
                     binades_mul_l, binades_mul_u,
                     saturate);
}

void superfp_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                          int M, int N, int K,
                                          int man_fma, int exp_fma,
                                          int binades_fma_l, int binades_fma_u,
                                          bool saturate)
{
  mm_superfp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                         c.data_ptr<float>(), M, K, N, man_fma, exp_fma,
                         binades_fma_l, binades_fma_u,
                         saturate);
  return;
}

void superfp_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_add, int exp_add,
                                       int man_mul, int exp_mul,
                                       int binades_add_l, int binades_add_u,
                                       int binades_mul_l, int binades_mul_u,
                                       bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_superfp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                        c.data_ptr<float>(), a.sizes()[0], M, K, N, man_add, exp_add,
                        man_mul, exp_mul, binades_add_l, binades_add_u,
                        binades_mul_l, binades_mul_u, saturate);
  else
    bmm_superfp_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                        c.data_ptr<float>(), 1, M, K, N, man_add, exp_add, man_mul,
                        exp_mul, binades_add_l, binades_add_u,
                        binades_mul_l, binades_mul_u, saturate);
  return;
}

void superfp_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma,
                                           int binades_fma_l, int binades_fma_u,
                                           bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_superfp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                            c.data_ptr<float>(), a.sizes()[0], M, K, N, man_fma,
                            exp_fma, binades_fma_l, binades_fma_u, saturate);
  else
    bmm_superfp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(),
                            c.data_ptr<float>(), 1, M, K, N, man_fma, exp_fma,
                            binades_fma_l, binades_fma_u, saturate);
  return;
}

void float_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_add, int exp_add, int rbits_add,
                                       int man_mul, int exp_mul, int rbits_mul,
                                       bool subnormals, bool saturate)
{
  mm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                   M, K, N,
                   man_add, exp_add, rbits_add,
                   man_mul, exp_mul, rbits_mul,
                   subnormals, saturate);
  return;
}

void float_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma, int rbits_fma,
                                           bool subnormals, bool saturate)
{
  mm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                       M, K, N,
                       man_fma, exp_fma, rbits_fma,
                       subnormals, saturate);
  return;
}

void float_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_add, int exp_add, int rbits_add,
                                        int man_mul, int exp_mul, int rbits_mul,
                                        bool subnormals, bool saturate)
{
  if (a.sizes().size() > 2)
    bmm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                      a.sizes()[0], M, K, N,
                      man_add, exp_add, rbits_add,
                      man_mul, exp_mul, rbits_mul,
                      subnormals, saturate);
  else
    bmm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                      1, M, K, N,
                      man_add, exp_add, rbits_add,
                      man_mul, exp_mul, rbits_mul,
                      subnormals, saturate);
}

void float_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                            int M, int N, int K,
                                            int man_fma, int exp_fma, int rbits_fma,
                                            bool subnormals, bool saturate)
{

  if (a.sizes().size() > 2)
    bmm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                          a.sizes()[0], M, K, N,
                          man_fma, exp_fma, rbits_fma,
                          subnormals, saturate);
  else
    bmm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                          1, M, K, N,
                          man_fma, exp_fma, rbits_fma,
                          subnormals, saturate);
}

void fixed_point_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                          int M, int N, int K,
                                          int wl_add, int fl_add,
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

void fixed_point_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int wl_add, int fl_add,
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
                                              int M, int N, int K,
                                              int wl_fma, int fl_fma,
                                              bool symmetric)
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
                                               int M, int N, int K,
                                               int wl_fma, int fl_fma,
                                               bool symmetric)
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
                                             int M, int N, int K,
                                             int wl_add, int fl_add,
                                             int wl_mul, int fl_mul,
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
                                              int M, int N, int K,
                                              int wl_add, int fl_add,
                                              int wl_mul, int fl_mul,
                                              bool symmetric)
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

static DimSizes partition_tensor(Tensor input, std::vector<int> &dims)
{
  DimSizes sizes;
  std::vector<int> real_dims(dims.size());
  for (int i = 0; i < dims.size(); i++)
  {
    real_dims[i] = (input.dim() + (dims[i] % input.dim())) % input.dim();
  }

  sizes.channel = 1;
  for (int dim : real_dims)
  {
    sizes.channel *= input.size(dim);
  }

  int min_dim = real_dims.back();
  int max_dim = real_dims.front();

  sizes.outer = 1;
  for (int i = 0; i < min_dim; i++)
  {
    sizes.outer *= input.size(i);
  }

  sizes.inner = 1;
  for (int i = max_dim + 1; i < input.dim(); i++)
  {
    sizes.inner *= input.size(i);
  }
  return sizes;
}

static DimSizes partition_tensor(Tensor a, int dim)
{
  DimSizes sizes;
  int real_dim = (a.dim() + (dim % a.dim())) % a.dim();
  sizes.outer = 1;
  sizes.channel = a.size(real_dim);
  sizes.inner = 1;
  for (int i = 0; i < real_dim; ++i)
  {
    sizes.outer *= a.size(i);
  }
  for (int i = real_dim + 1; i < a.dim(); ++i)
  {
    sizes.inner *= a.size(i);
  }
  return sizes;
}

void float_quantize_nearest_layernorm_forward_cuda(Tensor input, Tensor weight, Tensor bias,
                                                   Tensor output, Tensor mean, Tensor rstd,
                                                   float eps, std::vector<int> &dims,
                                                   int man_acc, int exp_acc,
                                                   int man_mul, int exp_mul,
                                                   int man_div, int exp_div,
                                                   int man_sqrt, int exp_sqrt,
                                                   bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_forward_fp_nearest(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                               output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                               eps, sizes,
                               man_acc, exp_acc,
                               man_mul, exp_mul,
                               man_div, exp_div,
                               man_sqrt, exp_sqrt,
                               subnormals, saturate);
}

void float_quantize_nearest_layernorm_backward_cuda(Tensor input, Tensor grad_output,
                                                    Tensor weight, Tensor bias,
                                                    Tensor mean, Tensor rstd,
                                                    Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                                    std::vector<int> &dims,
                                                    int man_acc, int exp_acc,
                                                    int man_mul, int exp_mul,
                                                    int man_div, int exp_div,
                                                    bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_backward_fp_nearest(input.data_ptr<float>(), grad_output.data_ptr<float>(),
                                weight.data_ptr<float>(), bias.data_ptr<float>(),
                                mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                grad_input.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
                                sizes,
                                man_acc, exp_acc,
                                man_mul, exp_mul,
                                man_div, exp_div,
                                subnormals, saturate);
}

void superfp_quantize_nearest_layernorm_forward_cuda(Tensor input, Tensor weight, Tensor bias,
                                                     Tensor output, Tensor mean, Tensor rstd,
                                                     float eps, std::vector<int> &dims,
                                                     int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                                     int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                                     int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                                     int man_sqrt, int exp_sqrt, int binades_sqrt_l, int binades_sqrt_u,
                                                     bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_forward_superfp_nearest(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                    output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                    eps, sizes,
                                    man_acc, exp_acc, binades_acc_l, binades_acc_u,
                                    man_mul, exp_mul, binades_mul_l, binades_mul_u,
                                    man_div, exp_div, binades_div_l, binades_div_u,
                                    man_sqrt, exp_sqrt, binades_sqrt_l, binades_sqrt_u,
                                    saturate);
}

void superfp_quantize_nearest_layernorm_backward_cuda(Tensor input, Tensor grad_output,
                                                      Tensor weight, Tensor bias,
                                                      Tensor mean, Tensor rstd,
                                                      Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                                      std::vector<int> &dims,
                                                      int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                                      int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                                      int man_div, int exp_div, int binades_div_l, int binades_div_u,
                                                      bool saturate)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_backward_superfp_nearest(input.data_ptr<float>(), grad_output.data_ptr<float>(),
                                     weight.data_ptr<float>(), bias.data_ptr<float>(),
                                     mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                     grad_input.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
                                     sizes,
                                     man_acc, exp_acc, binades_acc_l, binades_acc_u,
                                     man_mul, exp_mul, binades_mul_l, binades_mul_u,
                                     man_div, exp_div, binades_div_l, binades_div_u,
                                     saturate);
}

void binary8_quantize_nearest_layernorm_forward_cuda(Tensor input, Tensor weight, Tensor bias,
                                                     Tensor output, Tensor mean, Tensor rstd,
                                                     float eps, std::vector<int> &dims,
                                                     int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                                     int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                                     int P_div, OverflowPolicy op_div, bool signed_div,
                                                     int P_sqrt, OverflowPolicy op_sqrt, bool signed_sqrt,
                                                     bool subnormals)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_forward_binary8_nearest(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                                    output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                    eps, sizes,
                                    P_acc, op_acc, signed_acc,
                                    P_mul, op_mul, signed_mul,
                                    P_div, op_div, signed_div,
                                    P_sqrt, op_sqrt, signed_sqrt,
                                    subnormals);
}

void binary8_quantize_nearest_layernorm_backward_cuda(Tensor input, Tensor grad_output,
                                                      Tensor weight, Tensor bias,
                                                      Tensor mean, Tensor rstd,
                                                      Tensor grad_input, Tensor grad_gamma, Tensor grad_beta,
                                                      std::vector<int> &dims,
                                                      int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                                      int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                                      int P_div, OverflowPolicy op_div, bool signed_div,
                                                      bool subnormals)
{
  auto sizes = partition_tensor(input, dims);
  layernorm_backward_binary8_nearest(input.data_ptr<float>(), grad_output.data_ptr<float>(),
                                     weight.data_ptr<float>(), bias.data_ptr<float>(),
                                     mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                     grad_input.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
                                     sizes,
                                     P_acc, op_acc, signed_acc,
                                     P_mul, op_mul, signed_mul,
                                     P_div, op_div, signed_div,
                                     subnormals);
}

void float_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                 int man_exp, int exp_exp,
                                                 int man_off, int exp_off,
                                                 int man_acc, int exp_acc,
                                                 bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_forward_fp_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                             man_exp, exp_exp,
                             man_off, exp_off,
                             man_acc, exp_acc,
                             subnormals, saturate);
}

void float_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                     int man_off, int exp_off,
                                                     int man_lse, int exp_lse,
                                                     bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward_fp_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                 man_off, exp_off,
                                 man_lse, exp_lse,
                                 subnormals, saturate);
}

void float_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                  int man_add, int exp_add,
                                                  int man_mul, int exp_mul,
                                                  bool subnormals, bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_backward_fp_nearest(a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
                              man_add, exp_add,
                              man_mul, exp_mul,
                              subnormals, saturate);
}

void superfp_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                   int man_exp, int exp_exp, int binades_exp_l, int binades_exp_u,
                                                   int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                                   int man_acc, int exp_acc, int binades_acc_l, int binades_acc_u,
                                                   bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_forward_superfp_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                  man_exp, exp_exp, binades_exp_l, binades_exp_u,
                                  man_off, exp_off, binades_off_l, binades_off_u,
                                  man_acc, exp_acc, binades_acc_l, binades_acc_u,
                                  saturate);
}

void superfp_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                       int man_off, int exp_off, int binades_off_l, int binades_off_u,
                                                       int man_lse, int exp_lse, int binades_lse_l, int binades_lse_u,
                                                       bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward_superfp_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                      man_off, exp_off, binades_off_l, binades_off_u,
                                      man_lse, exp_lse, binades_lse_l, binades_lse_u,
                                      saturate);
}

void superfp_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                    int man_add, int exp_add, int binades_add_l, int binades_add_u,
                                                    int man_mul, int exp_mul, int binades_mul_l, int binades_mul_u,
                                                    bool saturate)
{
  auto sizes = partition_tensor(a, dim);
  softmax_backward_superfp_nearest(a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                   man_add, exp_add, binades_add_l, binades_add_u,
                                   man_mul, exp_mul, binades_mul_l, binades_mul_u,
                                   saturate);
}

void binary8_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                   int P_exp, OverflowPolicy op_exp, bool signed_exp,
                                                   int P_off, OverflowPolicy op_off, bool signed_off,
                                                   int P_acc, OverflowPolicy op_acc, bool signed_acc,
                                                   bool subnormals)
{
  auto sizes = partition_tensor(a, dim);
  softmax_forward_binary8_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                  P_exp, op_exp, signed_exp,
                                  P_off, op_off, signed_off,
                                  P_acc, op_acc, signed_acc,
                                  subnormals);
}

void binary8_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                       int P_off, OverflowPolicy op_off, bool signed_off,
                                                       int P_lse, OverflowPolicy op_lse, bool signed_lse,
                                                       bool subnormals)
{
  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward_binary8_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                      P_off, op_off, signed_off,
                                      P_lse, op_lse, signed_lse,
                                      subnormals);
}

void binary8_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                    int P_add, OverflowPolicy op_add, bool signed_add,
                                                    int P_mul, OverflowPolicy op_mul, bool signed_mul,
                                                    bool subnormals)
{
  auto sizes = partition_tensor(a, dim);
  softmax_backward_binary8_nearest(a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                   P_add, op_add, signed_add,
                                   P_mul, op_mul, signed_mul,
                                   subnormals);
}