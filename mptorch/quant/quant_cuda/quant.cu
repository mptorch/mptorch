#include "quant.h"
#include "quant_kernel.h"
#include "modes.h"
#include <ATen/ATen.h>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <tuple>
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  float_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
      man_bits, exp_bits, saturate, subnormal_mode);
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  float_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
      man_bits, exp_bits, prng_bits, saturate, subnormal_mode);
  return o;
}

Tensor float_quantize_nearest_even_cuda(Tensor a,
                                        int man_bits, int exp_bits,
                                        bool subnormals, bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  if (a.scalar_type() == kFloat) {
      float_kernel_nearest_even<<<blockNums, blockSize>>>(
          a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
          saturate, subnormal_mode);
  } else if (a.scalar_type() == kHalf) {
      int packed_size = size / 8;
      int packed_blockNums = (packed_size + blockSize - 1) / blockSize;
      
      if (packed_size > 0) {
          fp16_kernel_nearest_even_packed<<<packed_blockNums, blockSize>>>(
              reinterpret_cast<const float4*>(a.data_ptr<at::Half>()), 
              reinterpret_cast<float4*>(o.data_ptr<at::Half>()), 
              packed_size, man_bits, exp_bits);
      }
      
      int remainder_size = size % 8;
      if (remainder_size > 0) {
          int offset = packed_size * 8;
          fp16_kernel_nearest_even_scalar<<<1, remainder_size>>>(
              reinterpret_cast<const __half*>(a.data_ptr<at::Half>() + offset), 
              reinterpret_cast<__half*>(o.data_ptr<at::Half>() + offset), 
              remainder_size, man_bits, exp_bits);
      }
  } else if (a.scalar_type() == kBFloat16) {
      int packed_size = size / 8;
      int packed_blockNums = (packed_size + blockSize - 1) / blockSize;
      
      if (packed_size > 0) {
          bfloat16_kernel_nearest_even_packed<<<packed_blockNums, blockSize>>>(
              reinterpret_cast<const float4*>(a.data_ptr<at::BFloat16>()), 
              reinterpret_cast<float4*>(o.data_ptr<at::BFloat16>()), 
              packed_size, man_bits, exp_bits);
      }
      
      int remainder_size = size % 8;
      if (remainder_size > 0) {
          int offset = packed_size * 8;
          bfloat16_kernel_nearest_even_scalar<<<1, remainder_size>>>(
              reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>() + offset), 
              reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>() + offset), 
              remainder_size, man_bits, exp_bits);
      }
  } else {
      TORCH_CHECK(false, "Unsupported scalar type for float_quantize_nearest_even_cuda");
  }
  return o;
}

Tensor float_quantize_nearest_away_cuda(Tensor a,
                                        int man_bits, int exp_bits,
                                        bool subnormals, bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  float_kernel_nearest_away<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
      saturate, subnormal_mode);
  return o;
}

Tensor float_quantize_up_cuda(Tensor a,
                              int man_bits, int exp_bits,
                              bool subnormals, bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  float_kernel_up<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
      saturate, subnormal_mode);
  return o;
}

Tensor float_quantize_down_cuda(Tensor a,
                                int man_bits, int exp_bits,
                                bool subnormals, bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  float_kernel_down<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
      saturate, subnormal_mode);
  return o;
}

Tensor float_quantize_zero_cuda(Tensor a,
                                int man_bits, int exp_bits,
                                bool subnormals, bool saturate)
{
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  float_kernel_zero<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
      saturate, subnormal_mode);
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

Tensor fp_quantize_cuda(Tensor a, int man_bits, int exp_bits, int bias, int prng_bits,
                        bool saturate, RoundMode round_mode, SubnormalsMode subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel();
  if (round_mode != RoundMode::SR)
  {
    fp_kernel(
        a.data_ptr<float>(), o.data_ptr<float>(), size,
        man_bits, exp_bits, bias, saturate, round_mode, subnormals);
  }
  else
  {
    auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
    fp_kernel(
        a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(),
        size, man_bits, exp_bits, bias, prng_bits, saturate, round_mode, subnormals);
  }

  return o;
}

Tensor binaryK_quantize_cuda(Tensor a, int K, int P, int bias, int prng_bits, bool is_signed,
                             RoundMode round_mode, SaturationMode saturation_mode,
                             SubnormalsMode subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel();
  if (round_mode != RoundMode::SR)
  {
    binaryK_kernel(
        a.data_ptr<float>(), o.data_ptr<float>(), size,
        K, P, bias, is_signed,
        round_mode, saturation_mode, subnormals);
  }
  else
  {
    auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
    binaryK_kernel(
        a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(),
        size, K, P, bias, prng_bits, is_signed,
        round_mode, saturation_mode, subnormals);
  }

  return o;
}

Tensor superfp_quantize_cuda(Tensor a, int man_bits, int exp_bits, int bias, int prng_bits,
                             int binades_l, int binades_h, bool saturate, RoundMode round_mode)
{
  auto o = zeros_like(a);
  int size = a.numel();
  if (round_mode != RoundMode::SR)
  {
    superfp_kernel(
        a.data_ptr<float>(), o.data_ptr<float>(), size,
        man_bits, exp_bits, bias, binades_l, binades_h,
        saturate, round_mode);
  }
  else
  {
    auto rand_ints = randint_like(a, INT_MAX, device(kCUDA).dtype(kInt));
    superfp_kernel(
        a.data_ptr<float>(), rand_ints.data_ptr<int>(), o.data_ptr<float>(), size,
        man_bits, exp_bits, prng_bits, bias, binades_l, binades_h,
        saturate, round_mode);
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  mm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                M, K, N, man_add, exp_add, man_mul, exp_mul,
                saturate, subnormal_mode, compensated);
  return;
}

void float_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_fma, int exp_fma,
                                        bool subnormals,
                                        bool saturate,
                                        bool compensated)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  mm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                    M, K, N, man_fma, exp_fma,
                    saturate, subnormal_mode, compensated);
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  if (a.sizes().size() > 2)
    bmm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                   a.sizes()[0], M, K, N,
                   man_add, exp_add, man_mul, exp_mul,
                   saturate, subnormal_mode, compensated);
  else
    bmm_fp_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                   1, M, K, N,
                   man_add, exp_add, man_mul, exp_mul,
                   saturate, subnormal_mode, compensated);
  return;
}

void float_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                         int M, int N, int K,
                                         int man_fma, int exp_fma,
                                         bool subnormals,
                                         bool saturate,
                                         bool compensated)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  if (a.sizes().size() > 2)
    bmm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                       a.sizes()[0], M, K, N,
                       man_fma, exp_fma,
                       saturate, subnormal_mode, compensated);
  else
    bmm_fp_fma_nearest(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                       1, M, K, N,
                       man_fma, exp_fma,
                       saturate, subnormal_mode, compensated);
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  mm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                   M, K, N,
                   man_add, exp_add, rbits_add,
                   man_mul, exp_mul, rbits_mul,
                   saturate, subnormal_mode);
  return;
}

void float_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma, int rbits_fma,
                                           bool subnormals, bool saturate)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  mm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                       M, K, N,
                       man_fma, exp_fma, rbits_fma,
                       saturate, subnormal_mode);
  return;
}

void float_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_add, int exp_add, int rbits_add,
                                        int man_mul, int exp_mul, int rbits_mul,
                                        bool subnormals, bool saturate)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  if (a.sizes().size() > 2)
    bmm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                      a.sizes()[0], M, K, N,
                      man_add, exp_add, rbits_add,
                      man_mul, exp_mul, rbits_mul,
                      saturate, subnormal_mode);
  else
    bmm_fp_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                      1, M, K, N,
                      man_add, exp_add, rbits_add,
                      man_mul, exp_mul, rbits_mul,
                      saturate, subnormal_mode);
}

void float_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                            int M, int N, int K,
                                            int man_fma, int exp_fma, int rbits_fma,
                                            bool subnormals, bool saturate)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  if (a.sizes().size() > 2)
    bmm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                          a.sizes()[0], M, K, N,
                          man_fma, exp_fma, rbits_fma,
                          saturate, subnormal_mode);
  else
    bmm_fp_fma_stochastic(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(),
                          1, M, K, N,
                          man_fma, exp_fma, rbits_fma,
                          saturate, subnormal_mode);
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  auto sizes = partition_tensor(input, dims);
  layernorm_forward_fp_nearest(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                               output.data_ptr<float>(), mean.data_ptr<float>(), rstd.data_ptr<float>(),
                               eps, sizes,
                               man_acc, exp_acc,
                               man_mul, exp_mul,
                               man_div, exp_div,
                               man_sqrt, exp_sqrt,
                               saturate, subnormal_mode);
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
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  auto sizes = partition_tensor(input, dims);
  layernorm_backward_fp_nearest(input.data_ptr<float>(), grad_output.data_ptr<float>(),
                                weight.data_ptr<float>(), bias.data_ptr<float>(),
                                mean.data_ptr<float>(), rstd.data_ptr<float>(),
                                grad_input.data_ptr<float>(), grad_gamma.data_ptr<float>(), grad_beta.data_ptr<float>(),
                                sizes,
                                man_acc, exp_acc,
                                man_mul, exp_mul,
                                man_div, exp_div,
                                saturate, subnormal_mode);
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

void float_quantize_nearest_softmax_forward_cuda(Tensor a, Tensor o, int dim,
                                                 int man_exp, int exp_exp,
                                                 int man_off, int exp_off,
                                                 int man_acc, int exp_acc,
                                                 bool subnormals, bool saturate)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  auto sizes = partition_tensor(a, dim);
  softmax_forward_fp_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                             man_exp, exp_exp,
                             man_off, exp_off,
                             man_acc, exp_acc,
                             saturate, subnormal_mode);
}

void float_quantize_nearest_softmax_lse_forward_cuda(Tensor a, Tensor o, int dim,
                                                     int man_off, int exp_off,
                                                     int man_lse, int exp_lse,
                                                     bool subnormals, bool saturate)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  auto sizes = partition_tensor(a, dim);
  softmax_lse_forward_fp_nearest(a.data_ptr<float>(), o.data_ptr<float>(), sizes,
                                 man_off, exp_off,
                                 man_lse, exp_lse,
                                 saturate, subnormal_mode);
}

void float_quantize_nearest_softmax_backward_cuda(Tensor a, Tensor g, Tensor o, int dim,
                                                  int man_add, int exp_add,
                                                  int man_mul, int exp_mul,
                                                  bool subnormals, bool saturate)
{
  SubnormalsMode subnormal_mode = subnormals ? SubnormalsMode::SUBNORMALS : SubnormalsMode::NORMALS;

  auto sizes = partition_tensor(a, dim);
  softmax_backward_fp_nearest(a.data_ptr<float>(), g.data_ptr<float>(), o.data_ptr<float>(), sizes,
                              man_add, exp_add,
                              man_mul, exp_mul,
                              saturate, subnormal_mode);
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