#include "quant.h"
#include "quant_kernel.h"
#include "mm_ops.h"
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
              packed_size, man_bits, exp_bits, saturate, subnormal_mode);
      }
      
      int remainder_size = size % 8;
      if (remainder_size > 0) {
          int offset = packed_size * 8;
          fp16_kernel_nearest_even_scalar<<<1, remainder_size>>>(
              reinterpret_cast<const __half*>(a.data_ptr<at::Half>() + offset), 
              reinterpret_cast<__half*>(o.data_ptr<at::Half>() + offset), 
              remainder_size, man_bits, exp_bits, saturate, subnormal_mode);
      }
  } else if (a.scalar_type() == kBFloat16) {
      int packed_size = size / 8;
      int packed_blockNums = (packed_size + blockSize - 1) / blockSize;
      
      if (packed_size > 0) {
          bfloat16_kernel_nearest_even_packed<<<packed_blockNums, blockSize>>>(
              reinterpret_cast<const float4*>(a.data_ptr<at::BFloat16>()), 
              reinterpret_cast<float4*>(o.data_ptr<at::BFloat16>()), 
              packed_size, man_bits, exp_bits, saturate, subnormal_mode);
      }
      
      int remainder_size = size % 8;
      if (remainder_size > 0) {
          int offset = packed_size * 8;
          bfloat16_kernel_nearest_even_scalar<<<1, remainder_size>>>(
              reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>() + offset), 
              reinterpret_cast<__nv_bfloat16*>(o.data_ptr<at::BFloat16>() + offset), 
              remainder_size, man_bits, exp_bits, saturate, subnormal_mode);
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

void fp_mm_cuda(Tensor out, Tensor a, Tensor b, int64_t man_add, int64_t exp_add,
                int64_t man_mul, int64_t exp_mul, int64_t man_fma, int64_t exp_fma,
                int64_t round_mode, int64_t subnormals_mode, bool saturate,
                bool compensated, bool use_fma, int64_t rbits_add, int64_t rbits_mul,
                int64_t rbits_fma)
{
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  MmDims dims = infer_mm_dims(a, b);
  SubnormalsMode subnormal_mode =
      static_cast<SubnormalsMode>(subnormals_mode);
  RoundMode round = static_cast<RoundMode>(round_mode);
  float *pa = a.data_ptr<float>();
  float *pb = b.data_ptr<float>();
  float *pc = out.data_ptr<float>();

  const int man_add_i = static_cast<int>(man_add);
  const int exp_add_i = static_cast<int>(exp_add);
  const int man_mul_i = static_cast<int>(man_mul);
  const int exp_mul_i = static_cast<int>(exp_mul);
  const int man_fma_i = static_cast<int>(man_fma);
  const int exp_fma_i = static_cast<int>(exp_fma);
  const int rbits_add_i = static_cast<int>(rbits_add);
  const int rbits_mul_i = static_cast<int>(rbits_mul);
  const int rbits_fma_i = static_cast<int>(rbits_fma);

  if (round == RoundMode::SR)
  {
    if (use_fma)
    {
      if (dims.batch > 1)
        bmm_fp_fma_stochastic(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                              man_fma_i, exp_fma_i, rbits_fma_i, saturate,
                              subnormal_mode);
      else
        mm_fp_fma_stochastic(pa, pb, pc, dims.M, dims.K, dims.N, man_fma_i,
                             exp_fma_i, rbits_fma_i, saturate, subnormal_mode);
    }
    else if (dims.batch > 1)
    {
      bmm_fp_stochastic(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                        man_add_i, exp_add_i, rbits_add_i, man_mul_i,
                        exp_mul_i, rbits_mul_i, saturate, subnormal_mode);
    }
    else
    {
      mm_fp_stochastic(pa, pb, pc, dims.M, dims.K, dims.N, man_add_i,
                       exp_add_i, rbits_add_i, man_mul_i, exp_mul_i,
                       rbits_mul_i, saturate, subnormal_mode);
    }
  }
  else if (use_fma)
  {
    if (dims.batch > 1)
      bmm_fp_fma_nearest(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                         man_fma_i, exp_fma_i, saturate, subnormal_mode,
                         compensated);
    else
      mm_fp_fma_nearest(pa, pb, pc, dims.M, dims.K, dims.N, man_fma_i,
                        exp_fma_i, saturate, subnormal_mode, compensated);
  }
  else if (dims.batch > 1)
  {
    bmm_fp_nearest(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N, man_add_i,
                   exp_add_i, man_mul_i, exp_mul_i, saturate, subnormal_mode,
                   compensated);
  }
  else
  {
    mm_fp_nearest(pa, pb, pc, dims.M, dims.K, dims.N, man_add_i, exp_add_i,
                  man_mul_i, exp_mul_i, saturate, subnormal_mode, compensated);
  }
}

void superfp_mm_cuda(Tensor out, Tensor a, Tensor b, int64_t man_add,
                     int64_t exp_add, int64_t man_mul, int64_t exp_mul,
                     int64_t man_fma, int64_t exp_fma, int64_t binades_add_l,
                     int64_t binades_add_u, int64_t binades_mul_l,
                     int64_t binades_mul_u, int64_t binades_fma_l,
                     int64_t binades_fma_u, bool saturate, bool use_fma)
{
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  MmDims dims = infer_mm_dims(a, b);
  float *pa = a.data_ptr<float>();
  float *pb = b.data_ptr<float>();
  float *pc = out.data_ptr<float>();
  const int man_add_i = static_cast<int>(man_add);
  const int exp_add_i = static_cast<int>(exp_add);
  const int man_mul_i = static_cast<int>(man_mul);
  const int exp_mul_i = static_cast<int>(exp_mul);
  const int man_fma_i = static_cast<int>(man_fma);
  const int exp_fma_i = static_cast<int>(exp_fma);
  const int binades_add_l_i = static_cast<int>(binades_add_l);
  const int binades_add_u_i = static_cast<int>(binades_add_u);
  const int binades_mul_l_i = static_cast<int>(binades_mul_l);
  const int binades_mul_u_i = static_cast<int>(binades_mul_u);
  const int binades_fma_l_i = static_cast<int>(binades_fma_l);
  const int binades_fma_u_i = static_cast<int>(binades_fma_u);

  if (use_fma)
  {
    if (dims.batch > 1)
      bmm_superfp_fma_nearest(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                              man_fma_i, exp_fma_i, binades_fma_l_i,
                              binades_fma_u_i, saturate);
    else
      mm_superfp_fma_nearest(pa, pb, pc, dims.M, dims.K, dims.N, man_fma_i,
                             exp_fma_i, binades_fma_l_i, binades_fma_u_i,
                             saturate);
  }
  else if (dims.batch > 1)
  {
    bmm_superfp_nearest(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                        man_add_i, exp_add_i, man_mul_i, exp_mul_i,
                        binades_add_l_i, binades_add_u_i, binades_mul_l_i,
                        binades_mul_u_i, saturate);
  }
  else
  {
    mm_superfp_nearest(pa, pb, pc, dims.M, dims.K, dims.N, man_add_i,
                       exp_add_i, man_mul_i, exp_mul_i, binades_add_l_i,
                       binades_add_u_i, binades_mul_l_i, binades_mul_u_i,
                       saturate);
  }
}

void fxp_mm_cuda(Tensor out, Tensor a, Tensor b, int64_t wl_add, int64_t fl_add,
                 int64_t wl_mul, int64_t fl_mul, int64_t wl_fma, int64_t fl_fma,
                 int64_t round_mode, bool symmetric, bool use_fma)
{
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  MmDims dims = infer_mm_dims(a, b);
  RoundMode round = static_cast<RoundMode>(round_mode);
  float *pa = a.data_ptr<float>();
  float *pb = b.data_ptr<float>();
  float *pc = out.data_ptr<float>();
  const int wl_add_i = static_cast<int>(wl_add);
  const int fl_add_i = static_cast<int>(fl_add);
  const int wl_mul_i = static_cast<int>(wl_mul);
  const int fl_mul_i = static_cast<int>(fl_mul);
  const int wl_fma_i = static_cast<int>(wl_fma);
  const int fl_fma_i = static_cast<int>(fl_fma);

  if (use_fma)
  {
    int sigma_fma = -fl_fma_i;
    float t_min_fma, t_max_fma;
    fixed_min_max(wl_fma_i, fl_fma_i, symmetric, &t_min_fma, &t_max_fma);
    if (round == RoundMode::SR)
    {
      if (dims.batch > 1)
        bmm_fxp_fma_stochastic(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                               sigma_fma, t_min_fma, t_max_fma);
      else
        mm_fxp_fma_stochastic(pa, pb, pc, dims.M, dims.K, dims.N, sigma_fma,
                              t_min_fma, t_max_fma);
    }
    else if (dims.batch > 1)
    {
      bmm_fxp_fma_nearest(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                          sigma_fma, t_min_fma, t_max_fma);
    }
    else
    {
      mm_fxp_fma_nearest(pa, pb, pc, dims.M, dims.K, dims.N, sigma_fma,
                         t_min_fma, t_max_fma);
    }
  }
  else
  {
    int sigma_add = -fl_add_i;
    int sigma_mul = -fl_mul_i;
    float t_min_add, t_max_add, t_min_mul, t_max_mul;
    fixed_min_max(wl_add_i, fl_add_i, symmetric, &t_min_add, &t_max_add);
    fixed_min_max(wl_mul_i, fl_mul_i, symmetric, &t_min_mul, &t_max_mul);
    if (round == RoundMode::SR)
    {
      if (dims.batch > 1)
        bmm_fxp_stochastic(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N,
                           sigma_add, t_min_add, t_max_add, sigma_mul,
                           t_min_mul, t_max_mul);
      else
        mm_fxp_stochastic(pa, pb, pc, dims.M, dims.K, dims.N, sigma_add,
                          t_min_add, t_max_add, sigma_mul, t_min_mul,
                          t_max_mul);
    }
    else if (dims.batch > 1)
    {
      bmm_fxp_nearest(pa, pb, pc, dims.batch, dims.M, dims.K, dims.N, sigma_add,
                      t_min_add, t_max_add, sigma_mul, t_min_mul, t_max_mul);
    }
    else
    {
      mm_fxp_nearest(pa, pb, pc, dims.M, dims.K, dims.N, sigma_add, t_min_add,
                     t_max_add, sigma_mul, t_min_mul, t_max_mul);
    }
  }
}

void float_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                    int M, int N, int K,
                                    int man_add, int exp_add,
                                    int man_mul, int exp_mul,
                                    bool subnormals,
                                    bool saturate,
                                    bool compensated)
{
  (void)M;
  (void)N;
  (void)K;
  int64_t subnormal_mode = subnormals
                               ? static_cast<int64_t>(SubnormalsMode::SUBNORMALS)
                               : static_cast<int64_t>(SubnormalsMode::NORMALS);
  fp_mm_cuda(c, a, b, man_add, exp_add, man_mul, exp_mul, man_add, exp_add,
             static_cast<int64_t>(RoundMode::RNE), subnormal_mode, saturate,
             compensated, false, 0, 0, 0);
}

void float_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_fma, int exp_fma,
                                        bool subnormals,
                                        bool saturate,
                                        bool compensated)
{
  (void)M;
  (void)N;
  (void)K;
  int64_t subnormal_mode = subnormals
                               ? static_cast<int64_t>(SubnormalsMode::SUBNORMALS)
                               : static_cast<int64_t>(SubnormalsMode::NORMALS);
  fp_mm_cuda(c, a, b, man_fma, exp_fma, man_fma, exp_fma, man_fma, exp_fma,
             static_cast<int64_t>(RoundMode::RNE), subnormal_mode, saturate,
             compensated, true, 0, 0, 0);
}

void float_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                     int M, int N, int K,
                                     int man_add, int exp_add,
                                     int man_mul, int exp_mul,
                                     bool subnormals,
                                     bool saturate,
                                     bool compensated)
{
  float_quantize_nearest_mm_cuda(a, b, c, M, N, K, man_add, exp_add, man_mul,
                                 exp_mul, subnormals, saturate, compensated);
}

void float_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                         int M, int N, int K,
                                         int man_fma, int exp_fma,
                                         bool subnormals,
                                         bool saturate,
                                         bool compensated)
{
  float_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                     subnormals, saturate, compensated);
}

void superfp_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                      int M, int N, int K,
                                      int man_add, int exp_add,
                                      int man_mul, int exp_mul,
                                      int binades_add_l, int binades_add_u,
                                      int binades_mul_l, int binades_mul_u,
                                      bool saturate)
{
  (void)M;
  (void)N;
  (void)K;
  superfp_mm_cuda(c, a, b, man_add, exp_add, man_mul, exp_mul, man_add,
                  exp_add, binades_add_l, binades_add_u, binades_mul_l,
                  binades_mul_u, binades_add_l, binades_add_u, saturate, false);
}

void superfp_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                          int M, int N, int K,
                                          int man_fma, int exp_fma,
                                          int binades_fma_l, int binades_fma_u,
                                          bool saturate)
{
  (void)M;
  (void)N;
  (void)K;
  superfp_mm_cuda(c, a, b, man_fma, exp_fma, man_fma, exp_fma, man_fma,
                  exp_fma, binades_fma_l, binades_fma_u, binades_fma_l,
                  binades_fma_u, binades_fma_l, binades_fma_u, saturate, true);
}

void superfp_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_add, int exp_add,
                                       int man_mul, int exp_mul,
                                       int binades_add_l, int binades_add_u,
                                       int binades_mul_l, int binades_mul_u,
                                       bool saturate)
{
  superfp_quantize_nearest_mm_cuda(a, b, c, M, N, K, man_add, exp_add, man_mul,
                                   exp_mul, binades_add_l, binades_add_u,
                                   binades_mul_l, binades_mul_u, saturate);
}

void superfp_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma,
                                           int binades_fma_l, int binades_fma_u,
                                           bool saturate)
{
  superfp_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                       binades_fma_l, binades_fma_u, saturate);
}

void float_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                       int M, int N, int K,
                                       int man_add, int exp_add, int rbits_add,
                                       int man_mul, int exp_mul, int rbits_mul,
                                       bool subnormals, bool saturate)
{
  (void)M;
  (void)N;
  (void)K;
  int64_t subnormal_mode = subnormals
                               ? static_cast<int64_t>(SubnormalsMode::SUBNORMALS)
                               : static_cast<int64_t>(SubnormalsMode::NORMALS);
  fp_mm_cuda(c, a, b, man_add, exp_add, man_mul, exp_mul, man_add, exp_add,
             static_cast<int64_t>(RoundMode::SR), subnormal_mode, saturate,
             false, false, rbits_add, rbits_mul, 0);
}

void float_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int man_fma, int exp_fma, int rbits_fma,
                                           bool subnormals, bool saturate)
{
  (void)M;
  (void)N;
  (void)K;
  int64_t subnormal_mode = subnormals
                               ? static_cast<int64_t>(SubnormalsMode::SUBNORMALS)
                               : static_cast<int64_t>(SubnormalsMode::NORMALS);
  fp_mm_cuda(c, a, b, man_fma, exp_fma, man_fma, exp_fma, man_fma, exp_fma,
             static_cast<int64_t>(RoundMode::SR), subnormal_mode, saturate,
             false, true, 0, 0, rbits_fma);
}

void float_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                        int M, int N, int K,
                                        int man_add, int exp_add, int rbits_add,
                                        int man_mul, int exp_mul, int rbits_mul,
                                        bool subnormals, bool saturate)
{
  float_quantize_stochastic_mm_cuda(a, b, c, M, N, K, man_add, exp_add,
                                    rbits_add, man_mul, exp_mul, rbits_mul,
                                    subnormals, saturate);
}

void float_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                            int M, int N, int K,
                                            int man_fma, int exp_fma, int rbits_fma,
                                            bool subnormals, bool saturate)
{
  float_quantize_stochastic_mm_fma_cuda(a, b, c, M, N, K, man_fma, exp_fma,
                                        rbits_fma, subnormals, saturate);
}

void fixed_point_quantize_nearest_mm_cuda(Tensor a, Tensor b, Tensor c,
                                          int M, int N, int K,
                                          int wl_add, int fl_add,
                                          int wl_mul, int fl_mul,
                                          bool symmetric)
{
  (void)M;
  (void)N;
  (void)K;
  fxp_mm_cuda(c, a, b, wl_add, fl_add, wl_mul, fl_mul, wl_add, fl_add,
              static_cast<int64_t>(RoundMode::RNE), symmetric, false);
}

void fixed_point_quantize_nearest_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                           int M, int N, int K,
                                           int wl_add, int fl_add,
                                           int wl_mul, int fl_mul,
                                           bool symmetric)
{
  fixed_point_quantize_nearest_mm_cuda(a, b, c, M, N, K, wl_add, fl_add,
                                       wl_mul, fl_mul, symmetric);
}

void fixed_point_quantize_nearest_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K,
                                              int wl_fma, int fl_fma,
                                              bool symmetric)
{
  (void)M;
  (void)N;
  (void)K;
  fxp_mm_cuda(c, a, b, wl_fma, fl_fma, wl_fma, fl_fma, wl_fma, fl_fma,
              static_cast<int64_t>(RoundMode::RNE), symmetric, true);
}

void fixed_point_quantize_nearest_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                               int M, int N, int K,
                                               int wl_fma, int fl_fma,
                                               bool symmetric)
{
  fixed_point_quantize_nearest_mm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                           symmetric);
}

void fixed_point_quantize_stochastic_mm_cuda(Tensor a, Tensor b, Tensor c,
                                             int M, int N, int K,
                                             int wl_add, int fl_add,
                                             int wl_mul, int fl_mul,
                                             bool symmetric)
{
  (void)M;
  (void)N;
  (void)K;
  fxp_mm_cuda(c, a, b, wl_add, fl_add, wl_mul, fl_mul, wl_add, fl_add,
              static_cast<int64_t>(RoundMode::SR), symmetric, false);
}

void fixed_point_quantize_stochastic_bmm_cuda(Tensor a, Tensor b, Tensor c,
                                              int M, int N, int K,
                                              int wl_add, int fl_add,
                                              int wl_mul, int fl_mul,
                                              bool symmetric)
{
  fixed_point_quantize_stochastic_mm_cuda(a, b, c, M, N, K, wl_add, fl_add,
                                          wl_mul, fl_mul, symmetric);
}

void fixed_point_quantize_stochastic_mm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                 int M, int N, int K,
                                                 int wl_fma, int fl_fma,
                                                 bool symmetric)
{
  (void)M;
  (void)N;
  (void)K;
  fxp_mm_cuda(c, a, b, wl_fma, fl_fma, wl_fma, fl_fma, wl_fma, fl_fma,
              static_cast<int64_t>(RoundMode::SR), symmetric, true);
}

void fixed_point_quantize_stochastic_bmm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                  int M, int N, int K,
                                                  int wl_fma, int fl_fma,
                                                  bool symmetric)
{
  fixed_point_quantize_stochastic_mm_fma_cuda(a, b, c, M, N, K, wl_fma, fl_fma,
                                              symmetric);
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