#include "../mm_ops.h"
#include "fp_kernel.h"
#include "mm_dispatch.h"
#include "modes.h"
#include "quant.h"
#include <torch/library.h>

void fp_mm_cpu(Tensor out, Tensor a, Tensor b, int64_t man_add, int64_t exp_add,
               int64_t man_mul, int64_t exp_mul, int64_t man_fma, int64_t exp_fma,
               int64_t round_mode, int64_t subnormals_mode, bool saturate,
               bool compensated, bool use_fma, int64_t rbits_add, int64_t rbits_mul,
               int64_t rbits_fma)
{
  TORCH_CHECK(!out.is_cuda(), "out must be a CPU tensor");
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
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
  const int bias_add = (1 << (exp_add_i - 1)) - 1;
  const int bias_mul = (1 << (exp_mul_i - 1)) - 1;
  const int bias_fma = (1 << (exp_fma_i - 1)) - 1;

  if (round == RoundMode::SR)
  {
    if (use_fma)
    {
      launch_mm_fma(
          pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
          [man_fma_i, exp_fma_i, rbits_fma_i, bias_fma, saturate,
           subnormal_mode](float x)
          {
            return cast_fp_stochastic(x, man_fma_i, exp_fma_i, rbits_fma_i,
                                      bias_fma, saturate, subnormal_mode);
          });
    }
    else
    {
      launch_mm_addmul(
          pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
          [man_add_i, exp_add_i, rbits_add_i, bias_add, saturate,
           subnormal_mode](float x)
          {
            return cast_fp_stochastic(x, man_add_i, exp_add_i, rbits_add_i,
                                      bias_add, saturate, subnormal_mode);
          },
          [man_mul_i, exp_mul_i, rbits_mul_i, bias_mul, saturate,
           subnormal_mode](float x)
          {
            return cast_fp_stochastic(x, man_mul_i, exp_mul_i, rbits_mul_i,
                                      bias_mul, saturate, subnormal_mode);
          });
    }
  }
  else if (use_fma)
  {
    launch_mm_fma(
        pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, compensated,
        [man_fma_i, exp_fma_i, bias_fma, saturate, subnormal_mode](float x)
        {
          return cast_fp_nearest_even(x, man_fma_i, exp_fma_i, bias_fma,
                                      saturate, subnormal_mode);
        });
  }
  else
  {
    launch_mm_addmul(
        pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, compensated,
        [man_add_i, exp_add_i, bias_add, saturate, subnormal_mode](float x)
        {
          return cast_fp_nearest_even(x, man_add_i, exp_add_i, bias_add,
                                      saturate, subnormal_mode);
        },
        [man_mul_i, exp_mul_i, bias_mul, saturate, subnormal_mode](float x)
        {
          return cast_fp_nearest_even(x, man_mul_i, exp_mul_i, bias_mul,
                                      saturate, subnormal_mode);
        });
  }
}

void superfp_mm_cpu(Tensor out, Tensor a, Tensor b, int64_t man_add,
                    int64_t exp_add, int64_t man_mul, int64_t exp_mul,
                    int64_t man_fma, int64_t exp_fma, int64_t binades_add_l,
                    int64_t binades_add_u, int64_t binades_mul_l,
                    int64_t binades_mul_u, int64_t binades_fma_l,
                    int64_t binades_fma_u, bool saturate, bool use_fma)
{
  TORCH_CHECK(!out.is_cuda(), "out must be a CPU tensor");
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
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
    launch_mm_fma(
        pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
        [man_fma_i, exp_fma_i, binades_fma_l_i, binades_fma_u_i,
         saturate](float x)
        {
          return cast_superfp_nearest(x, man_fma_i, exp_fma_i, binades_fma_l_i,
                                      binades_fma_u_i, saturate);
        });
  }
  else
  {
    launch_mm_addmul(
        pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
        [man_add_i, exp_add_i, binades_add_l_i, binades_add_u_i,
         saturate](float x)
        {
          return cast_superfp_nearest(x, man_add_i, exp_add_i, binades_add_l_i,
                                      binades_add_u_i, saturate);
        },
        [man_mul_i, exp_mul_i, binades_mul_l_i, binades_mul_u_i,
         saturate](float x)
        {
          return cast_superfp_nearest(x, man_mul_i, exp_mul_i, binades_mul_l_i,
                                      binades_mul_u_i, saturate);
        });
  }
}

void fxp_mm_cpu(Tensor out, Tensor a, Tensor b, int64_t wl_add, int64_t fl_add,
                int64_t wl_mul, int64_t fl_mul, int64_t wl_fma, int64_t fl_fma,
                int64_t round_mode, bool symmetric, bool use_fma)
{
  TORCH_CHECK(!out.is_cuda(), "out must be a CPU tensor");
  TORCH_CHECK(!a.is_cuda(), "a must be a CPU tensor");
  TORCH_CHECK(!b.is_cuda(), "b must be a CPU tensor");
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
      launch_mm_fma(
          pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
          [sigma_fma, t_min_fma, t_max_fma](float x)
          {
            return cast_fxp_stochastic(x, sigma_fma, t_min_fma, t_max_fma);
          });
    }
    else
    {
      launch_mm_fma(
          pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
          [sigma_fma, t_min_fma, t_max_fma](float x)
          {
            return cast_fxp_nearest(x, sigma_fma, t_min_fma, t_max_fma);
          });
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
      launch_mm_addmul(
          pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
          [sigma_add, t_min_add, t_max_add](float x)
          {
            return cast_fxp_stochastic(x, sigma_add, t_min_add, t_max_add);
          },
          [sigma_mul, t_min_mul, t_max_mul](float x)
          {
            return cast_fxp_stochastic(x, sigma_mul, t_min_mul, t_max_mul);
          });
    }
    else
    {
      launch_mm_addmul(
          pa, pb, pc, dims.M, dims.K, dims.N, dims.batch, false,
          [sigma_add, t_min_add, t_max_add](float x)
          {
            return cast_fxp_nearest(x, sigma_add, t_min_add, t_max_add);
          },
          [sigma_mul, t_min_mul, t_max_mul](float x)
          {
            return cast_fxp_nearest(x, sigma_mul, t_min_mul, t_max_mul);
          });
    }
  }
}

TORCH_LIBRARY_IMPL(mptorch, CPU, m)
{
  m.impl("fp_mm", TORCH_FN(fp_mm_cpu));
  m.impl("superfp_mm", TORCH_FN(superfp_mm_cpu));
  m.impl("fxp_mm", TORCH_FN(fxp_mm_cpu));
}
