#pragma once

#include <ATen/ATen.h>

using namespace at;

struct MmDims
{
  int batch;
  int M;
  int K;
  int N;
};

MmDims infer_mm_dims(const Tensor &a, const Tensor &b);

void fp_mm_cpu(Tensor out, Tensor a, Tensor b, int64_t man_add, int64_t exp_add,
               int64_t man_mul, int64_t exp_mul, int64_t man_fma, int64_t exp_fma,
               int64_t round_mode, int64_t subnormals_mode, bool saturate,
               bool compensated, bool use_fma, int64_t rbits_add,
               int64_t rbits_mul, int64_t rbits_fma);

void superfp_mm_cpu(Tensor out, Tensor a, Tensor b, int64_t man_add,
                    int64_t exp_add, int64_t man_mul, int64_t exp_mul,
                    int64_t man_fma, int64_t exp_fma, int64_t binades_add_l,
                    int64_t binades_add_u, int64_t binades_mul_l,
                    int64_t binades_mul_u, int64_t binades_fma_l,
                    int64_t binades_fma_u, bool saturate, bool use_fma);

void fxp_mm_cpu(Tensor out, Tensor a, Tensor b, int64_t wl_add, int64_t fl_add,
                int64_t wl_mul, int64_t fl_mul, int64_t wl_fma, int64_t fl_fma,
                int64_t round_mode, bool symmetric, bool use_fma);

void fp_mm_cuda(Tensor out, Tensor a, Tensor b, int64_t man_add, int64_t exp_add,
                int64_t man_mul, int64_t exp_mul, int64_t man_fma, int64_t exp_fma,
                int64_t round_mode, int64_t subnormals_mode, bool saturate,
                bool compensated, bool use_fma, int64_t rbits_add,
                int64_t rbits_mul, int64_t rbits_fma);

void superfp_mm_cuda(Tensor out, Tensor a, Tensor b, int64_t man_add,
                     int64_t exp_add, int64_t man_mul, int64_t exp_mul,
                     int64_t man_fma, int64_t exp_fma, int64_t binades_add_l,
                     int64_t binades_add_u, int64_t binades_mul_l,
                     int64_t binades_mul_u, int64_t binades_fma_l,
                     int64_t binades_fma_u, bool saturate, bool use_fma);

void fxp_mm_cuda(Tensor out, Tensor a, Tensor b, int64_t wl_add, int64_t fl_add,
                 int64_t wl_mul, int64_t fl_mul, int64_t wl_fma, int64_t fl_fma,
                 int64_t round_mode, bool symmetric, bool use_fma);
