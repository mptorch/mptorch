#include "quant_cuda.h"
#include "quant_kernel.h"
#include <ATen/ATen.h>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <tuple>

using namespace at;

Tensor get_max_entry(Tensor a, int dim) {
  Tensor max_entry;
  if (dim == -1) {
    max_entry = at::max(at::abs(a)).expand_as(a).contiguous();
  } else if (dim == 0) {
    Tensor input_view = a.view({a.size(0), -1});
    max_entry = std::get<0>(input_view.abs().max(1, true))
                    .expand_as(input_view)
                    .view_as(a)
                    .contiguous();
  } else {
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

Tensor block_quantize_stochastic_cuda(Tensor a, int wl, int dim) {
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

Tensor block_quantize_nearest_cuda(Tensor a, int wl, int dim) {
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

Tensor block_quantize_sim_stochastic_cuda(Tensor a, int wl) {
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

Tensor block_quantize_sim_nearest_cuda(Tensor a, int wl) {
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
                                      bool subnormals, bool saturate) {
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
                                   bool subnormals, bool saturate) {
  // use external random number right now
  auto o = zeros_like(a);
  int size = a.numel();
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  float_kernel_nearest<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, man_bits, exp_bits,
      subnormals, saturate);
  return o;
}

void fixed_min_max(int wl, int fl, bool symmetric, float *t_min, float *t_max) {
  int sigma = -fl;
  *t_min = -ldexp(1.0, wl - fl - 1);
  *t_max = -*t_min - ldexp(1.0, sigma);
  if (symmetric)
    *t_min = *t_min + ldexp(1.0, sigma);
}

Tensor fixed_point_quantize_stochastic_cuda(Tensor a, int wl, int fl,
                                            bool use_clamp, bool symmetric) {
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
                                         bool use_clamp, bool symmetric) {
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
                                          bool symmetric) {
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
                                       bool symmetric) {
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

__global__ void init(unsigned int seed, curandState_t *state) {
  curand_init(seed, blockIdx.x * blockIdx.y, 0,
              &state[blockIdx.x * blockIdx.y]);
}

void float_quantize_nearest_gemm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                      int N, int K, int man_add, int exp_add,
                                      int man_mul, int exp_mul, bool subnormals,
                                      bool saturate) {

  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  gemm_fp_nearest<<<blocks, threads>>>(
      a.data<float>(), b.data<float>(), c.data<float>(), M, K, N, man_add,
      exp_add, man_mul, exp_mul, subnormals, saturate);

  return;
}

void float_quantize_nearest_gemm_fma_cuda(Tensor a, Tensor b, Tensor c, int M,
                                          int N, int K, int man_fma,
                                          int exp_fma, bool subnormals,
                                          bool saturate) {

  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  gemm_fp_fma_nearest<<<blocks, threads>>>(a.data<float>(), b.data<float>(),
                                           c.data<float>(), M, K, N, man_fma,
                                           exp_fma, subnormals, saturate);

  return;
}

void float_quantize_stochastic_gemm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                         int N, int K, int man_add, int exp_add,
                                         int man_mul, int exp_mul,
                                         bool subnormals, bool saturate) {

  // auto rand_ints = randint(INT_MAX, {(M + 8 - M % 8) * (N + 8 - N % 8) * (K +
  // 8 - K % 8) * 2},
  //                    device(kCUDA).dtype(kInt));
  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  curandState_t *state;
  cudaMalloc((void **)&state, blocks.x * blocks.y * sizeof(curandState_t));
  init<<<blocks, 1>>>(time(0), state);
  gemm_fp_stochastic<<<blocks, threads>>>(
      a.data<float>(), b.data<float>(), c.data<float>(),
      state, // rand_ints.data<int>(),
      M, K, N, man_add, exp_add, man_mul, exp_mul, subnormals, saturate);
  cudaFree(state);
  return;
}

void float_quantize_stochastic_gemm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                             int M, int N, int K, int man_fma,
                                             int exp_fma, bool subnormals,
                                             bool saturate) {

  auto rand_ints =
      randint(INT_MAX, {(M + 8 - M % 8) * (N + 8 - N % 8) * (K + 8 - K % 8)},
              device(kCUDA).dtype(kInt));
  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  curandState_t *state;
  cudaMalloc((void **)&state, blocks.x * blocks.y * sizeof(curandState_t));
  init<<<blocks, 1>>>(time(0), state);
  gemm_fp_fma_stochastic<<<blocks, threads>>>(
      a.data<float>(), b.data<float>(), c.data<float>(),
      state, // rand_ints.data<int>(),
      M, K, N, man_fma, exp_fma, subnormals, saturate);
  cudaFree(state);
  return;
}

void fixed_point_quantize_nearest_gemm_cuda(Tensor a, Tensor b, Tensor c, int M,
                                            int N, int K, int wl_add,
                                            int fl_add, int wl_mul, int fl_mul,
                                            bool symmetric) {
  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  int sigma_add = -fl_add;
  int sigma_mul = -fl_mul;
  float t_min_add, t_max_add, t_min_mul, t_max_mul;
  fixed_min_max(wl_add, fl_add, symmetric, &t_min_add, &t_max_add);
  fixed_min_max(wl_mul, fl_mul, symmetric, &t_min_mul, &t_max_mul);
  gemm_fxp_nearest<<<blocks, threads>>>(
      a.data<float>(), b.data<float>(), c.data<float>(), M, K, N, sigma_add,
      t_min_add, t_max_add, sigma_mul, t_min_mul, t_max_mul);
  return;
}

void fixed_point_quantize_nearest_gemm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                int M, int N, int K, int wl_fma,
                                                int fl_fma, bool symmetric) {
  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  int sigma_fma = -fl_fma;
  float t_min_fma, t_max_fma;
  fixed_min_max(wl_fma, fl_fma, symmetric, &t_min_fma, &t_max_fma);
  gemm_fxp_fma_nearest<<<blocks, threads>>>(a.data<float>(), b.data<float>(),
                                            c.data<float>(), M, K, N, sigma_fma,
                                            t_min_fma, t_max_fma);
  return;
}

void fixed_point_quantize_stochastic_gemm_cuda(Tensor a, Tensor b, Tensor c,
                                               int M, int N, int K, int wl_add,
                                               int fl_add, int wl_mul,
                                               int fl_mul, bool symmetric) {
  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  int sigma_add = -fl_add;
  int sigma_mul = -fl_mul;
  float t_min_add, t_max_add, t_min_mul, t_max_mul;
  fixed_min_max(wl_add, fl_add, symmetric, &t_min_add, &t_max_add);
  fixed_min_max(wl_mul, fl_mul, symmetric, &t_min_mul, &t_max_mul);
  // auto rand_probs = at::rand({(M + 8 - M % 8), (N + 8 - N % 8), (K + 8 - K %
  // 8) * 2},
  //                     device(kCUDA).dtype(kFloat));
  curandState_t *state;
  cudaMalloc((void **)&state, blocks.x * blocks.y * sizeof(curandState_t));
  // TODO: change this to a fixed seed?!
  init<<<blocks, 1>>>(time(0), state);
  gemm_fxp_stochastic<<<blocks, threads>>>(
      a.data<float>(), b.data<float>(),
      c.data<float>(), // rand_probs.data<float>(), M, K, N,
      state, M, K, N, sigma_add, t_min_add, t_max_add, sigma_mul, t_min_mul,
      t_max_mul);
  cudaFree(state);
  return;
}

void fixed_point_quantize_stochastic_gemm_fma_cuda(Tensor a, Tensor b, Tensor c,
                                                   int M, int N, int K,
                                                   int wl_fma, int fl_fma,
                                                   bool symmetric) {
  dim3 threads(8, 8);
  dim3 blocks((N + 8 - N % 8) / 8, (M + 8 - M % 8) / 8);
  int sigma_fma = -fl_fma;
  float t_min_fma, t_max_fma;
  fixed_min_max(wl_fma, fl_fma, symmetric, &t_min_fma, &t_max_fma);
  // auto rand_probs = at::rand({(M + 8 - M % 8) * (N + 8 - N % 8) * (K + 8 - K
  // % 8)},
  //                     device(kCUDA).dtype(kFloat));
  curandState_t *state;
  cudaMalloc((void **)&state, blocks.x * blocks.y * sizeof(curandState_t));
  init<<<blocks, 1>>>(time(0), state);
  gemm_fxp_fma_stochastic<<<blocks, threads>>>(
      a.data<float>(), b.data<float>(),
      c.data<float>(), // rand_probs.data<float>(), M, K, N,
      state, M, K, N, sigma_fma, t_min_fma, t_max_fma);
  cudaFree(state);
  return;
}