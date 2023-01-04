#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>

__device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
                                 bool subnormal_support = true,
                                 bool saturate = false) {
  unsigned int target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23);

  if (noquantize) {
    quantized = origin_float;
  } else {
    if (subnormal && subnormal_support) {
      float shift_float, val;
      int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val = origin_float + shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    } else {
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits =
          clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

__device__ float cast_fp_stochastic(float origin_float, unsigned int rand_prob,
                                    int man_bits, int exp_bits,
                                    bool subnormal_support = true,
                                    bool saturate = false) {
  unsigned int target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);

  if (subnormal && subnormal_support) {
    float shift_float, val;
    int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
    shift_float = BITS_TO_FLOAT(&shift_bits);
    val = origin_float + shift_float;
    target = FLOAT_TO_BITS(&val);
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
  } else {
    quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
    quantize_bits =
        clip_exponent(exp_bits, man_bits, target, quantize_bits, saturate);
    quantized = BITS_TO_FLOAT(&quantize_bits);
  }

  return quantized;
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits,
                                        bool subnormal_support, bool saturate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_stochastic(a[index], (unsigned int)r[index], man_bits,
                                  exp_bits, subnormal_support, saturate);
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     int man_bits, int exp_bits,
                                     bool subnormal_support, bool saturate) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size)
    o[index] = cast_fp_nearest(a[index], man_bits, exp_bits, subnormal_support,
                               saturate);
}

__global__ void gemm_fp_nearest(float *__restrict__ a, float *__restrict__ b,
                                float *__restrict__ c, int M, int K, int N,
                                int man_add, int exp_add, int man_mul,
                                int exp_mul, bool subnormal_support,
                                bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0f;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_fp_nearest(tmp + cast_fp_nearest(s_a[ty * blockDim.x + j] *
                                                      s_b[j * blockDim.x + tx],
                                                  man_mul, exp_mul,
                                                  subnormal_support, saturate),
                            man_add, exp_add, subnormal_support, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}

/* FABsum implementation */
/*__global__ void gemm_fp_nearest(float *__restrict__ a, float *__restrict__ b,
                                float *__restrict__ c, int M, int K, int N,
                                int man_add, int exp_add, int man_mul,
                                int exp_mul, bool subnormal_support,
                                bool saturate) {
  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float accSum = 0.0;
  float fastSum = 0.0;
  int blockFactor = 1;
  int currFactor = 0;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      fastSum = cast_fp_nearest(fastSum +
            cast_fp_nearest(s_a[ty * blockDim.x + j] *s_b[j * blockDim.x + tx],
            man_mul, exp_mul, subnormal_support, saturate),
            man_add, exp_add, subnormal_support, saturate);
    }
    currFactor++;
    currFactor %= blockFactor;
    if (currFactor == 0) {
      // do bfloat16 summation here by default
      accSum = cast_fp_nearest(accSum + fastSum, 8, 7, subnormal_support,
saturate); fastSum = 0.0f;
    }


    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  accSum = cast_fp_nearest(accSum + fastSum, 8, 7, subnormal_support, saturate);
  // write back results
  if (row < M && col < N)
    c[row * N + col] = accSum;
}*/

/* Kahan summation-based version */
/*__global__ void gemm_fp_nearest(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_add, int exp_add,
                                  int man_mul, int exp_mul,
                                  bool subnormal_support,
                                  bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0.0;
  float comp = 0.0;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
  // load in elements for this tile
    s_a[ty * blockDim.x + tx] = (row < M && i + tx < K) ? a[row * K + i + tx] :
0.0; s_b[ty * blockDim.x + tx] = (col < N && i + ty < K) ? b[i * N + ty * N +
col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      float y = cast_fp_nearest(cast_fp_nearest(s_a[ty * blockDim.x + j] *
          s_b[j * blockDim.x + tx], man_mul, exp_mul, subnormal_support,
saturate) - comp, man_add, exp_add, subnormal_support, saturate); float t =
cast_fp_nearest(sum + y, man_add, exp_add, subnormal_support, saturate); comp =
cast_fp_nearest(cast_fp_nearest(t - sum, man_add, exp_add, subnormal_support,
saturate) - y, man_add, exp_add, subnormal_support, saturate); sum = t;
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N) c[row * N + col] = sum;
}*/

__global__ void gemm_fp_fma_nearest(float *__restrict__ a,
                                    float *__restrict__ b,
                                    float *__restrict__ c, int M, int K, int N,
                                    int man_fma, int exp_fma,
                                    bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0f;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_fp_nearest(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp),
          man_fma, exp_fma, subnormal_support, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}

__global__ void gemm_fp_stochastic(float *__restrict__ a, float *__restrict__ b,
                                   float *__restrict__ c,
                                   curandState_t *state, // int *__restrict__ r,
                                   int M, int K, int N, int man_add,
                                   int exp_add, int man_mul, int exp_mul,
                                   bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x) * 2;
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  unsigned int radd, rmul;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      radd = curand(&state[sidx]);
      rmul = curand(&state[sidx]);
      // radd = (unsigned int)r[bidx + 2 * (i + j)];
      // rmul = (unsigned int)r[bidx + 2 * (i + j) + 1];
      tmp = cast_fp_stochastic(
          tmp + cast_fp_stochastic(
                    s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], rmul,
                    man_mul, exp_mul, subnormal_support, saturate),
          radd, man_add, exp_add, subnormal_support, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}

__global__ void
gemm_fp_fma_stochastic(float *__restrict__ a, float *__restrict__ b,
                       float *__restrict__ c,
                       curandState_t *state, // int *__restrict__ r,
                       int M, int K, int N, int man_fma, int exp_fma,
                       bool subnormal_support, bool saturate) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  // int bidx = (row * N + col) * (K + blockDim.x - K % blockDim.x);
  int sidx = blockIdx.x * blockDim.x + blockIdx.y;

  float tmp = 0.0f;
  unsigned int rfma;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      // rfma = (unsigned int)r[bidx + i + j];
      rfma = curand(&state[sidx]);
      tmp = cast_fp_stochastic(
          fmaf(s_a[ty * blockDim.x + j], s_b[j * blockDim.x + tx], tmp), rfma,
          man_fma, exp_fma, subnormal_support, saturate);
    }

    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = tmp;
}