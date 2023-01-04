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

__device__ float cast_mult_fp_in(float origin_float, int man_bits,
                                 int exp_bits) {
  unsigned int bits = *reinterpret_cast<unsigned int const *>(&origin_float);
  int sign = (bits & 0x80000000);

  unsigned int max_man = 0xFFFFFFFF << 9 >> 9 >> (23 - man_bits)
                                                     << (23 - man_bits);
  int max_exp = (1 << (exp_bits - 1)) + 127;
  unsigned int max_val = max_exp << 23;
  max_val = max_val | max_man;

  // calculate the minimum representable unbiased exp value and -ve smallest
  // value
  int MIN_EXP = -((1 << (exp_bits - 1)) - 2);
  unsigned int TIE = (1 << (22 - man_bits));
  unsigned int TIE_MASK = -(TIE << 1);
  unsigned int NEG_MAX_VAL = (max_val | 0x80000000);
  bool NO_QUANTIZE = (man_bits >= 23 && exp_bits >= 8); // sanity check

  // extract exp value from the input, and calculate unbiased exp value
  unsigned tgt_exp_raw = (bits << 1 >> 24);
  int tgt_exp = tgt_exp_raw - 127;

  unsigned int retval_bits;
  unsigned int tgt_round;
  float retval;
  bool cond;

  // apply quantization if needed
  if (NO_QUANTIZE) {
    retval = origin_float;
  } else {
    // apply rounding
    tgt_round = bits + TIE;
    tgt_round = tgt_round & TIE_MASK;

    unsigned MIN_ZERO =
        ((126 + MIN_EXP)
         << 23); // +ve 0 in the custom subnormal format (needs to be reserved
                 // --> truncate the corresponding float value to 0)
    unsigned NEG_MIN_ZERO =
        (MIN_ZERO | 0x80000000); // -ve 0 in the custom subnormal format (needs
                                 // to be reserved --> truncate the
                                 // corresponding float value to 0)
    cond = (MIN_EXP > tgt_exp + 1) || (tgt_round == MIN_ZERO) ||
           (tgt_round == NEG_MIN_ZERO);

    // extract the biased exp val of the value after rounding
    unsigned char exp_raw = (tgt_round >> 23);

    // temporarily saturate the overflowing value to max representable value
    if (exp_raw > max_exp || tgt_exp_raw > max_exp)
      tgt_round = max_val;

    // if input value will be in subnorm range of the target data type, truncate
    // it to 0
    if (cond)
      tgt_round = 0;

    // put the sign back and cast the value back to float format
    retval_bits = (sign | (tgt_round & 0x7FFFFFFF));
    retval = *reinterpret_cast<float const *>(&retval_bits);

    // if the value is saturated to +ve/-ve max value, saturate it to
    // corresponding Infinity
    if (tgt_round == max_val)
      retval = INFINITY;
    if (tgt_round == NEG_MAX_VAL)
      retval = -INFINITY;
  }

  return retval;
}

__device__ float cast_mult_fp_out(float origin_float, int man_bits,
                                  int exp_bits) {
  unsigned int bits = *reinterpret_cast<unsigned int const *>(&origin_float);
  int sign = (bits & 0x80000000);

  unsigned int max_man = 0xFFFFFFFF << 9 >> 9 >> (23 - man_bits)
                                                     << (23 - man_bits);
  int max_exp = (1 << (exp_bits - 1)) + 127;
  unsigned int max_val = max_exp << 23;
  max_val = max_val | max_man;

  // calculate the minimum representable unbiased exp value and -ve smallest
  // value
  int MIN_EXP = -((1 << (exp_bits - 1)) - 3);
  unsigned int TIE = (1 << (21 - man_bits));
  unsigned int TIE_MASK = -(TIE << 1);
  unsigned int NEG_MAX_VAL = (max_val | 0x80000000);
  bool NO_QUANTIZE = (man_bits >= 23 && exp_bits >= 8); // sanity check

  // extract exp value from the input, and calculate unbiased exp value
  unsigned tgt_exp_raw = (bits << 1 >> 24);
  int tgt_exp = tgt_exp_raw - 125;

  unsigned int retval_bits;
  unsigned int tgt_round;
  float retval;
  bool cond;

  // apply quantization if needed
  if (NO_QUANTIZE) {
    retval = origin_float;
  } else {
    // apply rounding
    tgt_round = bits + TIE;
    tgt_round = tgt_round & TIE_MASK;

    unsigned MIN_ZERO =
        ((125 + MIN_EXP)
         << 23); // +ve 0 in the custom subnormal format (needs to be reserved
                 // --> truncate the corresponding float value to 0)
    unsigned NEG_MIN_ZERO =
        (MIN_ZERO | 0x80000000); // -ve 0 in the custom subnormal format (needs
                                 // to be reserved --> truncate the
                                 // corresponding float value to 0)
    cond = (MIN_EXP > tgt_exp + 1) || (tgt_round == MIN_ZERO) ||
           (tgt_round == NEG_MIN_ZERO);

    // extract the biased exp val of the value after rounding
    unsigned char exp_raw = (tgt_round >> 23);

    // temporarily saturate the overflowing value to max representable value
    // (with NAN disabled)
    if (exp_raw > max_exp + 1)
      tgt_round = max_val;

    // if input value will be in subnorm range of the target data type, truncate
    // it to 0
    if (cond)
      tgt_round = 0;

    // put the sign back and cast the value back to float format
    retval_bits = (sign | (tgt_round & 0x7FFFFFFF));
    retval = *reinterpret_cast<float const *>(&retval_bits);

    // if the value is saturated to +ve/-ve max value, saturate it to
    // corresponding Infinity
    if (tgt_round == max_val)
      retval = INFINITY;
    if (tgt_round == NEG_MAX_VAL)
      retval = -INFINITY;
  }

  return retval;
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
    // o[index] = cast_fp_nearest(a[index], man_bits, exp_bits,
    // subnormal_support,
    //                           saturate);
    o[index] = cast_mult_fp_in(a[index], man_bits, exp_bits);
}

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
}*/

/* dot implementation */
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

  float accSum = 0.0;
  float dotSum = 0.0;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
    // load in elements for this tile
    s_a[ty * blockDim.x + tx] =
        (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] =
        (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do the adder tree accumulation for the dot operation
    float d1 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 0] * s_b[0 * blockDim.x + tx],
                        man_mul, exp_mul + 1);
    float d2 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 1] * s_b[1 * blockDim.x + tx],
                        man_mul, exp_mul + 1);
    float d3 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 2] * s_b[2 * blockDim.x + tx],
                        man_mul, exp_mul + 1);
    float d4 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 3] * s_b[3 * blockDim.x + tx],
                        man_mul, exp_mul + 1);

    d1 = cast_mult_fp_in(d1 + d2, man_mul + 1, exp_mul + 1);
    d2 = cast_mult_fp_in(d3 + d4, man_mul + 1, exp_mul + 1);
    d1 = cast_mult_fp_in(d1 + d2, man_mul + 2, exp_mul + 1);

    float d5 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 4] * s_b[4 * blockDim.x + tx],
                        man_mul, exp_mul + 1);
    float d6 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 5] * s_b[5 * blockDim.x + tx],
                        man_mul, exp_mul + 1);
    float d7 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 6] * s_b[6 * blockDim.x + tx],
                        man_mul, exp_mul + 1);
    float d8 =
        cast_mult_fp_in(s_a[ty * blockDim.x + 7] * s_b[7 * blockDim.x + tx],
                        man_mul, exp_mul + 1);

    d5 = cast_mult_fp_in(d5 + d6, man_mul + 1, exp_mul + 1);
    d6 = cast_mult_fp_in(d7 + d8, man_mul + 1, exp_mul + 1);
    d5 = cast_mult_fp_in(d5 + d6, man_mul + 2, exp_mul + 1);

    dotSum = cast_mult_fp_in(d1 + d5, man_mul + 3, exp_mul + 1);
    accSum = cast_mult_fp_in(accSum + dotSum, man_mul + 3, exp_mul + 1);
    // wait for all threads to finish using current tiles
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N)
    c[row * N + col] = accSum;
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