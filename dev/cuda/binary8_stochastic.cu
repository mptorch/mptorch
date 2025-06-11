/*
Kernels for IEEE-754 down casting from binary32 to a lower precision format.
Payload is still a binary32 value.

Compile example:
nvcc -O3 binary8_stochastic.cu -o binary8_stochastic -std=c++17 -lcublas

version 1 attempted to make the code as compact as possible, while also
maintaining readability; bit shifts and masking are used aplenty
./binary8_stochastic 1

*/

#include <cuda_runtime.h>
#include "common.h"
#include <random>
#include <climits>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <tuple>

enum class OverflowPolicy
{
  SATURATE_INFTY,
  SATURATE_MAXFLOAT,
  SATURATE_MAXFLOAT2
};

// --------------------------------------------------------------------------------------
// CPU reference code

uint32_t round_bitwise_stochastic_cpu(uint32_t target, uint32_t rand_prob, int man_bits)
{
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t add_r = target + (rand_prob & mask);
  uint32_t quantized = add_r & ~mask;
  return quantized;
}

uint32_t binary8_clip_exponent_cpu(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, OverflowPolicy overflow_policy, bool subnormal)
{
  if (quantized_num == 0)
  {
    return quantized_num;
  }

  uint32_t man_val = quantized_num & 0x7FFFFF;
  uint32_t old_sign = old_num >> 31 << 31;
  uint32_t max_man;

  int spec_exp = (man_bits == 0) ? 1 : 0; // if P = 1
  int special_unsigned_exp = 0;

  max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits); // max mantissa = 0xfe in the normal case

  if (exp_bits + man_bits == 7 && overflow_policy == OverflowPolicy::SATURATE_MAXFLOAT2)
  { // if signed and policy maxfloat_real then max mantissa = 0xff
    max_man = ((1u << man_bits) - 1u) << (23 - man_bits);
  }

  if (overflow_policy != OverflowPolicy::SATURATE_MAXFLOAT2)
  { // if we are not in OVERFLOW_MAXFLOAT_REALS policy :
    if (exp_bits == 8)
    {                           // unsigned and p=1
      special_unsigned_exp = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
    }
    else if (exp_bits == 7 && man_bits == 1)
    {                           // unsigned and p=2
      special_unsigned_exp = 1; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1
      max_man = ((1u << man_bits) - 1u) << (23 - man_bits);
    }
    else if (exp_bits + man_bits == 8)
    {                                                       // unsigned
      max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value is 0xfd = max_exp | max_mantissa - 1
    }
  }

  // if(exp_bits == 8 && overflow_policy == OverflowPolicy::OVERFLOW_INFTY){ // unsigned and p=1
  //     special_unsigned_exp = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
  // }else if (exp_bits == 7 &&  man_bits == 1 && overflow_policy == OverflowPolicy::OVERFLOW_INFTY){ // unsigned and p=2
  //     special_unsigned_exp = 1; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1
  // }else if(exp_bits + man_bits == 8){ // unsigned
  //     max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value is 0xfd = max_exp | max_mantissa - 1
  // }

  // Special because in unsigned we want our min to be 1 less because the space is taken by the Nan
  int quantized_exponent_store = (quantized_num << 1 >> 24);
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127 - special_unsigned_exp;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp;

  if (!subnormal)
  {
    min_exponent_store--;
  }

  if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && (man_val > max_man)))
  {
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY)
    {                                               // Overflow to infinity (exceeds the max value 0xfe or 0xfd if signed or unsigned)
      return quantized_num = old_sign | 0x7F800000; // INF
    }
    // Otherwise saturate to the max float value permitted by the policy reprensented by the max_man and max_exponent_store
    quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
  }
  if (quantized_exponent_store < min_exponent_store)
  {
    if (subnormal)
    {
      int subnormal_shift = min_exponent_store - quantized_exponent_store;
      int min_subnormals_exp = min_exponent_store - man_bits;
      uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
      uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
      if (subnormal_shift <= man_bits)
      {
        // quantized_num stays the same in this case
      }
      else if ((old_num & 0x7FFFFFFF) > middle_num)
      {
        quantized_num = old_sign | min_num;
      }
      else
      {
        quantized_num = 0;
      }
    }
    else
    { // no subnormal case; normalizing subnormal values
      uint32_t min_num = ((uint32_t)min_exponent_store << 23) | 1 << (23 - man_bits);
      uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23 | 1 << (23 - man_bits));
      if ((old_num & 0x7FFFFFFF) > middle_num)
      {
        return quantized_num = old_sign | min_num;
      }
      else
      {
        return quantized_num = 0;
      }
    }
  }
  return quantized_num;
}

float cast_binary8_signed_stochastic_cpu(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{

  const int exp_bits = 8 - P;
  const int man_bits = P - 1;
  const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  const uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  {
    return origin_float;
  }

  int subnormal_shift = 0;
  if (subnormals)
  {
    int spec_exp = (P == 1) ? 1 : 0;
    int max_exp = (1 << (exp_bits - 1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  rand_prob = rand_prob << 9 >> 9;
  // rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);
  rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

  uint32_t uval8 = round_bitwise_stochastic_cpu(uval32, rand_prob, man_bits - subnormal_shift);
  uval8 = binary8_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

float cast_binary8_unsigned_stochastic_cpu(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{

  if (origin_float < 0)
    return NAN;

  uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  const int exp_bits = 8 - P + 1;
  const int man_bits = P - 1;
  int subnormal_shift = 0;

  if (subnormals)
  {
    const int spec_exp = (P == 1) ? 1 : 0;
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  rand_prob = rand_prob << 9 >> 9;
  // rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);
  rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

  uint32_t uval8 = round_bitwise_stochastic_cpu(uval32, rand_prob, man_bits - subnormal_shift);
  uval8 = binary8_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

void binary8_signed_kernel_stochastic_cpu(float *__restrict__ a,
                                          int *__restrict__ r, float *o, int N,
                                          int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{
  for (int j = 0; j < N; ++j)
  {
    o[j] = cast_binary8_signed_stochastic_cpu(a[j], P, (uint32_t)r[j], prng_bits, overflow_policy, subnormals);
  }
}

void binary8_unsigned_kernel_stochastic_cpu(float *__restrict__ a,
                                            int *__restrict__ r, float *o, int N,
                                            int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{
  for (int j = 0; j < N; ++j)
  {
    o[j] = cast_binary8_unsigned_stochastic_cpu(a[j], P, (uint32_t)r[j], prng_bits, overflow_policy, subnormals);
  }
}

// --------------------------------------------------------------------------------------
// GPU kernels

__device__ __forceinline__ uint32_t
round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits)
{ // passing random_num with prng_bits; target is the original number
  uint32_t mask = (1 << (23 - man_bits)) - 1;
  uint32_t add_r = target + (rand_prob & mask); // adding random bits to target (which is not masked)
  uint32_t quantized = add_r & ~mask;           // masking out bits on the right hand side of the significant bits (truncating)
  return quantized;
}

__device__ __forceinline__ uint32_t
binary8_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, OverflowPolicy overflow_policy, bool subnormal)
{
  if (quantized_num == 0)
  {
    return quantized_num;
  }

  uint32_t man_val = quantized_num & 0x7FFFFF;
  uint32_t old_sign = old_num >> 31 << 31;
  uint32_t max_man;

  int spec_exp = (man_bits == 0) ? 1 : 0; // if P = 1
  int special_unsigned_exp = 0;

  max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits); // max mantissa = 0xfe in the normal case

  if (exp_bits + man_bits == 7 && overflow_policy == OverflowPolicy::SATURATE_MAXFLOAT2)
  { // if signed and policy maxfloat_real then max mantissa = 0xff
    max_man = ((1u << man_bits) - 1u) << (23 - man_bits);
  }

  if (overflow_policy != OverflowPolicy::SATURATE_MAXFLOAT2)
  { // if we are not in OVERFLOW_MAXFLOAT_REALS policy :
    if (exp_bits == 8)
    {                           // unsigned and p=1
      special_unsigned_exp = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
    }
    else if (exp_bits == 7 && man_bits == 1)
    {                           // unsigned and p=2
      special_unsigned_exp = 1; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1
      max_man = ((1u << man_bits) - 1u) << (23 - man_bits);
    }
    else if (exp_bits + man_bits == 8)
    {                                                       // unsigned
      max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value is 0xfd = max_exp | max_mantissa - 1
    }
  }

  // if(exp_bits == 8 && overflow_policy == OverflowPolicy::OVERFLOW_INFTY){ // unsigned and p=1
  //     special_unsigned_exp = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
  // }else if (exp_bits == 7 &&  man_bits == 1 && overflow_policy == OverflowPolicy::OVERFLOW_INFTY){ // unsigned and p=2
  //     special_unsigned_exp = 1; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1
  // }else if(exp_bits + man_bits == 8){ // unsigned
  //     max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value is 0xfd = max_exp | max_mantissa - 1
  // }

  // Special because in unsigned we want our min to be 1 less because the space is taken by the Nan
  int quantized_exponent_store = (quantized_num << 1 >> 24);
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127 - special_unsigned_exp;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp;

  if (!subnormal)
  {
    min_exponent_store--;
  }

  if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && (man_val > max_man)))
  {
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY)
    {                                               // Overflow to infinity (exceeds the max value 0xfe or 0xfd if signed or unsigned)
      return quantized_num = old_sign | 0x7F800000; // INF
    }
    // Otherwise saturate to the max float value permitted by the policy reprensented by the max_man and max_exponent_store
    quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
  }
  if (quantized_exponent_store < min_exponent_store)
  {
    if (subnormal)
    {
      int subnormal_shift = min_exponent_store - quantized_exponent_store;
      int min_subnormals_exp = min_exponent_store - man_bits;
      uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
      uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
      if (subnormal_shift <= man_bits)
      {
        // quantized_num stays the same in this case
      }
      else if ((old_num & 0x7FFFFFFF) > middle_num)
      {
        quantized_num = old_sign | min_num;
      }
      else
      {
        quantized_num = 0;
      }
    }
    else
    { // no subnormal case; normalizing subnormal values
      uint32_t min_num = ((uint32_t)min_exponent_store << 23) | 1 << (23 - man_bits);
      uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23 | 1 << (23 - man_bits));
      if ((old_num & 0x7FFFFFFF) > middle_num)
      {
        return quantized_num = old_sign | min_num;
      }
      else
      {
        return quantized_num = 0;
      }
    }
  }
  return quantized_num;
}

__device__ float cast_binary8_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{

  const int exp_bits = 8 - P;
  const int man_bits = P - 1;
  const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  const uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  {
    return origin_float;
  }

  int subnormal_shift = 0;
  if (subnormals)
  {
    int spec_exp = (P == 1) ? 1 : 0;
    int max_exp = (1 << (exp_bits - 1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  rand_prob = rand_prob << 9 >> 9;
  // rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);
  rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

  uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

__device__ float cast_binary8_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{

  if (origin_float < 0)
    return NAN;

  uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
  const int exp_val = (uval32 << 1 >> 24) - 127;
  uint32_t man_val = uval32 & 0x7FFFFF;

  // Early return for inf/NaN case
  if (exp_val == 128 && man_val != 0)
  { // if input nan return nan anyway
    return origin_float;
  }

  if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0)
  { // if input is infty and overflow_policy is SATURATE_INFTY
    return origin_float;
  }

  const int exp_bits = 8 - P + 1;
  const int man_bits = P - 1;
  int subnormal_shift = 0;

  if (subnormals)
  {
    const int spec_exp = (P == 1) ? 1 : 0;
    const int max_exp = (1 << (exp_bits - 1)) - 1;
    const int min_exp = spec_exp - max_exp;

    if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp)
    {
      subnormal_shift = min_exp - exp_val;
    }
  }

  rand_prob = rand_prob << 9 >> 9;
  // rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);
  rand_prob = rand_prob & ~(1 << (23 - man_bits - prng_bits) - 1);

  uint32_t uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
  uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);

  return BITS_TO_FLOAT(&uval8);
}

__global__ void binary8_signed_kernel_stochastic(float *__restrict__ a,
                                                 int *__restrict__ r, float *o, int N,
                                                 int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    o[idx] = cast_binary8_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_stochastic(float *__restrict__ a,
                                                   int *__restrict__ r, float *o, int N,
                                                   int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    o[idx] = cast_binary8_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
  }
}

// ---------------------------------------------------------------------------------------
// Kernel launchers

void binary8_signed_stochastic(float *a, int *r, float *o, int N, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals, const int blockSize)
{
  const int grid_size = ceil_div(N, blockSize);
  binary8_signed_kernel_stochastic<<<grid_size, blockSize>>>(a, r, o, N, P, prng_bits, overflow_policy, subnormals);
  cudaCheck(cudaGetLastError());
}

void binary8_unsigned_stochastic(float *a, int *r, float *o, int N, int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals, const int blockSize)
{
  const int grid_size = ceil_div(N, blockSize);
  binary8_unsigned_kernel_stochastic<<<grid_size, blockSize>>>(a, r, o, N, P, prng_bits, overflow_policy, subnormals);
  cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void binary8_stochastic(int kernel_num, // 1 for signed and 2 for unsighed
                        float *a,
                        int *r,
                        float *o,
                        int N,
                        int P,
                        int prng_bits,
                        OverflowPolicy overflow_policy,
                        bool subnormals,
                        const int blockSize)
{
  switch (kernel_num)
  {
  case 1:
    binary8_signed_stochastic(a, r, o, N, P, prng_bits, overflow_policy, subnormals, blockSize);
    break;
  case 2:
    binary8_unsigned_stochastic(a, r, o, N, P, prng_bits, overflow_policy, subnormals, blockSize);
    break;
  default:
    printf("Invalid kernel number\n");
    exit(EXIT_FAILURE);
  }
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
  bool is_signed = true;
  int kernel_num = 1;

  if (argc > 1)
  {
    kernel_num = atoi(argv[1]);
    is_signed = true;
  }

  int N = 1000;
  int P = 3;
  int prng_bits = 20;
  bool subnormals = true;
  OverflowPolicy overflow_policy = OverflowPolicy::SATURATE_MAXFLOAT;

  float *x = (float *)malloc(1000 * sizeof(float));
  int *r = (int *)malloc(1000 * sizeof(int)); // numbers added to round
  float *y = (float *)malloc(1000 * sizeof(float));
  int rnd_up = 0, rnd_down = 0;

  std::random_device rd;                                                 // Seed generator
  std::mt19937 gen(rd());                                                // Mersenne Twister engine
  std::uniform_real_distribution<float> distrib(0, std::pow(2, 23) - 1); // Uniform distribution in [0, 1)

  for (int i = 0; i < N; ++i)
  {
    r[i] = distrib(gen);
    x[i] = 2.1f;
  }

  printf("Using kernel %d\n", kernel_num);

  // compute reference CPU solution
  if (is_signed)
  {
    binary8_signed_kernel_stochastic_cpu(x, r, y, N, P, prng_bits, overflow_policy, subnormals);
  }
  else
  {
    binary8_unsigned_kernel_stochastic_cpu(x, r, y, N, P, prng_bits, overflow_policy, subnormals);
  }

  for (int i = 0; i < N; ++i)
  {
    // printf("%lf\n", y[i]);
    if (y[i] == 2.0)
    {
      rnd_down++;
    }
    else if (y[i] == 2.5)
    {
      rnd_up++;
    }
  }

  double prob_up = rnd_up / 1000.0;
  double prob_down = rnd_down / 1000.0;

  printf("Round up: %.2lf%% | Round down: %.2lf%%\n", (100.0 * prob_up), (100.0 * prob_down));

  // move data to the GPU
  float *d_x, *d_y;
  int *d_r;
  cudaCheck(cudaMalloc(&d_r, N * sizeof(int)));
  cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
  cudaCheck(cudaMalloc(&d_y, N * sizeof(float)));

  cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_r, r, N * sizeof(float), cudaMemcpyHostToDevice));

  // time the kernel at different block sizes
  int block_sizes[] = {32, 64, 128, 256, 512, 1024};
  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
  {
    int block_size = block_sizes[j];
    printf("Checking block size %d.\n", block_size);
    binary8_stochastic(kernel_num, d_x, d_r, d_y, N, P, prng_bits, overflow_policy, subnormals, block_size);

    float tol = 0.0f;

    validate_result(d_y, y, "y", N, tol); // checks if d_y and y are the same
  }

  printf("All results match. Starting benchmarks.\n\n");

  for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
  {
    int block_size = block_sizes[j];

    int repeat_times = 1000;

    float elapsed_time = benchmark_gpu_kernel(repeat_times, binary8_stochastic,
                                              kernel_num, d_y, d_r, d_x, N, P, prng_bits, overflow_policy,
                                              subnormals, block_size);

    // estimate memory bandwidth achieved
    // for each output element, we do 1 read and 1 write, 4 bytes each
    long memory_ops = N * 2 * (int)sizeof(float);
    float memory_bandwidth = memory_ops / elapsed_time / 1e6;

    printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
  }

  // free memory
  free(x);
  free(y);
  free(r);

  cudaCheck(cudaFree(d_x));
  cudaCheck(cudaFree(d_y));
  cudaCheck(cudaFree(d_r));

  return 0;
}
