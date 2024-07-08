/*
Kernels for IEEE-754 down casting from binary32 to a lower precision format.
Payload is still a binary32 value.

Compile example:
nvcc -O3 cast_binary8_stochastic.cu -o cast_binary8_stochastic -std=c++17 -lcublas

version 1 attempted to make the code as compact as possible, while also 
maintaining readability; bit shifts and masking are used aplenty
./cast_binary8_stochastic 1

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

enum class SaturateMode {
    SATURATE,
    NO_OVERFLOW,
    OVERFLOWS
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

uint32_t binary8_clip_exponent_cpu(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, SaturateMode saturation_mode, bool subnormal) {

  if (quantized_num == 0){
    return quantized_num;
  }
  
  uint32_t man_val = quantized_num & 0x7FFFFF;
  uint32_t old_sign = old_num >> 31 << 31;
  uint32_t max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
  
  int spec_exp = (man_bits == 0) ? 1 : 0; // if P = 1
  int special_p1 = 0;
  
  if(exp_bits == 8 && saturation_mode != SaturateMode::NO_OVERFLOW){ // unsigned and p=1
      special_p1 = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
  }else if (exp_bits == 7 &&  man_bits == 1 && saturation_mode != SaturateMode::NO_OVERFLOW){ // unsigned and p=2 
      special_p1 = 1;
  }else if(exp_bits + man_bits == 8){ // unsigned
      max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value 0xfd = mACax_exp | max_mantissa - 1 
  }

  // Special because in unsigned we want our min to be 1 less because the space is taken by the Nan
  int quantized_exponent_store = (quantized_num << 1 >> 24);
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127 - special_p1; 
  int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp;

  if (saturation_mode == SaturateMode::NO_OVERFLOW) { // Saturate to max without infinity
    max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
  }

  if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && (man_val > max_man))) 
  {
    if (saturation_mode == SaturateMode::OVERFLOWS){ // Overflow to infinity
        return quantized_num = old_sign | 0x7F800000; // INF
    } 
    return quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
  }
  if (quantized_exponent_store < min_exponent_store) {
    if (subnormal) {
        int subnormal_shift = min_exponent_store - quantized_exponent_store;
        int min_subnormals_exp = min_exponent_store - man_bits;
        uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
        uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
      if (subnormal_shift <= man_bits) {
        // quantized_num stays the same in this case
      } else if ((old_num & 0x7FFFFFFF) > middle_num) {
        quantized_num = old_sign | min_num;
      } else {
        quantized_num = 0;
      }
    } 
    if(!subnormal) {
      uint32_t min_num = (uint32_t)min_exponent_store << 23;
      uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
      if ((old_num & 0x7FFFFFFF) > middle_num) {
        quantized_num = old_sign | min_num;
      } else {
        quantized_num = 0;
      }
    }
  }
  return quantized_num;
}

float cast_binary8_signed_stochastic_cpu(float origin_float, int P, uint32_t rand_prob, int prng_bits, SaturateMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {          
        return origin_float;
    }

    if (subnormals){
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;

        if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);

    uval8 = round_bitwise_stochastic_cpu(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

float cast_binary8_unsigned_stochastic_cpu(float origin_float, int P, uint32_t rand_prob, int prng_bits, SaturateMode saturation_mode, bool subnormals) { 
    
    if(origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    uint32_t uval32, uval8;
    int subnormal_shift = 0;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {
        return origin_float;
    }

    if (subnormals){   
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;        

        if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);

    uval8 = round_bitwise_stochastic_cpu(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

void binary8_signed_kernel_stochastic_cpu(float *__restrict__ a,
                                      int *__restrict__ r, float *o, int N,
                                      int P, int prng_bits, SaturateMode saturation_mode, bool subnormals) {
   for (int j = 0; j < N; ++j) {
         o[j] = cast_binary8_signed_stochastic_cpu(a[j], P, (uint32_t)r[j], prng_bits, saturation_mode, subnormals);
    }  
}

void binary8_unsigned_kernel_stochastic_cpu(float *__restrict__ a, 
                                      int *__restrict__ r, float *o, int N,
                                      int P, int prng_bits, SaturateMode saturation_mode, bool subnormals) {
    for (int j = 0; j < N; ++j) {
         o[j] = cast_binary8_unsigned_stochastic_cpu(a[j], P, (uint32_t)r[j], prng_bits, saturation_mode, subnormals);
    }
}

// --------------------------------------------------------------------------------------
// GPU kernels

__device__ __forceinline__ uint32_t
round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits) { // passing random_num with prng_bits; target is the original number
  uint32_t mask = (1 << (23 - man_bits)) - 1; 
  uint32_t add_r = target + (rand_prob & mask); // adding random bits to target (which is not masked)
  uint32_t quantized = add_r & ~mask; // masking out bits on the right hand side of the significant bits (truncating)
  return quantized;
}

__device__ __forceinline__ uint32_t
binary8_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, SaturateMode saturation_mode, bool subnormal) {

  if (quantized_num == 0){
    return quantized_num;
  }
  
  uint32_t man_val = quantized_num & 0x7FFFFF;
  uint32_t old_sign = old_num >> 31 << 31;
  uint32_t max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
  
  int spec_exp = (man_bits == 0) ? 1 : 0; // if P = 1
  int special_p1 = 0;
  
  if(exp_bits == 8 && saturation_mode != SaturateMode::NO_OVERFLOW){ // unsigned and p=1
      special_p1 = 1; // 0 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 0
  }else if (exp_bits == 7 &&  man_bits == 1 && saturation_mode != SaturateMode::NO_OVERFLOW){ // unsigned and p=2 
      special_p1 = 1;
      //print_uint(max_man);
      //max_man = 1 << 22; // 1 bit of mantissa so the max value 0xfd = max_exp - 1 | mantissa = 1 
  }else if(exp_bits + man_bits == 8){ // unsigned
      max_man = ((1u << man_bits) - 3u) << (23 - man_bits); // 2+ bit of mantissa so the max value 0xfd = mACax_exp | max_mantissa - 1 
  }

  // Special because in unsigned we want our min to be 1 less because the space is taken by the Nan
  int quantized_exponent_store = (quantized_num << 1 >> 24);
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127 - special_p1; 
  int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp;

  if (saturation_mode == SaturateMode::NO_OVERFLOW) { // Saturate to max without infinity
    max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
  }

  if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && (man_val > max_man))) 
  {
    if (saturation_mode == SaturateMode::OVERFLOWS){ // Overflow to infinity
        return quantized_num = old_sign | 0x7F800000; // INF
    } 
    return quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
  }
  if (quantized_exponent_store < min_exponent_store) {
    if (subnormal) {
        int subnormal_shift = min_exponent_store - quantized_exponent_store;
        int min_subnormals_exp = min_exponent_store - man_bits;
        uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
        uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
      if (subnormal_shift <= man_bits) {
        // quantized_num stays the same in this case
      } else if ((old_num & 0x7FFFFFFF) > middle_num) {
        quantized_num = old_sign | min_num;
      } else {
        quantized_num = 0;
      }
    } 
    if(!subnormal) {
      uint32_t min_num = (uint32_t)min_exponent_store << 23;
      uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
      if ((old_num & 0x7FFFFFFF) > middle_num) {
        quantized_num = old_sign | min_num;
      } else {
        quantized_num = 0;
      }
    }
  }
  return quantized_num;
}

__device__ float cast_binary8_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, SaturateMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {          
        return origin_float;
    }

    if (subnormals){
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;

        if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);

    uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__device__ float cast_binary8_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, SaturateMode saturation_mode, bool subnormals) { 
    
    if(origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    uint32_t uval32, uval8;
    int subnormal_shift = 0;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {
        return origin_float;
    }

    if (subnormals){   
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;        

        if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
            subnormal_shift = min_exp - exp_val;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - prng_bits) - 1);

    uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__global__ void binary8_signed_kernel_stochastic(float *__restrict__ a,
                                      int *__restrict__ r, float *o, int N,
                                      int P, int prng_bits, SaturateMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
        o[idx] = cast_binary8_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, saturation_mode, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_stochastic(float *__restrict__ a, 
                                      int *__restrict__ r, float *o, int N,
                                      int P, int prng_bits, SaturateMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
        o[idx] = cast_binary8_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, saturation_mode, subnormals);
  }
}

// ---------------------------------------------------------------------------------------
// Kernel launchers

void binary8_signed_stochastic(float *a, int *r, float *o, int N, int P, int prng_bits, SaturateMode saturation_mode, bool subnormals, const int blockSize){
    const int grid_size = ceil_div(N, blockSize);
    binary8_signed_kernel_stochastic<<<grid_size, blockSize>>>(a, r, o, N, P, prng_bits, saturation_mode, subnormals);
    cudaCheck(cudaGetLastError());
}

void binary8_unsigned_stochastic(float *a, int *r, float *o, int N, int P, int prng_bits, SaturateMode saturation_mode, bool subnormals, const int blockSize){
    const int grid_size = ceil_div(N, blockSize);
    binary8_unsigned_kernel_stochastic<<<grid_size, blockSize>>>(a, r, o, N, P, prng_bits, saturation_mode, subnormals);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void binary8_stochastic(int kernel_num, // 1 for signed and 2 for unsighed
                    float*a,
                    int*r,
                    float*o,
                    int N,
                    int P,
                    int prng_bits,
                    SaturateMode saturation_mode,
                    bool subnormals,
                    const int blockSize)
{
    switch (kernel_num)
    {
    case 1:
        binary8_signed_stochastic(a, r, o, N, P, prng_bits, saturation_mode, subnormals, blockSize);
        break;
    case 2:
        binary8_unsigned_stochastic(a, r, o, N, P,  prng_bits, saturation_mode, subnormals, blockSize);
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
    SaturateMode saturation_mode = SaturateMode::SATURATE;

    float *x = (float*)malloc(1000 * sizeof(float));
    int *r = (int *)malloc(1000 * sizeof(int)); // numbers added to round
    float *y = (float*)malloc(1000 * sizeof(float));
    int rnd_up = 0, rnd_down = 0;

    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> distrib(0, std::pow(2, 23)-1); // Uniform distribution in [0, 1)

     for (int i = 0; i < N; ++i) {
        r[i] = distrib(gen);
        x[i] = 2.1f;
    }

    printf("Using kernel %d\n", kernel_num);

    // compute reference CPU solution
    if (is_signed){
        binary8_signed_kernel_stochastic_cpu(x, r, y, N, P, prng_bits, saturation_mode, subnormals);
    } else {
        binary8_unsigned_kernel_stochastic_cpu(x, r, y, N, P, prng_bits, saturation_mode, subnormals);
    }

    for (int i = 0; i < N; ++i){
        //printf("%lf\n", y[i]);
        if (y[i] == 2.0){
            rnd_down++;
        } else if (y[i] == 2.5){
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

    //time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        binary8_stochastic(kernel_num, d_x, d_r, d_y, N, P, prng_bits, saturation_mode, subnormals, block_size);

        float tol = 0.0f;

        validate_result(d_y, y, "y", N, tol); // checks if d_y and y are the same
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, binary8_stochastic, 
                kernel_num, d_y, d_r, d_x, N, P, prng_bits, saturation_mode,
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
