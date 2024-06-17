#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

template <bool subnormals>
__device__ float cast_p3109_signed_nearest(float origin_float, int P) {
    // P range from 1 to 7 in signed
    int exp_bits = 8-P;
    int man_bits = P-1; //P in unsigned
    
    int spec_exp = (P == 1) ? 1 : 0;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }
    
    int subnormal_shift = 0;
    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }

    uval8 = round_bitwise_nearest(uval32, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <>
__device__ float cast_p3109_signed_nearest<false>(float origin_float, int P) {
    // P range from 1 to 7 in signed
    int exp_bits = 8-P;
    int man_bits = P-1; //P in unsigned
    
    int spec_exp = (P == 1) ? 1 : 0;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_nearest(uval32, man_bits);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, false);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <bool subnormals>
__device__ float cast_p3109_signed_stochastic(float origin_float, int P, int prng_bits) {

    int exp_bits = 8-P;
    int man_bits = P-1;  
    
    int spec_exp = (P == 1) ? 1 : 0;
    
    bool subnormals = true;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;
    
        
    int subnormal_shift = 0;
    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }
    
    // Calculate the upper limit of the rand_prob
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - (man_bits - subnormal_shift))) - 1;

    // Random number generator
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Distribution that covers the range [0, 2^x - 1]
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);

    // Generate the random number
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, random_number, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <>
__device__ float cast_p3109_signed_stochastic<false>(float origin_float, int P, int prng_bits) {

    int exp_bits = 8-P;
    int man_bits = P-1;  
    
    int spec_exp = (P == 1) ? 1 : 0;
    
    bool subnormals = false;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;
    
    // Calculate the upper limit of the rand_prob
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - man_bits)) - 1;

    // Random number generator
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Distribution that covers the range [0, 2^x - 1]
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);

    // Generate the random number
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, random_number, man_bits);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <bool subnormals>
__device__ float cast_p3109_unsigned_nearest(float origin_float, int P) {

    if (origin_float < 0){
      return NAN;
    }
 
    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    
    int spec_exp = (P == 1) ? 1 : 0;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);


    int sign = uval32 >> 31;
    int exp_val = (uval32 << 1 >> 24) - 127;
    
    bool subnormals = true;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;
    

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }
    
    int subnormal_shift = 0;
    if((min_exp - exp_val) <= man_bits && exp_val < min_exp && subnormals){ 
      subnormal_shift = min_exp - exp_val;
    }

    uval8 = round_bitwise_nearest(uval32, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <>
__device__ float cast_p3109_unsigned_nearest<false>(float origin_float, int P) {

    if (origin_float < 0){
      return NAN;
    }
    
    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    
    int spec_exp = (P == 1) ? 1 : 0;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);


    int sign = uval32 >> 31;
    int exp_val = (uval32 << 1 >> 24) - 127;
    
    bool subnormals = false;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;


    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_nearest(uval32, man_bits);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <bool subnormals>
__device__ float cast_p3109_unsigned_stochastic(float origin_float, int P, int prng_bits) { // what is prng_bits? number of random bits?
    
    if(origin_float < 0){
        return NAN;
    }

    // P range from 1 to 7 in signed
    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    
    int spec_exp = (P == 1) ? 1 : 0;
    
    bool subnormals = true;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;
        
    int subnormal_shift = 0;
    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }
    
    // Calculate the upper limit of the rand_prob
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - (man_bits - subnormal_shift))) - 1;

    // Random number generator
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Distribution that covers the range [0, 2^x - 1]
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);

    // Generate the random number
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, random_number, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <>
__device__ float cast_p3109_unsigned_stochastic<false>(float origin_float, int P, int prng_bits) {

  if (origin_float < 0){
    return NAN;
  }

   // P range from 1 to 7 in signed
    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    
    int spec_exp = (P == 1) ? 1 : 0;
    
    bool subnormals = false;

    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    // minimal and maximal exponent value in binary8
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;

    if(sign == 1){
        return NAN;
    }
    
    // Calculate the upper limit of the rand_prob
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - man_bits)) - 1;

    // Random number generator
    std::random_device rd; // Seed
    std::mt19937 gen(rd()); // Mersenne Twister engine

    // Distribution that covers the range [0, 2^x - 1]
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);

    // Generate the random number
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, random_number, man_bits);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__global__ void p3109_signed_kernel_nearest(float *__restrict__ a, float *o, int size, int P, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      if (subnormals) {
          o[idx] = cast_p3109_signed_nearest<true>(a[idx], P);
      } else {
          o[idx] = cast_p3109_signed_nearest<false>(a[idx], P);
      }
  }
}

__global__ void p3109_unsigned_kernel_nearest(float *__restrict__ a, float *o, int size,
                                      int P, bool subnormals) {
  // TODO
}

__global__ void p3109_signed_kernel_stochastic(float *__restrict__ a, float *o, int size,
                                      int P, int prng_bits, bool subnormals) {
  // TODO
}

__global__ void p3109_unsigned_kernel_stochastic(float *__restrict__ a, float *o, int size,
                                      int P, int prng_bits, bool subnormals) {
  // TODO
}