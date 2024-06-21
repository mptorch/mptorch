#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

// template <bool subnormals>
__device__ float cast_p3109_signed_nearest(float origin_float, int P, bool subnormals) {

   

    int exp_bits = 8-P;
    int man_bits = P-1;    
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    if (subnormals){
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;    // minimal and maximal exponent value in binary8
        int min_exp = spec_exp - max_exp;
        if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
        subnormal_shift = min_exp - exp_val;
        }
    }

    uval8 = round_bitwise_nearest(uval32, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

// template <>
// __device__ float cast_p3109_signed_nearest<false>(float origin_float, int P) {

//     int exp_bits = 8-P;
//     int man_bits = P-1;    
//     int spec_exp = (P == 1) ? 1 : 0;
//     uint32_t uval32, uval8;
//     float fval8;

//     uval32 = FLOAT_TO_BITS(&origin_float);

//     int exp_val = (uval32 << 1 >> 24) - 127;
//     int max_exp = (1 << (exp_bits -1)) - 1;
//     int min_exp = spec_exp - max_exp;

//     if (exp_val == 128) {             // inf/Nan case
//         return origin_float;
//     }

//     uval8 = round_bitwise_nearest(uval32, man_bits);
//     uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, false);
//     fval8 = BITS_TO_FLOAT(&uval8);

//     return fval8;
// }

// template <bool subnormals>
__device__ float cast_p3109_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, bool subnormals) {

    int exp_val = (uval32 << 1 >> 24) - 127;

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;
    uval32 = FLOAT_TO_BITS(&origin_float);



    if (subnormals){



    }
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;
            
        if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
            subnormal_shift = min_exp - exp_val;
        }



    uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <>
__device__ float cast_p3109_signed_stochastic<false>(float origin_float, int P, uint32_t rand_prob, int prng_bits) {

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

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, prng_bits, man_bits);
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
__device__ float cast_p3109_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits) { // what is prng_bits? number of random bits?
    
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

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <>
__device__ float cast_p3109_unsigned_stochastic<false>(float origin_float, int P, uint32_t rand_prob, int prng_bits) {

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

    if (exp_val == 128) {             // inf/Nan case
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, rand_prob, man_bits);
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
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      if (subnormals) {
          o[idx] = cast_p3109_unsigned_nearest<true>(a[idx], P);
      } else {
          o[idx] = cast_p3109_unsigned_nearest<false>(a[idx], P);
      }
  }

}

__global__ void p3109_signed_kernel_stochastic(float *__restrict__ a,
                                      int *__restrict__ r, float *o, int size,
                                      int P, int prng_bits, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      if (subnormals) {
          o[idx] = cast_p3109_signed_stochastic<true>(a[idx], P, (uint32_t)r[idx], prng_bits);
      } else {
          o[idx] = cast_p3109_signed_stochastic<false>(a[idx], P, (uint32_t)r[idx], prng_bits);
      }
  }
}

__global__ void p3109_unsigned_kernel_stochastic(float *__restrict__ a, 
                                      int *__restrict__ r, float *o, int size,
                                      int P, int prng_bits, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
      if (subnormals) {
          o[idx] = cast_p3109_unsigned_stochastic<true>(a[idx], P, (uint32_t)r[idx], prng_bits);
      } else {
          o[idx] = cast_p3109_unsigned_stochastic<false>(a[idx], P, (uint32_t)r[idx], prng_bits);
      }
  }
}