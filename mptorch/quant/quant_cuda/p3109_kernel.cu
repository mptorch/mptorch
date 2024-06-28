#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "p3109_kernel.h"

__host__ __device__ float cast_p3109_signed_nearest(float origin_float, int P, SaturationMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;    
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturationMode::NO_OVERFLOW && man_val == 0)) {             // inf/Nan case
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
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__host__ __device__ float cast_p3109_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, SaturationMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturationMode::NO_OVERFLOW && man_val == 0)) {          
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
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__host__ __device__ float cast_p3109_signed_troncate(float origin_float, int P, SaturationMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturationMode::NO_OVERFLOW && man_val == 0)) {          
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

    uval8 = uval32 >> (23-man_bits+subnormal_shift) << (23-man_bits+subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}


__host__ __device__ float cast_p3109_unsigned_nearest(float origin_float, int P, SaturationMode saturation_mode, bool subnormals) {
    // we had talks abt the following for unsigned, P = 1:
    // 0: 0000 0000
    // NaN: FE or FF (currently. it is 1000 0000 in this revision)
    // inf: FE of FF (currently, it is FF)
    // max float: FD (currently, it is FE)
    // we have a variation that allows for this in the special condition where 
    
    if (origin_float < 0){
      return NAN;
    }
 
    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;
    
    if (exp_val == 128 && !(saturation_mode == SaturationMode::NO_OVERFLOW && man_val == 0)) {  // return inf/Nan expect in the case of no_overflow && inf
        return origin_float;
    }

    if (subnormals){
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;
           
        if((min_exp - exp_val) <= man_bits && exp_val < min_exp && subnormals){ 
            subnormal_shift = min_exp - exp_val;
        }
    }

    uval8 = round_bitwise_nearest(uval32, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__host__ __device__ float cast_p3109_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, SaturationMode saturation_mode, bool subnormals) { 
    
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

    if (exp_val == 128 && !(saturation_mode == SaturationMode::NO_OVERFLOW && man_val == 0)) {
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
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

__host__ __device__ float cast_p3109_unsigned_troncate(float origin_float, int P, SaturationMode saturation_mode, bool subnormals) { 
    
    if(origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturationMode::NO_OVERFLOW && man_val == 0)) {
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
    
    uval8 = uval32 >> (23-man_bits+subnormal_shift) << (23-man_bits+subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}


__global__ void p3109_signed_kernel_nearest(float *__restrict__ a, float *o, int size, int P, SaturationMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_p3109_signed_nearest(a[idx], P, saturation_mode, subnormals);
  }
}

__global__ void p3109_unsigned_kernel_nearest(float *__restrict__ a, float *o, int size,
                                      int P, SaturationMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_p3109_unsigned_nearest(a[idx], P, saturation_mode, subnormals);
  }
}

__global__ void p3109_signed_kernel_stochastic(float *__restrict__ a,
                                      int *__restrict__ r, float *o, int size,
                                      int P, int prng_bits, SaturationMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_p3109_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, saturation_mode, subnormals);
  }
}

__global__ void p3109_unsigned_kernel_stochastic(float *__restrict__ a, 
                                      int *__restrict__ r, float *o, int size,
                                      int P, int prng_bits, SaturationMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_p3109_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, saturation_mode, subnormals);
  }
}

__global__ void p3109_signed_kernel_troncate(float *__restrict__ a, float *o, int size,
                                      int P, SaturationMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_p3109_signed_troncate(a[idx], P, saturation_mode, subnormals);
  }
}

__global__ void p3109_unsigned_kernel_troncate(float *__restrict__ a, float *o, int size,
                                      int P, SaturationMode saturation_mode, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_p3109_unsigned_troncate(a[idx], P, saturation_mode, subnormals);
  }
}