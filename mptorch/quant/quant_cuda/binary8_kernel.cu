#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include "binary8_kernel.h"

__host__ __device__ float cast_binary8_signed_nearest(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {

    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }
    
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && (exp_val < min_exp)) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                              : round_bitwise_nearest(uval32, man_bits - subnormal_shift);

    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_signed_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {

    const int exp_bits = 8-P;
    const int man_bits = P-1; 
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }
    
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals){
        int spec_exp = (P == 1) ? 1 : 0;
        int max_exp = (1 << (exp_bits -1)) - 1;
        int min_exp = spec_exp - max_exp;

        if(((min_exp - exp_val) <= man_bits) && exp_val < min_exp){ 
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

__host__ __device__ float cast_binary8_signed_truncate(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {

    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    const uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    const uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }
    
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    int subnormal_shift = 0;
    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = uval32 >> (23-man_bits+subnormal_shift) << (23-man_bits+subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}


__host__ __device__ float cast_binary8_unsigned_nearest(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) {
    
    if (origin_float < 0) return NAN;   

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }
    
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 9 - P;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = (P == 1) - max_exp;
        
        if ((min_exp - exp_val) <= man_bits && exp_val < min_exp){ 
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = (P == 1) ? round_bitwise_nearest_p1(uval32, man_bits - subnormal_shift)
                               : round_bitwise_nearest(uval32, man_bits - subnormal_shift);
    
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    
    return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_binary8_unsigned_stochastic(float origin_float, int P, uint32_t rand_prob, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) { 
    
    if (origin_float < 0) return NAN;   

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }
    
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 8 - P + 1;
    const int man_bits = P - 1; 
    int subnormal_shift = 0;

    if (subnormals){   
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits -1)) - 1;
        const int min_exp = spec_exp - max_exp;        

        if(((min_exp - exp_val) <= man_bits) && exp_val < min_exp){ 
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

__host__ __device__ float cast_binary8_unsigned_truncate(float origin_float, int P, OverflowPolicy overflow_policy, bool subnormals) { 
    
    if (origin_float < 0) return NAN;   

    uint32_t uval32 = FLOAT_TO_BITS(&origin_float);
    const int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    // Early return for inf/NaN case
    if (exp_val == 128 && man_val != 0) { // if input nan return nan anyway
        return origin_float;
    }
    
    if (overflow_policy == OverflowPolicy::SATURATE_INFTY && exp_val == 128 && man_val == 0) { //if input is infty and overflow_policy is SATURATE_INFTY
        return origin_float;
    }

    const int exp_bits = 8 - P;
    const int man_bits = P - 1;
    int subnormal_shift = 0;

    if (subnormals) {
        const int spec_exp = (P == 1) ? 1 : 0;
        const int max_exp = (1 << (exp_bits - 1)) - 1;
        const int min_exp = spec_exp - max_exp;

        if (((min_exp - exp_val) <= man_bits) && exp_val < min_exp) {
            subnormal_shift = min_exp - exp_val;
        }
    }

    uint32_t uval8 = uval32 >> (23-man_bits+subnormal_shift) << (23-man_bits+subnormal_shift);
    uval8 = binary8_clip_exponent(exp_bits, man_bits, uval32, uval8, overflow_policy, subnormals);
    return BITS_TO_FLOAT(&uval8);
}

__host__ __device__ float cast_bfloat16_nearest(float origin_float) {   //THIS HAS SUBNORMALS AND IS UNTESTED

    uint32_t bits;

    bits = FLOAT_TO_BITS(&origin_float);
    
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  //unbiased exponent
    uint32_t sign = (bits >> 31) & 1;  //sign bit
    uint32_t mant = (bits & 0x7FFFFF); //mantissa
    
    uint32_t outbits;
    
    float out;
    
    if(exp == 128){ //infinity or NaN, 128 is inf/nan, 2^(8-1) -1 = 127 is emax for IEEE 754
        //infinity case
        if(mant == 0){
            outbits = (sign << 31| 0xFF << 23| 0); //infinities are propagated
            uint32_t *tempout = &outbits;
            out = *((float *)tempout);
            return out;
        } else {
        //NaN case
            outbits = (sign << 31| 0xFF << 23| mant);  //NaNs are propagated
            uint32_t *tempout = &outbits;
            out = *((float *)tempout);
            return out;
        }
    }
    
    if((mant & 0x1FFFF) == 0x8000) {//round to nearest even tie round down, 1ffff is bottom 23 - 7 + 1 = 17 bits of the mantissa, 8000 is the tie point
    
        mant = mant & 0x7F0000; //truncate mantissa, 7f0000 is top 7 bits of mantissa
        exp = exp + 127; //add bias
        outbits = (sign << 31| exp << 23| mant);
        uint32_t *tempout = &outbits;
        out = *((float *)tempout);
        return out;
    }

    mant = mant + (1 << (23 - 1 - 7)); //round to nearest

    if((mant >> 23) == 1) {//if overflow through rounding
        mant = 0; //truncate mantissa
        exp = exp + 1; //add bias
    }
    mant = mant & 0x7F0000; //truncate mantissa

    exp = exp + 127; //add bias
    outbits = (sign << 31| exp << 23| mant);
    uint32_t *tempout = &outbits;
    out = *((float *)tempout);
    return out;
}

__host__ __device__ float cast_bfloat16_stochastic(float origin_float, uint32_t rand_prob, int prng_bits) {   //THIS HAS SUBNORMALS AND IS UNTESTED

    uint32_t bits;

    bits = FLOAT_TO_BITS(&origin_float);
    
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  //unbiased exponent
    uint32_t sign = (bits >> 31) & 1;  //sign bit
    uint32_t mant = (bits & 0x7FFFFF); //mantissa
    
    uint32_t outbits;
    
    float out;
    
    if(exp == 128){ //infinity or NaN, 128 is inf/nan, 2^(8-1) -1 = 127 is emax for IEEE 754
        //infinity case
        if(mant == 0){
            outbits = (sign << 31| 0xFF << 23| 0); //infinities are propagated
            uint32_t *tempout = &outbits;
            out = *((float *)tempout);
            return out;
        } else {
        //NaN case
            outbits = (sign << 31| 0xFF << 23| mant);  //NaNs are propagated
            uint32_t *tempout = &outbits;
            out = *((float *)tempout);
            return out;
        }
    }

    rand_prob = rand_prob << 9 >> 9;
    rand_prob = rand_prob & ~(1 << (23 - 7 - prng_bits) - 1);

    uint32_t mask = (1 << (23 - 7)) - 1; //mask for random bits
    mant = mant + (rand_prob & mask); // adding random bits to target (which is not masked)

    // mant = mant + (1 << (23 - 1 - 7)); //round to nearest


    if((mant >> 23) == 1) {//if overflow through rounding
        mant = 0; //truncate mantissa
        exp = exp + 1; //add bias
    }
    mant = mant & 0x7F0000; //truncate mantissa

    exp = exp + 127; //add bias
    outbits = (sign << 31| exp << 23| mant);
    uint32_t *tempout = &outbits;
    out = *((float *)tempout);
    return out;
}


__global__ void binary8_signed_kernel_nearest(float *__restrict__ a, float *o, int size, int P, OverflowPolicy overflow_policy, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_binary8_signed_nearest(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_nearest(float *__restrict__ a, float *o, int size,
                                      int P, OverflowPolicy overflow_policy, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_binary8_unsigned_nearest(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void binary8_signed_kernel_stochastic(float *__restrict__ a,
                                      int *__restrict__ r, float *o, int size,
                                      int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_binary8_signed_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_stochastic(float *__restrict__ a, 
                                      int *__restrict__ r, float *o, int size,
                                      int P, int prng_bits, OverflowPolicy overflow_policy, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_binary8_unsigned_stochastic(a[idx], P, (uint32_t)r[idx], prng_bits, overflow_policy, subnormals);
  }
}

__global__ void binary8_signed_kernel_truncate(float *__restrict__ a, float *o, int size,
                                      int P, OverflowPolicy overflow_policy, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_binary8_signed_truncate(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void binary8_unsigned_kernel_truncate(float *__restrict__ a, float *o, int size,
                                      int P, OverflowPolicy overflow_policy, bool subnormals) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_binary8_unsigned_truncate(a[idx], P, overflow_policy, subnormals);
  }
}

__global__ void bfloat16_kernel_nearest(float *__restrict__ a, float *o, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_bfloat16_nearest(a[idx]);
  }
}

__global__ void bfloat16_kernel_stochastic(float *__restrict__ a, int *__restrict__ r,
                                           float *o, int size, int prng_bits) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
        o[idx] = cast_bfloat16_stochastic(a[idx], (uint32_t)r[idx], prng_bits);
  }
}