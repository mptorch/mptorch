#include <cuda_runtime.h>

#include "common.h"

// --------------------------------------------------------------------------------------

uint32_t test_inputs[] = {





};

uint32_t test_outputs[] = {




    
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

uint32_t p3109_clip_exponent_cpu(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, bool saturate, bool subnormal) 
{ 
    if (quantized_num == 0) 
        return quantized_num;
  
    int spec_exp = (man_bits == 0) ? 1 : 0; 
    int quantized_exponent_store = (quantized_num >> 23) & 0xFF;
    int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp; 
    uint32_t max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
    uint32_t man_val = quantized_num & 0x7FFFFF; 
    uint32_t old_sign = old_num & 0x80000000;

    if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && max_man < man_val)) {
        if (saturate) { 
            quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
        } else {
            quantized_num = old_sign | 0x7F800000;
        }
    } else if (quantized_exponent_store < min_exponent_store) {
        if (subnormal) {
            int subnormal_shift = min_exponent_store - quantized_exponent_store;
            int min_subnormals_exp = min_exponent_store - man_bits;
            uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
            uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
            if (subnormal_shift <= man_bits) {
     
            } else if ((old_num & 0x7FFFFFFF) > middle_num) {
            quantized_num = old_sign | min_num;
            } else {
            quantized_num = 0;
            }
        } else {
            uint32_t min_num = (uint32_t)min_exponent_store << 23;
            uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
            if ((quantized_num & 0x7FFFFFFF) > middle_num) {
            quantized_num = old_sign | min_num;
            } else {
            quantized_num = 0;
            }
        }
    }

    return quantized_num;
}

template<bool subnormals>
float cast_p3109_signed_stochastic_cpu(float origin_float, int P, int prng_bits) 
{

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = true;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1; 
    int min_exp = spec_exp - max_exp;
    int subnormal_shift = 0;

    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }
    
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - (man_bits - subnormal_shift))) - 1;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);

    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {           
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template<>
float cast_p3109_signed_stochastic_cpu<false>(float origin_float, int P, int prng_bits) 
{

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = false;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1; 
    int min_exp = spec_exp - max_exp;
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - man_bits)) - 1;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {            
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <bool subnormals>
float cast_p3109_unsigned_stochastic(float origin_float, int P, int prng_bits) 
{ 
    
    if(origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = true;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1;  
    int min_exp = spec_exp - max_exp;
    int subnormal_shift = 0;
    
    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }
    
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - (man_bits - subnormal_shift))) - 1;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {          
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}


template<>
float cast_p3109_unsigned_stochastic_cpu<false>(float origin_float, int P, int prng_bits) 
{

    if (origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = false;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (sign == 1){
        return NAN;
    }
    
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - man_bits)) - 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {             
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

void p3109_quantize_stochastic_cpu(Tensor a, int P, int prng_bits, bool is_signed, bool subnormals)
{   
    auto o = zeros_like(a);
    int size = a.numel;
    if (is_signed){
        for (int = 0; i < size; ++i){
            p3109_signed_kernel_stochastic( , , size, P, prng_bits, subnormals);
        }
    } else {
        for (int = 0; i < size; ++i){



        }
    }
  

    
}

// --------------------------------------------------------------------------------------
// GPU kernels

__device__ __forceinline__ uint32_t round_bitwise_stochastic(uint32_t target, uint32_t rand_prob, int man_bits) 
{ 
    uint32_t mask = (1 << (23 - man_bits)) - 1; 
    uint32_t add_r = target + (rand_prob & mask);
    uint32_t quantized = add_r & ~mask; 
    return quantized;
}

__device__ __forceinline__ uint32_t p3109_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, bool saturate, bool subnormal) 
{ 
    if (quantized_num == 0) 
        return quantized_num;
  
    int spec_exp = (man_bits == 0) ? 1 : 0; 
    int quantized_exponent_store = (quantized_num >> 23) & 0xFF;
    int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 1) + 127 + spec_exp; 
    uint32_t max_man = (((1u << man_bits) - 1u) & ~1u) << (23 - man_bits);
    uint32_t man_val = quantized_num & 0x7FFFFF; 
    uint32_t old_sign = old_num & 0x80000000;

    if (quantized_exponent_store > max_exponent_store || ((quantized_exponent_store == max_exponent_store) && max_man < man_val)) {
        if (saturate) { 
            quantized_num = old_sign | ((uint32_t)max_exponent_store << 23) | max_man;
        } else {
            quantized_num = old_sign | 0x7F800000;
        }
    } else if (quantized_exponent_store < min_exponent_store) {
        if (subnormal) {
            int subnormal_shift = min_exponent_store - quantized_exponent_store;
            int min_subnormals_exp = min_exponent_store - man_bits;
            uint32_t min_num = ((uint32_t)min_subnormals_exp << 23);
            uint32_t middle_num = ((uint32_t)(min_subnormals_exp - 1) << 23);
            if (subnormal_shift <= man_bits) {
     
            } else if ((old_num & 0x7FFFFFFF) > middle_num) {
            quantized_num = old_sign | min_num;
            } else {
            quantized_num = 0;
            }
        } else {
            uint32_t min_num = (uint32_t)min_exponent_store << 23;
            uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
            if ((quantized_num & 0x7FFFFFFF) > middle_num) {
            quantized_num = old_sign | min_num;
            } else {
            quantized_num = 0;
            }
        }
    }

    return quantized_num;
}

template<bool subnormals>
__global__ float cast_p3109_signed_stochastic(float origin_float, int P, int prng_bits) 
{

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = true;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1; 
    int min_exp = spec_exp - max_exp;
    int subnormal_shift = 0;

    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }
    
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - (man_bits - subnormal_shift))) - 1;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);

    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {           
        return origin_float;
    }

    uval8 = round_bitwise_stochastic(uval32, random_number, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template<>
__global__ float cast_p3109_signed_stochastic<false>(float origin_float, int P, int prng_bits) 
{

    int exp_bits = 8-P;
    int man_bits = P-1;  
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = false;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1; 
    int min_exp = spec_exp - max_exp;
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - man_bits)) - 1;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {            
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

template <bool subnormals>
__global__ float cast_p3109_unsigned_stochastic(float origin_float, int P, int prng_bits) 
{ 
    
    if(origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = true;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1;  
    int min_exp = spec_exp - max_exp;
    int subnormal_shift = 0;
    
    if(((min_exp - exp_val) <= man_bits) && (exp_val < min_exp) && (subnormals)){ 
      subnormal_shift = min_exp - exp_val;
    }
    
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - (man_bits - subnormal_shift))) - 1;
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {          
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}


template<>
__global__ float cast_p3109_unsigned_stochastic<false>(float origin_float, int P, int prng_bits) 
{

    if (origin_float < 0){
        return NAN;
    }

    int exp_bits = 8 - P + 1;
    int man_bits = P - 1; 
    int spec_exp = (P == 1) ? 1 : 0;
    bool subnormals = false;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    int max_exp = (1 << (exp_bits -1)) - 1;
    int min_exp = spec_exp - max_exp;

    if (sign == 1){
        return NAN;
    }
    
    uint64_t upper_limit = static_cast<uint32_t>(std::pow(2, 23 - man_bits)) - 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> distrib(0, upper_limit);
    uint64_t random_number = distrib(gen);

    if (exp_val == 128) {             
        return origin_float;
    }

    uval8 = round_bitwise_stochastic_cpu(uval32, random_number, man_bits);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, true, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

Tensor p3109_quantize_stochastic_cuda(Tensor a, int P, int prng_bits, bool is_signed, bool subnormals)
{
  auto o = zeros_like(a);
  int size = a.numel(); // gets number of elements in tensor a
  int blockSize = 1024;
  int blockNums = (size + blockSize - 1) / blockSize;

  if (is_signed){ // signed
      p3109_signed_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, P, prng_bits, subnormals);
  } else {  // unsigned
      p3109_unsigned_kernel_stochastic<<<blockNums, blockSize>>>(
      a.data_ptr<float>(), o.data_ptr<float>(), size, P, prng_bits, subnormals);
  }

  return o;
}


