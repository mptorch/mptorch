/*
Kernels for IEEE-754 down casting from binary32 to a lower precision format.
Payload is still a binary32 value.

Compile example:
nvcc -O3 p3109_signed_nearest.cu -o p3109_signed_nearest -std=c++17 -lcublas

version 1 attempted to make the code as compact as possible, while also 
maintaining readability; bit shifts and masking are used aplenty
./p3109_signed_nearest 1

*/

#include <cuda_runtime.h>
#include "common.h"

enum class SaturateMode {
    SATURATE,
    NO_OVERFLOW,
    OVERFLOWS
};

// --------------------------------------------------------------------------------------
// I/O pairs to sanity check the CPU reference code (E5M2 without subnormals, RNE)
uint32_t test_inputs[] = {
    0b00111000000000000000000000000000,
    0b00110111000000000000000000000000,
    0b00110110100000000000000000000000,
    0b00110110100000000000000000000001,
    0b01000111010000000000000000000000,
    0b01000111011000000000000000000000,
    0b00110110000000000000000000000000};
uint32_t test_outputs[] = {
    0b00111000000000000000000000000000,
    0b00110111000000000000000000000000,
    0b00000000000000000000000000000000,
    0b00110111000000000000000000000000,
    0b01000111010000000000000000000000,
    0b01000111010000000000000000000000,
    0b00000000000000000000000000000000};


uint32_t round_bitwise_nearest_cpu(uint32_t target, int man_bits) {
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 1 << (22 - man_bits);
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

uint32_t p3109_clip_exponent_cpu(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, SaturateMode saturation_mode, bool subnormal) {

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

float cast_p3109_signed_nearest_cpu(float origin_float, int P, SaturateMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;    
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {             // inf/Nan case
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

    uval8 = round_bitwise_nearest_cpu(uval32, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

float cast_p3109_unsigned_nearest_cpu(float origin_float, int P, SaturateMode saturation_mode, bool subnormals) {
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
    
    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {  // return inf/Nan expect in the case of no_overflow && inf
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

    uval8 = round_bitwise_nearest_cpu(uval32, man_bits - subnormal_shift);
    uval8 = p3109_clip_exponent_cpu(exp_bits, man_bits, uval32, uval8, saturation_mode, subnormals);
    fval8 = BITS_TO_FLOAT(&uval8);

    return fval8;
}

void p3109_signed_nearest_cpu(float *o, float *a, int N, int P, bool is_signed, SaturateMode saturation_mode, bool subnormals) {
  for (int i = 0; i < N; ++i){
    if(is_signed){
      o[i] = cast_p3109_signed_nearest_cpu(a[i], P, saturation_mode, subnormals);
    }
    else{
      o[i] = cast_p3109_unsigned_nearest_cpu(a[i], P, saturation_mode, subnormals);
    }
  }
}

// ---------------------------------------------------------------------------------------
// GPU kernels
__device__ __forceinline__ uint32_t
round_bitwise_nearest(uint32_t target, int man_bits) {
  uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
  uint32_t machine_eps = 1 << (22 - man_bits);
  int offset = (down == machine_eps);
  uint32_t add_r = target + machine_eps;
  return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

__device__ __forceinline__ uint32_t
p3109_clip_exponent(int exp_bits, int man_bits, uint32_t old_num, uint32_t quantized_num, SaturateMode saturation_mode, bool subnormal) {

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

__device__ float cast_p3109_signed_nearest(float origin_float, int P, SaturateMode saturation_mode, bool subnormals) {

    int exp_bits = 8-P;
    int man_bits = P-1;    
    int subnormal_shift = 0;
    uint32_t uval32, uval8;
    float fval8;

    uval32 = FLOAT_TO_BITS(&origin_float);

    int exp_val = (uval32 << 1 >> 24) - 127;
    uint32_t man_val = uval32 & 0x7FFFFF;

    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {             // inf/Nan case
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

__device__ float cast_p3109_unsigned_nearest(float origin_float, int P, SaturateMode saturation_mode, bool subnormals) {
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
    
    if (exp_val == 128 && !(saturation_mode == SaturateMode::NO_OVERFLOW && man_val == 0)) {  // return inf/Nan expect in the case of no_overflow && inf
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

__global__ void p3109_signed_nearest_gpu(float *o, float *__restrict__ a, int N, int P, bool is_signed, SaturateMode saturation_mode, bool subnormals)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N)
    {
      if(is_signed){
        o[index] = cast_p3109_signed_nearest(a[index], P, saturation_mode, subnormals);
      }else{
        o[index] = cast_p3109_unsigned_nearest(a[index], P, saturation_mode, subnormals);
      }
    }
}

// ---------------------------------------------------------------------------------------
// Kernel launchers
void p3109_signed_nearest1(float *o, float *a, int N, int P, const int block_size, bool is_signed, SaturateMode saturation_mode, bool subnormals)
{
    const int grid_size = ceil_div(N, block_size);
    p3109_signed_nearest_gpu<<<grid_size, block_size>>>(o, a, N, P, is_signed, saturation_mode, subnormals);
    cudaCheck(cudaGetLastError());
}

// kernel version dispatch
void p3109_signed_nearest(int kernel_num, float *o, float *a, int N, int P, const int block_size, bool is_signed, SaturateMode saturation_mode, bool subnormals)
{
    switch (kernel_num)
    {
    case 1:
        p3109_signed_nearest1(o, a, N, P, block_size, is_signed, saturation_mode, subnormals);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(EXIT_FAILURE);
    }
}


// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    setup_main();

    // Get signed or unsigned from user input
    bool is_signed;
    std::cout << "signed or unsigned, enter true (1) for signed, false (0) for unsigned: ";
    int is_signed_input;
    std::cin >> is_signed_input;
    is_signed = static_cast<bool>(is_signed_input);

    // Get subnormals from user input
    bool subnormals;
    std::cout << "subnormals or not, enter true (1) for subnormals, false (0) for not: ";
    int subnormals_input;
    std::cin >> subnormals_input;
    subnormals = static_cast<bool>(subnormals_input);

    SaturateMode saturation_mode = SaturateMode::SATURATE;


    // read the kernel number from the command line
    int kernel_num = 1;
    if (argc > 1)
    {
        kernel_num = atoi(argv[1]);
    }

    // sanity check the CPU reference code
    for (int j = 0; j < sizeof(test_inputs) / sizeof(uint32_t); ++j)
    {
        float fres = cast_p3109_signed_nearest_cpu(BITS_TO_FLOAT(&test_inputs[j]), 3, saturation_mode, subnormals);
        uint32_t res = FLOAT_TO_BITS(&fres);
        if (res != test_outputs[j])
        {
            printf("index = %d\n", j);
            print_float(res);
            printf("\nvs\n");
            print_uint32(test_outputs[j]);
            printf("\n");
            //exit(EXIT_FAILURE);
        }
    }

    int N = 1 << 24;
    int P = 3;

    float *x = make_random_float(N);
    float *y = (float *)malloc(N * sizeof(float));

    printf("Using kernel %d\n", kernel_num);

    // compute reference CPU solution
    p3109_signed_nearest_cpu(y, x, N, P, is_signed, saturation_mode, subnormals);

    // move data to the GPU
    float *d_x, *d_y;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_y, N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        p3109_signed_nearest(kernel_num, d_y, d_x, N, P, block_size, is_signed, saturation_mode, subnormals);

        float tol = 0.0f;
        validate_result(d_y, y, "y", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float elapsed_time = benchmark_kernel(repeat_times, p3109_signed_nearest, 
                kernel_num, d_y, d_x, N, P, block_size, is_signed, saturation_mode, subnormals);

        // estimate memory bandwidth achieved
        // for each output element, we do 1 read and 1 write, 4 bytes each
        long memory_ops = N * 2 * (int)sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // free memory
    free(x);
    free(y);

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_y));

    return 0;
}