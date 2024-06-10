#include "quant_kernel.h"
#include "sim_helper.cu"
#include "bit_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        float* __restrict__ max_entry,
                                        int man_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    uint32_t max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    uint32_t max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6*BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index]+base_float;
    uint32_t target_bits = FLOAT_TO_BITS(&target_rebase);
    uint32_t rand_prob = (uint32_t) r[index];
    uint32_t quantized = round_bitwise_stochastic(target_bits, rand_prob, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized)-base_float;

    uint32_t quantize_bits = FLOAT_TO_BITS(&quantize_float) ;
    uint32_t clip_quantize = clip_max_exponent(man_bits-2, max_exp, quantize_bits);
    quantize_float = BITS_TO_FLOAT(&clip_quantize);
    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     float* __restrict__ max_entry,
                                     int man_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    uint32_t max_entry_bits = FLOAT_TO_BITS(&max_entry[index]);
    uint32_t max_exp = max_entry_bits << 1 >> 24 << 23;
    float base_float = 6*BITS_TO_FLOAT(&max_exp);

    float target_rebase = a[index]+base_float;
    uint32_t target_bits = FLOAT_TO_BITS(&target_rebase);
    uint32_t quantized = round_bitwise_nearest(target_bits, man_bits);
    float quantize_float = BITS_TO_FLOAT(&quantized)-base_float;

    uint32_t quantize_bits = FLOAT_TO_BITS(&quantize_float); 
    uint32_t clip_quantize = clip_max_exponent(man_bits-2, max_exp, quantize_bits); // sign bit, virtual bit
    quantize_float = BITS_TO_FLOAT(&clip_quantize);

    o[index] = quantize_float;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_sim_stochastic(float* __restrict__ a,
                                            float* __restrict__ r,
                                            float* o, int size,
                                            float* max_entry,
                                            int wl) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int exponent = ((int) extract_exponent(max_entry));
    int sigma = exponent-(wl-1);
    o[index] = round(a[index], r[index], sigma);
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void block_kernel_sim_nearest(float* __restrict__ a,
                                         float* o, int size,
                                         float* max_entry,
                                         int wl) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int exponent = ((int) extract_exponent(max_entry));
    int sigma = exponent-(wl-1);
    o[index] = nearest_round(a[index], sigma);
  }
}
