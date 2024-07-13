/* 
Low-precision float softmax forward along any dimension.

Compile example:
nvcc -O3 lut_log.cu -o lut_log -lcublas

Version 1 kernel explicitly quant(logf(input))
./lut_log 1

Version 2 kernel access shared memory to directly retrieve quant(logf(x))
which was precomputed.
./lut_log 2
*/


#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include "common.h"


/* Quantization function and wrapper */
// ---------------------------------------------------------------------------------------

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

__host__ __device__ __forceinline__ uint32_t
round_bitwise_nearest(uint32_t target, int man_bits) {
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 1 << (22 - man_bits);
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    // apply the mask
    // this is the analogue of how you would do round 
    // to nearest integer using the floor function: 
    // round(x) = floor(x + 0.5)
    return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

__host__ __device__ __forceinline__ uint32_t
clip_exponent(int exp_bits, int man_bits, uint32_t old_num,
              uint32_t quantized_num, bool saturate = false) {
  if (quantized_num == 0)
    return quantized_num;

  int quantized_exponent_store = quantized_num << 1 >> 24;
  int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
  int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

  uint32_t old_sign = old_num >> 31 << 31;
  // saturate or overflow
  if (quantized_exponent_store > max_exponent_store) {
    if (saturate) {
      uint32_t max_man =
          (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
      uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
      quantized_num = old_sign | max_num;
    } else {
      quantized_num =
          ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
      quantized_num = quantized_num | old_sign;
    }
  } else if (quantized_exponent_store < min_exponent_store) {
    uint32_t min_num = ((uint32_t)min_exponent_store << 23);
    uint32_t middle_num = ((uint32_t)(min_exponent_store - 1) << 23);
    uint32_t unsigned_quantized_num = quantized_num << 1 >> 1;
    if (unsigned_quantized_num > middle_num) {
      uint32_t old_sign = old_num >> 31 << 31;
      quantized_num = old_sign | min_num;
    } else {
      quantized_num = 0;
    }
  }
  return quantized_num;
}

__host__ __device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
                                 bool subnormal_support = true,
                                 bool saturate = false) {
  uint32_t target, quantize_bits;
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

__host__ __device__ float quant(float origin_float) {
    return cast_fp_nearest(origin_float, 2, 5);
}


// ---------------------------------------------------------------------------------------
/* Host (CPU) implementation of a simple vectorized log */
static void vector_log_cpu(float *input_array, float *output_array, int N) {
    for(int i = 0; i < N; i++) {
        output_array[i] = quant(logf(input_array[i]));
    }
}

// ---------------------------------------------------------------------------------------
/* Device (CUDA) softmax kernels */
__global__ void vector_log_kernel1(float *input_array, float *output_array, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    output_array[i] = quant(logf(input_array[i]));
}

void vector_log_cuda1(float *input, float *output, int N, int block_size) {
    int blocks = N / block_size + (N % block_size != 0);
    vector_log_kernel1<<<blocks, block_size>>>(input, output, N);
}

__global__ void vector_log_kernel2(float *input_array, float *output_array, int N, float* log_table) {
    extern __shared__ float shared[];
    for(int i = threadIdx.x; i < 256; i += blockDim.x) {
        shared[i] = log_table[i];
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    output_array[i] = shared[(int)input_array[i]];
}

void vector_log_cuda2(float *input, float *output, int N, float *d_log_table, int block_size) {
    int blocks = N / block_size + (N % block_size != 0);
    vector_log_kernel2<<<blocks, block_size, 256 * sizeof(float)>>>(input, output, N, d_log_table);
}

// ---------------------------------------------------------------------------------------
void vector_log_cuda(int kernel_num, float *input, float *output, int N, float *d_log_table, int block_size) {
    switch (kernel_num) {
    case 1:
        vector_log_cuda1(input, output, N, block_size);
        break;
    case 2:
        vector_log_cuda2(input, output, N, d_log_table, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    setup_main();

    const int N = 1 << 24;
    float *h_input = make_zeros_float(N);
    float *h_output = make_zeros_float(N);
    // random values between 0 and 255
    for(int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 256);
    }

    // cpu reference
    vector_log_cpu(h_input, h_output, N);

    // which kernel version to use
    int version = 1;
    if(argc > 1) {
        version = atoi(argv[1]);
    }

    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    float h_log_table[256];
    for(int i = 0; i < 256; i++) {
        h_log_table[i] = quant(logf((float)i));
    }
    float* d_log_table;
    cudaCheck(cudaMalloc(&d_log_table, sizeof(h_log_table)));
    cudaCheck(cudaMemcpy(d_log_table, h_log_table, sizeof(h_log_table), cudaMemcpyHostToDevice));

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        vector_log_cuda(version, d_input, d_output, N, d_log_table, block_size);
        float tol = 0.0f;
        validate_result(d_output, h_output, "output", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, vector_log_cuda, version,
                                              d_input, d_output, N, d_log_table, block_size);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanup memory
    free(h_input);
    free(h_output);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    cudaCheck(cudaFree(d_log_table));
}
