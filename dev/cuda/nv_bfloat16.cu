/*
Compare results and performance of custom bfloat16 quantization with NVIDIA's __nv_bfloat16 routines.

Compile example:
nvcc -O3 nv_bfloat16.cu -o nv_bfloat16 -std=c++17 -lcublas

./nv_bfloat16
*/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "common.h"

__host__ __device__ float cast_bfloat16_nearest(float origin_float) {
    uint32_t bits;

    bits = FLOAT_TO_BITS(&origin_float);
    
    int32_t exp = ((bits >> 23) & 0xFF) - 127;  // unbiased exponent
    uint32_t sign = (bits >> 31) & 1;           // sign bit
    uint32_t mant = (bits & 0x7FFFFF);          // mantissa
    
    uint32_t outbits;
    
    float out;
    
    if(exp == 128) { // infinity or NaN, 128 is inf/nan, 2^(8-1) -1 = 127 is emax for IEEE 754
        // return the input unchanged
        return origin_float;
    }
    
    if((mant & 0x1FFFF) == 0x8000) { // round to nearest even tie round down, 1ffff is bottom 23 - 7 + 1 = 17 bits of the mantissa, 8000 is the tie point
    
        mant = mant & 0x7F0000; // truncate mantissa, 7f0000 is top 7 bits of mantissa
        exp = exp + 127;        // add bias
        outbits = (sign << 31 | exp << 23 | mant);
        out = BITS_TO_FLOAT(&outbits);
        return out;
    }

    mant = mant + (1 << (23 - 1 - 7)); // round to nearest

    if((mant >> 23) == 1) { // if overflow through rounding
        mant = 0;      // truncate mantissa
        exp = exp + 1; // add bias
    }
    mant = mant & 0x7F0000; // truncate mantissa

    exp = exp + 127; // add bias
    outbits = (sign << 31 | exp << 23 | mant);
    out = BITS_TO_FLOAT(&outbits);
    return out;
}

// ---------------------------------------------------------------------------------------
// CPU reference
void quantize_bfloat16_custom_cpu(const float *input, float *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = cast_bfloat16_nearest(input[i]);
    }
}


// ---------------------------------------------------------------------------------------
// CUDA kernels

// Use our custom quantization routine
__global__ void quantize_bfloat16_custom_kernel(const float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    output[i] = cast_bfloat16_nearest(input[i]);
}

void quantize_bfloat16_custom_cuda(const float *input, float *output, int N, int block_size) {
    int blocks = (N / block_size) + (N % block_size != 0);
    quantize_bfloat16_custom_kernel<<<blocks, block_size>>>(input, output, N);
}

// Use CUDA builtin __nv_bfloat16 casting
__global__ void quantize_bfloat16_nvidia_kernel(const float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    output[i] = __bfloat162float(__float2bfloat16(input[i]));
}

void quantize_bfloat16_nvidia_cuda(const float *input, float *output, int N, int block_size) {
    int blocks = (N / block_size) + (N % block_size != 0);
    quantize_bfloat16_nvidia_kernel<<<blocks, block_size>>>(input, output, N);
}

// Cast two floats at a time using CUDA __nv_bfloat162 cast
__global__ void quantize_bfloat16_2_nvidia_kernel(const float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(2*i >= N) return;
    if(2*i < N-1) {
        __nv_bfloat162 bf2 = __floats2bfloat162_rn(input[2*i], input[2*i+1]);
        // float2 f2 = __bfloat1622float2(bf2); // not found by compiler but exists in documentation?
        // output[2*i] = f2.x;
        // output[2*i+1] = f2.y;
        output[2*i] = __bfloat162float(bf2.x);
        output[2*i+1] = __bfloat162float(bf2.y);
    } else {
        output[2*i] = __bfloat162float(__float2bfloat16(input[2*i]));
    }
}

void quantize_bfloat16_2_nvidia_cuda(const float *input, float *output, int N, int block_size) {
    int threads = N / 2;
    int blocks = (threads / block_size) + (threads % block_size != 0);
    quantize_bfloat16_2_nvidia_kernel<<<blocks, block_size>>>(input, output, N);
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    setup_main();

    const int N = 1 << 24;

    float *h_input = make_random_float(N);
    float *h_output = make_zeros_float(N);

    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // cpu reference
    quantize_bfloat16_custom_cpu(h_input, h_output, N);

    printf("Comparing custom with __nv_bfloat16\n");
    quantize_bfloat16_nvidia_cuda(d_input, d_output, N, 64);
    validate_result(d_output, h_output, "output", N, 0.0f);

    printf("\nComparing custom with packed __nv_bfloat162\n");
    quantize_bfloat16_2_nvidia_cuda(d_input, d_output, N, 64);
    validate_result(d_output, h_output, "output", N, 0.0f);

    printf("All results match. Starting benchmarks.\n\n");

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];

        int repeat_times = 1000;

        float time_custom = benchmark_kernel(repeat_times, quantize_bfloat16_custom_cuda, d_input, d_output, N, block_size);
        float time_nvidia = benchmark_kernel(repeat_times, quantize_bfloat16_nvidia_cuda, d_input, d_output, N, block_size);
        float time_nvidia2 = benchmark_kernel(repeat_times, quantize_bfloat16_2_nvidia_cuda, d_input, d_output, N, block_size);

        printf("block_size %4d | time custom %.4f ms | time nvidia %.4f ms | time nvidia2 %.4f ms\n",
            block_size, time_custom, time_nvidia, time_nvidia2);
    }

    // memory cleanup
    free(h_input);
    free(h_output);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}