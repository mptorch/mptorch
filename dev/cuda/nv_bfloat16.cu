/*
Compare results and performance of custom bfloat16 quantization with NVIDIA's __nv_bfloat16 routines.

Compile example:
nvcc -O3 nv_bfloat16.cu -o nv_bfloat16 -std=c++17 -lcublas

./nv_bfloat16
*/

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "common.h"

// ---------------------------------------------------------------------------------------
// CPU + GPU kernels
__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest_impl(uint32_t target, int man_bits)
{
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

__host__ __device__ uint32_t clip_subnormal_range_exponent_impl1(int exp_bits, int man_bits, uint32_t old_num,
                                                        uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) - man_bits + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num = quantized_num | old_sign;
        quantized_num = offset * quantized_num;
    }
    return quantized_num;
}

__host__ __device__ uint32_t clip_normal_range_exponent_impl1(int exp_bits, int man_bits, uint32_t old_num,
                                                           uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int old_exponent_store = old_num << 1 >> 24;
    int quantized_exponent_store = quantized_num << 1 >> 24;
    int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // saturate or overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        if (saturate)
        {
            uint32_t max_man =
                (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
            uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
            quantized_num = old_sign | max_num;
        }
        else
        {
            quantized_num = ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
            quantized_num = quantized_num | old_sign;
        }
    } // underflow or round to smallest nonzero normal value
    else if (old_exponent_store < min_exponent_store)
    {
        uint32_t offset = (old_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > 0);
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= old_sign;
    }
    return quantized_num;
}

__host__ __device__ float cast_fp_nearest_impl1(float origin_float, int man_bits, int exp_bits,
                                       bool subnormal_support = true,
                                       bool saturate = false)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && subnormal_support)
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest_impl(target, exp_diff);
            quantize_bits =
                clip_subnormal_range_exponent_impl1(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 128)
        {
            quantized = origin_float;
        }
        // normal value range or overflow
        else
        {
            quantize_bits = round_bitwise_nearest_impl(target, man_bits);
            quantize_bits =
                clip_normal_range_exponent_impl1(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

// ---------------------------------------------------------------------------------------
// CPU reference
void quantize_bfloat16_custom_cpu(const float *input, float *output, int N) {
    for (int i = 0; i < N; i++) {
        output[i] = cast_fp_nearest_impl1(input[i], 7, 8, true, false);
    }
}

// ---------------------------------------------------------------------------------------
// CUDA kernels

// Use our custom quantization routine
__global__ void quantize_bfloat16_custom_kernel(const float *input, float *output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= N) return;
    output[i] = cast_fp_nearest_impl1(input[i], 7, 8, true, false);
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