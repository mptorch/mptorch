#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include "common.h"

/* Helper function for tensor striding */
// ---------------------------------------------------------------------------------------
void dim_striding(const int *norm_dims, int n_norm, const int *dims, int n_dims, int &B, int &T, int &C)
{
    int real_dims[n_norm];
    for (int i = 0; i < n_norm; i++)
    {
        real_dims[i] = (n_dims + (norm_dims[i] % n_dims)) % n_dims;
    }

    C = 1;
    for (int i : real_dims)
    {
        C *= dims[i];
    }

    int min_dim = real_dims[n_norm - 1];
    int max_dim = real_dims[0];

    B = 1;
    for (int i = 0; i < min_dim; i++)
    {
        B *= dims[i];
    }

    T = 1;
    for (int i = max_dim + 1; i < n_dims; i++)
    {
        T *= dims[i];
    }
}

/* Quantization function and wrapper */
// ---------------------------------------------------------------------------------------

#define FLOAT_TO_BITS(x) (*reinterpret_cast<uint32_t *>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float *>(x))

__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest(uint32_t target, int man_bits)
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

__host__ __device__ uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
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

__host__ __device__ uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                              uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

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
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= old_sign;
    }
    return quantized_num;
}

__host__ __device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
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
            quantize_bits = not_uflow * round_bitwise_nearest(target, exp_diff);
            quantize_bits =
                clip_exponent_with_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
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
            quantize_bits = round_bitwise_nearest(target, man_bits);
            quantize_bits =
                clip_exponent_without_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}

__host__ __device__ float quant_acc(float origin_float)
{
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

__host__ __device__ float quant_mul(float origin_float)
{
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

__host__ __device__ float quant_div(float origin_float)
{
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

__host__ __device__ float quant_sqrt(float origin_float)
{
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

// ---------------------------------------------------------------------------------------
// CPU version
static void layernorm_forward_cpu(const float *in_arr, float *out_arr,
                                  const float *w_array, const float *b_array,
                                  const float eps, int B, int T, int C)
{
    for (int i = 0; i < B * T; i++)
    {
        int b = i / T;
        int t = i % T;

        int base_index = (b * C * T) + t;
        const float *input = in_arr + base_index;
        float *output = out_arr + base_index;

        float m = 0.0f;
        for (int k = 0; k < C; k++)
        {
            int idx = k * T;
            m = quant_acc(m + input[idx]);
        }
        m = quant_div(m / C);

        float variance = 0;
        for (int k = 0; k < C; k++)
        {
            int idx = k * T;
            float shift = quant_acc(input[idx] - m);
            float shift_2 = quant_mul(shift * shift);
            variance = quant_acc(variance + shift_2);
        }
        variance = quant_div(variance / C);

        float rad = quant_acc(variance + eps);
        float std = quant_sqrt(sqrtf(rad));
        for (int k = 0; k < C; k++)
        {
            int idx = k * T;
            float numer = quant_acc(input[idx] - m);
            float norm = quant_div(numer / std);
            float out = quant_mul(w_array[k] * norm);
            output[idx] = out + b_array[k];
        }
    }
}

// ---------------------------------------------------------------------------------------
// GPU kernels

__global__ void layernorm_forward_kernel1(const float *__restrict__ in_arr, float *out_arr,
                                          const float *w_array, const float *b_array,
                                          const float eps, int B, int T, int C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > N)
        return;

    int b = i / T;
    int t = i % T;

    int base_index = (b * C * T) + t;
    const float *input = in_arr + base_index;
    float *output = out_arr + base_index;

    float m = 0.0f;
    for (int k = 0; k < C; k++)
    {
        int idx = k * T;
        m = quant_acc(m + input[idx]);
    }
    m = quant_div(m / C);

    float variance = 0;
    for (int k = 0; k < C; k++)
    {
        int idx = k * T;
        float shift = quant_acc(input[idx] - m);
        float shift_2 = quant_mul(shift * shift);
        variance = quant_acc(variance + shift_2);
    }
    variance = quant_div(variance / C);

    float rad = quant_acc(variance + eps);
    float std = quant_sqrt(sqrtf(rad));
    for (int k = 0; k < C; k++)
    {
        int idx = k * T;
        float numer = quant_acc(input[idx] - m);
        float norm = quant_div(numer / std);
        float out = quant_mul(w_array[k] * norm);
        output[idx] = out + b_array[k];
    }
}

__global__ void layernorm_forward_kernel2(const float *__restrict__ in_arr, float *out_arr,
                                          const float *w_array, const float *b_array,
                                          const float eps, int B, int T, int C)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // groups of 32 threads (which warp the thread belongs to)
    int lane = threadIdx.x % warpSize; // a warp has 32 lanes (id of the thread in a warp)

    int warpsPerBlock = blockDim.x / warpSize;

    int b = blockIdx.x / T;
    int t = blockIdx.x % T;

    int base_index = (b * C * T) + t;
    const float *input = in_arr + base_index;
    float *output = out_arr + base_index;

    // compute mean by reducing sum of elements then dividing
    float m_sum = 0.0f;
    for (int k = tid; k < C; k += blockDim.x)
    {
        int idx = k * T;
        m_sum = quant_acc(m_sum + input[idx]);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        m_sum = quant_acc(m_sum + __shfl_down_sync(0xffffffff, m_sum, offset));
    }
    if (lane == 0)
    {
        shared[warp] = m_sum;
    }
    __syncthreads();
    if (tid == 0)
    {
        m_sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            m_sum = quant_acc(m_sum + shared[i]);
        }
        shared[0] = m_sum / C;
    }
    __syncthreads();
    float m = shared[0];

    float v_sum = 0;
    for (int k = tid; k < C; k += blockDim.x)
    {
        int idx = k * T;
        float shift = quant_acc(input[idx] - m);
        float shift_2 = quant_mul(shift * shift);
        v_sum = quant_acc(v_sum + shift_2);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        v_sum = quant_acc(v_sum + __shfl_down_sync(0xffffffff, v_sum, offset));
    }
    if (lane == 0)
    {
        shared[warp] = v_sum;
    }
    __syncthreads();
    if (tid == 0)
    {
        v_sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            v_sum = quant_acc(v_sum + shared[i]);
        }
        shared[0] = v_sum / C;
    }
    __syncthreads();
    float variance = shared[0];

    float rad = quant_acc(variance + eps);
    float std = quant_sqrt(sqrtf(rad));
    for (int k = tid; k < C; k += blockDim.x)
    {
        int idx = k * T;
        float numer = quant_acc(input[idx] - m);
        float norm = quant_div(numer / std);
        float out = quant_mul(w_array[k] * norm);
        output[idx] = out + b_array[k];
    }
}

// TODO: Create kernel using cooperative groups
// __global__ void layernorm_forward_kernel2(const float* __restrict__ in_arr, float* out_arr,
//                                   const float* w_array, const float* b_array,
//                                   const float eps, int B, int T, int C){
// }

// ---------------------------------------------------------------------------------------
// Kernel launchers
void layernorm_forward_cuda1(const float *in_arr, float *out_arr,
                             const float *w_array, const float *b_array,
                             const float eps, int B, int T, int C,
                             int block_size)
{
    int N = B * T;
    int blocks = N / block_size + (N % block_size != 0);
    layernorm_forward_kernel1<<<blocks, block_size>>>(in_arr, out_arr, w_array, b_array, eps, B, T, C, N);
}

void layernorm_forward_cuda2(const float *in_arr, float *out_arr,
                             const float *w_array, const float *b_array,
                             const float eps, int B, int T, int C,
                             int block_size)
{
    int blocks = B * T;
    size_t shared_mem_size = (block_size / 32) * sizeof(float);
    layernorm_forward_kernel2<<<blocks, block_size, shared_mem_size>>>(in_arr, out_arr, w_array, b_array, eps, B, T, C);
}

void layernorm_forward_cuda(int kernel_num, const float *in_arr, float *out_arr,
                            const float *w_array, const float *b_array,
                            const float eps, int B, int T, int C,
                            int block_size)
{
    switch (kernel_num)
    {
    case 1:
        layernorm_forward_cuda1(in_arr, out_arr, w_array, b_array, eps, B, T, C, block_size);
        break;
    case 2:
        layernorm_forward_cuda2(in_arr, out_arr, w_array, b_array, eps, B, T, C, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    setup_main();

    const int norm_dims[] = {-1, -2};
    const int n_norm = sizeof(norm_dims) / sizeof(norm_dims[0]);
    const int dims[] = {40, 60, 80};
    const int n_dims = sizeof(dims) / sizeof(dims[0]);
    const float eps = 1e-5;

    int B, T, C;
    dim_striding(norm_dims, n_norm, dims, n_dims, B, T, C);

    // which kernel to use
    int version = 1;
    if (argc > 1)
    {
        version = atoi(argv[1]);
    }

    // host tensors
    int numel = 1;
    for (int i = 0; i < n_dims; i++)
    {
        numel *= dims[i];
    }
    float *h_input = make_random_float(numel);
    float *h_output = make_zeros_float(numel);
    float *h_weight = make_ones_float(C);
    float *h_bias = make_zeros_float(C);

    // compute cpu reference
    layernorm_forward_cpu(h_input, h_output, h_weight, h_bias, eps, B, T, C);

    // device tensors (move data to gpu)
    float *d_input, *d_output, *d_weight, *d_bias;
    cudaCheck(cudaMalloc(&d_input, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&d_weight, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_bias, C * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, numel * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input, numel * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_weight, h_weight, C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_bias, h_bias, C * sizeof(float), cudaMemcpyHostToDevice));

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        layernorm_forward_cuda(version, d_input, d_output, d_weight, d_bias, eps, B, T, C, block_size);

        float tol = 1e-1f;
        validate_result(d_output, h_output, "output", numel, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_gpu_kernel(repeat_times, layernorm_forward_cuda,
                                                  version, d_input, d_output, d_weight, d_bias,
                                                  eps, B, T, C, block_size);
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    printf("\nBenchmarking CPU version.");
    int repeat_times = 10;
    namespace chr = std::chrono;
    chr::steady_clock::time_point begin = chr::steady_clock::now();
    for (int i = 0; i < repeat_times; i++)
    {
        layernorm_forward_cpu(h_input, h_output, h_weight, h_bias, eps, B, T, C);
    }
    chr::steady_clock::time_point end = chr::steady_clock::now();
    auto elapsed_time_us = chr::duration_cast<chr::microseconds>(end - begin).count();
    float average_time_ms = ((float)elapsed_time_us / (float)repeat_times) / 1000.f;
    printf(" %.4f ms\n ", average_time_ms);

    free(h_input);
    free(h_output);
    free(h_weight);
    free(h_bias);

    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
    cudaCheck(cudaFree(d_weight));
    cudaCheck(cudaFree(d_bias));

    return 0;
}