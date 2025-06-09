/*
Low-precision float softmax forward along any dimension.

Compile example:
nvcc -O3 softmax_backward.cu -o softmax_backward -lcublas

Simple implementation parallelizing over the rows that were softmaxed, one thread per row
./softmax_backward 1 [dim=-3..2]

Efficient version using intra-warp and inter-warp reductions, block_size % 32 = 0
./softmax_backward 2 [dim=-3..2]

*/

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include "common.h"

struct DimSizes
{
    int outer;
    int inner;
    int channel;
};

/* Helper function for tensor striding */
// ---------------------------------------------------------------------------------------

DimSizes partition_tensor(const int *dims, int n_dims, int dim)
{
    DimSizes sizes;
    int real_dim = (n_dims + (dim % n_dims)) % n_dims;
    sizes.outer = 1;
    sizes.channel = dims[real_dim];
    sizes.inner = 1;
    for (int i = 0; i < real_dim; ++i)
    {
        sizes.outer *= dims[i];
    }
    for (int i = real_dim + 1; i < n_dims; ++i)
    {
        sizes.inner *= dims[i];
    }
    return sizes;
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
    bool noquantize = (man_bits >= 23) && (exp_bits >= 8);

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

__host__ __device__ float quant_add(float origin_float)
{
    return cast_fp_nearest(origin_float, 22, 8, true, false);
}

__host__ __device__ float quant_mul(float origin_float)
{
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

// ---------------------------------------------------------------------------------------
/* Host (CPU) implementation of a simple softmax backward */
static void softmax_backward_cpu(const float *input_array, const float *out_gradient, float *output_array, const int *dims, int n_dims, int dim)
{
    auto sizes = partition_tensor(dims, n_dims, dim);

    for (int i = 0; i < sizes.outer * sizes.inner; ++i)
    {
        int outer_idx = i / sizes.inner;
        int inner_idx = i % sizes.inner;

        int base_index = outer_idx * sizes.channel * sizes.inner + inner_idx;
        const float *input = input_array + base_index;
        const float *grad = out_gradient + base_index;
        float *output = output_array + base_index;

        float weighted_grad_sum = 0.f;
        for (int k = 0; k < sizes.channel; ++k)
        {
            int idx = k * sizes.inner;
            float prod = quant_mul(input[idx] * grad[idx]);
            weighted_grad_sum = quant_add(weighted_grad_sum + prod);
        }

        for (int k = 0; k < sizes.channel; ++k)
        {
            int idx = k * sizes.inner;
            float a = quant_add(grad[idx] - weighted_grad_sum);
            output[idx] = quant_mul(a * input[idx]);
        }
    }
}

// ---------------------------------------------------------------------------------------
/* Device (CUDA) softmax kernels */

__global__ void softmax_backward_kernel1(const float *__restrict__ input_array, const float *__restrict__ out_gradient, float *output_array,
                                         const DimSizes sizes, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id >= N)
        return;

    int outer_idx = id / sizes.inner;
    int inner_idx = id % sizes.inner;

    int base_index = outer_idx * sizes.channel * sizes.inner + inner_idx;
    const float *input = input_array + base_index;
    const float *grad = out_gradient + base_index;
    float *output = output_array + base_index;

    float weighted_grad_sum = 0.f;
    for (int k = 0; k < sizes.channel; ++k)
    {
        int idx = k * sizes.inner;
        float prod = quant_mul(input[idx] * grad[idx]);
        weighted_grad_sum = quant_add(weighted_grad_sum + prod);
    }

    for (int k = 0; k < sizes.channel; ++k)
    {
        int idx = k * sizes.inner;
        float a = quant_add(grad[idx] - weighted_grad_sum);
        output[idx] = quant_mul(a * input[idx]);
    }
}

__global__ void softmax_backward_kernel2(const float *__restrict__ input_array, const float *__restrict__ out_gradient, float *output_array,
                                         const DimSizes sizes)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / sizes.inner;
    int inner_idx = blockIdx.x % sizes.inner;

    int base_index = outer_idx * sizes.channel * sizes.inner + inner_idx;
    const float *input = input_array + base_index;
    const float *grad = out_gradient + base_index;
    float *output = output_array + base_index;

    // Compute the input row sum and weighted sum

    float *shared_weighted_grad_sum = &shared[0];

    float weighted_grad_sum = 0.f;
    for (int k = tid; k < sizes.channel; k += blockDim.x)
    {
        int idx = k * sizes.inner;
        float prod = quant_mul(input[idx] * grad[idx]);
        weighted_grad_sum = quant_add(weighted_grad_sum + prod);
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        weighted_grad_sum = quant_add(weighted_grad_sum + __shfl_down_sync(0xFFFFFFFF, weighted_grad_sum, offset));
    }
    if (lane == 0)
    {
        shared_weighted_grad_sum[warp] = weighted_grad_sum;
    }
    __syncthreads();
    if (tid == 0)
    {
        weighted_grad_sum = shared_weighted_grad_sum[0];
        for (int i = 1; i < warpsPerBlock; i++)
        {
            weighted_grad_sum = quant_add(weighted_grad_sum + shared_weighted_grad_sum[i]);
        }
        shared_weighted_grad_sum[0] = weighted_grad_sum;
    }
    __syncthreads();
    weighted_grad_sum = shared_weighted_grad_sum[0];

    // Last step, subsrtact the weighted sum from the gradient, and divide by input sum
    for (int k = tid; k < sizes.channel; k += blockDim.x)
    {
        int idx = k * sizes.inner;
        float a = quant_add(grad[idx] - weighted_grad_sum);
        output[idx] = quant_mul(a * input[idx]);
    }
}

// ---------------------------------------------------------------------------------------
/* Kernel launchers */

void softmax_backward_cuda1(float *input, float *out_grad, float *output, const int *dims, int n_dims, int dim, int block_size)
{
    DimSizes sizes = partition_tensor(dims, n_dims, dim);
    // one thread per row that was softmaxed
    int N = sizes.outer * sizes.inner; // number of rows
    int blocks = N / block_size + (N % block_size != 0);
    softmax_backward_kernel1<<<blocks, block_size>>>(input, out_grad, output, sizes, N);
}

void softmax_backward_cuda2(float *input, float *out_grad, float *output, const int *dims, int n_dims, int dim, int block_size)
{
    DimSizes sizes = partition_tensor(dims, n_dims, dim);
    // one block per row that was softmaxed
    int blocks = sizes.outer * sizes.inner; // number of rows
    // block_size must be multiple of 32
    size_t shared_mem_size = (block_size / 32) * sizeof(float);
    softmax_backward_kernel2<<<blocks, block_size, shared_mem_size>>>(input, out_grad, output, sizes);
}

void softmax_backward_cuda(int kernel_num, float *input, float *out_grad, float *output,
                           const int *dims, int n_dims, int dim, int block_size)
{
    switch (kernel_num)
    {
    case 1:
        softmax_backward_cuda1(input, out_grad, output, dims, n_dims, dim, block_size);
        break;
    case 2:
        softmax_backward_cuda2(input, out_grad, output, dims, n_dims, dim, block_size);
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

    const int dims[] = {125, 81, 384};
    const int n_dims = sizeof(dims) / sizeof(dims[0]);

    // which kernel version to use
    int version = 1;
    if (argc > 1)
    {
        version = atoi(argv[1]);
    }

    // which dimension to softmax
    int dim = 1;
    if (argc > 2)
    {
        dim = atoi(argv[2]);
    }
    if (dim <= -n_dims - 1 || dim >= n_dims)
    {
        exit(1);
    }

    // create host tensors
    int numel = 1;
    for (int i = 0; i < n_dims; i++)
    {
        numel *= dims[i];
    }
    float *h_input = make_random_float(numel);
    float *h_grad = make_random_float(numel);
    float *h_output = make_zeros_float(numel);

    // create cuda tensors
    float *d_input, *d_grad, *d_output;
    cudaCheck(cudaMalloc(&d_input, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&d_grad, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, numel * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input, numel * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_grad, h_grad, numel * sizeof(float), cudaMemcpyHostToDevice));

    // cpu reference
    softmax_backward_cpu(h_input, h_grad, h_output, dims, n_dims, dim);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_backward_cuda(version, d_input, d_grad, d_output, dims, n_dims, dim, block_size);
        float tol = 1e-1f;
        float rtol = 1e-1f;
        validate_result(d_output, h_output, "output", numel, tol, rtol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j)
    {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_gpu_kernel(repeat_times, softmax_backward_cuda, version,
                                                  d_input, d_grad, d_output, dims, n_dims, dim, block_size);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    printf("\nBenchmarking CPU version.");
    int repeat_times = 10;
    namespace chr = std::chrono;
    chr::steady_clock::time_point begin = chr::steady_clock::now();
    for (int i = 0; i < repeat_times; i++)
    {
        softmax_backward_cpu(h_input, h_grad, h_output, dims, n_dims, dim);
    }
    chr::steady_clock::time_point end = chr::steady_clock::now();
    auto elapsed_time_us = chr::duration_cast<chr::microseconds>(end - begin).count();
    float average_time_ms = ((float)elapsed_time_us / (float)repeat_times) / 1000.f;
    printf(" %.4f ms\n ", average_time_ms);

    // cleanup memory
    free(h_input);
    free(h_grad);
    free(h_output);

    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_grad));
    cudaCheck(cudaFree(d_output));
}
