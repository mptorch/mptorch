/* 
Low-precision float softmax.

Compile example:
nvcc -O3 softmax.cu -o softmax

Run with:
./softmax [dim=0|1|2]
*/


#include <cuda_runtime.h>
#include <cmath>
#include "common.h"


struct DimStrides
{
    int outer_size;
    int inner_size;
    int outer_stride;
    int dim_size;
    int dim_stride;
};

/* Helper function for tensor striding */
DimStrides dim_striding(const int *dims, int n_dims, int dim) {
    DimStrides strides;
    int real_dim = (n_dims + (dim % n_dims)) % n_dims;
    strides.outer_size = 1;
    strides.dim_size = dims[real_dim];
    strides.inner_size = 1;
    for (int i = 0; i < real_dim; ++i) {
        strides.outer_size *= dims[i];
    }
    for (int i = real_dim + 1; i < n_dims; ++i) {
        strides.inner_size *= dims[i];
    }
    strides.dim_stride = strides.inner_size;
    strides.outer_stride = strides.dim_size * strides.dim_stride;
    return strides;
}

// ---------------------------------------------------------------------------------------
/* Host (CPU) implementation of a simple softmax */
static void softmax_cpu(float *input_array, float *output_array, const int *dims, int n_dims, int dim) {
    auto strides = dim_striding(dims, n_dims, dim);

    for (int i = 0; i < strides.outer_size * strides.inner_size; ++i) {
        int outer_idx = i / strides.inner_size;
        int inner_idx = i % strides.inner_size;

        float* input = input_array + outer_idx * strides.outer_stride + inner_idx;
        float* output = output_array + outer_idx * strides.outer_stride + inner_idx;

        float max = input[0];
        for (int k = 1; k < strides.dim_size; ++k) {
            int idx = k * strides.dim_stride;
            if (input[idx] > max) {
                max = input[idx];
            }
        }

        float sum = 0.0f;
        for (int k = 0; k < strides.dim_size; ++k) {
            int idx = k * strides.dim_stride;
            float exp_x = expf(input[idx] - max);
            output[idx] = exp_x;
            sum += exp_x;
        }
        for (int k = 0; k < strides.dim_size; ++k) {
            int idx = k * strides.dim_stride;
            output[idx] = output[idx] / sum;
        }
    }
}

// ---------------------------------------------------------------------------------------
/* Device (CUDA) implementations of softmax */
__global__ void softmax_kernel(float *input_array, float *output_array, DimStrides *strides, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int outer_idx = id / strides->inner_size;
    int inner_idx = id % strides->inner_size;

    int base_index = outer_idx * strides->outer_stride + inner_idx;
    if(base_index >= N) return;

    float* input = input_array + base_index;
    float* output = output_array + base_index;

    float max = input[0];
    for (int k = 1; k < strides->dim_size; ++k) {
        int idx = k * strides->dim_stride;
        if (input[idx] > max) {
            max = input[idx];
        }
    }

    float sum = 0.0f;
    for (int k = 0; k < strides->dim_size; ++k) {
        int idx = k * strides->dim_stride;
        float exp_x = expf(input[idx] - max);
        output[idx] = exp_x;
        sum += exp_x;
    }
    for (int k = 0; k < strides->dim_size; ++k) {
        int idx = k * strides->dim_stride;
        output[idx] = output[idx] / sum;
    }
}

void softmax_cuda(float *input, float *output, const int *dims, int n_dims, int dim, int block_size) {
    DimStrides h_strides = dim_striding(dims, n_dims, dim);
    DimStrides *d_strides;
    cudaCheck(cudaMalloc(&d_strides, sizeof(DimStrides)));
    cudaCheck(cudaMemcpy(d_strides, &h_strides, sizeof(DimStrides), cudaMemcpyHostToDevice));
    int N = h_strides.outer_size * h_strides.inner_size * h_strides.dim_size;
    int blocks = N / block_size + (N % block_size != 0);
    softmax_kernel<<<blocks, block_size>>>(input, output, d_strides, N);
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    setup_main();

    const int dims[] = {100, 50, 259};
    const int n_dims = sizeof(dims)/sizeof(dims[0]);

    int dim = 1;
    if (argc > 1) {
        dim = atoi(argv[1]);
    }
    if(dim <= -n_dims-1 || dim >= n_dims) {
        exit(1);
    }

    // create host tensors
    int N = 1;
    for(int i = 0; i < n_dims; i++) {
        N *= dims[i];
    }
    float *h_input = make_random_float(N);
    float* h_output = make_zeros_float(N);

    // create cuda tensors
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // cpu reference
    softmax_cpu(h_input, h_output, dims, n_dims, dim);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_cuda(d_input, d_output, dims, n_dims, dim, block_size);
        float tol = 1e-7f;
        validate_result(d_output, h_output, "softmax output", N, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_cuda, d_input, d_output, dims, n_dims, dim, block_size);

        // estimate memory bandwidth achieved
        // for each output element, we do 3 reads and 2 writes, 4 bytes each
        long memory_ops = N * 5 * (int)sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // cleanup memory
    free(h_input);
    free(h_output);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}
