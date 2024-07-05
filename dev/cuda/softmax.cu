/* 
Low-precision float softmax along any dimension.

Compile example:
nvcc -O3 softmax.cu -o softmax

Simple implementation parallelizing over the rows to be softmaxed, one thread per row
./softmax 1 [dim=-3..2]

Efficient version using intra-warp and inter-warp reductions, block_size % 32 = 0
./softmax 2 [dim=-3..2]
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

        int base_index = outer_idx * strides.outer_stride + inner_idx;
        float* input = input_array + base_index;
        float* output = output_array + base_index;

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
/* Device (CUDA) softmax kernels */

__global__ void softmax_kernel1(float *input_array, float *output_array, DimStrides *strides, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if(id >= N) return;

    int outer_idx = id / strides->inner_size;
    int inner_idx = id % strides->inner_size;

    int base_index = outer_idx * strides->outer_stride + inner_idx;
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

void softmax_cuda1(float *input, float *output, const int *dims, int n_dims, int dim, int block_size) {
    DimStrides h_strides = dim_striding(dims, n_dims, dim);
    DimStrides *d_strides;
    cudaCheck(cudaMalloc(&d_strides, sizeof(DimStrides)));
    cudaCheck(cudaMemcpy(d_strides, &h_strides, sizeof(DimStrides), cudaMemcpyHostToDevice));
    // one thread per row to be softmaxed
    int N = h_strides.outer_size * h_strides.inner_size; // number of rows
    int blocks = N / block_size + (N % block_size != 0);
    softmax_kernel1<<<blocks, block_size>>>(input, output, d_strides, N);
    cudaCheck(cudaFree(d_strides));
}

__global__ void softmax_kernel2(float *input_array, float *output_array, DimStrides *strides) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    float* input = input_array + base_index;
    float* output = output_array + base_index;

    // compute the maximum value of the row
    
    float max = -INFINITY;
    // each thread computes part of the maximum by iterating over its corresponding
    // position along the row, as many times as required to cover all the row
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        if (input[idx] > max) {
            max = input[idx];
        }
    }
    // intra-warp maximum reduction
    // each thread now contains part of the maximum of the row, we combine these maximums
    // into a per-warp maximum, stored in the 0th thread of the warp (lane = 0)
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xFFFFFFFF, max, offset);
        if (other > max) {
            max = other;
        }
    }
    // store the warp-level maximum in shared memory
    if (lane == 0) {
        shared[warp] = max;
    }
    __syncthreads();
    // reduce the maximum of each warp into the 0th thread of the block (warp 0, lane 0)
    if(tid == 0) {
        max = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            if (shared[i] > max) {
                max = shared[i];
            }
        }
        // store the block's maximum (i.e row's maximum) in a fixed shared location
        shared[0] = max;
    }
    __syncthreads();
    // each thread reads the row's maximum
    max = shared[0];

    // each thread computes its exp(x[i] - max)
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = expf(input[idx] - max);
    }

    // compute the sum of exp(x[i] - max), using a similar approach as for the maximum,
    // but reducing a sum instead

    float sum = 0.f;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        sum += output[idx]; // we sum the previously computed exponentials
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    if (lane == 0) {
        shared[warp] = sum;
    }
    __syncthreads();
    if(tid == 0) {
        sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            sum += shared[i];
        }
        shared[0] = sum;
    }
    __syncthreads();
    sum = shared[0];

    // finally, divide each previously computed exponentials by the sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = output[idx] / sum;
    }
}

void softmax_cuda2(float *input, float *output, const int *dims, int n_dims, int dim, int block_size) {
    DimStrides h_strides = dim_striding(dims, n_dims, dim);
    DimStrides *d_strides;
    cudaCheck(cudaMalloc(&d_strides, sizeof(DimStrides)));
    cudaCheck(cudaMemcpy(d_strides, &h_strides, sizeof(DimStrides), cudaMemcpyHostToDevice));
    // one block per row to be softmaxed
    int blocks = h_strides.outer_size * h_strides.inner_size; // number of rows
    // block_size must be multiple of 32
    size_t shared_mem_size = block_size * sizeof(float);
    softmax_kernel2<<<blocks, block_size, shared_mem_size>>>(input, output, d_strides);
    cudaCheck(cudaFree(d_strides));
}


// ---------------------------------------------------------------------------------------
void softmax_cuda(int kernel_num, float *input, float *output,
                  const int *dims, int n_dims, int dim, int block_size) {
    switch (kernel_num) {
    case 1:
        softmax_cuda1(input, output, dims, n_dims, dim, block_size);
        break;
    case 2:
        softmax_cuda2(input, output, dims, n_dims, dim, block_size);
        break;
    default:
        printf("Invalid kernel number\n");
        exit(1);
    }
}


// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    setup_main();

    const int dims[] = {100, 50, 2549};
    const int n_dims = sizeof(dims)/sizeof(dims[0]);

    // which kernel version to use
    int version = 1;
    if(argc > 1) {
        version = atoi(argv[1]);
    }

    // which dimension to softmax
    int dim = 1;
    if (argc > 2) {
        dim = atoi(argv[2]);
    }
    if(dim <= -n_dims-1 || dim >= n_dims) {
        exit(1);
    }

    // create host tensors
    int numel = 1;
    for(int i = 0; i < n_dims; i++) {
        numel *= dims[i];
    }
    float *h_input = make_random_float(numel);
    float* h_output = make_zeros_float(numel);

    // create cuda tensors
    float *d_input, *d_output;
    cudaCheck(cudaMalloc(&d_input, numel * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, numel * sizeof(float)));
    cudaCheck(cudaMemcpy(d_input, h_input, numel * sizeof(float), cudaMemcpyHostToDevice));

    // cpu reference
    softmax_cpu(h_input, h_output, dims, n_dims, dim);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_cuda(version, d_input, d_output, dims, n_dims, dim, block_size);
        float tol = 1e-6f;
        validate_result(d_output, h_output, "softmax output", numel, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_cuda, version,
                                              d_input, d_output, dims, n_dims, dim, block_size);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // cleanup memory
    free(h_input);
    free(h_output);
    
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));
}
