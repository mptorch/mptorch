/* 
Low-precision float softmax.

Compile example:
nvcc -O3 softmax.cu -o softmax

Run with:
./softmax [dim]
*/


#include <cuda_runtime.h>
#include <cmath>
#include "common.h"


struct TensorStrides
{
    int outer_size;
    int inner_size;
    int outer_stride;
    int dim_size;
    int dim_stride;
};

/* Helper function for tensor striding */
void tensor_strides(const int *dims, int n_dims, int dim, TensorStrides &strides) {
  // dimension we are iterating across, needed for negative dims
  int real_dim = (n_dims + (dim % n_dims)) % n_dims;
  
  // columns to iterate over
  strides.outer_size = 1;
  for (int i = 0; i < real_dim; ++i) {
    strides.outer_size *= dims[i];
  }

  // rows to iterate over
  strides.inner_size = 1;
  for (int i = real_dim + 1; i < n_dims; ++i) {
    strides.inner_size *= dims[i];
  }

  // size of the dimesnion we are iterating accross
  strides.dim_stride = strides.inner_size;

  // how much to iterate to get to next set of elements in dim
  int dim_size = dims[real_dim];
  strides.dim_size = dim_size;
  strides.outer_stride = dim_size * strides.dim_stride;
}

// ---------------------------------------------------------------------------------------
/* Host (CPU) implementation of a simple softmax */
static void softmax_cpu(const float *input, float *output, const int *dims, int n_dims, int dim) {
    TensorStrides strides;
    tensor_strides(dims, n_dims, dim, strides);

    for (int L = 0; L < strides.outer_size * strides.inner_size; ++L) {
        int i = L / strides.inner_size;
        int j = L % strides.inner_size;

        // get max value of the current elements softmax is done on,
        // this is for ensuring the exponentials don't overflow
        float max = input[(i*strides.outer_stride) + j];
        for (int k = 0; k < strides.dim_size; ++k) {
            int idx = (i*strides.outer_stride) + (k*strides.dim_stride) + j;
            if (input[idx] > max) {
                max = input[idx];
            }
        }

        float sum = 0.0f;
        // calculates the sum of exponentials and store the intermediate exponentials
        for (int k = 0; k < strides.dim_size; ++k) {
            int idx = (i*strides.outer_stride) + (k*strides.dim_stride) + j;
            float exp_x = expf(input[idx] - max);
            output[idx] = exp_x;
            sum += exp_x;
        }
        // Divides all outputs by the current sum
        for (int k = 0; k < strides.dim_size; ++k) {
            int idx = (i*strides.outer_stride) + (k*strides.dim_stride) + j;
            output[idx] /= sum;
        }
    }
}


__global__ void softmax_kernel(const float *input, float *output, int inner_size, int outer_stride, int dim_size, int dim_stride, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int i = id / inner_size;
    int j = id % inner_size;

    if(i*outer_stride + j > N) return;

    float max = input[i*outer_stride + j];
    for (int k = 0; k < dim_size; ++k) {
        int idx = (i*outer_stride) + (k*dim_stride) + j;
        if (input[idx] > max) {
            max = input[idx];
        }
    }

    float sum = 0.0f;
    for (int k = 0; k < dim_size; ++k) {
        int idx = (i*outer_stride) + (k*dim_stride) + j;
        float exp_x = expf(input[idx] - max);
        output[idx] = exp_x;
        sum += exp_x;
    }
    for (int k = 0; k < dim_size; ++k) {
        int idx = (i*outer_stride) + (k*dim_stride) + j;
        output[idx] /= sum;
    }
}

void softmax_cuda(const float *input, float *output, const int *dims, int n_dims, int dim, int threads_per_block) {
    TensorStrides strides;
    tensor_strides(dims, n_dims, dim, strides);
    int num_threads = strides.outer_size * strides.inner_size * strides.dim_size;
    int blocks = num_threads / threads_per_block + (num_threads % threads_per_block != 0);
    softmax_kernel<<<blocks, threads_per_block>>>(input, output, strides.inner_size, strides.outer_stride, strides.dim_size, strides.dim_stride, num_threads);
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    setup_main();

    // create host tensors
    const int dims[] = {100, 50, 259};
    const int n_dims = sizeof(dims)/sizeof(dims[0]);
    int size = 1;
    for(int i = 0; i < n_dims; i++) {
        size *= dims[i];
    }
    float *h_input = make_random_float(size);
    float* h_output = make_zeros_float(size);

    // create cuda tensors
    float *d_input, *d_output;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    int dim = 1;
    if (argc > 1) {
        dim = atoi(argv[1]);
    }

    // cpu reference
    softmax_cpu(h_input, h_output, dims, n_dims, dim);

    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax_cuda(d_input, d_output, dims, n_dims, dim, block_size);
        float tol = 1e-7f;
        validate_result(d_output, h_output, "softmax output", size, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); ++j) {
        int block_size = block_sizes[j];
        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, softmax_cuda, d_input, d_output, dims, n_dims, dim, block_size);

        // estimate memory bandwidth achieved
        // for each output element, we do 3 reads and 2 writes, 4 bytes each
        long memory_ops = size * 5 * (int)sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    // cleanup memory
    free(h_input);
    free(h_output);
    
    cudaFree(d_input);
    cudaFree(d_output);
}
