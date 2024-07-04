/* 
Low-precision float softmax.

Compile example:
nvcc -O3 softmax.cu -o softmax

Run with:
./softmax
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
static void softmax_cpu(float *input, float *output, const int *dims, int n_dims, int dim) {
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

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    setup_main();

    const int dims[] = {100, 50, 259};
    const int n_dims = sizeof(dims)/sizeof(dims[0]);

    int size = 1;
    for(int i = 0; i < n_dims; i++) {
        size *= dims[i];
    }
    float *h_input = make_random_float(size);

    int dim = 1;
    if (argc > 1) {
        dim = atoi(argv[1]);
    }

    // cpu reference
    float* h_output = make_zeros_float(size);
    softmax_cpu(h_input, h_output, dims, n_dims, dim);
}
