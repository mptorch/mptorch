#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include "common.h"

struct DimStrides
{
    int B;
    int T;
    int outer_stride;
    int dim_size;
    int dim_stride;
};

/* Helper function for tensor striding */
// ---------------------------------------------------------------------------------------
DimStrides dim_striding(const int *norm_dims, int n_norm, const int *dims, int n_dims) {
    DimStrides strides;
    
	int real_dims[n_norm];
	for (int i = 0; i < n_norm; i++){
		real_dims[i] = (n_dims + (norm_dims[i] % n_dims)) % n_dims;
	}

	strides.dim_size = 1;
	for (int i : real_dims){
		strides.dim_size *= dims[i];
	}

	int min_dim = real_dims[n_norm - 1];
	int max_dim = real_dims[0];

    printf("\nmin: %d, max: %d\n", min_dim, max_dim);

	strides.B = 1;
	for (int i = 0; i < min_dim; i++){
		strides.B *= dims[i];
	}

	strides.T = 1;
	for (int i = max_dim + 1; i < n_dims; i++){
		strides.T *= dims[i];
	}

    printf("\nT: %d, B: %d\n", strides.T, strides.B);

	strides.outer_stride = strides.dim_size * strides.T;

    return strides;
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

__host__ __device__ float quant_acc(float origin_float) {
    return cast_fp_nearest(origin_float, 7, 8, true, false);
}

__host__ __device__ float quant_mul(float origin_float) {
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

__host__ __device__ float quant_div(float origin_float) {
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

__host__ __device__ float quant_sqrt(float origin_float) {
    return cast_fp_nearest(origin_float, 10, 8, true, false);
}

// ---------------------------------------------------------------------------------------
// CPU version
static void cpu_quantize_layernorm_forward(const float *in_arr, float *out_arr, 
                                           const float *w_array, const float *b_array, 
                                           const float eps, DimStrides &strides){
	for (int i = 0; i < strides.B * strides.T; i++){
		int b = i / strides.T;
		int t = i % strides.T;

		int base_index = (b * strides.outer_stride) + t;
		const float* input = in_arr + base_index;
		float* output = out_arr + base_index;

		float m = 0.0f;
		for (int k = 0; k < strides.dim_size; k++){
	  		int idx = k * strides.T;
	  		m = quant_acc(m + input[idx]);
		}
		m = quant_div(m/strides.dim_size);

		float variance = 0;
        for (int k = 0; k < strides.dim_size; k++){
            int idx = k * strides.T;
            float shift = quant_acc(input[idx] - m);
            float shift_2 = quant_mul(shift * shift);
            variance = quant_acc(variance + shift_2);
        }
        variance = quant_div(variance/strides.dim_size);

		float rad = quant_acc(variance + eps);
		float std = quant_sqrt(sqrtf(rad));
		for (int k = 0; k < strides.dim_size; k++){
	  		int idx = k * strides.T;
	  		float numer = quant_acc(input[idx] - m);
	  		float norm = quant_div(numer/std);
	  		float out = quant_mul(w_array[k] * norm);
	  		output[idx] = out + b_array[k];
		}
	}
}

// ---------------------------------------------------------------------------------------
// GPU kernels




// ---------------------------------------------------------------------------------------
// Kernel launchers


// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv) {
    setup_main();

    const int norm_dims[] = {-1, -2};
    const int n_norm = sizeof(norm_dims)/sizeof(norm_dims[0]);
    const int dims[] = {2, 3, 4};
    const int n_dims = sizeof(dims)/sizeof(dims[0]);
    const float eps = 1e-5;

    printf("\nn_norm: %d, n_dim: %d\n", n_norm, n_dims);

    auto strides = dim_striding(norm_dims, n_norm, dims, n_dims);

    // host tensors
    int numel = 1;
    for (int i = 0; i < n_dims; i++){
    	numel *= dims[i];
    }
    float* h_input = make_random_float(numel);
    float* h_output = make_zeros_float(numel);

    float* w_array = make_ones_float(strides.dim_size);
    float* b_array = make_zeros_float(strides.dim_size);

    // cpu reference
    cpu_quantize_layernorm_forward(h_input, h_output, w_array, b_array, eps, strides);

    for (int i = 0; i < numel; i++){
        std::cout << h_output[i] << " \n";
    }


    free(h_input);
    free(h_output);
    free(w_array);
    free(b_array);

    return 0;
}