#pragma once
#include "quant_kernel.h"

// forward pass of layer normalization
// outputs means, rstds, and normalized tensor
// means and rstds used later for the backward pass
template<class Qacc, class Qmul, class Qdiv, class Qsqrt>
__global__ void layernorm_forward_impl(const float* __restrict__ in_arr, const float* __restrict__ w_array, const float* __restrict__ b_array,
                                    float* out_arr, float* m_array, float* r_array,
                                    float eps, const DimSizes sizes,
                                    Qacc quant_acc, Qmul quant_mul, Qdiv quant_div, Qsqrt quant_sqrt)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // groups of 32 threads (which warp the thread belongs to)
    int lane = threadIdx.x % warpSize; // a warp has 32 lanes (id of the thread in a warp)

    int warpsPerBlock = blockDim.x / warpSize;

    int b = blockIdx.x / sizes.inner;
    int t = blockIdx.x % sizes.inner;

    int base_index = (b * (sizes.channel * sizes.inner)) + t;
    const float* input = in_arr + base_index;
    float* output = out_arr + base_index;

    // compute mean by reducing sum of elements then dividing by size of channel
    float m_sum = 0.0f;
    for (int k = tid; k < sizes.channel; k += blockDim.x){
        int idx = k * sizes.inner;
        m_sum = quant_acc(m_sum + input[idx]);
    }
    // each thread contains part of the sum, so we combine these values
    for (int offset = warpSize/2; offset > 0; offset /= 2){
        m_sum = quant_acc(m_sum + __shfl_down_sync(0xffffffff, m_sum, offset));
    }
    // store sum for mean in shared memory when at 0th thread of warp
    if (lane == 0){
        shared[warp] = m_sum;
    }
    __syncthreads();
    // reduce sum of each warp into 0th thread of block, warp 0 and lane 0
    if (tid == 0){
        m_sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++){
            m_sum = quant_acc(m_sum + shared[i]);
        }
        // store computed mean in fixed location of memory
        shared[0] = m_sum/sizes.channel;
    }
    __syncthreads();

    // get computed mean from shared memory
    float m = shared[0];

    // compute variance by reducing sum of elements then dividing by size of channel
    // done similarly to mean
    float v_sum = 0;
    for (int k = tid; k < sizes.channel; k += blockDim.x){
        int idx = k * sizes.inner;
        float shift = quant_acc(input[idx] - m);
        float shift_2 = quant_mul(shift * shift);
        v_sum = quant_acc(v_sum + shift_2);
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2){
        v_sum = quant_acc(v_sum + __shfl_down_sync(0xffffffff, v_sum, offset));
    }
    if (lane == 0){
        shared[warp] = v_sum;
    }
    __syncthreads();
    if (tid == 0){
        v_sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++){
            v_sum = quant_acc(v_sum + shared[i]);
        }
        shared[0] = v_sum/sizes.channel;
    }
    __syncthreads();

    // get computer variance from shared memory
    float variance = shared[0];
    
    // computer the layer normalization formula with quantization
    float rad = quant_acc(variance + eps);
    float std = quant_sqrt(sqrtf(rad));
    for (int k = tid; k < sizes.channel; k += blockDim.x){
        int idx = k * sizes.inner;
        float numer = quant_acc(input[idx] - m);
        float norm = quant_div(numer/std);
        float out = quant_mul(w_array[k] * norm);
        output[idx] = out + b_array[k];
    }

    // output mean and rstd to respective arrays
    // these values will be used later in the backward pass 
    m_array[b * sizes.inner + t] = m;
    r_array[b * sizes.inner + t] = quant_div(1.0f/std); 
}

// first pass of the backward pass
// used to calculate only the input gradient
// saves some values for later second pass to calculate gradient of weights and biases
template<class Qacc, class Qmul, class Qdiv>
__global__ void layernorm_backward_first_pass_impl(const float* __restrict__ in_arr, const float* __restrict__ out_grad, 
                                                const float* __restrict__ w_array, const float* __restrict__ b_array,
                                                const float* __restrict__ m_array, const float* __restrict__ r_array,
                                                float* out_arr, float* xhat_gradient, const DimSizes sizes,
                                                Qacc quant_acc, Qmul quant_mul, Qdiv quant_div)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= sizes.outer*sizes.inner) return;

    int b = i / sizes.inner;
    int t = i % sizes.inner;

    int base_index = (b * sizes.channel * sizes.inner) + t;
    const float* input = in_arr + base_index;
    const float* gradient = out_grad + base_index;
    float* output = out_arr + base_index;
    float* xhat_grad = xhat_gradient + base_index;

    // get out current mean and rstd from earlier
    float m = m_array[b * sizes.inner + t];
    float r = r_array[b * sizes.inner + t];

    // two reduce operations
    float grad_sum = 0.0f;
    float grad_sum_xhat = 0.0f;
    for (int k = 0; k < sizes.channel; k++){
        int idx = k * sizes.inner;
        float in_m = quant_acc(input[idx] - m);
        float xhat = quant_mul(in_m * r);
        float norm_grad = quant_mul(w_array[k] * gradient[idx]);
        float dot_xhat = quant_mul(xhat * norm_grad);
        grad_sum = quant_acc(grad_sum + norm_grad);
        grad_sum_xhat = quant_acc(grad_sum_xhat + dot_xhat);
    }
    grad_sum = quant_div(grad_sum/sizes.channel);
    grad_sum_xhat = quant_div(grad_sum_xhat/sizes.channel);

    // iterate and accumulate 
    for (int k = 0; k < sizes.channel; k++){
        int idx = k * sizes.inner;
        float in_m = quant_acc(input[idx] - m);
        float xhat = quant_mul(in_m * r);

        // save the xhat gradient for a second pass
        // used for gradients of gamma (weights) and beta (biases)
        xhat_grad[idx] = quant_mul(xhat * gradient[idx]);

        float norm_grad = quant_mul(w_array[k] * gradient[idx]);
        float weighted_grad_sum = quant_mul(xhat * grad_sum_xhat);

        // accumulate the input gradient
        float grad_input = norm_grad;
        grad_input = quant_acc(grad_input - grad_sum);
        grad_input = quant_acc(grad_input - weighted_grad_sum);
        grad_input = quant_mul(grad_input * r);

        output[idx] = grad_input;
    }
}

// second pass of the backward pass
// uses output gradient from forward pass and xhat_gradient from first backward pass
// used to calculate grad_gamma (gradient of weights) and grad_beta (gradient of biases)
template<class Qacc>
__global__ void layernorm_backward_second_pass_impl(const float* __restrict__ xhat_gradient, const float* __restrict__ out_grad,
                                                    float* grad_gamma, float* grad_beta, const DimSizes sizes,
                                                    Qacc quant_acc){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over size of channel rather than B*T
    if (k >= sizes.channel) return;

    int idx = k * sizes.inner;

    // temp values for accumulation
    float grad_gamma_sum = 0.0f;
    float grad_beta_sum = 0.0f;

    for (int i = 0; i < sizes.outer * sizes.inner; i++){
        int b = i / sizes.inner;
        int t = i % sizes.inner;
        int base_index = (b * sizes.channel * sizes.inner) + t;

        // get xhat gradient and output gradient from earlier
        const float* gradient = out_grad + base_index;
        const float* xhat_grad = xhat_gradient + base_index;

        // increment gradient of weights and biases by respective values
        grad_gamma_sum = quant_acc(grad_gamma_sum + xhat_grad[idx]);
        grad_beta_sum = quant_acc(grad_beta_sum + gradient[idx]);
    }
    // output to respective arrays (gamma is weight, beta is bias)
    grad_gamma[k] = grad_gamma_sum;
    grad_beta[k] = grad_beta_sum;
}

// calls forward pass kernel
template<class Qacc, class Qmul, class Qdiv, class Qsqrt>
void layernorm_forward(const float* in_arr, const float* w_array, const float* b_array,
                    float* out_arr, float* m_array, float* r_array,
                    float eps, const DimSizes &sizes,
                    Qacc quant_acc, Qmul quant_mul, Qdiv quant_div, Qsqrt quant_sqrt)
{
    int blocks = sizes.outer * sizes.inner;
    int block_size = 64;
    size_t shared_mem_size = (block_size / 32) * sizeof(float);
    layernorm_forward_impl<<<blocks, block_size, shared_mem_size>>>(
        in_arr, w_array, b_array, out_arr, m_array, r_array, eps, sizes,
        quant_acc, quant_mul, quant_div, quant_sqrt);
}

// calls backward pass kernel(s)
template<class Qacc, class Qmul, class Qdiv>
void layernorm_backward(const float* in_arr, const float* out_grad, 
                        const float* w_array, const float* b_array,
                        const float* m_array, const float* r_array,
                        float* grad_input, float* grad_gamma, float* grad_beta, float* xhat_gradient,
                        const DimSizes &sizes, Qacc quant_acc, Qmul quant_mul, Qdiv quant_div)
{
    int blocks = sizes.outer * sizes.inner;
    int block_size = 64;
    layernorm_backward_first_pass_impl<<<blocks, block_size>>>(
        in_arr, out_grad, w_array, b_array, m_array, r_array, 
        grad_input, xhat_gradient, sizes, 
        quant_acc, quant_mul, quant_div);
    blocks = sizes.channel / block_size + (sizes.channel % block_size != 0);
    layernorm_backward_second_pass_impl<<<blocks, block_size>>>(
        xhat_gradient, out_grad, 
        grad_gamma, grad_beta, sizes, 
        quant_acc);
}