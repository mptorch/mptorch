#pragma once
#include "quant_kernel.h"

template<class Qacc, class Qmul, class Qdiv, class Qsqrt>
__global__ void layernorm_forward_impl(const float* __restrict__ in_arr, const float* w_array, const float* b_array,
                                    float* out_arr, float* m_array, float* r_array,
                                    float eps, const DimSizes &sizes,
                                    Qacc quant_acc, Qmul quant_mul, Qdiv quant_div, Qsqrt quant_sqrt)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // groups of 32 threads (which warp the thread belongs to)
    int lane = threadIdx.x % warpSize; // a warp has 32 lanes (id of the thread in a warp)

    int warpsPerBlock= blockDim.x / warpSize;

    int b = blockIdx.x / sizes.inner;
    int t = blockIdx.x % sizes.inner;

    int base_index = (b * sizes.channel * sizes.inner) + t;
    const float* input = in_arr + base_index;
    float* output = out_arr + base_index;

    // compute mean by reducing sum of elements then dividing
    float m_sum = 0.0f;
    for (int k = tid; k < sizes.channel; k += blockDim.x){
        int idx = k * sizes.inner;
        m_sum = quant_acc(m_sum + input[idx]);
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2){
        m_sum = quant_acc(m_sum + __shfl_down_sync(0xffffffff, m_sum, offset));
    }
    if (lane == 0){
        shared[warp] = m_sum;
    }
    __syncthreads();
    if (tid == 0){
        m_sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++){
            m_sum = quant_acc(m_sum + shared[i]);
        }
        shared[0] = m_sum/sizes.channel;
    }
    __syncthreads();
    float m = shared[0];

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
    float variance = shared[0];
    
    float rad = quant_acc(variance + eps);
    float std = quant_sqrt(sqrtf(rad));
    for (int k = tid; k < sizes.channel; k += blockDim.x){
        int idx = k * sizes.inner;
        float numer = quant_acc(input[idx] - m);
        float norm = quant_div(numer/std);
        float out = quant_mul(w_array[k] * norm);
        output[idx] = out + b_array[k];
    }
    m_array[b * sizes.inner + t] = m;
    r_array[b * sizes.inner + t] = quant_div(1.0f/std); 
}

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