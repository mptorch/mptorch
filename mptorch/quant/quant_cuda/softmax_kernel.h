#pragma once

#include "quant_kernel.h"

template<class Qexp, class Qoff, class Qacc, class Qdiv>
__global__ void softmax_forward_impl(
    const float* __restrict__ input_array, float *output_array, const DimStrides *strides,
    Qexp quant_exp, Qoff quant_off, Qacc quant_acc, Qdiv quant_div)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    const float* input = input_array + base_index;
    float* output = output_array + base_index;

    // compute the maximum value of the row
    
    float max = -INFINITY;
    // each thread computes part of the maximum by iterating over its corresponding
    // position along the row, as many times as required to cover all the row
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        max = fmaxf(max, input[idx]);
    }
    // intra-warp maximum reduction
    // each thread now contains part of the maximum of the row, we combine these maximums
    // into a per-warp maximum, stored in the 0th thread of the warp (lane = 0)
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, max, offset);
        max = fmaxf(max, other_max);
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
            max = fmaxf(max, shared[i]);
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
        output[idx] = quant_exp(expf(quant_off(input[idx] - max)));
    }

    // compute the sum of exp(x[i] - max), using a similar approach as for the maximum,
    // but reducing a sum instead

    float sum = 0.f;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        sum = quant_acc(sum + output[idx]); // we sum the previously computed exponentials
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum = quant_acc(sum + __shfl_down_sync(0xFFFFFFFF, sum, offset));
    }
    if (lane == 0) {
        shared[warp] = sum;
    }
    __syncthreads();
    if(tid == 0) {
        sum = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            sum = quant_acc(sum + shared[i]);
        }
        shared[0] = sum;
    }
    __syncthreads();
    sum = shared[0];

    // finally, divide each previously computed exponentials by the sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = quant_div(output[idx] / sum);
    }
}

template<class Qexp, class Qoff, class Qlse>
__global__ void softmax_lse_forward_impl(
    const float* __restrict__ input_array, float *output_array, const DimStrides *strides,
    Qexp quant_exp, Qoff quant_off, Qlse quant_lse)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    const float* input = input_array + base_index;
    float* output = output_array + base_index;

    // compute the maximum value of the row

    float max = -INFINITY;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        max = fmaxf(max, input[idx]);
    }
    // intra-warp maximum reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xFFFFFFFF, max, offset);
        max = fmaxf(max, other_max);
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
            max = fmaxf(max, shared[i]);
        }
        // store the block's maximum (i.e row's maximum) in a fixed shared location
        shared[0] = max;
    }
    __syncthreads();
    // each thread reads the row's maximum
    max = shared[0];

    // compute log(exp(x[i] - max) + ...) using LogSumExp iterations
    float lgs = -INFINITY;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        float x = quant_off(input[idx] - max);
        output[idx] = x;
        lgs = quant_lse(logf(expf(lgs) + expf(x)));
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        float other_lgs = __shfl_down_sync(0xFFFFFFFF, lgs, offset);
        lgs = quant_lse(logf(expf(lgs) + expf(other_lgs)));
    }
    if (lane == 0) {
        shared[warp] = lgs;
    }
    __syncthreads();
    if(tid == 0) {
        lgs = shared[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            lgs = quant_lse(logf(expf(lgs) + expf(shared[i])));
        }
        shared[0] = lgs;
    }
    __syncthreads();
    lgs = shared[0];

    // finally, divide each previously computed exponentials by the sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        output[idx] = quant_exp(expf(quant_off(output[idx] - lgs)));
    }
}

template<class Qadd, class Qmul, class Qdiv>
__global__ void softmax_backward_impl(
  const float* __restrict__ input_array, const float* __restrict__ out_gradient, float* output_array,
  const DimStrides *strides,
  Qadd quant_add, Qmul quant_mul, Qdiv quant_div)
{
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int warp = threadIdx.x / warpSize; // which warp within this block the thread belongs to
    int lane = threadIdx.x % warpSize; // the id of the thread in its warp

    int warpsPerBlock = blockDim.x / warpSize;

    int outer_idx = blockIdx.x / strides->inner_size;
    int inner_idx = blockIdx.x % strides->inner_size;
    
    int base_index = outer_idx * strides->outer_stride + inner_idx;
    const float* input = input_array + base_index;
    const float* grad = out_gradient + base_index;
    float* output = output_array + base_index;

    // Compute the input row sum and weighted sum

    float* shared_input_sum = &shared[0];
    float* shared_weighted_grad_sum = &shared[warpsPerBlock];

    float input_sum = 0.f;
    float weighted_grad_sum = 0.f;
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        input_sum = quant_add(input_sum + input[idx]);
        float prod = quant_mul(input[idx] * grad[idx]);
        weighted_grad_sum = quant_add(weighted_grad_sum + prod);
    }
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        input_sum = quant_add(input_sum + __shfl_down_sync(0xFFFFFFFF, input_sum, offset));
        weighted_grad_sum = quant_add(weighted_grad_sum + __shfl_down_sync(0xFFFFFFFF, weighted_grad_sum, offset));
    }
    if (lane == 0) {
        shared_input_sum[warp] = input_sum;
        shared_weighted_grad_sum[warp] = weighted_grad_sum;
    }
    __syncthreads();
    if(tid == 0) {
        input_sum = shared_input_sum[0];
        weighted_grad_sum = shared_weighted_grad_sum[0];
        for (int i = 1; i < warpsPerBlock; i++) {
            input_sum = quant_add(input_sum + shared_input_sum[i]);
            weighted_grad_sum = quant_add(weighted_grad_sum + shared_weighted_grad_sum[i]);
        }
        shared_input_sum[0] = input_sum;
        shared_weighted_grad_sum[0] = weighted_grad_sum;
    }
    __syncthreads();
    input_sum = shared_input_sum[0];
    weighted_grad_sum = shared_weighted_grad_sum[0];

    // Last step, subsrtact the weighted sum from the gradient, and divide by input sum
    for (int k = tid; k < strides->dim_size; k += blockDim.x) {
        int idx = k * strides->dim_stride;
        float a = quant_add(grad[idx] - weighted_grad_sum);
        float b = quant_mul(a * input[idx]);
        output[idx] = quant_div(b / input_sum);
    }
}