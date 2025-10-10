#pragma once

#include "quant.h"
#include <torch/torch.h>
#include <cmath>

using namespace at;

// Uses the standard softmax formula e^xi/sum(e^xj)
// Returns ouput tensor of softmaxxed and quantized values
template<class Qexp, class Qoff, class Qacc>
void softmax_forward(const float *a_array, float *o_array, const DimSizes& sizes, Qexp quant_exp, Qoff quant_off, Qacc quant_acc)
{
  for (int i = 0; i < sizes.outer * sizes.inner; ++i) {
    int outer_idx = i / sizes.inner;
    int inner_idx = i % sizes.inner;

    // Get beginning pointers to the row being softmaxed
    int base_index = outer_idx * sizes.channel * sizes.inner + inner_idx;
    const float* input = a_array + base_index;
    float* output = o_array + base_index;
  
    // Get the maximum of the current row being softmaxed.
    // This ensures no overflow and more stable exponentials.
    float max = input[0];
    for (int k = 1; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      max = fmaxf(max, input[idx]);
    }

    // Calculate the sum and store quantized values of exp(input) into output
    float sum = 0.0f;
    for (int k = 0; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      float x = quant_off(input[idx] - max);
      float e_x = quant_exp(expf(x));
      output[idx] = e_x;
      sum = quant_acc(sum + e_x);
    }
    // Divide all outputs by the current sum
    for (int k = 0; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      output[idx] = output[idx] / sum; // quantization handled by output_quant
    }
  }
}

// Uses the LogSumExp formula to compute softmax without divisions
// Returns ouput tensor of softmaxxed and quantized values
template<class Qoff, class Qlse>
void softmax_lse_forward(const float *a_array, float *o_array, const DimSizes& sizes, Qoff quant_off, Qlse quant_lse)
{
  for (int i = 0; i < sizes.outer * sizes.inner; ++i) {
    int outer_idx = i / sizes.inner;
    int inner_idx = i % sizes.inner;

    // Get beginning pointers to the row being softmaxed
    int base_index = outer_idx * sizes.channel * sizes.inner + inner_idx;
    const float* input = a_array + base_index;
    float* output = o_array + base_index;

    // Get the maximum of the current row being softmaxed.
    // This ensures no overflow and more stable exponentials.
    float max = input[0];
    for (int k = 1; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      max = fmaxf(max, input[idx]);
    }

    // Calculate LogSumExp
    float x0 = quant_off(input[0] - max);
    output[0] = x0;
    float lgs = x0; // = log(exp(x[0] - max))
    for (int k = 1; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      float x = quant_off(input[idx] - max);
      output[idx] = x;
      lgs = quant_lse(logf(expf(lgs) + expf(x)));
    }
    // Compute output tensor elements
    for (int k = 0; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      float x = quant_off(output[idx] - lgs);
      output[idx] = expf(x); // quantization handled by output_quant
    }
  }
}

template<class Qadd, class Qmul>
void softmax_backward(const float *a_array, const float *g_array, float *o_array, const DimSizes& sizes, Qadd quant_add, Qmul quant_mul)
{
  for (int i = 0; i < sizes.outer * sizes.inner; ++i) {
    int outer_idx = i / sizes.inner;
    int inner_idx = i % sizes.inner;

    int base_index = outer_idx * sizes.channel * sizes.inner + inner_idx;
    const float* input = a_array + base_index;
    const float* grad = g_array + base_index;
    float* output = o_array + base_index;

    float weighted_grad_sum = 0.f;
    for (int k = 0; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      float prod = quant_mul(input[idx] * grad[idx]);
      weighted_grad_sum = quant_add(weighted_grad_sum + prod);
    }

    for (int k = 0; k < sizes.channel; ++k) {
      int idx = k * sizes.inner;
      float a = quant_add(grad[idx] - weighted_grad_sum);
      output[idx] = a * input[idx]; // quantization handled by grad_quant
    }
  }
}
