#pragma once

#include "quant.h"
#include <torch/torch.h>
#include <cmath>

using namespace at;

// forward pass of layer normalization
template<class Qacc, class Qmul, class Qdiv, class Qsqrt>
void layernorm_forward(const float *a_array, const float *w_array, const float *b_array,
					float *o_array, float *m_array, float *r_array,
                    float eps, const DimSizes& sizes,
                    Qacc quant_acc, Qmul quant_mul, Qdiv quant_div, Qsqrt quant_sqrt)
{
  for (int i = 0; i < sizes.outer * sizes.inner; i++){
    int b = i / sizes.inner;
    int t = i % sizes.inner;

    int base_index = (b * sizes.channel * sizes.inner) + t;
    const float* input = a_array + base_index;
    float* output = o_array + base_index;

    // calculate mean
    float m = 0.0f;
    for (int k = 0; k < sizes.channel; k++){
      int idx = k*sizes.inner;
      m = quant_acc(m + input[idx]);
    }
    m = quant_div(m/sizes.channel);

    // calculate variance
    float variance = 0.0f;
    for (int k = 0; k < sizes.channel; k++){
      int idx = k * sizes.inner;
      float shift = quant_acc(input[idx] - m);
      float shift_2 = quant_mul(shift * shift);
      variance = quant_acc(variance + shift_2);
    }
    variance = quant_div(variance/sizes.channel);

    // calculate layer normalization formula
    float rad = quant_acc(variance + eps);
    float std = quant_sqrt(sqrtf(rad));
    for (int k = 0; k < sizes.channel; k++){
      int idx = k * sizes.inner;
      float numer = quant_acc(input[idx] - m);
      float norm = quant_div(numer/std);
      float out = quant_mul(w_array[k] * norm);
      output[idx] = out + b_array[k];
    }
    // output mean and rstd calculated earlier
    // will be used in backward pass
    m_array[b * sizes.inner + t] = m;
    r_array[b * sizes.inner + t] = quant_div(1.0f/std); 
  }
}

// backward pass of layer normalization
template<class Qacc, class Qmul, class Qdiv>
void layernorm_backward(const float *a_array, const float *g_array, 
						const float *w_array, const float *b_array, const float *m_array, const float *r_array,
						float *o_array, float *grad_gamma, float *grad_beta, const DimSizes& sizes,
                    	Qacc quant_acc, Qmul quant_mul, Qdiv quant_div)
{
  for (int i = 0; i < sizes.outer * sizes.inner; i++){
    int b = i / sizes.inner;
    int t = i % sizes.inner;

    int base_index = (b * sizes.channel * sizes.inner) + t;
    const float* input = a_array + base_index;
    const float* gradient = g_array + base_index;
    float* output = o_array + base_index;

    const float m = m_array[b * sizes.inner + t];
    const float r = r_array[b * sizes.inner + t];

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
    // normalized gradients
    grad_sum = quant_div(grad_sum/sizes.channel);
    grad_sum_xhat = quant_div(grad_sum_xhat/sizes.channel);

    // iterate and accumulate 
    for (int k = 0; k < sizes.channel; k++){
      int idx = k * sizes.inner;
      float in_m = quant_acc(input[idx] - m);
      float xhat = quant_mul(in_m * r);
      float xhat_gradient = quant_mul(xhat * gradient[idx]);
      float norm_grad = quant_mul(w_array[k] * gradient[idx]);

      // calculate the grad_beta (gradient of biases) and grad_gamma (gradient of weights)
      grad_beta[k] = quant_acc((grad_beta[k] + gradient[idx]));
      grad_gamma[k] = quant_acc(grad_gamma[k] + xhat_gradient);

      // accumulate the input gradient
      float weighted_grad_sum = quant_mul(xhat * grad_sum_xhat);
      float grad_input = norm_grad;
      grad_input = quant_acc(grad_input - grad_sum);
      grad_input = quant_acc(grad_input - weighted_grad_sum);
      grad_input = quant_mul(grad_input * r);

      output[idx] = grad_input;
    }
  }
}