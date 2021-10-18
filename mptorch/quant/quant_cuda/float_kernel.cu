#include "quant_kernel.h"
#include "bit_helper.cu"
#include "sim_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float* __restrict__ a,
                                        int* __restrict__ r,
                                        float* o, int size,
                                        int man_bits,
                                        int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int rand_prob = (unsigned int) r[index];
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float* __restrict__ a,
                                     float* o, int size,
                                     int man_bits,
                                     int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target,quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal){
      float shift_float,val;
      int shift_bits = ((127+min_exp)<<23) | (target >> 31 <<31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val=a[index]+shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    }
    else{
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max) return max;
  else if (a < min) return min;
  else return a;
}

template <typename T>
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max, uint8_t* mask) {
  if (a > max) {
    *mask = 1;
    return max;
  } else if (a < min) {
    *mask = 1;
    return min;
  }
  *mask = 0;
  return a;
}

__device__ float cast_fxp(double origin_float, 
                          int sigma, double t_min, double t_max) {
  origin_float = nearest_round(origin_float, sigma);
  origin_float = clamp_helper(origin_float, t_min, t_max);
  return origin_float;
}

__device__ float cast_fp(float origin_float, int man_bits, int exp_bits)
{
    unsigned int target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) -127; 
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23);

	if(noquantize){
		quantized = origin_float;
	}
	else{
		if (subnormal){
			float shift_float, val;
			int shift_bits = ((127 + min_exp) << 23) | (target >> 31 <<31);
			shift_float = BITS_TO_FLOAT(&shift_bits);
			val = origin_float + shift_float;
			target = FLOAT_TO_BITS(&val);
			quantize_bits = round_bitwise_nearest(target, man_bits);
			quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
		}
		else{
			quantize_bits = round_bitwise_nearest(target, man_bits);
			quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
			quantized = BITS_TO_FLOAT(&quantize_bits);
		}
	}
	
    return quantized;
}

__global__ void tvm_gemm_fp(float *__restrict__ feature,
                         float *__restrict__ kernel, float *__restrict__ gemm,
                         int M, int K, int N, int man_bits, int exp_bits) {
  float gemm_local[4];
  float gemm_local_rest[4];
  __shared__ float feature_shared[128];
  __shared__ float kernel_shared[128];
  float feature_shared_local[4];
  float kernel_shared_local[4];
  float gemm_local1[4];
  float gemm_local1_rest[4];
  float feature_shared_local1[4];
  float kernel_shared_local1[4];
  if (((int)blockIdx.x) < (M / 16)) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        gemm_local[((i_c_init * 2) + j_c_init)] = 0.000000e+00f;
        gemm_local_rest[((i_c_init * 2) + j_c_init)] = 0.000000e+00f;
      }
    }
    for (int rx_outer = 0; rx_outer < ((K + 7) / 8); ++rx_outer) {
      __syncthreads();
      for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner * 8)) +
                          ((int)threadIdx.x))] =
              feature[(((rx_outer * 8) + ((((((int)blockIdx.x) * 16) +
                                            (((int)threadIdx.y) * 2)) +
                                           ax0_inner) *
                                          K)) +
                       ((int)threadIdx.x))];
        }
      }
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner)];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer = 0; rx_inner_outer < 4; ++rx_inner_outer) {
        for (int ax0 = 0; ax0 < 2; ++ax0) {
          for (int ax1 = 0; ax1 < 2; ++ax1) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax1)) {
              feature_shared_local[((ax0 * 2) + ax1)] =
                  feature_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) +
                                   (rx_inner_outer * 2)) +
                                  ax1)];
            }
          }
        }
        for (int ax01 = 0; ax01 < 2; ++ax01) {
          for (int ax11 = 0; ax11 < 2; ++ax11) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax01)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax11)) {
                kernel_shared_local[((ax01 * 2) + ax11)] =
                    kernel_shared[((((rx_inner_outer * 32) + (ax01 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax11)];
              }
            }
          }
        }
        for (int i_c = 0; i_c < 2; ++i_c) {
          for (int j_c = 0; j_c < 2; ++j_c) {
            for (int rx_inner_inner = 0; rx_inner_inner < 2; ++rx_inner_inner) {
              if (((rx_outer * 8) + (rx_inner_outer * 2)) <
                  (K - rx_inner_inner)) {
                if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                    (N - j_c)) {
                  gemm_local[((i_c * 2) + j_c)] =
                    cast_fp(gemm_local[((i_c * 2) + j_c)] +
                    cast_fp(feature_shared_local[((i_c * 2) +
                    rx_inner_inner)] * kernel_shared_local[((rx_inner_inner *
                    2) + j_c)], man_bits, exp_bits), man_bits, exp_bits);
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
            (N - j_inner_inner)) {
          gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                 ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                   i_inner_inner) *
                  N)) +
                j_inner_inner)] =
              gemm_local[((i_inner_inner * 2) + j_inner_inner)];
        }
      }
    }
  } else {
    for (int i_c_init1 = 0; i_c_init1 < 2; ++i_c_init1) {
      for (int j_c_init1 = 0; j_c_init1 < 2; ++j_c_init1) {
        gemm_local1[((i_c_init1 * 2) + j_c_init1)] = 0.000000e+00f;
      }
    }
    for (int rx_outer1 = 0; rx_outer1 < ((K + 7) / 8); ++rx_outer1) {
      for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) <
            (M - ax0_inner1)) {
          if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
            feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner1 * 8)) +
                            ((int)threadIdx.x))] =
                feature[(((rx_outer1 * 8) + ((((((int)blockIdx.x) * 16) +
                                               (((int)threadIdx.y) * 2)) +
                                              ax0_inner1) *
                                             K)) +
                         ((int)threadIdx.x))];
          }
        }
      }
      for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
        if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner1)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner1)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer1 * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner1)];
          }
        }
      }
      for (int rx_inner_outer1 = 0; rx_inner_outer1 < 4; ++rx_inner_outer1) {
        for (int ax02 = 0; ax02 < 2; ++ax02) {
          for (int ax12 = 0; ax12 < 2; ++ax12) {
            if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                (M - ax02)) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax12)) {
                feature_shared_local1[((ax02 * 2) + ax12)] =
                    feature_shared[((((((int)threadIdx.x) * 16) + (ax02 * 8)) +
                                     (rx_inner_outer1 * 2)) +
                                    ax12)];
              }
            }
          }
        }
        for (int ax03 = 0; ax03 < 2; ++ax03) {
          for (int ax13 = 0; ax13 < 2; ++ax13) {
            if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax03)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax13)) {
                kernel_shared_local1[((ax03 * 2) + ax13)] =
                    kernel_shared[((((rx_inner_outer1 * 32) + (ax03 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax13)];
              }
            }
          }
        }
        for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
          for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
            for (int rx_inner_inner1 = 0; rx_inner_inner1 < 2;
                 ++rx_inner_inner1) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) <
                  (K - rx_inner_inner1)) {
                if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                    (M - i_c1)) {
                  if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                      (N - j_c1)) {
                      gemm_local1[((i_c1 * 2) + j_c1)] =
                      cast_fp(gemm_local1[((i_c1 * 2) + j_c1)] +
                      cast_fp(feature_shared_local1[((i_c1 * 2) +
                      rx_inner_inner1)] *
                      kernel_shared_local1[((rx_inner_inner1 * 2) +
                      j_c1)],man_bits,exp_bits),man_bits,exp_bits);
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner1 = 0; i_inner_inner1 < 2; ++i_inner_inner1) {
      for (int j_inner_inner1 = 0; j_inner_inner1 < 2; ++j_inner_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
            (M - i_inner_inner1)) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - j_inner_inner1)) {
            gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                   ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                     i_inner_inner1) *
                    N)) +
                  j_inner_inner1)] =
                gemm_local1[((i_inner_inner1 * 2) + j_inner_inner1)];
          }
        }
      }
    }
  }
}


__global__ void tvm_gemm_fp_fp(float *__restrict__ feature, float *__restrict__ kernel,
                         float *__restrict__ gemm, int M, int K, int N, 
                         int m_man_bits, int m_exp_bits, int a_man_bits, int a_exp_bits)
{
  float gemm_local[4];
  __shared__ float feature_shared[128];
  __shared__ float kernel_shared[128];
  float feature_shared_local[4];
  float kernel_shared_local[4];
  float gemm_local1[4];
  float feature_shared_local1[4];
  float kernel_shared_local1[4];
  if (((int)blockIdx.x) < (M / 16)) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        gemm_local[((i_c_init * 2) + j_c_init)] = 0.0f;
      }
    }
    for (int rx_outer = 0; rx_outer < ((K + 7) / 8); ++rx_outer) {
      __syncthreads();
      for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner * 8)) +
                          ((int)threadIdx.x))] =
              feature[(((rx_outer * 8) + ((((((int)blockIdx.x) * 16) +
                                            (((int)threadIdx.y) * 2)) +
                                           ax0_inner) *
                                          K)) +
                       ((int)threadIdx.x))];          
        }
      }
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner)];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer = 0; rx_inner_outer < 4; ++rx_inner_outer) {
        for (int ax0 = 0; ax0 < 2; ++ax0) {
          for (int ax1 = 0; ax1 < 2; ++ax1) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax1)) {
              feature_shared_local[((ax0 * 2) + ax1)] =
                  feature_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) +
                                (rx_inner_outer * 2)) + ax1)];
            }
          }
        }
        for (int ax01 = 0; ax01 < 2; ++ax01) {
          for (int ax11 = 0; ax11 < 2; ++ax11) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax01)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) < (N - ax11)) {
                kernel_shared_local[((ax01 * 2) + ax11)] =
                    kernel_shared[((((rx_inner_outer * 32) + (ax01 * 16)) +
                                  (((int)threadIdx.y) * 2)) + ax11)];
              }
            }
          }
        }
        for (int i_c = 0; i_c < 2; ++i_c) {
          for (int j_c = 0; j_c < 2; ++j_c) {
            for (int rx_inner_inner = 0; rx_inner_inner < 2; ++rx_inner_inner) {
              if (((rx_outer * 8) + (rx_inner_outer * 2)) <
                  (K - rx_inner_inner)) {
                if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) < (N - j_c)) {
                  gemm_local[((i_c * 2) + j_c)] = cast_fp(
                      gemm_local[((i_c * 2) + j_c)] + cast_fp(feature_shared_local[((i_c * 2) + rx_inner_inner)] * kernel_shared_local[((rx_inner_inner * 2) + j_c)],
                      m_man_bits, m_exp_bits), a_man_bits, a_exp_bits);
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
            (N - j_inner_inner)) {
          gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                 ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                   i_inner_inner) *
                  N)) +
                j_inner_inner)] =
              gemm_local[((i_inner_inner * 2) + j_inner_inner)];
        }        
      }
    }
  } else {
    for (int i_c_init1 = 0; i_c_init1 < 2; ++i_c_init1) {
      for (int j_c_init1 = 0; j_c_init1 < 2; ++j_c_init1) {
        gemm_local1[((i_c_init1 * 2) + j_c_init1)] = 0.000000e+00f;
      }
    }
    for (int rx_outer1 = 0; rx_outer1 < ((K + 7) / 8); ++rx_outer1) {
      for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) <
            (M - ax0_inner1)) {
          if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
            feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner1 * 8)) +
                            ((int)threadIdx.x))] =
                feature[(((rx_outer1 * 8) + ((((((int)blockIdx.x) * 16) +
                                               (((int)threadIdx.y) * 2)) +
                                              ax0_inner1) *
                                             K)) +
                         ((int)threadIdx.x))];
          }
        }
      }
      for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
        if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner1)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner1)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer1 * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner1)];
          }
        }
      }
      for (int rx_inner_outer1 = 0; rx_inner_outer1 < 4; ++rx_inner_outer1) {
        for (int ax02 = 0; ax02 < 2; ++ax02) {
          for (int ax12 = 0; ax12 < 2; ++ax12) {
            if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                (M - ax02)) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax12)) {
                feature_shared_local1[((ax02 * 2) + ax12)] =
                    feature_shared[((((((int)threadIdx.x) * 16) + (ax02 * 8)) +
                                     (rx_inner_outer1 * 2)) +
                                    ax12)];
              }
            }
          }
        }
        for (int ax03 = 0; ax03 < 2; ++ax03) {
          for (int ax13 = 0; ax13 < 2; ++ax13) {
            if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax03)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax13)) {
                kernel_shared_local1[((ax03 * 2) + ax13)] =
                    kernel_shared[((((rx_inner_outer1 * 32) + (ax03 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax13)];
              }
            }
          }
        }
        for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
          for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
            for (int rx_inner_inner1 = 0; rx_inner_inner1 < 2;
                 ++rx_inner_inner1) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) <
                  (K - rx_inner_inner1)) {
                if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                    (M - i_c1)) {
                  if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                      (N - j_c1)) {
                    gemm_local1[((i_c1 * 2) + j_c1)] = cast_fp(
                      gemm_local1[((i_c1 * 2) + j_c1)] + cast_fp(
                      feature_shared_local1[((i_c1 * 2) + rx_inner_inner1)] *
                      kernel_shared_local1[((rx_inner_inner1 * 2) + j_c1)],
                      m_man_bits, m_exp_bits), a_man_bits, a_exp_bits);
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner1 = 0; i_inner_inner1 < 2; ++i_inner_inner1) {
      for (int j_inner_inner1 = 0; j_inner_inner1 < 2; ++j_inner_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
            (M - i_inner_inner1)) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - j_inner_inner1)) {
            gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                   ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                     i_inner_inner1) *
                    N)) +
                  j_inner_inner1)] =
                gemm_local1[((i_inner_inner1 * 2) + j_inner_inner1)];
          }
        }
      }
    }
  }
}

__global__ void tvm_gemm_fp_fxp(float *__restrict__ feature, 
                        float *__restrict__ kernel,
                        float *__restrict__ gemm, int M, int K, int N, 
                        int m_man_bits, int m_exp_bits,
                        int Qi, int Qf, bool symmetric)
{
  float gemm_local[4];
  __shared__ float feature_shared[128];
  __shared__ float kernel_shared[128];
  float feature_shared_local[4];
  float kernel_shared_local[4];
  float gemm_local1[4];
  float feature_shared_local1[4];
  float kernel_shared_local1[4];
  double t_min = -ldexp(1.0, Qi-1);
  double t_max = -t_min;
  if (!symmetric) 
    t_max -= ldexp(1.0, -Qf);
  if (((int)blockIdx.x) < (M / 16)) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        gemm_local[((i_c_init * 2) + j_c_init)] = 0.0f;
      }
    }
    for (int rx_outer = 0; rx_outer < ((K + 7) / 8); ++rx_outer) {
      __syncthreads();
      for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner * 8)) +
                          ((int)threadIdx.x))] =
              feature[(((rx_outer * 8) + ((((((int)blockIdx.x) * 16) +
                                            (((int)threadIdx.y) * 2)) +
                                           ax0_inner) *
                                          K)) +
                       ((int)threadIdx.x))];          
        }
      }
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner)];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer = 0; rx_inner_outer < 4; ++rx_inner_outer) {
        for (int ax0 = 0; ax0 < 2; ++ax0) {
          for (int ax1 = 0; ax1 < 2; ++ax1) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax1)) {
              feature_shared_local[((ax0 * 2) + ax1)] =
                  feature_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) +
                                (rx_inner_outer * 2)) + ax1)];
            }
          }
        }
        for (int ax01 = 0; ax01 < 2; ++ax01) {
          for (int ax11 = 0; ax11 < 2; ++ax11) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax01)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) < (N - ax11)) {
                kernel_shared_local[((ax01 * 2) + ax11)] =
                    kernel_shared[((((rx_inner_outer * 32) + (ax01 * 16)) +
                                  (((int)threadIdx.y) * 2)) + ax11)];
              }
            }
          }
        }
        for (int i_c = 0; i_c < 2; ++i_c) {
          for (int j_c = 0; j_c < 2; ++j_c) {
            for (int rx_inner_inner = 0; rx_inner_inner < 2; ++rx_inner_inner) {
              if (((rx_outer * 8) + (rx_inner_outer * 2)) <
                  (K - rx_inner_inner)) {
                if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) < (N - j_c)) {
                  gemm_local[((i_c * 2) + j_c)] = cast_fxp(
                      cast_fxp(gemm_local[((i_c * 2) + j_c)], -Qf, t_min, t_max) + 
                      cast_fxp(cast_fp(feature_shared_local[((i_c * 2) + rx_inner_inner)] * kernel_shared_local[((rx_inner_inner * 2) + j_c)],
                      m_man_bits, m_exp_bits), -Qf, t_min, t_max), -Qf, t_min, t_max);
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
            (N - j_inner_inner)) {
          gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                 ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                   i_inner_inner) *
                  N)) +
                j_inner_inner)] =
              gemm_local[((i_inner_inner * 2) + j_inner_inner)];
        }        
      }
    }
  } else {
    for (int i_c_init1 = 0; i_c_init1 < 2; ++i_c_init1) {
      for (int j_c_init1 = 0; j_c_init1 < 2; ++j_c_init1) {
        gemm_local1[((i_c_init1 * 2) + j_c_init1)] = 0.000000e+00f;
      }
    }
    for (int rx_outer1 = 0; rx_outer1 < ((K + 7) / 8); ++rx_outer1) {
      for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) <
            (M - ax0_inner1)) {
          if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
            feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner1 * 8)) +
                            ((int)threadIdx.x))] =
                feature[(((rx_outer1 * 8) + ((((((int)blockIdx.x) * 16) +
                                               (((int)threadIdx.y) * 2)) +
                                              ax0_inner1) *
                                             K)) +
                         ((int)threadIdx.x))];
          }
        }
      }
      for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
        if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner1)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner1)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer1 * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner1)];
          }
        }
      }
      for (int rx_inner_outer1 = 0; rx_inner_outer1 < 4; ++rx_inner_outer1) {
        for (int ax02 = 0; ax02 < 2; ++ax02) {
          for (int ax12 = 0; ax12 < 2; ++ax12) {
            if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                (M - ax02)) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax12)) {
                feature_shared_local1[((ax02 * 2) + ax12)] =
                    feature_shared[((((((int)threadIdx.x) * 16) + (ax02 * 8)) +
                                     (rx_inner_outer1 * 2)) +
                                    ax12)];
              }
            }
          }
        }
        for (int ax03 = 0; ax03 < 2; ++ax03) {
          for (int ax13 = 0; ax13 < 2; ++ax13) {
            if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax03)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax13)) {
                kernel_shared_local1[((ax03 * 2) + ax13)] =
                    kernel_shared[((((rx_inner_outer1 * 32) + (ax03 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax13)];
              }
            }
          }
        }
        for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
          for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
            for (int rx_inner_inner1 = 0; rx_inner_inner1 < 2;
                 ++rx_inner_inner1) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) <
                  (K - rx_inner_inner1)) {
                if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                    (M - i_c1)) {
                  if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                      (N - j_c1)) {
                    gemm_local1[((i_c1 * 2) + j_c1)] = cast_fxp(
                      cast_fxp(gemm_local1[((i_c1 * 2) + j_c1)], -Qf, t_min, t_max) + 
                      cast_fxp(cast_fp(
                      feature_shared_local1[((i_c1 * 2) + rx_inner_inner1)] *
                      kernel_shared_local1[((rx_inner_inner1 * 2) + j_c1)],
                      m_man_bits, m_exp_bits), -Qf, t_min, t_max), -Qf, t_min, t_max);
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner1 = 0; i_inner_inner1 < 2; ++i_inner_inner1) {
      for (int j_inner_inner1 = 0; j_inner_inner1 < 2; ++j_inner_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
            (M - i_inner_inner1)) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - j_inner_inner1)) {
            gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                   ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                     i_inner_inner1) *
                    N)) +
                  j_inner_inner1)] =
                gemm_local1[((i_inner_inner1 * 2) + j_inner_inner1)];
          }
        }
      }
    }
  }
}

__global__ void tvm_gemm_fma(float *__restrict__ feature, 
                        float *__restrict__ kernel,
                        float *__restrict__ gemm, int M, int K, int N, 
                        int man_bits, int exp_bits)
{
  float gemm_local[4];
  __shared__ float feature_shared[128];
  __shared__ float kernel_shared[128];
  float feature_shared_local[4];
  float kernel_shared_local[4];
  float gemm_local1[4];
  float feature_shared_local1[4];
  float kernel_shared_local1[4];
  if (((int)blockIdx.x) < (M / 16)) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        gemm_local[((i_c_init * 2) + j_c_init)] = 0.0f;
      }
    }
    for (int rx_outer = 0; rx_outer < ((K + 7) / 8); ++rx_outer) {
      __syncthreads();
      for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner * 8)) +
                          ((int)threadIdx.x))] =
              feature[(((rx_outer * 8) + ((((((int)blockIdx.x) * 16) +
                                            (((int)threadIdx.y) * 2)) +
                                           ax0_inner) *
                                          K)) +
                       ((int)threadIdx.x))];          
        }
      }
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if ((rx_outer * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner)];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer = 0; rx_inner_outer < 4; ++rx_inner_outer) {
        for (int ax0 = 0; ax0 < 2; ++ax0) {
          for (int ax1 = 0; ax1 < 2; ++ax1) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax1)) {
              feature_shared_local[((ax0 * 2) + ax1)] =
                  feature_shared[((((((int)threadIdx.x) * 16) + (ax0 * 8)) +
                                (rx_inner_outer * 2)) + ax1)];
            }
          }
        }
        for (int ax01 = 0; ax01 < 2; ++ax01) {
          for (int ax11 = 0; ax11 < 2; ++ax11) {
            if (((rx_outer * 8) + (rx_inner_outer * 2)) < (K - ax01)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) < (N - ax11)) {
                kernel_shared_local[((ax01 * 2) + ax11)] =
                    kernel_shared[((((rx_inner_outer * 32) + (ax01 * 16)) +
                                  (((int)threadIdx.y) * 2)) + ax11)];
              }
            }
          }
        }
        for (int i_c = 0; i_c < 2; ++i_c) {
          for (int j_c = 0; j_c < 2; ++j_c) {
            for (int rx_inner_inner = 0; rx_inner_inner < 2; ++rx_inner_inner) {
              if (((rx_outer * 8) + (rx_inner_outer * 2)) <
                  (K - rx_inner_inner)) {
                if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) < (N - j_c)) {
                  gemm_local[((i_c * 2) + j_c)] = cast_fp(
                      fma(feature_shared_local[((i_c * 2) + rx_inner_inner)], kernel_shared_local[((rx_inner_inner * 2) + j_c)], gemm_local[((i_c * 2) + j_c)]),
                      man_bits, exp_bits);
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
            (N - j_inner_inner)) {
          gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                 ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                   i_inner_inner) *
                  N)) +
                j_inner_inner)] =
              gemm_local[((i_inner_inner * 2) + j_inner_inner)];
        }        
      }
    }
  } else {
    for (int i_c_init1 = 0; i_c_init1 < 2; ++i_c_init1) {
      for (int j_c_init1 = 0; j_c_init1 < 2; ++j_c_init1) {
        gemm_local1[((i_c_init1 * 2) + j_c_init1)] = 0.000000e+00f;
      }
    }
    for (int rx_outer1 = 0; rx_outer1 < ((K + 7) / 8); ++rx_outer1) {
      for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.y) * 2)) <
            (M - ax0_inner1)) {
          if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
            feature_shared[(((((int)threadIdx.y) * 16) + (ax0_inner1 * 8)) +
                            ((int)threadIdx.x))] =
                feature[(((rx_outer1 * 8) + ((((((int)blockIdx.x) * 16) +
                                               (((int)threadIdx.y) * 2)) +
                                              ax0_inner1) *
                                             K)) +
                         ((int)threadIdx.x))];
          }
        }
      }
      for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
        if ((rx_outer1 * 8) < (K - ((int)threadIdx.x))) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - ax1_inner1)) {
            kernel_shared[(
                ((((int)threadIdx.x) * 16) + (((int)threadIdx.y) * 2)) +
                ax1_inner1)] =
                kernel[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                         (((rx_outer1 * 8) + ((int)threadIdx.x)) * N)) +
                        ax1_inner1)];
          }
        }
      }
      for (int rx_inner_outer1 = 0; rx_inner_outer1 < 4; ++rx_inner_outer1) {
        for (int ax02 = 0; ax02 < 2; ++ax02) {
          for (int ax12 = 0; ax12 < 2; ++ax12) {
            if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                (M - ax02)) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax12)) {
                feature_shared_local1[((ax02 * 2) + ax12)] =
                    feature_shared[((((((int)threadIdx.x) * 16) + (ax02 * 8)) +
                                     (rx_inner_outer1 * 2)) +
                                    ax12)];
              }
            }
          }
        }
        for (int ax03 = 0; ax03 < 2; ++ax03) {
          for (int ax13 = 0; ax13 < 2; ++ax13) {
            if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) < (K - ax03)) {
              if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                  (N - ax13)) {
                kernel_shared_local1[((ax03 * 2) + ax13)] =
                    kernel_shared[((((rx_inner_outer1 * 32) + (ax03 * 16)) +
                                    (((int)threadIdx.y) * 2)) +
                                   ax13)];
              }
            }
          }
        }
        for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
          for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
            for (int rx_inner_inner1 = 0; rx_inner_inner1 < 2;
                 ++rx_inner_inner1) {
              if (((rx_outer1 * 8) + (rx_inner_outer1 * 2)) <
                  (K - rx_inner_inner1)) {
                if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
                    (M - i_c1)) {
                  if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
                      (N - j_c1)) {
                    gemm_local1[((i_c1 * 2) + j_c1)] = cast_fp(
                      fma(feature_shared_local1[((i_c1 * 2) + rx_inner_inner1)],
                      kernel_shared_local1[((rx_inner_inner1 * 2) + j_c1)],
                      gemm_local1[((i_c1 * 2) + j_c1)]), man_bits, exp_bits);
                  }
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner1 = 0; i_inner_inner1 < 2; ++i_inner_inner1) {
      for (int j_inner_inner1 = 0; j_inner_inner1 < 2; ++j_inner_inner1) {
        if (((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) <
            (M - i_inner_inner1)) {
          if (((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) <
              (N - j_inner_inner1)) {
            gemm[((((((int)blockIdx.y) * 16) + (((int)threadIdx.y) * 2)) +
                   ((((((int)blockIdx.x) * 16) + (((int)threadIdx.x) * 2)) +
                     i_inner_inner1) *
                    N)) +
                  j_inner_inner1)] =
                gemm_local1[((i_inner_inner1 * 2) + j_inner_inner1)];
          }
        }
      }
    }
  }
}
