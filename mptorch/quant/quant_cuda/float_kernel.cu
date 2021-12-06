#include "bit_helper.cu"
#include "quant_kernel.h"
#include "sim_helper.cu"

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_stochastic(float *__restrict__ a,
                                        int *__restrict__ r, float *o, int size,
                                        int man_bits, int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int rand_prob = (unsigned int)r[index];
    unsigned int target, quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal) {
      float shift_float, val;
      int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val = a[index] + shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    } else {
      quantize_bits = round_bitwise_stochastic(target, rand_prob, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

// quantize a float into a floating point with [exp_bits] exponent and
// [man_bits] mantissa
__global__ void float_kernel_nearest(float *__restrict__ a, float *o, int size,
                                     int man_bits, int exp_bits) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    unsigned int target, quantize_bits;
    target = FLOAT_TO_BITS(&a[index]);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    if (subnormal) {
      float shift_float, val;
      int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val = a[index] + shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    } else {
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
    o[index] = quantized;
  }
}

template <typename T>
__device__ __forceinline__ T clamp_helper(T a, T min, T max) {
  if (a > max)
    return max;
  else if (a < min)
    return min;
  else
    return a;
}

template <typename T>
__device__ __forceinline__ T clamp_mask_helper(T a, T min, T max,
                                               uint8_t *mask) {
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

__device__ float cast_fxp(double origin_float, int sigma, double t_min,
                          double t_max) {
  origin_float = nearest_round(origin_float, sigma);
  origin_float = clamp_helper(origin_float, t_min, t_max);
  return origin_float;
}

__device__ float cast_fp(float origin_float, int man_bits, int exp_bits) {
  unsigned int target, quantize_bits;
  target = FLOAT_TO_BITS(&origin_float);
  float quantized;

  int target_exp = (target << 1 >> 1 >> 23) - 127;
  int min_exp = -((1 << (exp_bits - 1)) - 2);
  bool subnormal = (target_exp < min_exp);
  bool noquantize = (man_bits >= 23);

  if (noquantize) {
    quantized = origin_float;
  } else {
    if (subnormal) {
      float shift_float, val;
      int shift_bits = ((127 + min_exp) << 23) | (target >> 31 << 31);
      shift_float = BITS_TO_FLOAT(&shift_bits);
      val = origin_float + shift_float;
      target = FLOAT_TO_BITS(&val);
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits) - shift_float;
    } else {
      quantize_bits = round_bitwise_nearest(target, man_bits);
      quantize_bits = clip_exponent(exp_bits, man_bits, target, quantize_bits);
      quantized = BITS_TO_FLOAT(&quantize_bits);
    }
  }

  return quantized;
}

// ALGORITHM 0: naive GEMM kernel
// code adapted from: 
// https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final 
// (slides 17 - 28)
// see also https://github.com/DmitryLyakh/CUDA_Tutorial/blob/master/bla_lib.cu
// (function gpu_gemm_nn)
__global__ void gemm_fp_algo0(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits) {
  size_t ty = blockIdx.y * blockDim.y + threadIdx.y; // global thread index Y
  size_t tx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index X

  size_t n_pos = tx;
  while (n_pos < N) {

    size_t m_pos = ty;
    while (m_pos < M) {

      float tmp = static_cast<float>(0.0);
      for (size_t k_pos = 0; k_pos < K; ++k_pos) {
        tmp = cast_fp(tmp + cast_fp(a[m_pos * K + k_pos] * b[k_pos * N + n_pos],
                                    man_bits, exp_bits),
                      man_bits, exp_bits);
      }
      c[m_pos * N + n_pos] =
          cast_fp(c[m_pos * N + n_pos] + tmp, man_bits, exp_bits);

      m_pos += gridDim.y * blockDim.y;
    }

    n_pos += gridDim.x * blockDim.x;
  }
  return;
}

// ALGORITHM 1: shared memory GEMM kernel (v1)
// code adapted from:
// https://github.com/CoffeeBeforeArch/cuda_programming/blob/8711be08e2060680820304594f3691cc0016edd1/02_matrix_mul/nonMultiple/mmul.cu
// (see function matrixMul)
__global__ void gemm_fp_algo1(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits) {

  // declare shared memory matrices for A and B matrices
  __shared__ float s_a[64];
  __shared__ float s_b[64];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.0;

  // sweep tile across matrix
  for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x) {
  // load in elements for this tile
    s_a[ty * blockDim.x + tx] = (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0;
    s_b[ty * blockDim.x + tx] = (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0;

    // wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // do matrix multiplication on the small matrices
    for (int j = 0; j < blockDim.x; j++) {
      tmp = cast_fp(tmp + cast_fp(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx], 
                    man_bits, exp_bits), man_bits, exp_bits);
    }

    // wait for all threads to finish using current tiles 
    // before loading in new ones
    __syncthreads();
  }

  // write back results
  if (row < M && col < N) c[row * N + col] = tmp;
}

// ALGORITHM 2: shared memory GEMM kernel (v2)
// code adapted from: 
// https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final 
// (slides 30 - 36)
// see also https://github.com/DmitryLyakh/CUDA_Tutorial/blob/master/bla_lib.cu
// (function gpu_gemm_sh_nn)
__global__ void gemm_fp_algo2(float *__restrict__ a, float *__restrict__ b,
                                  float *__restrict__ c, int M, int K, int N,
                                  int man_bits, int exp_bits) {
  __shared__ float lbuf[64][16], rbuf[16][64];
  
  for (int m_pos = blockIdx.y * blockDim.y; m_pos < M; 
                          m_pos += gridDim.y * blockDim.y) { // the offset in the Y dimension
    for (int n_pos = blockIdx.x * blockDim.x; n_pos < N;
                          n_pos += gridDim.x * blockDim.x) { // the offset in the X dimension

      float tmp = 0.0f; // accumulator
      for (int k_pos = 0; k_pos < K; k_pos += 64) { // k_pos is the position of the CUDA
                                                            // thread along the K dimension
        int k_end = k_pos + 64;
        if (k_end > K)
          k_end = K;
        // load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K)
        if (m_pos + threadIdx.y < M) {
          for (int k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x) {
            lbuf[k_loc-k_pos][threadIdx.y] = a[(m_pos + threadIdx.y) * K + k_loc];
          }
        }

        // load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:_TILE_EXT_N)
        if (n_pos + threadIdx.x < N) {
          for (int k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y) {
            rbuf[threadIdx.x][k_loc-k_pos] = b[k_loc * N + n_pos + threadIdx.x];
          }
        }
        __syncthreads();

        // multiply two loaded tiles to produce a tile of 
        // matrix C(m_pos:TILE_EXT_M, n_pos:TILE_EXT_N)
        if (m_pos + threadIdx.y < M && n_pos + threadIdx.x < N) {
          if (k_end - k_pos == 64) { // number of loop iterations is known at compile time
            #pragma unroll
            for (int l = 0; l < 64; ++l) {
              tmp = cast_fp(tmp + cast_fp(lbuf[l][threadIdx.y] * rbuf[threadIdx.x][l], 
                      man_bits, exp_bits), man_bits, exp_bits);
            }
          } else { // number of iterations is not known at compile time
            for (int l = 0; l < (k_end - k_pos); ++l) {
              tmp = cast_fp(tmp + cast_fp(lbuf[l][threadIdx.y] * rbuf[threadIdx.x][l], 
                man_bits, exp_bits), man_bits, exp_bits);
            }
          }
        }
        __syncthreads();
      } // k_pos
      if (m_pos + threadIdx.y < M && n_pos + threadIdx.x < N)
        c[(m_pos + threadIdx.y) * N + (n_pos + threadIdx.x)] = cast_fp(
            c[(m_pos + threadIdx.y) * N + (n_pos + threadIdx.x)] + tmp, man_bits, exp_bits);

    } // n_pos
  } // m_pos
}

// ALGORITHM 3: shared memory + registers GEMM kernel (v1)
// code adapted from: 
// https://bluewaters.ncsa.illinois.edu/liferay-content/image-gallery/content/BLA-final 
// (slides 38 - 45)
// see also https://github.com/DmitryLyakh/CUDA_Tutorial/blob/master/bla_lib.cu
// (function gpu_gemm_sh_reg_nn)
__global__ void gemm_fp_algo3(float *__restrict__ a, float *__restrict__ b,
  float *__restrict__ c, int M, int K, int N,
  int man_bits, int exp_bits) {
    __shared__ float lbuf[16][64], rbuf[64][16];

    for (int m_pos = blockIdx.y * 64; m_pos < M; m_pos += gridDim.y * 64) {
      int m_end = m_pos + 64;
      if (m_end > M) m_end = M;
      for ( int n_pos = blockIdx.x * 64; n_pos < N; n_pos += gridDim.x * 64) {
        int n_end = n_pos + 64;
        if (n_end > N) n_end = N;
  
        if ((m_end - m_pos == 64) && (n_end - n_pos == 64)) {
          // initialize registers to zero:
          float dreg[4][4] = {0.0f};
          float rreg[4] = {0.0f};
          float lreg[4] = {0.0f};
  
          for(int k_pos = 0; k_pos < K; k_pos += 16) { 
            int k_end = k_pos + 16; 
            if (k_end > K) k_end = K;
       
            for(int m_loc = m_pos + threadIdx.y; m_loc < m_end; m_loc += blockDim.y) {
              for(int k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x) {
                lbuf[k_loc - k_pos][m_loc - m_pos] = a[m_loc * K + k_loc];
              }
            }
  
            for (int n_loc = n_pos + threadIdx.x; n_loc < n_end; n_loc += blockDim.x) {
              for (int k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y) {
                rbuf[n_loc - n_pos][k_loc - k_pos] = b[k_loc * N + n_loc];
              }
            }
            __syncthreads();
  
            if (k_end - k_pos == 16) {
              #pragma unroll
              for (int l = 0; l < 16; ++l) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) rreg[j] = rbuf[threadIdx.x + blockDim.x * j][l];
                #pragma unroll
                for (int j = 0; j < 4; ++j) lreg[j] = lbuf[l][threadIdx.y + blockDim.y * j];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                  #pragma unroll
                  for (int i = 0; i < 4; ++i) {
                    dreg[j][i] = cast_fp(dreg[j][i] + cast_fp(lreg[i] * rreg[j],
                      man_bits, exp_bits), man_bits, exp_bits);
                  }
                }
              }
            } else {
              for(int l = 0; l < (k_end - k_pos); ++l) {
                #pragma unroll
                for(int j = 0; j < 4; ++j) rreg[j] = rbuf[threadIdx.x + blockDim.x*j][l];
                #pragma unroll
                for(int j = 0; j < 4; ++j) lreg[j] = lbuf[l][threadIdx.y + blockDim.y*j];
                #pragma unroll
                for(int j = 0; j < 4; ++j){
                  #pragma unroll
                  for(int i = 0; i < 4; ++i){
                    dreg[j][i] = cast_fp(dreg[j][i] + cast_fp(lreg[i] * rreg[j], 
                      man_bits, exp_bits), man_bits, exp_bits);
                  }
                }
              }            
            }
            __syncthreads();
          } // k_pos
  
          #pragma unroll
          for(int j = 0; j < 4; ++j) {
            #pragma unroll
            for(int i = 0; i < 4; ++i) {
              c[(m_pos + threadIdx.y + blockDim.y * i) * N + 
                  (n_pos + threadIdx.x + blockDim.x * j)] = cast_fp(
                    c[(m_pos + threadIdx.y + blockDim.y * i) * N + 
                    (n_pos + threadIdx.x + blockDim.x * j)] + dreg[j][i],
                    man_bits, exp_bits);
            }
          }
        } else {
  
          float dreg[4][4] = {0.0f};
          float rreg[4] = {0.0f};
          float lreg[4] = {0.0f};
  
          for(int k_pos = 0; k_pos < K; k_pos += 16) { 
            int k_end = k_pos + 16; 
            if(k_end > K) k_end = K;
  
            for(int m_loc = m_pos + threadIdx.y; m_loc < m_end; m_loc += blockDim.y) {
              for(int k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x) {
                lbuf[k_loc - k_pos][m_loc - m_pos] = a[m_loc * K + k_loc];
              }
            }
  
            for(int n_loc = n_pos + threadIdx.x; n_loc < n_end; n_loc += blockDim.x) {
              for(int k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y) {
                rbuf[n_loc - n_pos][k_loc - k_pos] = b[k_loc * N + n_loc];
              }
            }
            __syncthreads();
  
            for (int l = 0; l < (k_end - k_pos); ++l) {
              for (int i = 0, j = threadIdx.x; j < n_end - n_pos; j += blockDim.x, i++) 
                rreg[i] = rbuf[j][l];
              for (int i = 0, j = threadIdx.y; j < m_end - m_pos; j += blockDim.y, i++) 
                lreg[i] = lbuf[l][j];
              #pragma unroll
              for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                  dreg[j][i] = cast_fp(dreg[j][i] + cast_fp(lreg[i] * rreg[j],
                    man_bits, exp_bits), man_bits, exp_bits);
                }
              }
            }
            __syncthreads();          
  
          } // k_pos
  
          for(int j = 0, n_loc = n_pos + threadIdx.x; n_loc < n_end; n_loc += blockDim.x, j++) {
            for(int i = 0, m_loc = m_pos + threadIdx.y; m_loc < m_end; m_loc += blockDim.y, i++) {
              c[m_loc * N + n_loc] = cast_fp(c[m_loc * N + n_loc] + dreg[j][i], 
                man_bits, exp_bits);
            }
          }
  
        }
      } // n_pos
    } // m_pos
    return;
}

// ALGORITHM 4: shared memory + registers GEMM kernel (v2)
// code adapted from: 
// https://github.com/drcut/CPD/blob/master/CPDtorch/quant/quant_cuda/float_kernel.cu
// (function tvm_gemm)
__global__ void gemm_fp_algo4(float *__restrict__ a,
                                 float *__restrict__ b,
                                 float *__restrict__ c, int M, int K, int N,
                                 int man_bits, int exp_bits) {
  float c_local[4];
  __shared__ float a_shared[128];
  __shared__ float b_shared[128];
  float a_shared_local[4];
  float b_shared_local[4];
  float c_local1[4];
  float a_shared_local1[4];
  float b_shared_local1[4];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  if (bx < (M / 16)) {
    for (int i_c_init = 0; i_c_init < 2; ++i_c_init) {
      for (int j_c_init = 0; j_c_init < 2; ++j_c_init) {
        c_local[i_c_init * 2 + j_c_init] = 0.0f;
      }
    }
    for (int rx_outer = 0; rx_outer < ((K + 7) / 8); ++rx_outer) {
      __syncthreads();
      for (int ax0_inner = 0; ax0_inner < 2; ++ax0_inner) {
        if (rx_outer * 8 < K - tx) {
          a_shared[ty * 16 + ax0_inner * 8 + tx] =
                  a[rx_outer * 8 + (bx * 16 + ty * 2 + ax0_inner) * K + tx];
        }
      }
      for (int ax1_inner = 0; ax1_inner < 2; ++ax1_inner) {
        if (rx_outer * 8 < K - tx) {
          if (by * 16 + ty * 2 < N - ax1_inner) {
            b_shared[tx * 16 + ty * 2 + ax1_inner] =
                  b[by * 16 + ty * 2 + (rx_outer * 8 + tx) * N + ax1_inner];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer = 0; rx_inner_outer < 4; ++rx_inner_outer) {
        for (int ax0 = 0; ax0 < 2; ++ax0) {
          for (int ax1 = 0; ax1 < 2; ++ax1) {
            if (rx_outer * 8 + rx_inner_outer * 2 < K - ax1) {
              a_shared_local[ax0 * 2 + ax1] = 
                      a_shared[tx * 16 + ax0 * 8 + rx_inner_outer * 2 + ax1];
            }
          }
        }
        for (int ax01 = 0; ax01 < 2; ++ax01) {
          for (int ax11 = 0; ax11 < 2; ++ax11) {
            if (rx_outer * 8 + rx_inner_outer * 2 < K - ax01) {
              if (by * 16 + ty * 2 < N - ax11) {
                b_shared_local[ax01 * 2 + ax11] =
                        b_shared[rx_inner_outer * 32 + ax01 * 16 + ty * 2 + ax11];
              }
            }
          }
        }
        for (int i_c = 0; i_c < 2; ++i_c) {
          for (int j_c = 0; j_c < 2; ++j_c) {
            for (int rx_inner_inner = 0; rx_inner_inner < 2; ++rx_inner_inner) {
              if (rx_outer * 8 + rx_inner_outer * 2 < K - rx_inner_inner) {
                if (by * 16 + ty * 2 < N - j_c) {
                  c_local[i_c * 2 + j_c] = cast_fp(c_local[i_c * 2 + j_c] +
                        cast_fp(a_shared_local[i_c * 2 + rx_inner_inner] * 
                        b_shared_local[rx_inner_inner * 2 + j_c], 
                        man_bits, exp_bits), man_bits, exp_bits);
                }
              }
            }
          }
        }
      }
    }
    for (int i_inner_inner = 0; i_inner_inner < 2; ++i_inner_inner) {
      for (int j_inner_inner = 0; j_inner_inner < 2; ++j_inner_inner) {
        if (by * 16 + ty * 2 < N - j_inner_inner) {
          c[by * 16 + ty * 2 + (bx * 16 + tx * 2 + i_inner_inner) * N + j_inner_inner] =
              c_local[i_inner_inner * 2 + j_inner_inner];
        }
      }
    }
  } else {
    for (int i_c_init1 = 0; i_c_init1 < 2; ++i_c_init1) {
      for (int j_c_init1 = 0; j_c_init1 < 2; ++j_c_init1) {
        c_local1[i_c_init1 * 2 + j_c_init1] = 0.0f;
      }
    }
    for (int rx_outer1 = 0; rx_outer1 < (K + 7) / 8; ++rx_outer1) {
      __syncthreads();
      for (int ax0_inner1 = 0; ax0_inner1 < 2; ++ax0_inner1) {
        if (bx * 16 + ty * 2 < M - ax0_inner1) {
          if (rx_outer1 * 8 < K - tx) {
            a_shared[ty * 16 + ax0_inner1 * 8 + tx] =
                a[rx_outer1 * 8 + (bx * 16 + ty * 2 + ax0_inner1) * K + tx];
          }
        }
      }
      for (int ax1_inner1 = 0; ax1_inner1 < 2; ++ax1_inner1) {
        if (rx_outer1 * 8 < K - tx) {
          if (by * 16 + ty * 2 < N - ax1_inner1) {
            b_shared[tx * 16 + ty * 2 + ax1_inner1] =
                b[by * 16 + ty * 2 + (rx_outer1 * 8 + tx) * N + ax1_inner1];
          }
        }
      }
      __syncthreads();
      for (int rx_inner_outer1 = 0; rx_inner_outer1 < 4; ++rx_inner_outer1) {
        for (int ax02 = 0; ax02 < 2; ++ax02) {
          for (int ax12 = 0; ax12 < 2; ++ax12) {
            if (bx * 16 + tx * 2 < M - ax02) {
              if (rx_outer1 * 8 + rx_inner_outer1 * 2 < K - ax12) {
                a_shared_local1[ax02 * 2 + ax12] = 
                    a_shared[tx * 16 + ax02 * 8 + rx_inner_outer1 * 2 + ax12];
              }
            }
          }
        }
        for (int ax03 = 0; ax03 < 2; ++ax03) {
          for (int ax13 = 0; ax13 < 2; ++ax13) {
            if (rx_outer1 * 8 + rx_inner_outer1 * 2 < K - ax03) {
              if (by * 16 + ty * 2 < N - ax13) {
                b_shared_local1[ax03 * 2 + ax13] =
                    b_shared[rx_inner_outer1 * 32 + ax03 * 16 + ty * 2 + ax13];
              }
            }
          }
        }
        for (int i_c1 = 0; i_c1 < 2; ++i_c1) {
          for (int j_c1 = 0; j_c1 < 2; ++j_c1) {
            for (int rx_inner_inner1 = 0; rx_inner_inner1 < 2;
                 ++rx_inner_inner1) {
              if (rx_outer1 * 8 + rx_inner_outer1 * 2 < K - rx_inner_inner1) {
                if (bx * 16 + tx * 2 < M - i_c1) {
                  if (by * 16 + ty * 2 < N - j_c1) {
                    c_local1[i_c1 * 2 + j_c1] = cast_fp(c_local1[i_c1 * 2 + j_c1] +
                      cast_fp(a_shared_local1[i_c1 * 2 + rx_inner_inner1] * 
                      b_shared_local1[rx_inner_inner1 * 2 + j_c1], 
                      man_bits, exp_bits), man_bits, exp_bits);
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
        if (bx * 16 + tx * 2 < M - i_inner_inner1) {
          if (by * 16 + ty * 2 < N - j_inner_inner1) {
            c[by * 16 + ty * 2 + (bx * 16 + tx * 2 + i_inner_inner1) * N + j_inner_inner1] =
                c_local1[i_inner_inner1 * 2 + j_inner_inner1];
          }
        }
      }
    }
  }
}