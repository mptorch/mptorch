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

__global__ void tvm_gemm_fp(float *__restrict__ a, float *__restrict__ b, float *__restrict__ c,
                        int M, int K, int N, int man_bits, int exp_bits) {

	//declare shared memory matrices for A and B matrices
	__shared__ float shared_a_tile[2][2];
	__shared__ float shared_b_tile[2][2];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//check if thread directly maps to the dimensions of the resulting matrix
	if (row < M && col < N)
	{
		float result = 0;
		int k;
		int phase;
		
		//calculate C matrix indexes in phases. Each phase shares 
		//2 * 2 data copied to the shared matrix A and matrix B.
		for (phase = 0; phase <= K/2; phase++)
		{
			shared_a_tile[ty][tx] = a[row * K + phase * 2 + tx];
			shared_b_tile[ty][tx] = b[(phase * 2 + ty) * N + col];
			__syncthreads();
			
			for (k = 0; k < 2; k++)
			{
				if (k + (phase * 2) < K) 
				{
					result = cast_fp(result + cast_fp(shared_a_tile[ty][k] * shared_b_tile[k][tx], 
                      man_bits, exp_bits), man_bits, exp_bits);
				}
			}
			__syncthreads();
		}	
		c[row * N + col] = result;
	}
}

__global__ void tvm_gemm_fp_fp(float *__restrict__ a, float *__restrict__ b,
                         float *__restrict__ c, int M, int K, int N, 
                         int m_man_bits, int m_exp_bits, int a_man_bits, int a_exp_bits)
{
	//declare shared memory matrices for A and B matrices
	__shared__ float shared_a_tile[2][2];
	__shared__ float shared_b_tile[2][2];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//check if thread directly maps to the dimensions of the resulting matrix
	if (row < M && col < N)
	{
		float result = 0;
		int k;
		int phase;
		
		//calculate C matrix indexes in phases. Each phase shares 
		//2 * 2 data copied to the shared matrix A and matrix B.
		for (phase = 0; phase <= K/2; phase++)
		{
			shared_a_tile[ty][tx] = a[row * K + phase * 2 + tx];
			shared_b_tile[ty][tx] = b[(phase * 2 + ty) * N + col];
			__syncthreads();
			
			for (k = 0; k < 2; k++)
			{
				if (k + (phase * 2) < K) 
				{
					result = cast_fp(result + cast_fp(shared_a_tile[ty][k] * shared_b_tile[k][tx], 
                      m_man_bits, m_exp_bits), a_man_bits, a_exp_bits);
				}
			}
			__syncthreads();
		}	
		c[row * N + col] = result;
	}
}

__global__ void tvm_gemm_fp_fxp(float *__restrict__ a, 
                        float *__restrict__ b,
                        float *__restrict__ c, int M, int K, int N, 
                        int m_man_bits, int m_exp_bits,
                        int Qi, int Qf, bool symmetric)
{
	//declare shared memory matrices for A and B matrices
	__shared__ float shared_a_tile[2][2];
	__shared__ float shared_b_tile[2][2];
  double t_min = -ldexp(1.0, Qi-1);
  double t_max = -t_min;
  if (!symmetric) 
    t_max -= ldexp(1.0, -Qf);

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//check if thread directly maps to the dimensions of the resulting matrix
	if (row < M && col < N)
	{
		float result = 0;
		int k;
		int phase;
		
		//calculate C matrix indexes in phases. Each phase shares 
		//2 * 2 data copied to the shared matrix A and matrix B.
		for (phase = 0; phase <= K/2; phase++)
		{
			shared_a_tile[ty][tx] = a[row * K + phase * 2 + tx];
			shared_b_tile[ty][tx] = b[(phase * 2 + ty) * N + col];
			__syncthreads();
			
			for (k = 0; k < 2; k++)
			{
				if (k + (phase * 2) < K) 
				{
					result = cast_fxp(cast_fxp(result, -Qf, t_min, t_max) + 
                    cast_fxp(cast_fp(shared_a_tile[ty][k] * shared_b_tile[k][tx], 
                      m_man_bits, m_exp_bits), -Qf, t_min, t_max), -Qf, t_min, t_max);
				}
			}
			__syncthreads();
		}	
		c[row * N + col] = result;
	}
}

__global__ void tvm_gemm_fma(float *__restrict__ a, 
                        float *__restrict__ b,
                        float *__restrict__ c, int M, int K, int N, 
                        int man_bits, int exp_bits)
{
	//declare shared memory matrices for A and B matrices
	__shared__ float shared_a_tile[2][2];
	__shared__ float shared_b_tile[2][2];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//check if thread directly maps to the dimensions of the resulting matrix
	if (row < M && col < N)
	{
		float result = 0;
		int k;
		int phase;
		
		//calculate C matrix indexes in phases. Each phase shares 
		//2 * 2 data copied to the shared matrix A and matrix B.
		for (phase = 0; phase <= K/2; phase++)
		{
			shared_a_tile[ty][tx] = a[row * K + phase * 2 + tx];
			shared_b_tile[ty][tx] = b[(phase * 2 + ty) * N + col];
			__syncthreads();
			
			for (k = 0; k < 2; k++)
			{
				if (k + (phase * 2) < K) 
				{
					result = cast_fp(fma(shared_a_tile[ty][k], shared_b_tile[k][tx], 
                    result), man_bits, exp_bits);
				}
			}
			__syncthreads();
		}	
		c[row * N + col] = result;
	}
}