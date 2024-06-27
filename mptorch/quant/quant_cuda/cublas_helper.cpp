#include "quant.h"

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <stdio.h>


using namespace at;


static cublasHandle_t cublas_handle = NULL;


cublasHandle_t get_cublas_handle() {
    if (cublas_handle == NULL) {
        throw std::runtime_error("cuBLAS not initialized.");
    }
    return cublas_handle;
}

void create_cublas_handle() {
    if (cublas_handle != NULL) {
        return;
    }
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to create cuBLAS.");
    }
}

void delete_cublas_handle() {
    cublasStatus_t status = cublasDestroy(cublas_handle);
    if(status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("Failed to destroy cuBLAS.");
    }
}


static const char* datatype_string(cudaDataType t) {
    switch (t) {
    case CUDA_R_32F:  return "CUDA_R_32F";
    case CUDA_R_16F:  return "CUDA_R_16F";
    case CUDA_R_16BF: return "CUDA_R_16BF";
    default:
        throw std::runtime_error("Not implemented.");
    }
}

static const char* computetype_string(cublasComputeType_t t) {
    switch (t) {
    case CUBLAS_COMPUTE_32F:           return "CUBLAS_COMPUTE_32F";
    case CUBLAS_COMPUTE_32F_PEDANTIC:  return "CUBLAS_COMPUTE_32F_PEDANTIC";
    case CUBLAS_COMPUTE_16F:           return "CUBLAS_COMPUTE_16F";
    case CUBLAS_COMPUTE_16F_PEDANTIC:  return "CUBLAS_COMPUTE_16F_PEDANTIC";
    case CUBLAS_COMPUTE_32F_FAST_16F:  return "CUBLAS_COMPUTE_32F_FAST_16F";
    case CUBLAS_COMPUTE_32F_FAST_16BF: return "CUBLAS_COMPUTE_32F_FAST_16BF";
    case CUBLAS_COMPUTE_32F_FAST_TF32: return "CUBLAS_COMPUTE_32F_FAST_TF32";
    default:
        throw std::runtime_error("Not implemented.");
    }
}


void cublas_config::summary() const {
    printf("matrix A: %s\n", datatype_string(matrix_a));
    printf("matrix B: %s\n", datatype_string(matrix_b));
    printf("matrix C: %s\n", datatype_string(matrix_c));
    printf("compute type: %s\n", computetype_string(compute));
}


void get_cublas_configuration(
    cublas_matrix_dt inp_matrix_type,
    cublas_matrix_dt out_matrix_type,
    cublas_compute_dt compute_type,
    bool pedantic,
    cublas_config& config
) {
    auto to_datatype = [](cublas_matrix_dt t) {
        switch (t) {
        case cublas_matrix_dt::kF32:
            return CUDA_R_32F;
        case cublas_matrix_dt::kF16:
            return CUDA_R_16F;
        case cublas_matrix_dt::kBF16:
            return CUDA_R_16BF;
        default:
            throw std::invalid_argument("Invalid data type.");
        }
    };
    cudaDataType inp_type = to_datatype(inp_matrix_type);
    cudaDataType out_type = to_datatype(out_matrix_type);
    config.matrix_a = inp_type;
    config.matrix_b = inp_type;
    config.matrix_c = out_type;

    auto to_scalartype = [](cublas_compute_dt t) {
        switch (t) {
        case cublas_compute_dt::kF16:
            return CUDA_R_16F;
        default:
            return CUDA_R_32F;
        }
    };
    config.scalar = to_scalartype(compute_type);

    auto types_match = [&](cudaDataType t_in, cudaDataType t_out) {
        return t_in == inp_type && t_out == out_type;
    };

    auto assert_types = [](bool valid_type) {
        if (!valid_type) {
            throw std::invalid_argument(
                "Invalid input/output combination for the "
                "given accumulator/compute type."
            );
        }
    };

    // compatibility table from: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
    auto to_computetype = [&](cublas_compute_dt t) {
        switch (t) {
        case cublas_compute_dt::kF32:
            assert_types(
               types_match(CUDA_R_32F,  CUDA_R_32F)
            || types_match(CUDA_R_16F,  CUDA_R_32F)
            || types_match(CUDA_R_16BF, CUDA_R_32F)
            || types_match(CUDA_R_16F,  CUDA_R_16F)
            || types_match(CUDA_R_16BF, CUDA_R_16BF)
            );
            if (pedantic) {
                return CUBLAS_COMPUTE_32F_PEDANTIC;
            }
            return CUBLAS_COMPUTE_32F;
        
        case cublas_compute_dt::kF16:
            assert_types(types_match(CUDA_R_16F, CUDA_R_16F));
            if (pedantic) {
                return CUBLAS_COMPUTE_16F_PEDANTIC;
            }
            return CUBLAS_COMPUTE_16F;
        
        case cublas_compute_dt::kFastF16:
            assert_types(types_match(CUDA_R_32F, CUDA_R_32F));
            if (pedantic) {
                throw std::invalid_argument("FAST_F16 cannot be pedantic");
            }
            return CUBLAS_COMPUTE_32F_FAST_16F;
        
        case cublas_compute_dt::kFastBF16:
            assert_types(types_match(CUDA_R_32F, CUDA_R_32F));
            if (pedantic) {
                throw std::invalid_argument("FAST_BF16 cannot be pedantic");
            }
            return CUBLAS_COMPUTE_32F_FAST_16BF;
        
        case cublas_compute_dt::kFastTF32:
            assert_types(types_match(CUDA_R_32F, CUDA_R_32F));
            if (pedantic) {
                throw std::invalid_argument("FAST_TF32 cannot be pedantic");
            }
            return CUBLAS_COMPUTE_32F_FAST_TF32;
        
        default:
            throw std::invalid_argument("Invalid compute type.");
        }
    };
    config.compute = to_computetype(compute_type);
}


void float_mm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                     cublas_matrix_dt AB_type, cublas_matrix_dt C_type,
                     cublas_compute_dt compute_type, bool pedantic)
{
  // Tensors a, b, and c are assumed to have the right datatype and be properly transposed.
  cublas_config config;
  get_cublas_configuration(AB_type, C_type, compute_type, pedantic, config);

  cublasMath_t math = pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH;
  math = (cublasMath_t)(math | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  cublasSetMathMode(get_cublas_handle(), math);

  // special case for scalar types: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex
  switch (config.scalar) {
  case CUDA_R_16F:
    {
    half alpha = __float2half(1.f);
    half beta = __float2half(0.f);
    cublasGemmEx(get_cublas_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                  a.data_ptr(), config.matrix_a, M,
                  b.data_ptr(), config.matrix_b, K, &beta,
                  c.data_ptr(), config.matrix_c, M,
                  config.compute,
                  CUBLAS_GEMM_DEFAULT);
    }
    break;
  default:
    {
    float alpha = 1.f;
    float beta = 0.f;
    cublasGemmEx(get_cublas_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                  a.data_ptr(), config.matrix_a, M,
                  b.data_ptr(), config.matrix_b, K, &beta,
                  c.data_ptr(), config.matrix_c, M,
                  config.compute,
                  CUBLAS_GEMM_DEFAULT);
    }
    break;
  }
}


void float_bmm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                      cublas_matrix_dt AB_type, cublas_matrix_dt C_type,
                      cublas_compute_dt compute_type, bool pedantic)
{
  // Tensors a, b, and c are assumed to have the right datatype and be properly transposed.
  cublas_config config;
  get_cublas_configuration(AB_type, C_type, compute_type, pedantic, config);

  cublasMath_t math = pedantic ? CUBLAS_PEDANTIC_MATH : CUBLAS_DEFAULT_MATH;
  math = (cublasMath_t)(math | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  cublasSetMathMode(get_cublas_handle(), math);

  int B = a.sizes().size() > 2 ? a.size(0) : 1; // batch count

  // Allocate the array of pointers for each matrix  
  auto get_ptrs = [B](void** arr, Tensor a, cudaDataType t, int stride) {
    switch (t) {
    case CUDA_R_32F: {
        float *p = a.data_ptr<float>();
        for (int i = 0; i < B; i++) arr[i] = p + i * stride;
      }
      break;
    case CUDA_R_16F: {
        at::Half *p = a.data_ptr<at::Half>();
        for (int i = 0; i < B; i++) arr[i] = p + i * stride;
      }
      break;
    case CUDA_R_16BF: {
        at::BFloat16 *p = a.data_ptr<at::BFloat16>();
        for (int i = 0; i < B; i++) arr[i] = p + i * stride;
      }
      break;
    default:
      throw std::invalid_argument("Invalid matrix datatype.");
    }
  };

  void *h_dA[B], *h_dB[B], *h_dC[B];
  void **d_dA, **d_dB, **d_dC;

  get_ptrs(h_dA, a, config.matrix_a, M*K);
  get_ptrs(h_dB, b, config.matrix_b, K*N);
  get_ptrs(h_dC, c, config.matrix_c, M*N);
  cudaMalloc(&d_dA, B * sizeof(float*));
  cudaMalloc(&d_dB, B * sizeof(float*));
  cudaMalloc(&d_dC, B * sizeof(float*));
  cudaMemcpy(d_dA, h_dA, B * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dB, h_dB, B * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dC, h_dC, B * sizeof(float*), cudaMemcpyHostToDevice);

  // special case for scalar types: https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex
  switch (config.scalar) {
  case CUDA_R_16F:
    {
    half alpha = __float2half(1.f);
    half beta = __float2half(0.f);
    cublasGemmBatchedEx(get_cublas_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                  d_dA, config.matrix_a, M,
                  d_dB, config.matrix_b, K, &beta,
                  d_dC, config.matrix_c, M, B,
                  config.compute,
                  CUBLAS_GEMM_DEFAULT);
    }
    break;
  default:
    {
    float alpha = 1.f;
    float beta = 0.f;
    cublasGemmBatchedEx(get_cublas_handle(),
                  CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                  d_dA, config.matrix_a, M,
                  d_dB, config.matrix_b, K, &beta,
                  d_dC, config.matrix_c, M, B,
                  config.compute,
                  CUBLAS_GEMM_DEFAULT);
    }
    break;
  }

  cudaFree(d_dA);
  cudaFree(d_dB);
  cudaFree(d_dC);
}