#include "cublas_helper.h"

#include <cublas_v2.h>
#include <stdexcept>
#include <stdio.h>


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
        case CUBLAS_COMPUTE_32F:            return "CUBLAS_COMPUTE_32F";
        case CUBLAS_COMPUTE_32F_PEDANTIC:   return "CUBLAS_COMPUTE_32F_PEDANTIC";
        case CUBLAS_COMPUTE_16F:            return "CUBLAS_COMPUTE_16F";
        case CUBLAS_COMPUTE_16F_PEDANTIC:   return "CUBLAS_COMPUTE_16F_PEDANTIC";
        case CUBLAS_COMPUTE_32F_FAST_16F:   return "CUBLAS_COMPUTE_32F_FAST_16F";
        case CUBLAS_COMPUTE_32F_FAST_16BF:  return "CUBLAS_COMPUTE_32F_FAST_16BF";
        case CUBLAS_COMPUTE_32F_FAST_TF32:  return "CUBLAS_COMPUTE_32F_FAST_TF32";
        default:
            throw std::runtime_error("Not implemented.");
    }
}

void get_cublas_configuration(
    cublas_matrix_dt inp_matrix_type,
    cublas_matrix_dt out_matrix_type,
    cublas_compute_dt compute_type,
    bool pedantic,
    cublas_config& config
) {
    config.algo = CUBLAS_GEMM_DEFAULT; // uses heuristic for the best algorithm

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

    auto types_match = [&](cudaDataType t_in, cudaDataType t_out) {
        return t_in == inp_type && t_out == out_type;
    };

    auto assert_types = [](bool valid_type) {
        if (!valid_type) {
            throw std::invalid_argument(
                "Invalid input/output combination for the "
                "given accumulator type."
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

