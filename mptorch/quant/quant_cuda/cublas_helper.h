#pragma once

#include <cublas_v2.h>


struct cublas_config
{
    cudaDataType matrix_a;
    cudaDataType matrix_b;
    cudaDataType matrix_c;
    cublasComputeType_t compute;
    cublasGemmAlgo_t algo;
};


enum class cublas_matrix_dt {
    kF32, kF16, kBF16
};

enum class cublas_compute_t {
    kF32, kF16,
    kFastF16,
    kFastBF16,
    kFastTF32
};


cublasHandle_t get_cublas_handle();
void create_cublas_handle();
void delete_cublas_handle();

void get_cublas_configuration(
    cublas_matrix_dt AB_type,
    cublas_matrix_dt C_type,
    cublas_compute_t compute_type,
    bool pedantic,
    cublas_config& config
);
