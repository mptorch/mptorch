#pragma once

#include <ATen/ATen.h>
#include <cublas_v2.h>


using namespace at;


struct cublas_config
{
    cudaDataType matrix_a;
    cudaDataType matrix_b;
    cudaDataType matrix_c;
    cudaDataType scalar;
    cublasComputeType_t compute;

    void summary() const;
};


enum class cublas_matrix_dt {
    kF32, kF16, kBF16
};

enum class cublas_compute_dt {
    kF32, kF16,
    kFastF16,
    kFastBF16,
    kFastTF32
};


/**
 * Creates a new cuBLAS handle, if none hasn't been created yet. If it
 * already exist, nothing happens.
*/
void create_cublas_handle();

/**
 * Deletes the current cuBLAS handle, if has already been created. If not
 * throws an error.
*/
void delete_cublas_handle();

/**
 * Performs a matrix multiplication using cuBLAS API with the precision
 * configuration defined by the user.
*/
void float_mm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                     cublas_matrix_dt AB_type, cublas_matrix_dt C_type,
                     cublas_compute_dt compute_type, bool pedantic);


/**
 * Performs a batched matrix multiplication using cuBLAS API with the
 * precision configuration defined by the user.
*/
void float_bmm_cublas(Tensor a, Tensor b, Tensor c, int M, int N, int K,
                      cublas_matrix_dt AB_type, cublas_matrix_dt C_type,
                      cublas_compute_dt compute_type, bool pedantic);
