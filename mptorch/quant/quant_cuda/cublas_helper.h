#pragma once

#include <ATen/ATen.h>
#include <cublas_v2.h>


using namespace at;

/**
 * Precision configuration structure. Holds information
 * about the I/O matrix datatypes and compute precision
 * used during the CUBLAS (B)MM calls.
 */
struct cublas_config
{
    cudaDataType matrix_a;
    cudaDataType matrix_b;
    cudaDataType matrix_c;
    cudaDataType scalar;
    cublasComputeType_t compute;

    void summary() const;
};

/**
 * Possible I/O matrix datatypes (binary32, binary16 and bfloat16).
 */
enum class cublas_matrix_dt {
    kF32, kF16, kBF16
};

/**
 * Compute precision/reduction configuration for CUBLAS computations.
 * The **fast** variations allow the use of tensor core with automatic
 * downconversion and binary16/bfloat16/tfloat32 compute for binary32
 * I/O matrices.
 */
enum class cublas_compute_dt {
    kF32, kF16,
    kFastF16,
    kFastBF16,
    kFastTF32
};


/**
 * Creates a new cuBLAS handle, if none hasn't been created yet. If it
 * already exists, nothing happens.
*/
void create_cublas_handle();

/**
 * Deletes the current cuBLAS handle, if has already been created. If not,
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
