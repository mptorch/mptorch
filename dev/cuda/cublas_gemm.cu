/* Matrix-matrix multiply using cublasGemmEx function.

Compile example:
nvcc -O3 cublas_gemm.cu -o cublas_gemm -lcublas

Run with:
./cublas_gemm
*/


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "common.h"


/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int M, int N, int K, float alpha, const float *A, const float *B,
                         float beta, float *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float prod = 0.0f;
            for (int k = 0; k < K; ++k) {
                prod = prod + A[k * M + i] * B[j * K + k];
            }
            C[j * M + i] = alpha * prod + beta * C[j * M + i];
        }
    }
}

void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLASLt error: %s:%d\n%s\n", file, line,
            cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

#define cublasCheck(status) check_cublas(status, __FILE__, __LINE__)


int main(int argc, char **argv) {
    setup_main();

    const int M = 1024;
    const int N = 1024;
    const int K = 2048;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;

    printf("CUBLAS GEMM %dx%dx%d (MxNxK)...\n", M, N, K);

    /* Initialize CUBLAS */
    cublasCheck(cublasCreate(&handle));

    h_A = make_random_float(M*K);
    h_B = make_random_float(K*N);
    h_C = (float*)(malloc(M*N * sizeof(h_C[0])));

    /* Allocate device memory for the matrices */
    cudaCheck(cudaMalloc(&d_A, M*K * sizeof(d_A[0])));
    cudaCheck(cudaMalloc(&d_B, K*N * sizeof(d_B[0])));
    cudaCheck(cudaMalloc(&d_C, M*N * sizeof(d_C[0])));

    /* Initialize the device matrices with the host matrices */
    cublasCheck(cublasSetVector(M*K, sizeof(h_A[0]), h_A, 1, d_A, 1));
    cublasCheck(cublasSetVector(K*N, sizeof(h_B[0]), h_B, 1, d_B, 1));
    cublasCheck(cublasSetVector(M*N, sizeof(h_C[0]), h_C, 1, d_C, 1));

    /* Performs operation using plain C code */
    simple_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C);

    /* Performs operation using cublas */
    int lda = M;
    int ldb = K;
    int ldc = M;
    cublasCheck(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, 
                        d_A, CUDA_R_32F, lda,           /* 32-bit float A */
                        d_B, CUDA_R_32F, ldb, &beta,    /* 32-bit float B */
                        d_C, CUDA_R_32F, ldc,           /* 32-bit float C */
                        CUBLAS_COMPUTE_32F,             /* 32-bit computation */
                        CUBLAS_GEMM_DEFAULT));

    /* Check result against reference */
    validate_result(d_C, h_C, "C", M*N);

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));
}
