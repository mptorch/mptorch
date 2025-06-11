/*
Matrix-matrix multiply using cublasGemmEx function.

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

// ---------------------------------------------------------------------------------------
/* Host (CPU) implementation of a simple version of sgemm */
static void simple_sgemm(int M, int N, int K, float alpha, const float *A, const float *B,
                         float beta, float *C)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float prod = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                prod = prod + A[k * M + i] * B[j * K + k];
            }
            C[j * M + i] = alpha * prod + beta * C[j * M + i];
        }
    }
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
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

    h_A = make_random_float(M * K);
    h_B = make_random_float(K * N);
    h_C = make_zeros_float(M * N);

    /* Allocate device memory for the matrices */
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(float)));

    /* Copy host matrices content into the device matrices */
    cudaCheck(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    int lda = M;
    int ldb = K;
    int ldc = M;

    /* Perform matrix multiply operation using cublas */
    auto cublas_gemm = [&]()
    {
        cublasCheck(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                                 d_A, CUDA_R_32F, lda,        /* 32-bit float A */
                                 d_B, CUDA_R_32F, ldb, &beta, /* 32-bit float B */
                                 d_C, CUDA_R_32F, ldc,        /* 32-bit float C */
                                 CUBLAS_COMPUTE_32F,          /* 32-bit computation */
                                 CUBLAS_GEMM_DEFAULT));
    };
    cublas_gemm();

    /* Perform matrix multipy using reference CPU C code */
    simple_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C);

    /* Check CUBLAS GEMM result against reference */
    validate_result(d_C, h_C, "C", M * N, 1.0e-2f);
    printf("All results match. Starting benchmarks.\n\n");

    /* Benchmark */
    int repeat_times = 1000;
    float elapsed_time = benchmark_gpu_kernel(repeat_times, cublas_gemm);
    printf("time %.4f ms\n", elapsed_time);

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));

    return 0;
}
