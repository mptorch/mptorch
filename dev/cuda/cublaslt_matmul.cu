/*
Matrix-matrix multiply using cublasLtMatmul function.
Adapted from:
https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu

Compile example:
nvcc -O3 cublaslt_matmul.cu -o cublaslt_matmul -lcublas -lcublasLt

Run with:
./cublaslt_matmul
*/

#include <cublas_v2.h>
#include <cublasLt.h>
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
    cublasLtHandle_t handle;

    printf("CUBLASLt Matmul %dx%dx%d (MxNxK)...\n", M, N, K);

    /* Initialize CUBLASLt */
    cublasCheck(cublasLtCreate(&handle));

    h_A = make_random_float(M * K);
    h_B = make_random_float(K * N);
    h_C = (float *)(malloc(M * N * sizeof(h_C[0])));

    /* Allocate device memory for the matrices */
    cudaCheck(cudaMalloc(&d_A, M * K * sizeof(d_A[0])));
    cudaCheck(cudaMalloc(&d_B, K * N * sizeof(d_B[0])));
    cudaCheck(cudaMalloc(&d_C, M * N * sizeof(d_C[0])));

    /* Copy host matrices content into the device matrices */
    cublasCheck(cublasSetVector(M * K, sizeof(h_A[0]), h_A, 1, d_A, 1));
    cublasCheck(cublasSetVector(K * N, sizeof(h_B[0]), h_B, 1, d_B, 1));
    cublasCheck(cublasSetVector(M * N, sizeof(h_C[0]), h_C, 1, d_C, 1));

    /* Perform matrix multiply operation using cublasLt */
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    int lda = M;
    int ldb = K;
    int ldc = M;

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    cublasCheck(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // create matrix descriptors, we are good with the details here so no need to set any extra attributes
    cublasCheck(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? M : K, transa == CUBLAS_OP_N ? K : M, lda));
    cublasCheck(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? K : N, transb == CUBLAS_OP_N ? N : K, ldb));
    cublasCheck(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, M, N, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    cublasLtMatmulPreference_t preference = NULL;
    cublasCheck(cublasLtMatmulPreferenceCreate(&preference));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    cublasCheck(cublasLtMatmulAlgoGetHeuristic(
        handle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        cublasCheck(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    auto cublasLt_matmul = [&]()
    {
        cublasCheck(cublasLtMatmul(handle, operationDesc, &alpha,
                                   d_A, Adesc,
                                   d_B, Bdesc, &beta,
                                   d_C, Cdesc,
                                   d_C, Cdesc,
                                   &heuristicResult.algo,
                                   NULL, 0, 0));
    };
    cublasLt_matmul();

    /* Performs operation using reference CPU C code */
    simple_sgemm(M, N, K, alpha, h_A, h_B, beta, h_C);

    /* Check result against reference */
    validate_result(d_C, h_C, "C", M * N, 1.0e-2f);
    printf("All results match. Starting benchmarks.\n\n");

    /* Benchmark */
    int repeat_times = 1000;
    float elapsed_time = benchmark_gpu_kernel(repeat_times, cublasLt_matmul);
    printf("time %.4f ms\n", elapsed_time);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference)
        cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc)
        cublasCheck(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc)
        cublasCheck(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc)
        cublasCheck(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc)
        cublasCheck(cublasLtMatmulDescDestroy(operationDesc));

    /* Memory clean up */
    free(h_A);
    free(h_B);
    free(h_C);

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));

    return 0;
}
