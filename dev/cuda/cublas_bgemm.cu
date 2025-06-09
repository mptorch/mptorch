/*
Batched matrix-matrix multiply using cublasGemmBatchedEx function.

Compile example:
nvcc -O3 cublas_bgemm.cu -o cublas_bgemm -lcublas

Version 1 uses separate memory allocation for each matrices A[i], B[i] and C[i]
./cublas_bgemm 1

Version 2 stores batch matrcies A, B and C in a single array and passes the pointer
to the beginning of each matrix A[i], B[i] and C[i] to cublasGemmBatchedEx
./cublas_bgemm 2
*/

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "common.h"

// ---------------------------------------------------------------------------------------
/* Host (CPU) implementation of a simple version of sgemm */
static void simple_bsgemm(int P, int M, int N, int K, float alpha, const float **A, const float **B,
                          float beta, float **C)
{
    for (int p = 0; p < P; p++)
    {
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float prod = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    prod = prod + A[p][k * M + i] * B[p][j * K + k];
                }
                C[p][j * M + i] = alpha * prod + beta * C[p][j * M + i];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
/*This function allocates distinct memory locations for each matrices A[i], B[i] and C[i] */
void main_separate_arrays(cublasHandle_t handle, int P, int M, int N, int K)
{
    printf("CUBLAS GEMM %dx%dx%d (MxNxK) using separate arrays...\n", M, N, K);

    float **h_A, **h_B, **h_C;
    float **h_dA, **h_dB, **h_dC;
    float **d_dA, **d_dB, **d_dC;
    float alpha = 1.0f;
    float beta = 0.0f;

    h_A = new float *[P];
    h_B = new float *[P];
    h_C = new float *[P];
    h_dA = new float *[P];
    h_dB = new float *[P];
    h_dC = new float *[P];

    for (int i = 0; i < P; i++)
    {
        h_A[i] = make_random_float(M * K);
        h_B[i] = make_random_float(K * N);
        h_C[i] = make_zeros_float(M * N);
    }

    /* Allocate device memory for the matrices */
    for (int i = 0; i < P; i++)
    {
        cudaCheck(cudaMalloc(&h_dA[i], M * K * sizeof(float)));
        cudaCheck(cudaMalloc(&h_dB[i], K * N * sizeof(float)));
        cudaCheck(cudaMalloc(&h_dC[i], M * N * sizeof(float)));

        cudaCheck(cudaMemcpy(h_dA[i], h_A[i], M * K * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(h_dB[i], h_B[i], K * N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(h_dC[i], h_C[i], M * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    cudaCheck(cudaMalloc(&d_dA, P * sizeof(float *)));
    cudaCheck(cudaMalloc(&d_dB, P * sizeof(float *)));
    cudaCheck(cudaMalloc(&d_dC, P * sizeof(float *)));

    cudaCheck(cudaMemcpy(d_dA, h_dA, P * sizeof(float *), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dB, h_dB, P * sizeof(float *), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dC, h_dC, P * sizeof(float *), cudaMemcpyHostToDevice));

    int lda = M;
    int ldb = K;
    int ldc = M;

    /* Performs operation using cublas */
    auto cublas_bgemm = [&]()
    {
        cublasCheck(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                                        (void **)d_dA, CUDA_R_32F, lda,
                                        (void **)d_dB, CUDA_R_32F, ldb, &beta,
                                        (void **)d_dC, CUDA_R_32F, ldc, P,
                                        CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT));
    };
    cublas_bgemm();

    /* Performs operation using reference CPU C code */
    simple_bsgemm(P, M, N, K, alpha, (const float **)h_A, (const float **)h_B, beta, h_C);

    /* Check CUBLAS batched GEMM result against reference */
    for (int i = 0; i < P; i++)
    {
        validate_result(h_dC[i], h_C[i], "C", M * N, 1.0e-3f);
        printf("\n");
    }
    printf("All results match. Starting benchmarks.\n\n");

    /* Benchmark */
    int repeat_times = 1000;
    float elapsed_time = benchmark_gpu_kernel(repeat_times, cublas_bgemm);
    printf("time %.4f ms\n", elapsed_time);

    /* Memory clean up */
    for (int i = 0; i < P; i++)
    {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);

        cudaCheck(cudaFree(h_dA[i]));
        cudaCheck(cudaFree(h_dB[i]));
        cudaCheck(cudaFree(h_dC[i]));
    }

    cudaCheck(cudaFree(d_dA));
    cudaCheck(cudaFree(d_dB));
    cudaCheck(cudaFree(d_dC));

    delete[] h_dA;
    delete[] h_dB;
    delete[] h_dC;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

/*
 * This version uses a single contiguous array to store the batches of matrices A, B and C,
 * and the beginning address of each matrix is computed and passed to cublasGemmBatchedEx.
 */
void main_shared_arrays(cublasHandle_t handle, int P, int M, int N, int K)
{
    printf("CUBLAS GEMM %dx%dx%d (MxNxK) using shared arrays...\n", M, N, K);

    float **h_A, **h_B, **h_C;
    float *d_A, *d_B, *d_C;
    float **h_dA, **h_dB, **h_dC;
    float **d_dA, **d_dB, **d_dC;
    float alpha = 1.0f;
    float beta = 0.0f;

    h_A = new float *[P];
    h_B = new float *[P];
    h_C = new float *[P];
    h_dA = new float *[P];
    h_dB = new float *[P];
    h_dC = new float *[P];

    for (int i = 0; i < P; i++)
    {
        h_A[i] = make_random_float(M * K);
        h_B[i] = make_random_float(K * N);
        h_C[i] = make_zeros_float(M * N);
    }

    /* Allocate device memory for the matrices */
    cudaCheck(cudaMalloc(&d_A, P * M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_B, P * K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_C, P * M * N * sizeof(float)));

    for (int i = 0; i < P; i++)
    {
        cudaCheck(cudaMemcpy(d_A + i * M * K, h_A[i], M * K * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_B + i * K * N, h_B[i], K * N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_C + i * M * N, h_C[i], M * N * sizeof(float), cudaMemcpyHostToDevice));
    }

    // Retrieve each pointers to each matrices
    for (int i = 0; i < P; i++)
    {
        h_dA[i] = d_A + i * M * K;
        h_dB[i] = d_B + i * K * N;
        h_dC[i] = d_C + i * M * N;
    }

    cudaCheck(cudaMalloc(&d_dA, P * sizeof(float *)));
    cudaCheck(cudaMalloc(&d_dB, P * sizeof(float *)));
    cudaCheck(cudaMalloc(&d_dC, P * sizeof(float *)));

    cudaCheck(cudaMemcpy(d_dA, h_dA, P * sizeof(float *), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dB, h_dB, P * sizeof(float *), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dC, h_dC, P * sizeof(float *), cudaMemcpyHostToDevice));

    int lda = M;
    int ldb = K;
    int ldc = M;

    /* Performs operation using cublas */
    auto cublas_bgemm = [&]()
    {
        cublasCheck(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                                        (void **)d_dA, CUDA_R_32F, lda,
                                        (void **)d_dB, CUDA_R_32F, ldb, &beta,
                                        (void **)d_dC, CUDA_R_32F, ldc, P,
                                        CUBLAS_COMPUTE_32F,
                                        CUBLAS_GEMM_DEFAULT));
    };
    cublas_bgemm();

    // /* Performs operation using reference CPU C code */
    simple_bsgemm(P, M, N, K, alpha, (const float **)h_A, (const float **)h_B, beta, h_C);

    /* Check result against reference */
    for (int i = 0; i < P; i++)
    {
        validate_result(d_C + i * M * N, h_C[i], "C", M * N, 1.0e-3f);
        printf("\n");
    }
    printf("All results match. Starting benchmarks.\n\n");

    /* Benchmark */
    int repeat_times = 1000;
    float elapsed_time = benchmark_gpu_kernel(repeat_times, cublas_bgemm);
    printf("time %.4f ms\n", elapsed_time);

    /* Memory clean up */
    for (int i = 0; i < P; i++)
    {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);
    }

    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));

    cudaCheck(cudaFree(d_dA));
    cudaCheck(cudaFree(d_dB));
    cudaCheck(cudaFree(d_dC));

    delete[] h_dA;
    delete[] h_dB;
    delete[] h_dC;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// ---------------------------------------------------------------------------------------
int main(int argc, const char **argv)
{
    setup_main();

    const int M = 512;
    const int N = 512;
    const int K = 1024;
    const int P = 3; // batch count
    cublasHandle_t handle;

    /* Initialize CUBLAS */
    cublasCheck(cublasCreate(&handle));

    int version = 1;
    if (argc > 1)
    {
        version = atoi(argv[1]);
    }

    switch (version)
    {
    case 1:
        main_separate_arrays(handle, P, M, N, K);
        break;
    case 2:
        main_shared_arrays(handle, P, M, N, K);
        break;
    default:
        printf("Invalid version number\n");
        exit(1);
    }

    return 0;
}
