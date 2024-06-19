/* Matrix-matrix multiply using cublasGemmEx function.

Compile example:
nvcc -O3 cublas_bgemm.cu -o cublas_bgemm -lcublas

Run with:
./cublas_bgemm
*/


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "common.h"


/* Host implementation of a simple version of sgemm */
static void simple_bsgemm(int P, int M, int N, int K, float alpha, const float **A, const float **B,
                         float beta, float **C) {
    for(int p = 0; p < P; p++) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float prod = 0.0f;
                for (int k = 0; k < K; ++k) {
                    prod = prod + A[p][k * M + i] * B[p][j * K + k];
                }
                C[p][j * M + i] = alpha * prod + beta * C[p][j * M + i];
            }
        }
    }
}

void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error: %s:%d\n%s\n", file, line,
            cublasGetStatusString(status));
        exit(EXIT_FAILURE);
    }
}

#define cublasCheck(status) check_cublas(status, __FILE__, __LINE__)


int main(int argc, char **argv) {
    setup_main();

    const int M = 512;
    const int N = 512;
    const int K = 1024;
    const int P = 3; // batch count
    float *h_A[P], *h_B[P], *h_C[P];
    float *d_A[P], *d_B[P], *d_C[P];
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;

    printf("CUBLAS GEMM %dx%dx%d (MxNxK)...\n", M, N, K);

    /* Initialize CUBLAS */
    cublasCheck(cublasCreate(&handle));

    for(int i = 0; i < P; i++) {
        h_A[i] = make_random_float(M*K);
        h_B[i] = make_random_float(K*N);
        h_C[i] = (float*)(malloc(M*N * sizeof(float)));
    }

    /* Allocate device memory for the matrices */
    for(int i = 0; i < P; i++) {
        cudaCheck(cudaMalloc(&d_A[i], M*K * sizeof(float)));
        cudaCheck(cudaMalloc(&d_B[i], K*N * sizeof(float)));
        cudaCheck(cudaMalloc(&d_C[i], M*N * sizeof(float)));

        cudaCheck(cudaMemcpy(d_A[i], h_A[i], M*K * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_B[i], h_B[i], K*N * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(d_C[i], h_C[i], M*N * sizeof(float), cudaMemcpyHostToDevice));
    }

    int lda = M;
    int ldb = K;
    int ldc = M;

    /* Performs operation using cublas */
    auto cublas_bgemm = [&]() {
    cublasCheck(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                                (void**)d_A, CUDA_R_32F, lda,
                                (void**)d_B, CUDA_R_32F, ldb, &beta,
                                (void**)d_C, CUDA_R_32F, ldc, P,
                                CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT));
    };
    cublas_bgemm();

    /* Performs operation using plain C code */
    simple_bsgemm(P, M, N, K, alpha, (const float**)h_A, (const float**)h_B, beta, h_C);

    /* Check result against reference */
    for(int i = 0; i < P; i++) {
        validate_result(d_C[i], h_C[i], "C", M*N, 1.0e-3f);
        printf("\n");
    }


    // /* Benchmark */
    int repeat_times = 1000;
    float elapsed_time = benchmark_kernel(repeat_times, cublas_bgemm);
    printf("time %.4f ms\n", elapsed_time);


    /* Memory clean up */
    for(int i = 0; i < P; i++) {
        free(h_A[i]);
        free(h_B[i]);
        free(h_C[i]);

        cudaCheck(cudaFree(d_A[i]));
        cudaCheck(cudaFree(d_B[i]));
        cudaCheck(cudaFree(d_C[i]));
    }
}
