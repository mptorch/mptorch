/*
Custom precision matrix-matrix multiply lambda vs non-lambda version

Compile example:
nvcc -O3 custom_matmul.cu -o custom_matmul -lcublas --extended-lambda

Run with:
./custom_matmul
*/

#include <cuda_runtime.h>
#include <cmath>
#include <chrono>

#include "common.h"

// ------------------------------------------------------------------------------------
// Quantization functions and wrappers

__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest(uint32_t target, int man_bits)
{
    uint32_t down = target << (8 + man_bits) >> (8 + man_bits);
    uint32_t machine_eps = 1 << (22 - man_bits);
    // tie breaking rule offset
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    // apply the mask
    // this is the analogue of how you would do round
    // to nearest integer using the floor function:
    // round(x) = floor(x + 0.5)
    return add_r & ~((1 << (23 - man_bits + offset)) - 1);
}

__host__ __device__ uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                           uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) - man_bits + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // underflow or round to smallest non zero subnormal value
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num += offset * (1u << 23);
        quantized_num = quantized_num | old_sign;
        quantized_num = offset * quantized_num;
    }
    return quantized_num;
}

__host__ __device__ uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                              uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = quantized_num << 1 >> 24;
    int max_exponent_store = (1 << (exp_bits - 1)) - 1 + 127;
    int min_exponent_store = -((1 << (exp_bits - 1)) - 2) + 127;

    uint32_t old_sign = old_num >> 31 << 31;
    // saturate or overflow
    if (quantized_exponent_store > max_exponent_store)
    {
        if (saturate)
        {
            uint32_t max_man =
                (uint32_t)-1 << 9 >> 9 >> (23 - man_bits) << (23 - man_bits);
            uint32_t max_num = ((uint32_t)max_exponent_store << 23) | max_man;
            quantized_num = old_sign | max_num;
        }
        else
        {
            quantized_num = ((((uint32_t)1 << 31) - 1) ^ (((uint32_t)1 << 23) - 1));
            quantized_num = quantized_num | old_sign;
        }
    } // underflow or round to smallest nonzero normal value
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) && ((old_num << 9 >> 9) > (1 << 22));
        quantized_num = offset * (min_exponent_store << 23);
        quantized_num |= old_sign;
    }
    return quantized_num;
}

__host__ __device__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
                                          bool subnormal_support = true,
                                          bool saturate = false)
{
    uint32_t target, quantize_bits;
    target = FLOAT_TO_BITS(&origin_float);
    float quantized;

    int target_exp = (target << 1 >> 1 >> 23) - 127;
    int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);
    bool noquantize = (man_bits >= 23);

    if (noquantize)
    {
        quantized = origin_float;
    }
    else
    {
        // handle subnormal inputs (if subnormal mode is active)
        if (subnormal && subnormal_support)
        {
            int exp_diff = man_bits - (min_exp - target_exp);
            int not_uflow = exp_diff > -1 || ((exp_diff == -1) && ((target << 9) > 0));
            quantize_bits = not_uflow * round_bitwise_nearest(target, exp_diff);
            quantize_bits =
                clip_exponent_with_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
        // handle NaN/inf inputs
        else if (target_exp == 128)
        {
            quantized = origin_float;
        }
        // normal value range or overflow
        else
        {
            quantize_bits = round_bitwise_nearest(target, man_bits);
            quantize_bits =
                clip_exponent_without_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
            quantized = BITS_TO_FLOAT(&quantize_bits);
        }
    }

    return quantized;
}
// ------------------------------------------------------------------------------------
// CPU Kernels
template <class Qadd, class Qmul>
void mm_cpu_kernel1(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    // naive version
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc = quant_add(acc + quant_mul(a[i * K + k] * b[k * N + j]));
            c[i * N + j] = acc;
        }
}

template <class Qadd, class Qmul>
void mm_cpu_kernel2(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    // cache-aware version
    for (int i = 0; i < M * N; ++i)
        c[i] = 0.f;

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
            {
                c[i * N + j] = quant_add(c[i * N + j] + quant_mul(a[i * K + k] * b[k * N + j]));
            }
}

// ------------------------------------------------------------------------------------
// GPU Kernels

template <size_t SHMEM_SIZE>
__global__ void mm_kernel1(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int K, int N,
                           int man_add, int exp_add, int man_mul,
                           int exp_mul, bool subnormals,
                           bool saturate)
{

    // declare shared memory matrices for A and B matrices
    __shared__ float s_a[SHMEM_SIZE];
    __shared__ float s_b[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float inner_sum = 0.0f;
    float outer_sum = 0.0f;
    int blockFactor = 1;
    int currFactor = 0;

    // sweep tile across matrix
    for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
    {
        // load in elements for this tile
        s_a[ty * blockDim.x + tx] =
            (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
        s_b[ty * blockDim.x + tx] =
            (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

        // wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // do matrix multiplication on the small matrices
        for (int j = 0; j < blockDim.x; j++)
        {
            inner_sum = cast_fp_nearest(inner_sum + cast_fp_nearest(s_a[ty * blockDim.x + j] *
                                                                        s_b[j * blockDim.x + tx],
                                                                    man_mul, exp_mul, subnormals,
                                                                    saturate),
                                        man_add, exp_add, subnormals, saturate);
        }
        currFactor++;
        currFactor %= blockFactor;
        if (currFactor == 0)
        {
            outer_sum = cast_fp_nearest(outer_sum + inner_sum, man_add, exp_add, subnormals, saturate);
            inner_sum = 0.0f;
        }

        // wait for all threads to finish using current tiles
        // before loading in new ones
        __syncthreads();
    }

    // write back results
    if (row < M && col < N)
        c[row * N + col] = outer_sum;
}

template <size_t BLOCK_FACTOR, size_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_kernel2(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int K, int N,
                           Qadd quant_add, Qmul quant_mul)
{

    // declare shared memory matrices for A and B matrices
    __shared__ float s_a[SHMEM_SIZE];
    __shared__ float s_b[SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float inner_sum = 0.0f;
    float outer_sum = 0.0f;
    int currFactor = 0;

    // sweep tile across matrix
    for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
    {
        // load in elements for this tile
        s_a[ty * blockDim.x + tx] =
            (row < M && i + tx < K) ? a[row * K + i + tx] : 0.0f;
        s_b[ty * blockDim.x + tx] =
            (col < N && i + ty < K) ? b[i * N + ty * N + col] : 0.0f;

        // wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // do matrix multiplication on the small matrices
        for (int j = 0; j < blockDim.x; j++)
        {
            inner_sum = quant_add(inner_sum + quant_mul(s_a[ty * blockDim.x + j] * s_b[j * blockDim.x + tx]));
        }
        currFactor++;
        currFactor %= BLOCK_FACTOR;
        if (currFactor == 0)
        {
            outer_sum = quant_add(outer_sum + inner_sum);
            inner_sum = 0.0f;
        }

        // wait for all threads to finish using current tiles
        // before loading in new ones
        __syncthreads();
    }

    // write back results
    if (row < M && col < N)
        c[row * N + col] = outer_sum;
}

template <size_t BLOCKSIZE, class Qadd, class Qmul>
__global__ void mm_kernel3(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int K, int N,
                           Qadd quant_add, Qmul quant_mul)
{

    // the output block that we want to compute in this threadblock
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    a += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    b += cCol * BLOCKSIZE;                        // row=0, col=cCol
    c += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;

    int cId = cCol * BLOCKSIZE + threadCol;
    int rId = cRow * BLOCKSIZE + threadRow;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = (rId < M && bkIdx + threadCol < K) ? a[threadRow * K + threadCol] : 0.0f;
        Bs[threadRow * BLOCKSIZE + threadCol] = (bkIdx + threadRow < K && cId < N) ? b[threadRow * N + threadCol] : 0.0f;

        // block threads in this block until cache is fully populated
        __syncthreads();
        a += BLOCKSIZE;
        b += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            tmp = quant_add(tmp + quant_mul(As[threadRow * BLOCKSIZE + dotIdx] *
                                            Bs[dotIdx * BLOCKSIZE + threadCol]));
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    if (rId < M && cId < N)
    {
        c[threadRow * N + threadCol] = tmp;
    }
}

template <size_t BM, size_t BN, size_t BK, size_t TM, class Qadd, class Qmul>
__global__ void mm_kernel4(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int K, int N,
                           Qadd quant_add, Qmul quant_mul)
{
    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit race.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move block tile at the beginning of A's row and B's column
    a += cRow * BM * K;
    b += cCol * BN;
    c += cRow * BM * N + cCol * BN;

    // TODO: adjust this to each thread to load multiple entries and
    // better exploit the cache sizes
    // assert(BM * BK == blockDim.x);
    // assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    int rId = cRow * BM + threadRow;
    int cId = cCol * BN + threadCol;

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = (rId < M && bkIdx + innerColA < K) ? a[innerRowA * K + innerColA] : 0.0f;
        Bs[innerRowB * BN + innerColB] = (bkIdx + innerRowB < K && cId < N) ? b[innerRowB * N + innerColB] : 0.0f;
        __syncthreads();

        // advance blocktiles
        a += BK;
        b += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
            // we make the dot product loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can later cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx)
            {
                threadResults[resIdx] = quant_add(threadResults[resIdx] + quant_mul(As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB));
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx)
    {
        if (rId < M && cId + resIdx < N)
            c[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
    }
}

// ------------------------------------------------------------------------------------
// Kernel Launchers

void mm_cpu1(float *a, float *b, float *c, int M, int K, int N,
             int man_add, int exp_add, int man_mul, int exp_mul,
             bool subnormals, bool saturate)
{
    mm_cpu_kernel1(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
}

void mm_cpu2(float *a, float *b, float *c, int M, int K, int N,
             int man_add, int exp_add, int man_mul, int exp_mul,
             bool subnormals, bool saturate)
{
    mm_cpu_kernel2(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
}

void mm_cuda1(float *a, float *b, float *c, int M, int K, int N,
              int man_add, int exp_add, int man_mul, int exp_mul,
              bool subnormals, bool saturate)
{

    constexpr size_t THREADS_X{8U};
    constexpr size_t THREADS_Y{8U};
    constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
    dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
    dim3 const block_dim{
        (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
        (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
    mm_kernel1<SHMEM_SIZE>
        <<<block_dim, thread_dim>>>(a, b, c, M, K, N, man_add, exp_add, man_mul,
                                    exp_mul, subnormals, saturate);
}

void mm_cuda2(float *a, float *b, float *c, int M, int K, int N,
              int man_add, int exp_add, int man_mul, int exp_mul,
              bool subnormals, bool saturate)
{
    constexpr size_t THREADS_X{8U};
    constexpr size_t THREADS_Y{8U};
    constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
    dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
    dim3 const block_dim{
        (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
        (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
    mm_kernel2<1u, SHMEM_SIZE>
        <<<block_dim, thread_dim>>>(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
}

void mm_cuda3(float *a, float *b, float *c, int M, int K, int N,
              int man_add, int exp_add, int man_mul, int exp_mul,
              bool subnormals, bool saturate)
{

    dim3 const thread_dim{1024U, 1U, 1U};
    dim3 const block_dim{(uint)ceil_div(M, 32), (uint)ceil_div(N, 32), 1U};
    mm_kernel3<32>
        <<<block_dim, thread_dim>>>(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); });
}

void mm_cuda4(float *a, float *b, float *c, int M, int K, int N,
              int man_add, int exp_add, int man_mul, int exp_mul,
              bool subnormals, bool saturate)
{
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    dim3 const block_dim((uint)ceil_div(N, BN), (uint)ceil_div(M, BM));
    dim3 const thread_dim((BM * BN) / TM);
    mm_kernel4<BM, BN, BK, TM><<<block_dim, thread_dim>>>(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                                          { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                                          { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); });
}

int main(int argc, const char **argv)
{
    setup_main();

    int M = 1000;
    int K = 1000;
    int N = 1000;
    float *a = make_random_float(M * K);
    float *b = make_random_float(K * N);
    float *c = make_zeros_float(M * N);

    // move data to the GPU
    float *d_a, *d_b, *d_c;
    cudaCheck(cudaMalloc(&d_a, M * K * sizeof(float)));
    cudaCheck(cudaMalloc(&d_b, K * N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_c, M * N * sizeof(float)));

    cudaCheck(cudaMemcpy(d_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice));

    mm_cpu1(a, b, c, M, K, N, 23, 8, 23, 8, true, true);
    mm_cuda1(d_a, d_b, d_c, M, K, N, 23, 8, 23, 8, true, true);
    printf("Checking if kernel results match...\n");
    float tol = 1e-4f;
    validate_result(d_c, c, "c", M * N, tol);
    printf("All results match. Starting benchmarks...\n\n");

    printf("CPU benchmarking...\n");
    int repeat_times = 1;
    float elapsed_time1 = benchmark_cpu_kernel(repeat_times, mm_cpu1, a, b, c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time2 = benchmark_cpu_kernel(repeat_times, mm_cpu2, a, b, c, M, K, N, 10, 5, 10, 5, true, true);
    printf("time mm_cpu1 %.4f ms | time mm_cpu2 %.4f ms\n", elapsed_time1, elapsed_time2);

    printf("CUDA benchmarking...\n");
    repeat_times = 1000;
    float elapsed_time3 = benchmark_gpu_kernel(repeat_times, mm_cuda1, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time4 = benchmark_gpu_kernel(repeat_times, mm_cuda2, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time5 = benchmark_gpu_kernel(repeat_times, mm_cuda3, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time6 = benchmark_gpu_kernel(repeat_times, mm_cuda4, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);

    printf("time mm_cuda1 %.4f ms | time mm_cuda2 %.4f ms | time mm_cuda3 %.4f ms | time mm_cuda3 %.4f ms\n", elapsed_time3, elapsed_time4, elapsed_time5, elapsed_time6);

    // free memory
    free(a);
    free(b);
    free(c);

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    return 0;
}