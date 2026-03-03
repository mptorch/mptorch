/*
Custom precision matrix-matrix multiply lambda vs non-lambda version

Compile example:
  nvcc -O3 custom_matmul.cu -o custom_matmul -lcublas --extended-lambda
  (parallel CPU: add -Xcompiler -fopenmp)

Run with:
  ./custom_matmul
*/

#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"

// ------------------------------------------------------------------------------------
// Quantization functions and wrappers

__host__ __device__ __forceinline__ uint32_t round_bitwise_nearest(uint32_t target, int man_bits)
{
    const int shift_lo = 8 + man_bits;
    const int shift_hi = 23 - man_bits;
    uint32_t down = target << shift_lo >> shift_lo;
    uint32_t machine_eps = 1u << (22 - man_bits);
    int offset = (down == machine_eps);
    uint32_t add_r = target + machine_eps;
    return add_r & ~((1u << (shift_hi + offset)) - 1u);
}

__host__ __device__ __forceinline__ uint32_t clip_exponent_with_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                                           uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)(quantized_num << 1u >> 24u);
    const int exp_half_range = (1 << (exp_bits - 1)) - 2;
    int min_exponent_store = -exp_half_range - man_bits + 127;

    uint32_t old_sign = old_num & 0x80000000u;
    if (quantized_exponent_store < min_exponent_store)
    {
        int offset = (quantized_exponent_store == (min_exponent_store - 1));
        quantized_num = (quantized_num + (offset * (1u << 23))) | old_sign;
        quantized_num = offset * quantized_num;
    }
    return quantized_num;
}

__host__ __device__ __forceinline__ uint32_t clip_exponent_without_subnormals(int exp_bits, int man_bits, uint32_t old_num,
                                                                              uint32_t quantized_num, bool saturate = false)
{
    if (quantized_num == 0)
        return quantized_num;

    int quantized_exponent_store = (int)(quantized_num << 1u >> 24u);
    const int exp_half_range = (1 << (exp_bits - 1)) - 2;
    int max_exponent_store = exp_half_range + 1 + 127;
    int min_exponent_store = -exp_half_range + 127;

    uint32_t old_sign = old_num & 0x80000000u;
    if (quantized_exponent_store > max_exponent_store)
    {
        if (saturate)
        {
            const int man_shift = 23 - man_bits;
            uint32_t max_man = ((uint32_t)-1u >> 9u) >> man_shift << man_shift;
            quantized_num = old_sign | ((uint32_t)max_exponent_store << 23u) | max_man;
        }
        else
        {
            quantized_num = old_sign | 0x7F7FFFFFu;
        }
    }
    else if (quantized_exponent_store < min_exponent_store)
    {
        uint32_t offset = (quantized_exponent_store == (min_exponent_store - 1)) & (unsigned)((old_num << 9u >> 9u) > (1u << 22u));
        quantized_num = old_sign | (offset * (uint32_t)(min_exponent_store << 23));
    }
    return quantized_num;
}

__host__ __device__ __forceinline__ float cast_fp_nearest(float origin_float, int man_bits, int exp_bits,
                                                          bool subnormal_support = true,
                                                          bool saturate = false)
{
#if defined(__CUDA_ARCH__)
    uint32_t target = __float_as_uint(origin_float);
#else
    uint32_t target = FLOAT_TO_BITS(&origin_float);
#endif

    if (man_bits >= 23)
        return origin_float;

    int target_exp = (int)((target & 0x7FFFFFFFu) >> 23u) - 127;
    const int min_exp = -((1 << (exp_bits - 1)) - 2);
    bool subnormal = (target_exp < min_exp);

    if (subnormal && subnormal_support)
    {
        int exp_diff = man_bits - (min_exp - target_exp);
        int not_uflow = exp_diff > -1 || ((exp_diff == -1) & ((target << 9u) > 0u));
        uint32_t quantize_bits = not_uflow * round_bitwise_nearest(target, exp_diff);
        quantize_bits = clip_exponent_with_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
#if defined(__CUDA_ARCH__)
        return __uint_as_float(quantize_bits);
#else
        return BITS_TO_FLOAT(&quantize_bits);
#endif
    }
    if (target_exp == 128)
        return origin_float;

    uint32_t quantize_bits = round_bitwise_nearest(target, man_bits);
    quantize_bits = clip_exponent_without_subnormals(exp_bits, man_bits, target, quantize_bits, saturate);
#if defined(__CUDA_ARCH__)
    return __uint_as_float(quantize_bits);
#else
    return BITS_TO_FLOAT(&quantize_bits);
#endif
}

// Templated quantizer for compile-time precision (enables constant propagation in kernels).
template <int man_bits, int exp_bits, bool subnormals, bool saturate>
struct QuantizeFn {
    __device__ __forceinline__ float operator()(float x) const
    {
        return cast_fp_nearest(x, man_bits, exp_bits, subnormals, saturate);
    }
};

// ------------------------------------------------------------------------------------
// CPU Kernels
template <class Qadd, class Qmul>
void mm_cpu_kernel1(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
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
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    // cache-aware version
    for (int i = 0; i < M * N; ++i)
        c[i] = 0.f;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
            {
                c[i * N + j] = quant_add(c[i * N + j] + quant_mul(a[i * K + k] * b[k * N + j]));
            }
}

template <class Qadd, class Qmul>
void mm_cpu_kernel3(float *a, float *b, float *c, int M, int K, int N, Qadd quant_add, Qmul quant_mul)
{
    constexpr int TI = 16;
    constexpr int TJ = 16;
    constexpr int TK = 16;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i0 = 0; i0 < M; i0 += TI)
    {
        int ti = (i0 + TI <= M) ? TI : (M - i0);
        for (int j0 = 0; j0 < N; j0 += TJ)
        {
            int tj = (j0 + TJ <= N) ? TJ : (N - j0);

            float c_tile[TI * TJ];
            for (int i = 0; i < ti; ++i)
                for (int j = 0; j < tj; ++j)
                    c_tile[i * TJ + j] = 0.f;

            for (int k0 = 0; k0 < K; k0 += TK)
            {
                int tk = (k0 + TK <= K) ? TK : (K - k0);

                float a_tile[TI * TK];
                float b_tile[TK * TJ];
                for (int i = 0; i < ti; ++i)
                    for (int k = 0; k < tk; ++k)
                        a_tile[i * TK + k] = a[(i0 + i) * K + (k0 + k)];
                for (int k = 0; k < tk; ++k)
                    for (int j = 0; j < tj; ++j)
                        b_tile[k * TJ + j] = b[(k0 + k) * N + (j0 + j)];

                for (int i = 0; i < ti; ++i)
                    for (int k = 0; k < tk; ++k)
                    {
                        float a_ik = a_tile[i * TK + k];
                        for (int j = 0; j < tj; ++j)
                            c_tile[i * TJ + j] = quant_add(c_tile[i * TJ + j] + quant_mul(a_ik * b_tile[k * TJ + j]));
                    }
            }

            for (int i = 0; i < ti; ++i)
                for (int j = 0; j < tj; ++j)
                    c[(i0 + i) * N + (j0 + j)] = c_tile[i * TJ + j];
        }
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

    float tmp = 0.0f;

    // sweep tile across matrix
    for (int i = 0; i < K + blockDim.x - K % blockDim.x; i += blockDim.x)
    {
        // load in elements for this tile (__ldg for read-only cache)
        s_a[ty * blockDim.x + tx] =
            (row < M && i + tx < K) ? __ldg(&a[row * K + i + tx]) : 0.0f;
        s_b[ty * blockDim.x + tx] =
            (col < N && i + ty < K) ? __ldg(&b[i * N + ty * N + col]) : 0.0f;

        // wait for both tiles to be loaded in before doing computation
        __syncthreads();

        // do matrix multiplication on the small matrices
#pragma unroll
        for (int j = 0; j < blockDim.x; j++)
        {
            tmp = cast_fp_nearest(tmp + cast_fp_nearest(s_a[ty * blockDim.x + j] *
                                                                        s_b[j * blockDim.x + tx],
                                                                    man_mul, exp_mul, subnormals,
                                                                    saturate),
                                        man_add, exp_add, subnormals, saturate);
        }

        // wait for all threads to finish using current tiles
        // before loading in new ones
        __syncthreads();
    }

    // write back results
    if (row < M && col < N)
        c[row * N + col] = tmp;
}

template <ssize_t SHMEM_SIZE, class Qadd, class Qmul>
__global__ void mm_kernel2(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int K, int N,
                           Qadd quant_add, Qmul quant_mul)
{

    // double-buffered shared memory
    __shared__ float s_a[2][SHMEM_SIZE];
    __shared__ float s_b[2][SHMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0f;

    const int numTiles = (K + blockDim.x - 1) / blockDim.x;

    // load first tile into buffer 0
    s_a[0][ty * blockDim.x + tx] =
        (row < M && tx < K) ? __ldg(&a[row * K + tx]) : 0.0f;
    s_b[0][ty * blockDim.x + tx] =
        (col < N && ty < K) ? __ldg(&b[ty * N + col]) : 0.0f;

    for (int i = 0; i < numTiles; ++i)
    {
        __syncthreads();

        float *cur_a = s_a[i % 2];
        float *cur_b = s_b[i % 2];

        // do matrix multiplication on the current tile
#pragma unroll
        for (int j = 0; j < blockDim.x; j++)
        {
            tmp = quant_add(tmp + quant_mul(cur_a[ty * blockDim.x + j] * cur_b[j * blockDim.x + tx]));
        }


        // load next tile into the other buffer
        if (i + 1 < numTiles)
        {
            int nextK = (i + 1) * blockDim.x;
            s_a[(i + 1) % 2][ty * blockDim.x + tx] =
                (row < M && nextK + tx < K) ? __ldg(&a[row * K + nextK + tx]) : 0.0f;
            s_b[(i + 1) % 2][ty * blockDim.x + tx] =
                (col < N && nextK + ty < K) ? __ldg(&b[nextK * N + ty * N + col]) : 0.0f;
        }
        __syncthreads();
    }

    // write back results
    if (row < M && col < N)
        c[row * N + col] = tmp;
}

template <size_t BLOCKSIZE, class Qadd, class Qmul>
__global__ void mm_kernel3(float *__restrict__ a, float *__restrict__ b,
                           float *__restrict__ c, int M, int K, int N,
                           Qadd quant_add, Qmul quant_mul)
{

    // the output block that we want to compute in this threadblock
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    // double-buffered shared memory
    __shared__ float As[2][BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[2][BLOCKSIZE * BLOCKSIZE];

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

    const int numTiles = (K + BLOCKSIZE - 1) / BLOCKSIZE;

    // load first tile into buffer 0
    As[0][threadRow * BLOCKSIZE + threadCol] = (rId < M && threadCol < K) ? __ldg(&a[threadRow * K + threadCol]) : 0.0f;
    Bs[0][threadRow * BLOCKSIZE + threadCol] = (threadRow < K && cId < N) ? __ldg(&b[threadRow * N + threadCol]) : 0.0f;

    for (int t = 0; t < numTiles; ++t)
    {
        __syncthreads();

        float *curAs = As[t % 2];
        float *curBs = Bs[t % 2];

        // execute the dotproduct on the currently cached block
#pragma unroll
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            tmp = quant_add(tmp + quant_mul(curAs[threadRow * BLOCKSIZE + dotIdx] *
                                            curBs[dotIdx * BLOCKSIZE + threadCol]));
        }

        // load next tile into the other buffer (from a+BLOCKSIZE, b+BLOCKSIZE*N)
        if (t + 1 < numTiles)
        {
            int nextK = (t + 1) * BLOCKSIZE;
            As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] = (rId < M && nextK + threadCol < K) ? __ldg(&a[BLOCKSIZE + threadRow * K + threadCol]) : 0.0f;
            Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] = (nextK + threadRow < K && cId < N) ? __ldg(&b[BLOCKSIZE * N + threadRow * N + threadCol]) : 0.0f;
        }
        a += BLOCKSIZE;
        b += BLOCKSIZE * N;
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

    // Single-buffer SMEM (double-buffering + BK>8 cost too much occupancy on most GPUs)
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // move block tile at the beginning of A's row and B's column
    a += cRow * BM * K;
    b += cCol * BN;
    c += cRow * BM * N + cCol * BN;

    const uint innerColA = threadIdx.x % BK;
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;
    const uint innerRowB = threadIdx.x / BN;

    float threadResults[TM] = {0.0};

    int rId = cRow * BM + threadRow;
    int cId = cCol * BN + threadCol;

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        As[innerRowA * BK + innerColA] = (rId < M && bkIdx + innerColA < K) ? __ldg(&a[innerRowA * K + innerColA]) : 0.0f;
        Bs[innerRowB * BN + innerColB] = (bkIdx + innerRowB < K && cId < N) ? __ldg(&b[innerRowB * N + innerColB]) : 0.0f;
        __syncthreads();

        a += BK;
        b += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx)
        {
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

void mm_cpu3(float *a, float *b, float *c, int M, int K, int N,
             int man_add, int exp_add, int man_mul, int exp_mul,
             bool subnormals, bool saturate)
{
    mm_cpu_kernel3(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate](float x)
                   { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
}

void mm_cuda1(float *a, float *b, float *c, int M, int K, int N,
              int man_add, int exp_add, int man_mul, int exp_mul,
              bool subnormals, bool saturate)
{

    constexpr size_t THREADS_X{16U};
    constexpr size_t THREADS_Y{16U};
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
    constexpr size_t THREADS_X{16U};
    constexpr size_t THREADS_Y{16U};
    constexpr size_t SHMEM_SIZE{THREADS_X * THREADS_Y};
    dim3 const thread_dim{THREADS_X, THREADS_Y, 1U};
    dim3 const block_dim{
        (static_cast<uint32_t>(N) + thread_dim.x - 1U) / thread_dim.x,
        (static_cast<uint32_t>(M) + thread_dim.y - 1U) / thread_dim.y, 1U};
    mm_kernel2<SHMEM_SIZE>
        <<<block_dim, thread_dim>>>(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
}

void mm_cuda3(float *a, float *b, float *c, int M, int K, int N,
              int man_add, int exp_add, int man_mul, int exp_mul,
              bool subnormals, bool saturate)
{

    dim3 const thread_dim{256U, 1U, 1U};
    dim3 const block_dim{(uint)ceil_div(M, 16), (uint)ceil_div(N, 16), 1U};
    mm_kernel3<16>
        <<<block_dim, thread_dim>>>(a, b, c, M, K, N, [man_add, exp_add, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate] __device__(float x)
                                    { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
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
                                                          { return cast_fp_nearest(x, man_add, exp_add, subnormals, saturate); }, [man_mul, exp_mul, subnormals, saturate] __device__(float x)
                                                          { return cast_fp_nearest(x, man_mul, exp_mul, subnormals, saturate); });
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
    int repeat_times = 10;
    float elapsed_time1 = benchmark_cpu_kernel(repeat_times, mm_cpu1, a, b, c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time2 = benchmark_cpu_kernel(repeat_times, mm_cpu2, a, b, c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time3 = benchmark_cpu_kernel(repeat_times, mm_cpu3, a, b, c, M, K, N, 10, 5, 10, 5, true, true);
    printf("time mm_cpu1 %.4f ms | time mm_cpu2 %.4f ms | time mm_cpu3 %.4f ms\n", elapsed_time1, elapsed_time2, elapsed_time3);

    printf("CUDA benchmarking...\n");
    repeat_times = 1000;
    float elapsed_time4 = benchmark_gpu_kernel(repeat_times, mm_cuda1, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time5 = benchmark_gpu_kernel(repeat_times, mm_cuda2, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time6 = benchmark_gpu_kernel(repeat_times, mm_cuda3, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);
    float elapsed_time7 = benchmark_gpu_kernel(repeat_times, mm_cuda4, d_a, d_b, d_c, M, K, N, 10, 5, 10, 5, true, true);

    printf("time mm_cuda1 %.4f ms | time mm_cuda2 %.4f ms | time mm_cuda3 %.4f ms | time mm_cuda4 %.4f ms\n", elapsed_time4, elapsed_time5, elapsed_time6, elapsed_time7);

    // free memory
    free(a);
    free(b);
    free(c);

    cudaCheck(cudaFree(d_a));
    cudaCheck(cudaFree(d_b));
    cudaCheck(cudaFree(d_c));
    return 0;
}