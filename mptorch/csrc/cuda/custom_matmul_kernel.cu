#include "../common/gemm_policy.h"
#include "../quant_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

using namespace at;

namespace
{

    // Double-buffered tiled GEMM, ported from the mm_kernel3 prototype in
    // dev/cuda/custom_matmul.cu (benchmarked slightly faster than the
    // register-blocked mm_kernel4 variant it originally replaced). Each
    // thread block computes a BLOCKSIZE x BLOCKSIZE output tile; each thread
    // computes exactly one output element, accumulated via Accumulator (Mac
    // -generic -- see gemm_policy.h).
    constexpr int BLOCKSIZE = 16;

    template <typename scalar_t>
    __device__ __forceinline__ float load_a(const scalar_t *A, int64_t M, int64_t K, bool trans_a,
                                             int64_t row, int64_t col)
    {
        // logical A is M x K; trans_a means A's storage is actually K x M.
        return trans_a ? static_cast<float>(A[col * M + row]) : static_cast<float>(A[row * K + col]);
    }

    template <typename scalar_t>
    __device__ __forceinline__ float load_b(const scalar_t *B, int64_t K, int64_t N, bool trans_b,
                                             int64_t row, int64_t col)
    {
        // logical B is K x N; trans_b means B's storage is actually N x K.
        return trans_b ? static_cast<float>(B[col * K + row]) : static_cast<float>(B[row * N + col]);
    }

    template <typename scalar_t, class Accumulator>
    __global__ __launch_bounds__(BLOCKSIZE * BLOCKSIZE)
    void custom_matmul_kernel(
        const scalar_t *__restrict__ A, const scalar_t *__restrict__ B, scalar_t *__restrict__ C,
        int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
        Accumulator acc_proto)
    {
        __shared__ float As[2][BLOCKSIZE * BLOCKSIZE];
        __shared__ float Bs[2][BLOCKSIZE * BLOCKSIZE];

        const int64_t cRow = blockIdx.y;
        const int64_t cCol = blockIdx.x;

        const int threadCol = threadIdx.x % BLOCKSIZE;
        const int threadRow = threadIdx.x / BLOCKSIZE;

        const int64_t rowBase = cRow * BLOCKSIZE;
        const int64_t colBase = cCol * BLOCKSIZE;
        const int64_t rId = rowBase + threadRow;
        const int64_t cId = colBase + threadCol;

        const int64_t numTiles = (K + BLOCKSIZE - 1) / BLOCKSIZE;

        Accumulator acc = acc_proto;

        // load first tile into buffer 0
        As[0][threadRow * BLOCKSIZE + threadCol] =
            (rId < M && threadCol < K) ? load_a<scalar_t>(A, M, K, trans_a, rId, threadCol) : 0.0f;
        Bs[0][threadRow * BLOCKSIZE + threadCol] =
            (threadRow < K && cId < N) ? load_b<scalar_t>(B, K, N, trans_b, threadRow, cId) : 0.0f;

        for (int64_t t = 0; t < numTiles; ++t)
        {
            __syncthreads();

            float *curAs = As[t % 2];
            float *curBs = Bs[t % 2];

#pragma unroll
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            {
                acc.accumulate(curAs[threadRow * BLOCKSIZE + dotIdx], curBs[dotIdx * BLOCKSIZE + threadCol]);
            }

            // load next tile into the other buffer
            if (t + 1 < numTiles)
            {
                int64_t nextK = (t + 1) * BLOCKSIZE;
                As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                    (rId < M && nextK + threadCol < K) ? load_a<scalar_t>(A, M, K, trans_a, rId, nextK + threadCol) : 0.0f;
                Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                    (nextK + threadRow < K && cId < N) ? load_b<scalar_t>(B, K, N, trans_b, nextK + threadRow, cId) : 0.0f;
            }
            __syncthreads();
        }

        if (rId < M && cId < N)
            C[rId * N + cId] = static_cast<scalar_t>(acc.finalize());
    }

    template <typename scalar_t, class Accumulator>
    void launch_custom_matmul(const scalar_t *a, const scalar_t *b, scalar_t *c,
                              int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
                              Accumulator acc_proto)
    {
        dim3 block_dim(static_cast<unsigned>((N + BLOCKSIZE - 1) / BLOCKSIZE),
                       static_cast<unsigned>((M + BLOCKSIZE - 1) / BLOCKSIZE));
        dim3 thread_dim(BLOCKSIZE * BLOCKSIZE);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        custom_matmul_kernel<scalar_t, Accumulator><<<block_dim, thread_dim, 0, stream>>>(
            a, b, c, M, K, N, trans_a, trans_b, acc_proto);
    }

    void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                             int64_t round_mode, int64_t accumulate_algorithm)
    {
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op_name, " expects 2D tensors, got ",
                   a.dim(), "D and ", b.dim(), "D");
        TORCH_CHECK(static_cast<RoundMode>(round_mode) == RoundMode::RNE, op_name,
                   ": only RoundMode.RNE is supported in this build");
        TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) == AccumulateAlgorithm::NAIVE,
                   op_name, ": only AccumulateAlgorithm.NAIVE is supported in this build");
    }

    void matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b,
                             const char *op_name, int64_t &M, int64_t &K, int64_t &N)
    {
        M = trans_a ? a.size(1) : a.size(0);
        K = trans_a ? a.size(0) : a.size(1);
        int64_t K_b = trans_b ? b.size(1) : b.size(0);
        N = trans_b ? b.size(0) : b.size(1);
        TORCH_CHECK(K == K_b, op_name, ": inner dimensions must match (got ", K, " vs ", K_b, ")");
    }

} // namespace

Tensor binaryK_matmul_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK", M, K, N);

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return c;

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);

    int mul_man_bits = static_cast<int>(mul_P - 1);
    int mul_exp_bits = static_cast<int>(mul_is_signed ? mul_K - mul_P : mul_K - mul_P + 1);
    int acc_man_bits = static_cast<int>(acc_P - 1);
    int acc_exp_bits = static_cast<int>(acc_is_signed ? acc_K - acc_P : acc_K - acc_P + 1);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_matmul_cuda", [&]
                                    {
        BinaryKMultiplier mul{mul_man_bits, mul_exp_bits, static_cast<int>(mul_bias), mul_is_signed, sat, sub};
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (accumulate_quant)
        {
            BinaryKAdder add{acc_man_bits, acc_exp_bits, static_cast<int>(acc_bias), acc_is_signed, sat, sub};
            using Mac = SplitMac<BinaryKMultiplier, BinaryKAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        }
        else
        {
            using Mac = SplitMac<BinaryKMultiplier, IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        } });

    return c;
}

Tensor superfp_matmul_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp", M, K, N);

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return c;

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_matmul_cuda", [&]
                                    {
        SuperfpMultiplier mul{static_cast<int>(mul_man_bits), static_cast<int>(mul_exp_bits),
                              static_cast<int>(mul_normal_binades), static_cast<int>(mul_bias), mul_is_signed, sat};
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (accumulate_quant)
        {
            SuperfpAdder add{static_cast<int>(acc_man_bits), static_cast<int>(acc_exp_bits),
                             static_cast<int>(acc_normal_binades), static_cast<int>(acc_bias), acc_is_signed, sat};
            using Mac = SplitMac<SuperfpMultiplier, SuperfpAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        }
        else
        {
            using Mac = SplitMac<SuperfpMultiplier, IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        } });

    return c;
}

Tensor binaryK_matmul_fma_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK_fma", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma", M, K, N);

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return c;

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);

    int fma_man_bits = static_cast<int>(fma_P - 1);
    int fma_exp_bits = static_cast<int>(fma_is_signed ? fma_K - fma_P : fma_K - fma_P + 1);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "binaryK_matmul_fma_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (fma_quant)
        {
            BinaryKAdder add{fma_man_bits, fma_exp_bits, static_cast<int>(fma_bias), fma_is_signed, sat, sub};
            using Mac = FusedMac<BinaryKAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        }
        else
        {
            using Mac = FusedMac<IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        } });

    return c;
}

Tensor superfp_matmul_fma_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades,
    int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp_fma", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_fma", M, K, N);

    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return c;

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, a.scalar_type(), "superfp_matmul_fma_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (fma_quant)
        {
            SuperfpAdder add{static_cast<int>(fma_man_bits), static_cast<int>(fma_exp_bits),
                             static_cast<int>(fma_normal_binades), static_cast<int>(fma_bias), fma_is_signed, sat};
            using Mac = FusedMac<SuperfpAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        }
        else
        {
            using Mac = FusedMac<IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto);
        } });

    return c;
}
