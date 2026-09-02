#include "../common/gemm_policy.h"
#include "../common/dispatch.h"
#include "../quant_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <initializer_list>
#include <vector>
#include <mutex>

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

    // MIXED selects whether this instantiation carries the spatially-varying
    // FormatPalette prologue below. It has to be a template parameter rather
    // than the runtime `pal.n > 0` test it used to be: a *possible* write to
    // acc.mac forces the Mac policy's format constants to live in registers
    // for the whole K-loop, instead of being re-read from the constant bank
    // where acc_proto already sits. That costs the split-mac kernels ~29 extra
    // registers (109 vs 80 at BLOCKSIZE^2 threads) and roughly halves their
    // throughput -- 34.7 -> 68.3 ms at 1024^3 on sm_89 -- on every
    // single-format launch, which is nearly all of them. Instantiating the two
    // shapes separately doubles this kernel's instantiation count; see
    // dev/gemm_perf_audit.md (finding G4) for why that trade is worth taking
    // and dev/benchmarks/gemm_kernel_tuning.cu for the A/B.
    template <typename scalar_t, bool MIXED, class Accumulator>
    __global__ __launch_bounds__(BLOCKSIZE * BLOCKSIZE)
    void custom_matmul_kernel(
        const scalar_t *__restrict__ A, const scalar_t *__restrict__ B, scalar_t *__restrict__ C,
        int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
        Accumulator acc_proto, bool use_rng, at::PhiloxCudaState rng_args,
        FormatPalette<typename Accumulator::mac_type> pal,
        const int32_t *__restrict__ prec_idx, int64_t idx_row_stride, int64_t idx_col_stride)
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

        // RoundMode::SR: seed this thread's output element's Philox stream
        // once, before any accumulate() call -- keyed by its own global
        // linear index, so the result is independent of BLOCKSIZE/grid
        // geometry (see gemm_policy.h's NaiveAccumulator::seed_rng and
        // dev/gemm_core_roadmap.md for why this granularity, not a shared
        // per-block RNG state).
        if (use_rng && rId < M && cId < N)
        {
            auto seeds = at::cuda::philox::unpack(rng_args);
            acc.seed_rng(std::get<0>(seeds), static_cast<uint64_t>(rId) * N + cId, std::get<1>(seeds));
        }

        // Spatially-varying mixed format: bind this output element's Mac
        // policy from the palette before the K-loop (see gemm_policy.h's
        // FormatPalette). Compiled out entirely on the single-format path.
        if constexpr (MIXED)
        {
            if (pal.n > 0 && rId < M && cId < N)
                acc.mac = pal.slot(prec_idx[rId * idx_row_stride + cId * idx_col_stride]);
        }

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

            // Unroll 4, not the full 16: every accumulate() inlines the whole
            // runtime RoundMode switch -- each mode's cast body, twice over for
            // a split mac -- so unrolling all BLOCKSIZE steps expands this loop
            // past any instruction cache and the kernel spends its time
            // fetching code it never executes. 4 is at or within 3% of the
            // measured optimum for all four mac policies and both the
            // deterministic and SR paths (dev/benchmarks/gemm_kernel_tuning.cu;
            // binaryK split at 1024^3: 16 -> 34.7 ms, 8 -> 29.8, 4 -> 26.6,
            // 2 -> 25.7, 1 -> 28.6), and it is what keeps the runtime switch
            // competitive with templating the kernel on RoundMode, so the
            // instantiation count stays bounded. Worth nothing before finding
            // G1 landed -- until the accumulator was register-resident this
            // loop was bound by local-memory traffic, not instruction fetch.
            // See dev/gemm_perf_audit.md (finding G2).
#pragma unroll 4
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

    template <typename scalar_t, bool MIXED = false, class Accumulator>
    void launch_custom_matmul(const scalar_t *a, const scalar_t *b, scalar_t *c,
                              int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
                              Accumulator acc_proto, bool use_rng, at::PhiloxCudaState rng_args,
                              FormatPalette<typename Accumulator::mac_type> pal = {},
                              const int32_t *prec_idx = nullptr,
                              int64_t idx_row_stride = 0, int64_t idx_col_stride = 0)
    {
        dim3 block_dim(static_cast<unsigned>((N + BLOCKSIZE - 1) / BLOCKSIZE),
                       static_cast<unsigned>((M + BLOCKSIZE - 1) / BLOCKSIZE));
        dim3 thread_dim(BLOCKSIZE * BLOCKSIZE);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        custom_matmul_kernel<scalar_t, MIXED, Accumulator><<<block_dim, thread_dim, 0, stream>>>(
            a, b, c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args,
            pal, prec_idx, idx_row_stride, idx_col_stride);
    }

    void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                             int64_t round_mode, int64_t accumulate_algorithm)
    {
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op_name, " expects 2D tensors, got ",
                   a.dim(), "D and ", b.dim(), "D");
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

    // Draws (seed, offset) from ATen's default CUDA generator (respecting
    // torch.manual_seed, same as the elementwise binaryK_quantize/
    // superfp_quantize SR path's quant_rng_engine_inputs in cuda/utils.cuh),
    // reserving `counter_offset`
    // 128-bit Philox blocks so a subsequent unrelated RNG-consuming op
    // doesn't reuse the same (seed, offset) pair -- the standard native
    // CUDA RNG kernel idiom (see e.g. native/cuda/Dropout.cu upstream).
    // Only called when RoundMode::SR is selected. draws_per_thread is a
    // safe upper bound on how many random values any single output
    // element's thread may draw over its K-step reduction (2*K for
    // SplitMac's independent mul/add draws, K for FusedMac's single draw
    // per step) -- Philox batches 4 draws per 128-bit block.
    at::PhiloxCudaState matmul_rng_engine_inputs(uint64_t draws_per_thread)
    {
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
        uint64_t counter_offset = (draws_per_thread + 3) / 4;
        std::lock_guard<std::mutex> lock(gen->mutex_);
        return gen->philox_cuda_state(counter_offset);
    }

    // Remembers precision maps that have already passed the bounds check
    // below, so a map reused across calls is checked once rather than every
    // time.
    //
    // The check needs the index *values* on the host, and on CUDA that means
    // a device-to-host copy, which drains the stream before the GEMM is even
    // launched: 0.23 ms per call on an RTX 4060 laptop under WSL2, against
    // 0.27 ms for an entire 64^3 mixed GEMM. A map is normally built once
    // and reused, so this skips the copy while the same tensor comes back
    // unchanged. A miss runs exactly the check it always ran, with the same
    // error and the same message.
    //
    // The key is the TensorImpl's address plus its version counter, and the
    // entry holds a weak reference to that impl. The weak reference is what
    // makes comparing raw addresses sound: it keeps the impl's control block
    // (not its storage -- the map itself is not pinned) alive, so no other
    // tensor can be constructed at that address while an entry still names
    // it. The version counter catches every in-place write that goes through
    // ATen. Two cases are deliberately never memoized: a CPU tensor, which
    // has no sync to save, and a tensor with no version counter (created
    // under torch.inference_mode), which has nothing to invalidate against.
    // Both take the full check on every call.
    //
    // FormatPalette::slot masks the index into range regardless, so nothing
    // here can turn a stale entry into an out-of-bounds read.
    // See dev/gemm_perf_audit.md (finding G5).
    struct ValidatedPrecIdx
    {
        c10::weak_intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl> impl;
        const c10::TensorImpl *raw;
        uint32_t version;
        int64_t n_formats;
    };

    constexpr size_t PREC_IDX_MEMO_SLOTS = 4;
    std::mutex g_prec_idx_memo_mutex;
    std::vector<ValidatedPrecIdx> g_prec_idx_memo;

    // (impl address, version) if this tensor is a candidate for the memo.
    bool prec_idx_memo_key(const at::Tensor &pidx, const c10::TensorImpl *&raw, uint32_t &version)
    {
        if (!pidx.is_cuda())
            return false;
        c10::TensorImpl *impl = pidx.unsafeGetTensorImpl();
        if (!impl->version_counter().enabled())
            return false;
        raw = impl;
        version = impl->version_counter().current_version();
        return true;
    }

    bool prec_idx_already_validated(const at::Tensor &pidx, int64_t n_formats)
    {
        const c10::TensorImpl *raw = nullptr;
        uint32_t version = 0;
        if (!prec_idx_memo_key(pidx, raw, version))
            return false;
        std::lock_guard<std::mutex> lock(g_prec_idx_memo_mutex);
        for (const ValidatedPrecIdx &e : g_prec_idx_memo)
            if (e.raw == raw && e.version == version && e.n_formats == n_formats)
                return true;
        return false;
    }

    void remember_validated_prec_idx(const at::Tensor &pidx, int64_t n_formats)
    {
        const c10::TensorImpl *raw = nullptr;
        uint32_t version = 0;
        if (!prec_idx_memo_key(pidx, raw, version))
            return;
        std::lock_guard<std::mutex> lock(g_prec_idx_memo_mutex);
        // drop entries whose tensor is gone, and any stale record of this one
        auto dead = std::remove_if(g_prec_idx_memo.begin(), g_prec_idx_memo.end(),
                                   [&](const ValidatedPrecIdx &e)
                                   { return e.impl.expired() || e.raw == raw; });
        g_prec_idx_memo.erase(dead, g_prec_idx_memo.end());
        if (g_prec_idx_memo.size() >= PREC_IDX_MEMO_SLOTS)
            g_prec_idx_memo.erase(g_prec_idx_memo.begin());
        g_prec_idx_memo.push_back(ValidatedPrecIdx{
            c10::weak_intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>(pidx.getIntrusivePtr()),
            raw, version, n_formats});
    }

    // Validates a spatially-varying mixed-format op's per-output-element
    // precision index and derives the (row_stride, col_stride) pair the
    // kernel reads it with -- accepting a dense [M, N] map, a per-row
    // [M, 1] map, or a per-column [1, N] map (see gemm_policy.h's
    // FormatPalette). Casts to int32 on the operand's device, bounds-checks
    // every entry against the palette size (see the memo above for when that
    // costs a device sync), and hands back the contiguous tensor to keep
    // alive across the launch.
    void resolve_prec_idx(const at::Tensor &prec_idx, const at::Tensor &ref, int64_t M, int64_t N,
                          const char *op_name, int64_t n_formats, at::Tensor &pidx_out,
                          int64_t &idx_row_stride, int64_t &idx_col_stride)
    {
        TORCH_CHECK(prec_idx.dim() == 2, op_name, ": prec_idx must be 2D, got ", prec_idx.dim(), "D");
        at::Tensor pidx = prec_idx.to(ref.device(), at::kInt).contiguous();
        int64_t r = pidx.size(0), c = pidx.size(1);
        if (r == M && c == N)
        {
            idx_row_stride = N;
            idx_col_stride = 1;
        }
        else if (r == M && c == 1)
        {
            idx_row_stride = 1;
            idx_col_stride = 0;
        }
        else if (r == 1 && c == N)
        {
            idx_row_stride = 0;
            idx_col_stride = 1;
        }
        else
        {
            TORCH_CHECK(false, op_name, ": prec_idx shape must be [M, N], [M, 1] or [1, N] (M=", M,
                       ", N=", N, "), got [", r, ", ", c, "]");
        }
        if (!prec_idx_already_validated(pidx, n_formats))
        {
            // aminmax is one reduction and one 2-element copy back, where
            // min().item() and max().item() were two of each.
            auto mm = at::aminmax(pidx);
            auto bounds = at::stack({std::get<0>(mm), std::get<1>(mm)}).cpu();
            const int32_t *b = bounds.data_ptr<int32_t>();
            int64_t lo = b[0], hi = b[1];
            TORCH_CHECK(lo >= 0 && hi < n_formats, op_name, ": prec_idx entries must be in [0, ", n_formats,
                       "), got range [", lo, ", ", hi, "]");
            remember_validated_prec_idx(pidx, n_formats);
        }
        pidx_out = pidx;
    }

    void check_palette_lengths(int64_t n, const char *op_name, std::initializer_list<int64_t> other_lens)
    {
        TORCH_CHECK(n >= 1 && n <= MAX_GEMM_FORMATS, op_name, ": expected 1..", MAX_GEMM_FORMATS,
                   " palette formats, got ", n);
        for (int64_t l : other_lens)
            TORCH_CHECK(l == n, op_name, ": every palette parameter list must have length ", n,
                       " (got one of length ", l, ")");
    }

} // namespace

Tensor binaryK_matmul_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK", M, K, N);

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    int mul_man_bits = static_cast<int>(mul_P - 1);
    int mul_exp_bits = static_cast<int>(mul_is_signed ? mul_K - mul_P : mul_K - mul_P + 1);
    int acc_man_bits = static_cast<int>(acc_P - 1);
    int acc_exp_bits = static_cast<int>(acc_is_signed ? acc_K - acc_P : acc_K - acc_P + 1);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(2 * static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_cuda", [&]
                                    {
        BinaryKMultiplier mul{mul_man_bits, mul_exp_bits, static_cast<int>(mul_bias), mul_is_signed, sat, rm, sub,
                              static_cast<int>(mul_prng_bits)};
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (accumulate_quant)
        {
            BinaryKAdder add{acc_man_bits, acc_exp_bits, static_cast<int>(acc_bias), acc_is_signed, sat, rm, sub,
                             static_cast<int>(acc_prng_bits)};
            using Mac = SplitMac<BinaryKMultiplier, BinaryKAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
        else
        {
            using Mac = SplitMac<BinaryKMultiplier, IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        } });

    return mptorch::widen_float64(c, widen_f64);
}

Tensor superfp_matmul_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias, bool mul_is_signed,
    bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits, int64_t acc_normal_binades,
    int64_t acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp", M, K, N);

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(2 * static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_cuda", [&]
                                    {
        SuperfpMultiplier mul{static_cast<int>(mul_man_bits), static_cast<int>(mul_exp_bits),
                              static_cast<int>(mul_normal_binades), static_cast<int>(mul_bias), mul_is_signed, sat, rm,
                              static_cast<int>(mul_prng_bits)};
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (accumulate_quant)
        {
            SuperfpAdder add{static_cast<int>(acc_man_bits), static_cast<int>(acc_exp_bits),
                             static_cast<int>(acc_normal_binades), static_cast<int>(acc_bias), acc_is_signed, sat, rm,
                             static_cast<int>(acc_prng_bits)};
            using Mac = SplitMac<SuperfpMultiplier, SuperfpAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
        else
        {
            using Mac = SplitMac<SuperfpMultiplier, IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{mul, IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        } });

    return mptorch::widen_float64(c, widen_f64);
}

Tensor binaryK_matmul_fma_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t subnormals_mode,
    int64_t fma_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK_fma", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma", M, K, N);

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    int fma_man_bits = static_cast<int>(fma_P - 1);
    int fma_exp_bits = static_cast<int>(fma_is_signed ? fma_K - fma_P : fma_K - fma_P + 1);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_fma_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (fma_quant)
        {
            BinaryKAdder add{fma_man_bits, fma_exp_bits, static_cast<int>(fma_bias), fma_is_signed, sat, rm, sub,
                             static_cast<int>(fma_prng_bits)};
            using Mac = FusedMac<BinaryKAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
        else
        {
            using Mac = FusedMac<IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        } });

    return mptorch::widen_float64(c, widen_f64);
}

Tensor superfp_matmul_fma_cuda(
    Tensor a, Tensor b, bool trans_a, bool trans_b,
    bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades,
    int64_t fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t fma_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp_fma", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_fma", M, K, N);

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_fma_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        if (fma_quant)
        {
            SuperfpAdder add{static_cast<int>(fma_man_bits), static_cast<int>(fma_exp_bits),
                             static_cast<int>(fma_normal_binades), static_cast<int>(fma_bias), fma_is_signed, sat, rm,
                             static_cast<int>(fma_prng_bits)};
            using Mac = FusedMac<SuperfpAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{add}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        }
        else
        {
            using Mac = FusedMac<IdentityAdder>;
            NaiveAccumulator<Mac> acc_proto{Mac{IdentityAdder{}}, 0.f};
            launch_custom_matmul<scalar_t>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args);
        } });

    return mptorch::widen_float64(c, widen_f64);
}

// Spatially-varying (per-output-element) mixed-format binaryK GEMM: same
// SplitMac arithmetic as binaryK_matmul_cuda, but the multiply/accumulate
// format for each output element C[i, j] is picked from a palette of up to
// MAX_GEMM_FORMATS entries by prec_idx[i, j] (see gemm_policy.h's
// FormatPalette). mul_K/mul_P/mul_bias and acc_K/acc_P/acc_bias are
// per-palette-entry lists (all the same length); round_mode/saturation/
// sign/prng_bits are shared across the palette, matching the mm_impl
// prototype where only the format widths are tabulated.
at::Tensor binaryK_matmul_mixed_cuda(
    at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b,
    c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed,
    bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P, c10::IntArrayRef acc_bias,
    bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK_mixed", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_mixed", M, K, N);

    int64_t n_fmt = static_cast<int64_t>(mul_K.size());
    check_palette_lengths(n_fmt, "custom_matmul_binaryK_mixed",
                          {static_cast<int64_t>(mul_P.size()), static_cast<int64_t>(mul_bias.size()),
                           static_cast<int64_t>(acc_K.size()), static_cast<int64_t>(acc_P.size()),
                           static_cast<int64_t>(acc_bias.size())});

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    at::Tensor pidx;
    int64_t sr = 0, sc = 0;
    resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_binaryK_mixed", n_fmt, pidx, sr, sc);
    const int32_t *p_idx = pidx.data_ptr<int32_t>();

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(2 * static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_mixed_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        auto make_mul = [&](int64_t i) {
            int mmb = static_cast<int>(mul_P[i] - 1);
            int meb = static_cast<int>(mul_is_signed ? mul_K[i] - mul_P[i] : mul_K[i] - mul_P[i] + 1);
            return BinaryKMultiplier{mmb, meb, static_cast<int>(mul_bias[i]), mul_is_signed, sat, rm, sub,
                                     static_cast<int>(mul_prng_bits)};
        };

        if (accumulate_quant)
        {
            using Mac = SplitMac<BinaryKMultiplier, BinaryKAdder>;
            static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
            FormatPalette<Mac> pal;
            pal.n = static_cast<int>(n_fmt);
            for (int64_t i = 0; i < n_fmt; ++i)
            {
                int amb = static_cast<int>(acc_P[i] - 1);
                int aeb = static_cast<int>(acc_is_signed ? acc_K[i] - acc_P[i] : acc_K[i] - acc_P[i] + 1);
                BinaryKAdder add{amb, aeb, static_cast<int>(acc_bias[i]), acc_is_signed, sat, rm, sub,
                                 static_cast<int>(acc_prng_bits)};
                pal.slots[i] = Mac{make_mul(i), add};
            }
            NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
            launch_custom_matmul<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                                 use_rng, rng_args, pal, p_idx, sr, sc);
        }
        else
        {
            using Mac = SplitMac<BinaryKMultiplier, IdentityAdder>;
            static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
            FormatPalette<Mac> pal;
            pal.n = static_cast<int>(n_fmt);
            for (int64_t i = 0; i < n_fmt; ++i)
                pal.slots[i] = Mac{make_mul(i), IdentityAdder{}};
            NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
            launch_custom_matmul<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                                 use_rng, rng_args, pal, p_idx, sr, sc);
        } });

    return mptorch::widen_float64(c, widen_f64);
}

// superfp analogue of binaryK_matmul_mixed_cuda -- see its comment.
at::Tensor superfp_matmul_mixed_cuda(
    at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b,
    c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits, c10::IntArrayRef mul_normal_binades,
    c10::IntArrayRef mul_bias, bool mul_is_signed,
    bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
    c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t mul_prng_bits, int64_t acc_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp_mixed", round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_mixed", M, K, N);

    int64_t n_fmt = static_cast<int64_t>(mul_man_bits.size());
    check_palette_lengths(n_fmt, "custom_matmul_superfp_mixed",
                          {static_cast<int64_t>(mul_exp_bits.size()),
                           static_cast<int64_t>(mul_normal_binades.size()),
                           static_cast<int64_t>(mul_bias.size()),
                           static_cast<int64_t>(acc_man_bits.size()),
                           static_cast<int64_t>(acc_exp_bits.size()),
                           static_cast<int64_t>(acc_normal_binades.size()),
                           static_cast<int64_t>(acc_bias.size())});

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    at::Tensor pidx;
    int64_t sr = 0, sc = 0;
    resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_superfp_mixed", n_fmt, pidx, sr, sc);
    const int32_t *p_idx = pidx.data_ptr<int32_t>();

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(2 * static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_mixed_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        auto make_mul = [&](int64_t i) {
            return SuperfpMultiplier{static_cast<int>(mul_man_bits[i]), static_cast<int>(mul_exp_bits[i]),
                                     static_cast<int>(mul_normal_binades[i]), static_cast<int>(mul_bias[i]),
                                     mul_is_signed, sat, rm, static_cast<int>(mul_prng_bits)};
        };

        if (accumulate_quant)
        {
            using Mac = SplitMac<SuperfpMultiplier, SuperfpAdder>;
            static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
            FormatPalette<Mac> pal;
            pal.n = static_cast<int>(n_fmt);
            for (int64_t i = 0; i < n_fmt; ++i)
            {
                SuperfpAdder add{static_cast<int>(acc_man_bits[i]), static_cast<int>(acc_exp_bits[i]),
                                 static_cast<int>(acc_normal_binades[i]), static_cast<int>(acc_bias[i]),
                                 acc_is_signed, sat, rm, static_cast<int>(acc_prng_bits)};
                pal.slots[i] = Mac{make_mul(i), add};
            }
            NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
            launch_custom_matmul<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                                 use_rng, rng_args, pal, p_idx, sr, sc);
        }
        else
        {
            using Mac = SplitMac<SuperfpMultiplier, IdentityAdder>;
            static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
            FormatPalette<Mac> pal;
            pal.n = static_cast<int>(n_fmt);
            for (int64_t i = 0; i < n_fmt; ++i)
                pal.slots[i] = Mac{make_mul(i), IdentityAdder{}};
            NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
            launch_custom_matmul<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                                 use_rng, rng_args, pal, p_idx, sr, sc);
        } });

    return mptorch::widen_float64(c, widen_f64);
}

// Spatially-varying (per-output-element) mixed-format binaryK FMA GEMM: same
// FusedMac arithmetic as binaryK_matmul_fma_cuda (one rounding per K-step),
// but the FMA format for each output element C[i, j] is picked from a
// palette of up to MAX_GEMM_FORMATS entries by prec_idx[i, j] (see
// gemm_policy.h's FormatPalette). fma_K/fma_P/fma_bias are per-palette-entry
// lists (all the same length); round_mode/saturation/sign/prng_bits are
// shared across the palette. fma_quant=false is rejected: FusedMac<
// IdentityAdder> carries no format, so a palette of it would make prec_idx a
// no-op -- use custom_matmul_binaryK_fma for the unquantized fused step.
at::Tensor binaryK_matmul_fma_mixed_cuda(
    at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b,
    bool fma_quant, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P, c10::IntArrayRef fma_bias,
    bool fma_is_signed, int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode,
    int64_t subnormals_mode, int64_t fma_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_binaryK_fma_mixed", round_mode, accumulate_algorithm);
    TORCH_CHECK(fma_quant, "custom_matmul_binaryK_fma_mixed: fma_quant=false has no per-element "
                           "format to vary; use custom_matmul_binaryK_fma");

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_binaryK_fma_mixed", M, K, N);

    int64_t n_fmt = static_cast<int64_t>(fma_K.size());
    check_palette_lengths(n_fmt, "custom_matmul_binaryK_fma_mixed",
                          {static_cast<int64_t>(fma_P.size()), static_cast<int64_t>(fma_bias.size())});

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    at::Tensor pidx;
    int64_t sr = 0, sc = 0;
    resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_binaryK_fma_mixed", n_fmt, pidx, sr, sc);
    const int32_t *p_idx = pidx.data_ptr<int32_t>();

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    SubnormalsMode sub = static_cast<SubnormalsMode>(subnormals_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "binaryK_matmul_fma_mixed_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        using Mac = FusedMac<BinaryKAdder>;
        static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
        FormatPalette<Mac> pal;
        pal.n = static_cast<int>(n_fmt);
        for (int64_t i = 0; i < n_fmt; ++i)
        {
            int mb = static_cast<int>(fma_P[i] - 1);
            int eb = static_cast<int>(fma_is_signed ? fma_K[i] - fma_P[i] : fma_K[i] - fma_P[i] + 1);
            BinaryKAdder add{mb, eb, static_cast<int>(fma_bias[i]), fma_is_signed, sat, rm, sub,
                             static_cast<int>(fma_prng_bits)};
            pal.slots[i] = Mac{add};
        }
        NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
        launch_custom_matmul<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                             use_rng, rng_args, pal, p_idx, sr, sc); });

    return mptorch::widen_float64(c, widen_f64);
}

// superfp analogue of binaryK_matmul_fma_mixed_cuda -- see its comment.
at::Tensor superfp_matmul_fma_mixed_cuda(
    at::Tensor a, at::Tensor b, at::Tensor prec_idx, bool trans_a, bool trans_b,
    bool fma_quant, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
    c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed,
    int64_t accumulate_algorithm, int64_t round_mode, int64_t saturation_mode, int64_t fma_prng_bits)
{
    check_matmul_inputs(a, b, "custom_matmul_superfp_fma_mixed", round_mode, accumulate_algorithm);
    TORCH_CHECK(fma_quant, "custom_matmul_superfp_fma_mixed: fma_quant=false has no per-element "
                           "format to vary; use custom_matmul_superfp_fma");

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, "custom_matmul_superfp_fma_mixed", M, K, N);

    int64_t n_fmt = static_cast<int64_t>(fma_man_bits.size());
    check_palette_lengths(n_fmt, "custom_matmul_superfp_fma_mixed",
                          {static_cast<int64_t>(fma_exp_bits.size()),
                           static_cast<int64_t>(fma_normal_binades.size()),
                           static_cast<int64_t>(fma_bias.size())});

    // float64 in, float64 out, narrowed here instead of on every load so the
    // dispatch below need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    auto a_c = a.contiguous();
    auto b_c = b.contiguous();
    auto c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
        return mptorch::widen_float64(c, widen_f64);

    at::Tensor pidx;
    int64_t sr = 0, sc = 0;
    resolve_prec_idx(prec_idx, a, M, N, "custom_matmul_superfp_fma_mixed", n_fmt, pidx, sr, sc);
    const int32_t *p_idx = pidx.data_ptr<int32_t>();

    SaturationMode sat = static_cast<SaturationMode>(saturation_mode);
    RoundMode rm = static_cast<RoundMode>(round_mode);

    bool use_rng = (rm == RoundMode::SR);
    at::PhiloxCudaState rng_args = use_rng ? matmul_rng_engine_inputs(static_cast<uint64_t>(K))
                                            : at::PhiloxCudaState{};

    MPTORCH_DISPATCH_QUANT_TYPES(a.scalar_type(), "superfp_matmul_fma_mixed_cuda", [&]
                                    {
        const scalar_t *p_a = a_c.data_ptr<scalar_t>();
        const scalar_t *p_b = b_c.data_ptr<scalar_t>();
        scalar_t *p_c = c.data_ptr<scalar_t>();

        using Mac = FusedMac<SuperfpAdder>;
        static_assert(std::is_trivially_copyable_v<Mac>, "Mac must be trivially copyable to ride in FormatPalette");
        FormatPalette<Mac> pal;
        pal.n = static_cast<int>(n_fmt);
        for (int64_t i = 0; i < n_fmt; ++i)
        {
            SuperfpAdder add{static_cast<int>(fma_man_bits[i]), static_cast<int>(fma_exp_bits[i]),
                             static_cast<int>(fma_normal_binades[i]), static_cast<int>(fma_bias[i]),
                             fma_is_signed, sat, rm, static_cast<int>(fma_prng_bits)};
            pal.slots[i] = Mac{add};
        }
        NaiveAccumulator<Mac> acc_proto{pal.slots[0], 0.f};
        launch_custom_matmul<scalar_t, true>(p_a, p_b, p_c, M, K, N, trans_a, trans_b, acc_proto,
                                             use_rng, rng_args, pal, p_idx, sr, sc); });

    return mptorch::widen_float64(c, widen_f64);
}
