#pragma once

// The CUDA GEMM kernel, its launch helper, and the host-side prologue every
// entry point repeats -- shared by the four custom_matmul_*.cu translation
// units that hold the eight entry points themselves.
//
// All of this lived in one custom_matmul_kernel.cu until that file became the
// build's long pole: 101.9 s of a 111.2 s build once -g stopped reaching the
// host compiler (finding B1). Compiled on its own it was 87.2 s, of which
// ~43 s was the fixed cost of putting ATen's headers through nvcc -- a cost
// every .cu pays whatever it contains (cuda_ops.cu, which holds no kernels at
// all, was 43.1 s). So the file could only be shortened by putting the other
// ~44 s on more than one core. setup.py globs this directory, so splitting it
// is a matter of adding files: the largest of the four is 43.9 s. See
// dev/gemm_roadmap.md (finding B2), and note the include set below, which is
// what took the fixed cost from ~43 s to ~25 s.
//
// The cut is one TU per (format family x mac mode), which keeps each family's
// cast template (cast_binaryK.h / cast_superfp.h -- a ~150-instruction body)
// instantiated once rather than twice, and keeps that family's single-format
// and mixed-format entry points, whose Mac types are the same, together. No
// kernel specialization is therefore compiled into two objects. Four is not
// an arbitrary number: with 15 TUs the build is already throughput-bound
// rather than long-pole-bound on this 16-thread machine, so splitting further
// would only multiply that ~25 s fixed cost again for nothing.
//
// These were file-local (an anonymous namespace) before the split and are a
// named namespace now: a header's anonymous namespace would give each
// including TU its own copy of the memo below, where there was one. The
// kernel is a template either way, so nvcc still sees a full specialization
// per launch -- the eight GEMM kernels' measured throughput is unchanged.

#include "../common/gemm_policy.h"
#include "../common/dispatch.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/PhiloxUtils.cuh>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/stack.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <initializer_list>
#include <vector>
#include <mutex>

namespace mptorch::gemm_cuda
{
    // Confined to this namespace, rather than dumped into the file scope of
    // everything that includes this header.
    using namespace at;

    // Double-buffered tiled GEMM, ported from the mm_kernel3 prototype in
    // dev/cuda/custom_matmul.cu (benchmarked slightly faster than the
    // register-blocked mm_kernel4 variant it originally replaced). Each
    // thread block computes a BLOCKSIZE x BLOCKSIZE output tile; each thread
    // computes exactly one output element, accumulated via Accumulator (Mac
    // -generic -- see gemm_policy.h).
    constexpr int BLOCKSIZE = 16;

    // The operands' storage dtype is a kernel *argument*, not a template
    // parameter -- see mptorch::GemmDtype in common/dispatch.h for why
    // (finding K2). The switch is warp-uniform, it is reached twice per
    // BLOCKSIZE K-steps rather than once per accumulate, and it sits next to
    // the global load whose latency it hides behind.
    __device__ __forceinline__ float load_elem(const void *p, int64_t i, mptorch::GemmDtype dt)
    {
        switch (dt)
        {
        case mptorch::GemmDtype::Half:
            return static_cast<float>(static_cast<const at::Half *>(p)[i]);
        case mptorch::GemmDtype::BFloat16:
            return static_cast<float>(static_cast<const at::BFloat16 *>(p)[i]);
        default:
            return static_cast<const float *>(p)[i];
        }
    }

    __device__ __forceinline__ void store_elem(void *p, int64_t i, mptorch::GemmDtype dt, float v)
    {
        switch (dt)
        {
        case mptorch::GemmDtype::Half:
            static_cast<at::Half *>(p)[i] = static_cast<at::Half>(v);
            break;
        case mptorch::GemmDtype::BFloat16:
            static_cast<at::BFloat16 *>(p)[i] = static_cast<at::BFloat16>(v);
            break;
        default:
            static_cast<float *>(p)[i] = v;
            break;
        }
    }

    __device__ __forceinline__ float load_a(const void *A, mptorch::GemmDtype dt,
                                            int64_t M, int64_t K, bool trans_a,
                                            int64_t row, int64_t col)
    {
        // logical A is M x K; trans_a means A's storage is actually K x M.
        return load_elem(A, trans_a ? col * M + row : row * K + col, dt);
    }

    __device__ __forceinline__ float load_b(const void *B, mptorch::GemmDtype dt,
                                            int64_t K, int64_t N, bool trans_b,
                                            int64_t row, int64_t col)
    {
        // logical B is K x N; trans_b means B's storage is actually N x K.
        return load_elem(B, trans_b ? col * K + row : row * N + col, dt);
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
    template <bool MIXED, class Accumulator>
    __global__ __launch_bounds__(BLOCKSIZE * BLOCKSIZE)
    void custom_matmul_kernel(
        const void *__restrict__ A, const void *__restrict__ B, void *__restrict__ C,
        mptorch::GemmDtype dt,
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
        // dev/gemm_roadmap.md for why this granularity, not a shared
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
            (rId < M && threadCol < K) ? load_a(A, dt, M, K, trans_a, rId, threadCol) : 0.0f;
        Bs[0][threadRow * BLOCKSIZE + threadCol] =
            (threadRow < K && cId < N) ? load_b(B, dt, K, N, trans_b, threadRow, cId) : 0.0f;

        for (int64_t t = 0; t < numTiles; ++t)
        {
            __syncthreads();

            float *curAs = As[t % 2];
            float *curBs = Bs[t % 2];

            // UNROLL was 4 for both paths for as long as accumulate() inlined
            // a runtime RoundMode switch -- seven cast bodies, twice over for a
            // split mac -- because unrolling that 16 times expanded the loop
            // past any instruction cache and the kernel spent its time fetching
            // code it never executed (finding G2). K1 made the mode a template
            // parameter, so one body is live and the full unroll fits again --
            // but only where the registers do, which is why the two paths now
            // differ. Re-measured on the shipped kernel at 1024^3, min ms, at
            // unroll 2 / 4 / 8 / 16:
            //
            //   binaryK split RNE       9.49 / 8.58 / 8.47 / 8.28
            //   binaryK fma RNE         4.65 / 4.31 / 4.21 / 3.97
            //   superfp split SR nb=8  22.62 / 22.37 / 22.97 / 21.37
            //   binaryK mixed RNE      11.54 / 10.82 / 11.29 / 13.82
            //   superfp mixed SR nb=1  24.03 / 23.36 / 23.59 / 29.84
            //   superfp mixed SR nb=8  35.63 / 35.69 / 38.95 / 50.01
            //
            // 16 wins every single-format row; on the mixed rows it is a
            // 1.3-1.4x loss and 4 wins or ties. The mixed kernel carries a
            // whole extra Mac per output element and the palette besides, so it
            // reaches the register cliff two doublings sooner -- the same
            // pressure finding G10 traced to the Lean params. Worth nothing
            // before finding G1 landed: until the accumulator was
            // register-resident this loop was bound by local-memory traffic,
            // not instruction fetch.
            //
            // The full unroll is bought with binary size: .nv_fatbin over the
            // same four settings is 2.94 / 3.75 / 5.34 / 8.48 MB, and this
            // 16-and-4 split lands at 6.50 MB against 5.29 MB before K1/K2.
            // So the last 3-8% on the single-format kernels costs 2.8 MB. That
            // is the trade to revisit if binary size ever matters here: set
            // both paths to 4 and the extension is 2.8 MB smaller.
            constexpr int UNROLL = MIXED ? 4 : 16;
#pragma unroll UNROLL
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            {
                acc.accumulate(curAs[threadRow * BLOCKSIZE + dotIdx], curBs[dotIdx * BLOCKSIZE + threadCol]);
            }

            // load next tile into the other buffer
            if (t + 1 < numTiles)
            {
                int64_t nextK = (t + 1) * BLOCKSIZE;
                As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                    (rId < M && nextK + threadCol < K) ? load_a(A, dt, M, K, trans_a, rId, nextK + threadCol) : 0.0f;
                Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                    (nextK + threadRow < K && cId < N) ? load_b(B, dt, K, N, trans_b, nextK + threadRow, cId) : 0.0f;
            }
            __syncthreads();
        }

        if (rId < M && cId < N)
            store_elem(C, rId * N + cId, dt, acc.finalize());
    }

    template <bool MIXED = false, class Accumulator>
    void launch_custom_matmul(const void *a, const void *b, void *c, mptorch::GemmDtype dt,
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
        custom_matmul_kernel<MIXED, Accumulator><<<block_dim, thread_dim, 0, stream>>>(
            a, b, c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args,
            pal, prec_idx, idx_row_stride, idx_col_stride);
    }

    inline void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                             int64_t round_mode, int64_t accumulate_algorithm)
    {
        TORCH_CHECK(a.dim() == 2 && b.dim() == 2, op_name, " expects 2D tensors, got ",
                   a.dim(), "D and ", b.dim(), "D");
        TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) == AccumulateAlgorithm::NAIVE,
                   op_name, ": only AccumulateAlgorithm.NAIVE is supported in this build");
    }

    inline void matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b,
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
    inline at::PhiloxCudaState matmul_rng_engine_inputs(uint64_t draws_per_thread)
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
    inline std::mutex g_prec_idx_memo_mutex;
    inline std::vector<ValidatedPrecIdx> g_prec_idx_memo;

    // (impl address, version) if this tensor is a candidate for the memo.
    inline bool prec_idx_memo_key(const at::Tensor &pidx, const c10::TensorImpl *&raw, uint32_t &version)
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

    inline bool prec_idx_already_validated(const at::Tensor &pidx, int64_t n_formats)
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

    inline void remember_validated_prec_idx(const at::Tensor &pidx, int64_t n_formats)
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
    inline void resolve_prec_idx(const at::Tensor &prec_idx, const at::Tensor &ref, int64_t M, int64_t N,
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

    inline void check_palette_lengths(int64_t n, const char *op_name, std::initializer_list<int64_t> other_lens)
    {
        TORCH_CHECK(n >= 1 && n <= MAX_GEMM_FORMATS, op_name, ": expected 1..", MAX_GEMM_FORMATS,
                   " palette formats, got ", n);
        for (int64_t l : other_lens)
            TORCH_CHECK(l == n, op_name, ": every palette parameter list must have length ", n,
                       " (got one of length ", l, ")");
    }
} // namespace mptorch::gemm_cuda
