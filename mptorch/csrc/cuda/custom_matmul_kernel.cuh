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
// named namespace now. The kernel is a template either way, so nvcc still
// sees a full specialization per launch -- the eight GEMM kernels' measured
// throughput is unchanged.
//
// H1 then took everything that was not the kernel out of here: the host
// prologue and the prec_idx memo are common/gemm_host.h, the policy factories
// are common/gemm_args.h, and the entry points are custom_matmul_entry.cpp.
// What is left carries no ATen at all, which is what makes the four .cu files
// cheap to compile -- see cuda/gemm_backend.h.
#include "../common/gemm_args.h"
#include "gemm_backend.h"
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::GemmShape;

    // Double-buffered tiled GEMM, ported from the mm_kernel3 prototype in
    // dev/cuda/custom_matmul.cu (benchmarked slightly faster than the
    // register-blocked mm_kernel4 variant it originally replaced). Each
    // thread block computes a BLOCKSIZE x BLOCKSIZE output tile; each thread
    // computes exactly one output element, accumulated via Accumulator (Mac
    // -generic -- see gemm_policy.h).
    constexpr int BLOCKSIZE = 16;

    // The operands' storage dtype is a kernel *argument*, not a template
    // parameter -- see mptorch::GemmDtype in common/gemm_dtype.h for why
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

    // The launcher the four .cu files instantiate. The stream arrives from
    // the driver rather than being fetched here: getCurrentCUDAStream() lives
    // behind <ATen/cuda/CUDAContext.h>, which is most of what this file no
    // longer includes.
    template <bool MIXED = false, class Accumulator>
    void launch_custom_matmul(const void *a, const void *b, void *c, mptorch::GemmDtype dt,
                              int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
                              Accumulator acc_proto, bool use_rng, at::PhiloxCudaState rng_args,
                              cudaStream_t stream,
                              FormatPalette<typename Accumulator::mac_type> pal = {},
                              const int32_t *prec_idx = nullptr,
                              int64_t idx_row_stride = 0, int64_t idx_col_stride = 0)
    {
        dim3 block_dim(static_cast<unsigned>((N + BLOCKSIZE - 1) / BLOCKSIZE),
                       static_cast<unsigned>((M + BLOCKSIZE - 1) / BLOCKSIZE));
        dim3 thread_dim(BLOCKSIZE * BLOCKSIZE);
        custom_matmul_kernel<MIXED, Accumulator><<<block_dim, thread_dim, 0, stream>>>(
            a, b, c, dt, M, K, N, trans_a, trans_b, acc_proto, use_rng, rng_args,
            pal, prec_idx, idx_row_stride, idx_col_stride);
    }

    // One body for all eight ops. The Args is the only thing that differs
    // between them, and all it does is name which policy to build
    // (common/gemm_args.h); the eight entry points that used to spell this out
    // one at a time are now two calls in custom_matmul_entry.cpp.
    //
    // LEAN=true is this backend's answer to the one question gemm_args.h asks
    // it: the superfp mixed palette holds two full policies per slot and runs
    // out of registers at 100, where the Lean spelling rebuilds the fast-path
    // floats instead of carrying them and buys a resident block back. The CPU
    // has no such cliff and answers false. See dev/gemm_perf_audit.md (G10).
    template <class Args>
    void CudaBackend::launch(const GemmShape &s, const Args &args, const LaunchContext &ctx)
    {
        mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
        {
            constexpr RoundMode RM = decltype(rm_c)::value;
            if constexpr (Args::mixed)
            {
                args.template with_palette<RM, true>([&](auto acc, const auto &pal)
                {
                    launch_custom_matmul<true>(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a,
                                               s.trans_b, acc, s.use_rng, ctx.rng, ctx.stream,
                                               pal, s.prec_idx, s.idx_row_stride, s.idx_col_stride);
                });
            }
            else
            {
                args.template with_accumulator<RM, true>([&](auto acc)
                {
                    launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                         acc, s.use_rng, ctx.rng, ctx.stream);
                });
            }
        });
    }
} // namespace mptorch::gemm_cuda
