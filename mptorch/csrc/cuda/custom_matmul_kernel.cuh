#pragma once

// The CUDA GEMM kernel and its launcher, shared by the eight
// custom_matmul_*.cu translation units, each of which explicitly instantiates
// CudaBackend::launch_as (declared in gemm_backend.h) for its two ops.
//
// This header includes no ATen tensor headers, on purpose. Under nvcc a
// translation unit that includes <ATen/core/Tensor.h> pays about 25 s of
// fixed front-end cost before it compiles a single kernel; one that includes
// only the policy headers and the CUDA fp16/bf16 headers pays about 3 s. So
// the kernels see raw pointers and plain structs (GemmShape and an Args from
// common/gemm_args.h), and everything that touches at::Tensor stays in
// custom_matmul_entry.cpp, which g++ compiles. The only ATen headers here are
// the Philox state unpacker and the Half/BFloat16 value types, both
// header-only and cheap.
//
// Why eight .cu files rather than one: the build is throughput-bound across
// sixteen threads, and a single kernel file was its long pole (over 100 s of
// a 111 s build). The cut is one translation unit per (format family x mac
// mode x carrier): custom_matmul_{binaryK,superfp}{,_fma}.cu hold the
// binary32 kernels and their *_f64.cu twins the binary64 ones. A family's
// cast template is a body of ~150 instructions, and a file that holds both
// the single-format and the mixed-format op of one (family, mac mode)
// instantiates it once; no kernel specialization is compiled into two
// objects. The *_f64 files are separate so that MPTORCH_NO_FP64=1 can leave
// the binary64 kernels out of the build. Splitting further would only repeat
// the ~3 s fixed cost per file with nothing left to parallelize.
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

    // Double-buffered shared-memory tiled GEMM. Each thread block computes
    // one BLOCKSIZE x BLOCKSIZE tile of C with BLOCKSIZE^2 threads, and each
    // thread owns exactly one output element and its whole K-reduction,
    // driven through an Accumulator policy (common/gemm_policy.h) so that the
    // format and the rounding are the policy's, not the kernel's. Per K-step
    // of BLOCKSIZE the block stages one tile of A and one of B in shared
    // memory; the two buffers let the next tile's global loads overlap the
    // current tile's arithmetic. This one-element-per-thread shape was
    // benchmarked against a register-blocked variant (several outputs per
    // thread) in dev/cuda/custom_matmul.cu and came out slightly faster here.
    constexpr int BLOCKSIZE = 16;

    // Element load and store by storage dtype. The dtype is a kernel
    // argument (mptorch::GemmDtype, common/gemm_dtype.h), not a template
    // parameter: the storage type decides nothing but these two conversions,
    // and instantiating the kernel per dtype would triple the cast bodies,
    // the unrolled K-loop, the compile time and the binary for the sake of
    // two loads per BLOCKSIZE K-steps and one store per output element. The
    // switch is warp-uniform and sits next to the global load whose latency
    // hides it.
    //
    // T is the carrier the kernel computes in. The primary template is
    // binary32's and converts any of its three dtypes to float; binary64 has
    // one dtype (double), so its specializations are a plain load and a
    // plain store. They are specializations rather than an `if constexpr`
    // around the switch so the binary32 bodies compile to the same code
    // whether or not the binary64 kernels are built.
    template <class T>
    __device__ __forceinline__ T load_elem(const void *p, int64_t i, mptorch::GemmDtype dt)
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

    template <class T>
    __device__ __forceinline__ void store_elem(void *p, int64_t i, mptorch::GemmDtype dt, T v)
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

    template <>
    __device__ __forceinline__ double load_elem<double>(const void *p, int64_t i, mptorch::GemmDtype)
    {
        return static_cast<const double *>(p)[i];
    }

    template <>
    __device__ __forceinline__ void store_elem<double>(void *p, int64_t i, mptorch::GemmDtype, double v)
    {
        static_cast<double *>(p)[i] = v;
    }

    // Logical element (row, col) of an operand. `off` is this batch
    // element's base offset into the operand, in elements: 0 for a 2D call
    // and for an operand broadcast across the batch (stride 0).
    template <class T>
    __device__ __forceinline__ T load_a(const void *A, mptorch::GemmDtype dt,
                                        int64_t M, int64_t K, bool trans_a,
                                        int64_t row, int64_t col, int64_t off)
    {
        // logical A is M x K; trans_a means A's storage is actually K x M.
        return load_elem<T>(A, off + (trans_a ? col * M + row : row * K + col), dt);
    }

    template <class T>
    __device__ __forceinline__ T load_b(const void *B, mptorch::GemmDtype dt,
                                        int64_t K, int64_t N, bool trans_b,
                                        int64_t row, int64_t col, int64_t off)
    {
        // logical B is K x N; trans_b means B's storage is actually N x K.
        return load_elem<T>(B, off + (trans_b ? col * K + row : row * N + col), dt);
    }

    // The kernel. A is logically [M, K] and B [K, N] (trans_* says the
    // storage is the transpose); C is written densely as [batch, M, N]. The
    // batch element is batch_base + blockIdx.z, and stride_a / stride_b are
    // element offsets between consecutive batch elements, 0 meaning the
    // operand is broadcast. acc_proto is the fully built Accumulator every
    // thread copies; rng_args is read only when use_rng (RoundMode::SR). On
    // the mixed path pal holds up to MAX_GEMM_FORMATS Mac policies and
    // prec_idx picks one per output element through the three strides (0 for
    // a dimension the index does not vary along).
    //
    // MIXED selects whether this instantiation carries the per-element
    // palette prologue. It is a template parameter rather than a runtime
    // `pal.n > 0` test because a merely *possible* write to acc.mac forces
    // the Mac policy's format constants into registers for the whole K-loop,
    // instead of being re-read from the constant bank where acc_proto sits.
    // That cost the split-mac kernels ~29 extra registers (109 vs 80 at
    // BLOCKSIZE^2 threads) and roughly halved their throughput (34.7 -> 68.3
    // ms at 1024^3 on sm_89) on every single-format launch, which is nearly
    // all of them. Two instantiations double the kernel count, and the
    // single-format path keeps its speed. dev/benchmarks/gemm_kernel_tuning.cu
    // holds the A/B.
    template <bool MIXED, class Accumulator>
    __global__ __launch_bounds__(BLOCKSIZE * BLOCKSIZE)
    void custom_matmul_kernel(
        const void *__restrict__ A, const void *__restrict__ B, void *__restrict__ C,
        mptorch::GemmDtype dt,
        int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
        int64_t batch_base, int64_t stride_a, int64_t stride_b,
        Accumulator acc_proto, bool use_rng, at::PhiloxCudaState rng_args,
        PaletteArg<MIXED, typename Accumulator::mac_type> pal,
        const int32_t *__restrict__ prec_idx, int64_t idx_row_stride, int64_t idx_col_stride,
        int64_t idx_batch_stride)
    {
        // The carrier: every operand is loaded into it, every step computed
        // in it, and the output stored from it. The two double buffers are
        // 4 KB of shared memory for binary32 and 8 KB for binary64.
        using T = typename Accumulator::value_t;
        __shared__ T As[2][BLOCKSIZE * BLOCKSIZE];
        __shared__ T Bs[2][BLOCKSIZE * BLOCKSIZE];

        const int64_t cRow = blockIdx.y;
        const int64_t cCol = blockIdx.x;

        const int threadCol = threadIdx.x % BLOCKSIZE;
        const int threadRow = threadIdx.x / BLOCKSIZE;

        const int64_t rowBase = cRow * BLOCKSIZE;
        const int64_t colBase = cCol * BLOCKSIZE;
        const int64_t rId = rowBase + threadRow;
        const int64_t cId = colBase + threadCol;

        const int64_t numTiles = (K + BLOCKSIZE - 1) / BLOCKSIZE;

        // The batch dimension is blockIdx.z, offset by batch_base because the
        // launcher splits a batch larger than the 65,535 grid-z limit into
        // several launches. The two operand offsets are computed once and
        // added to every index: stride 0 makes an operand broadcast across
        // the batch, and a 2D call is base 0 with both strides 0, so it
        // computes exactly the addresses an unbatched kernel would. The
        // offsets stay live across the K-loop and cost +2 to +11 registers;
        // a separate unbatched instantiation would buy back at most 3.6%
        // for 86 more kernels and 4.7 MB more fatbin, so there is none.
        const int64_t bId = batch_base + static_cast<int64_t>(blockIdx.z);
        const int64_t aOff = bId * stride_a;
        const int64_t bOff = bId * stride_b;

        Accumulator acc = acc_proto;

        // RoundMode::SR: seed this output element's Philox stream once,
        // before the K-loop. The subsequence is the element's global linear
        // index into the whole [batch, M, N] output, so the draws it sees
        // depend on nothing but which element it is: not on BLOCKSIZE, the
        // grid, the chunking of the batch, or whether the product was spelled
        // as a batched call or as a 2D call with the batch folded into M.
        // Batch element 0 gets the subsequence the 2D call gives, which is
        // what makes the two spellings bit-identical. A stream shared by a
        // block's threads would make results depend on launch geometry.
        if (use_rng && rId < M && cId < N)
        {
            auto seeds = at::cuda::philox::unpack(rng_args);
            acc.seed_rng(std::get<0>(seeds),
                         (static_cast<uint64_t>(bId) * M + rId) * N + cId, std::get<1>(seeds));
        }

        // Mixed format: bind this output element's Mac policy from the
        // palette before the K-loop, so the loop itself is the same code as
        // the single-format path. Compiled out entirely when MIXED is false,
        // whose instantiation does not even carry the palette argument. No
        // emptiness test: the mixed packers require at least one format, so
        // MIXED implies a populated palette.
        if constexpr (MIXED)
        {
            if (rId < M && cId < N)
                acc.mac = pal.slot(prec_idx[bId * idx_batch_stride + rId * idx_row_stride +
                                            cId * idx_col_stride]);
        }

        // Stage the first K-tile in buffer 0; out-of-range lanes read 0.
        As[0][threadRow * BLOCKSIZE + threadCol] =
            (rId < M && threadCol < K) ? load_a<T>(A, dt, M, K, trans_a, rId, threadCol, aOff) : 0.0f;
        Bs[0][threadRow * BLOCKSIZE + threadCol] =
            (threadRow < K && cId < N) ? load_b<T>(B, dt, K, N, trans_b, threadRow, cId, bOff) : 0.0f;

        for (int64_t t = 0; t < numTiles; ++t)
        {
            __syncthreads();

            T *curAs = As[t % 2];
            T *curBs = Bs[t % 2];

            // The unroll factor is where the cast bodies meet the instruction
            // cache and the register file. With the rounding mode a template
            // parameter only one cast body is live per step, so the full
            // unroll (16) fits and wins 3-8% on every binary32 single-format
            // kernel (binaryK fma RNE at 1024^3: 4.31 ms at 4, 3.97 ms at
            // 16). The mixed kernels carry an extra Mac per output element
            // plus the palette, reach the register cliff sooner, and lose
            // 1.3-1.4x at 16, so they stay at 4. The full unroll is paid for
            // in binary size, about 2.8 MB of fatbin over the extension. The
            // binary64 kernels are bound by the device's FP64 rate rather
            // than by instruction fetch, so their unroll is flat within 1%
            // from 2 to 8; 2 has the fewest spills and the smallest binary.
            // Measured with dev/benchmarks/gemm_unroll_sweep.cu.
            constexpr int UNROLL = std::is_same_v<T, double> ? 2 : (MIXED ? 4 : 16);
#pragma unroll UNROLL
            for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            {
                acc.accumulate(curAs[threadRow * BLOCKSIZE + dotIdx], curBs[dotIdx * BLOCKSIZE + threadCol]);
            }

            // Prefetch the next K-tile into the other buffer; the barrier
            // below publishes it before the next iteration reads it.
            if (t + 1 < numTiles)
            {
                int64_t nextK = (t + 1) * BLOCKSIZE;
                As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                    (rId < M && nextK + threadCol < K) ? load_a<T>(A, dt, M, K, trans_a, rId, nextK + threadCol, aOff) : 0.0f;
                Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                    (nextK + threadRow < K && cId < N) ? load_b<T>(B, dt, K, N, trans_b, nextK + threadRow, cId, bOff) : 0.0f;
            }
            __syncthreads();
        }

        // C is dense [batch, M, N], so its own stride needs no argument.
        if (rId < M && cId < N)
            store_elem<T>(C, (bId * M + rId) * N + cId, dt, acc.finalize());
    }

    // Launches the kernel over a grid of [M, N] tiles per batch element. The
    // stream and the Philox state arrive from the entry point:
    // getCurrentCUDAStream() lives behind <ATen/cuda/CUDAContext.h>, which
    // this header must not include (see the top of the file).
    //
    // The batch rides on gridDim.z, whose extent is capped at 65,535, two
    // orders of magnitude below gridDim.x/y's 2^31 - 1 and reachable in
    // practice (a per-head attention call is batch * heads). A larger batch
    // is split into successive launches with batch_base advancing. Chunking
    // here rather than folding the batch into gridDim.y keeps each block's
    // (row, col) mapping and shared-memory tiling untouched, and a chunk
    // boundary is not observable in the output: each block writes its own
    // tile of its own batch element, and RoundMode::SR keys on the global
    // element index, not on the launch.
    template <bool MIXED = false, class Accumulator>
    void launch_custom_matmul(const void *a, const void *b, void *c, mptorch::GemmDtype dt,
                              int64_t M, int64_t K, int64_t N, bool trans_a, bool trans_b,
                              int64_t batch, int64_t stride_a, int64_t stride_b,
                              Accumulator acc_proto, bool use_rng, at::PhiloxCudaState rng_args,
                              cudaStream_t stream,
                              PaletteArg<MIXED, typename Accumulator::mac_type> pal = {},
                              const int32_t *prec_idx = nullptr,
                              int64_t idx_row_stride = 0, int64_t idx_col_stride = 0,
                              int64_t idx_batch_stride = 0)
    {
        constexpr int64_t MAX_GRID_Z = 65535;
        dim3 block_dim(static_cast<unsigned>((N + BLOCKSIZE - 1) / BLOCKSIZE),
                       static_cast<unsigned>((M + BLOCKSIZE - 1) / BLOCKSIZE), 1u);
        dim3 thread_dim(BLOCKSIZE * BLOCKSIZE);
        for (int64_t base = 0; base < batch; base += MAX_GRID_Z)
        {
            const int64_t left = batch - base;
            block_dim.z = static_cast<unsigned>(left < MAX_GRID_Z ? left : MAX_GRID_Z);
            custom_matmul_kernel<MIXED, Accumulator><<<block_dim, thread_dim, 0, stream>>>(
                a, b, c, dt, M, K, N, trans_a, trans_b, base, stride_a, stride_b,
                acc_proto, use_rng, rng_args,
                pal, prec_idx, idx_row_stride, idx_col_stride, idx_batch_stride);
        }
    }

    // One host body for all eight ops in carrier T. Args names which policy
    // to build (common/gemm_args.h) and whether the op is mixed. The round
    // mode is turned from a runtime value into a template parameter here: a
    // runtime switch inside accumulate() kept seven cast bodies live in the
    // unrolled K-loop and cost 1.2-2.3x of kernel time. Each of the eight
    // .cu files explicitly instantiates this for its two Args and its
    // carrier.
    template <class T, class Args>
    void CudaBackend::launch_as(const GemmShape &s, const Args &args, const LaunchContext &ctx)
    {
        mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
        {
            constexpr RoundMode RM = decltype(rm_c)::value;
            if constexpr (Args::mixed)
            {
                args.template with_palette<T, RM>([&](auto acc, const auto &pal)
                {
                    launch_custom_matmul<true>(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a,
                                               s.trans_b, s.batch, s.stride_a, s.stride_b,
                                               acc, s.use_rng, ctx.rng, ctx.stream,
                                               pal, s.prec_idx, s.idx_row_stride, s.idx_col_stride,
                                               s.idx_batch_stride);
                });
            }
            else
            {
                args.template with_accumulator<T, RM>([&](auto acc)
                {
                    launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                         s.batch, s.stride_a, s.stride_b,
                                         acc, s.use_rng, ctx.rng, ctx.stream);
                });
            }
        });
    }
} // namespace mptorch::gemm_cuda
