#pragma once

// The CUDA GEMM kernel, custom_matmul_kernel, and the four CudaBackend
// templates that launch it, all declared in cuda/gemm_backend.h: launch_as
// for the NAIVE GEMM ops, launch_accumulated_as for the GEMM ops under
// KAHAN, BLOCK and TREE (common/gemm_accumulate.h), launch_conv_as for the
// conv ops and launch_block_as for the block GEMM. Only the instantiation
// units in cuda/inst/ include this header.
//
// This header includes no ATen tensor header, on purpose. Under nvcc a
// translation unit that includes <ATen/core/Tensor.h> pays about 25 s of
// fixed front-end cost before it compiles a single kernel; one that includes
// only the policy headers and the CUDA fp16/bf16 headers pays about 3 s. So
// the kernels see raw pointers and plain structs (GemmShape and an Args from
// common/gemm_args.h), and everything that touches at::Tensor stays in the
// entry files (cuda/custom_*_entry.cpp), which g++ compiles. The only ATen
// headers here are the Philox state unpacker and the Half and BFloat16
// value types, both header-only and cheap.
//
// How the instantiations are divided among the units. Each file in
// cuda/inst/ holds nothing but explicit instantiations of these templates,
// for a few closely related ops, under one accumulate algorithm and in one
// carrier. There are three reasons:
//   - The kernel is expensive to compile, since every policy is instantiated
//     once per round mode, and separate units compile in parallel. When this
//     was measured, one file that held every kernel was the build's longest
//     step (over 100 s of a 111 s build). A format family's single-format
//     and palette ops share a file: the family's cast template, about 150
//     instructions, is then instantiated once for both, and separating them
//     would only repeat the fixed cost of about 3 s per file.
//   - setup.py leaves a group of units out of the build by its file stem:
//     *_f64 under MPTORCH_NO_FP64=1, *_odd and *_odd_f64 under
//     MPTORCH_NO_ODD=1, and *_{kahan,block,tree}_odd{,_f64} unless
//     MPTORCH_ODD_ACCUMULATED=1 is set.
//   - A new op gets units of its own rather than lines in an existing one, so
//     that the existing units' objects stay byte-identical.
// The units of the odd ops (carrier_rounding=RoundMode.RO) also include the
// common/carrier_odd*.h header that defines their Args. Those under KAHAN,
// BLOCK and TREE include common/carrier_odd_accumulate.h, which defines
// AccumulateFamily for their Args and without which they do not compile;
// tests/test_accumulators.py checks that they do.
#include "../common/gemm_accumulate.h"
#include "../common/gemm_args.h"
#include "../common/gemm_block.h"
#include "../common/gemm_gather.h"
#include "gemm_backend.h"
#include <ATen/cuda/PhiloxUtils.cuh>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace mptorch::gemm_cuda
{
    using mptorch::gemm::GemmShape;

    // A tiled GEMM with double-buffered shared memory. Each thread block
    // computes one BLOCKSIZE x BLOCKSIZE tile of C with BLOCKSIZE^2 threads,
    // and each thread computes exactly one output element, its whole
    // reduction over K, through an Accumulator policy (common/gemm_policy.h),
    // so that the format and the rounding are the policy's and not the
    // kernel's. For each BLOCKSIZE steps of K the block stages one tile of A
    // and one of B in shared memory; with two buffers, the next tiles' global
    // loads overlap the current tiles' arithmetic. One element per thread
    // measured slightly faster than a register-blocked variant that computes
    // several outputs per thread.
    constexpr int BLOCKSIZE = 16;

    // The K-loop's unroll factor for every binary64 kernel (the K-loop's
    // comment has the measurements). It is a macro only so that
    // dev/benchmarks/timing/gemm_f64_unroll.cu can time the shipped kernels at
    // other factors; nothing in the build defines it.
#if !defined(MPTORCH_GEMM_F64_UNROLL)
#define MPTORCH_GEMM_F64_UNROLL 2
#endif

    // The K-loop's unroll factor for a single-format binary32 kernel: a
    // positional accumulator's own factor, and 16 for a NaiveAccumulator (the
    // K-loop's comment has the measurements).
    template <class Accumulator>
    constexpr int unroll_of()
    {
        if constexpr (is_positional_accumulator_v<Accumulator>)
            return Accumulator::unroll;
        else
            return 16;
    }

    // Element load and store by storage dtype. The dtype is a kernel
    // argument (mptorch::GemmDtype, common/gemm_dtype.h), not a template
    // parameter: the storage type decides nothing but these two conversions,
    // and instantiating the kernel per dtype would triple the cast bodies,
    // the unrolled K-loop, the compile time and the binary, for the sake of
    // two loads per BLOCKSIZE steps of K and one store per output element.
    // Every thread of a warp takes the same arm of the switch, and the switch
    // sits next to the global load, whose latency hides it.
    //
    // T is the carrier the kernel computes in (the float type of its
    // arithmetic). The primary template serves binary32 and converts any of
    // its three storage dtypes to float; binary64 has one storage dtype,
    // float64, so its specializations are a plain load and a plain store.
    // They are specializations rather than an `if constexpr` around the
    // switch so that the binary32 bodies compile to the same code whether or
    // not the binary64 kernels are built.
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
    // element's offset into the operand, in elements: 0 for a 2D call and
    // for an operand broadcast across the batch (stride 0).
    template <class T>
    __device__ __forceinline__ T load_a(const void *A, mptorch::GemmDtype dt,
                                        int64_t M, int64_t K, bool trans_a,
                                        int64_t row, int64_t col, int64_t off)
    {
        // Logical A is M x K; under trans_a, A is stored as K x M.
        return load_elem<T>(A, off + (trans_a ? col * M + row : row * K + col), dt);
    }

    template <class T>
    __device__ __forceinline__ T load_b(const void *B, mptorch::GemmDtype dt,
                                        int64_t K, int64_t N, bool trans_b,
                                        int64_t row, int64_t col, int64_t off)
    {
        // Logical B is K x N; under trans_b, B is stored as N x K.
        return load_elem<T>(B, off + (trans_b ? col * K + row : row * N + col), dt);
    }

    // What a thread of a gathered policy (the conv ops; common/gemm_gather.h)
    // keeps across the K-loop: its batch element as (sample, group), its
    // output row's part of A's offsets, its output column's part of B's, and
    // the result column it stores into. That column is its own column except
    // in the input gradient, whose GEMMs each cover one residue class of the
    // stride. A policy that is not gathered keeps nothing (NoConvThread).
    struct ConvThread
    {
        int64_t b;
        int32_t grp;
        int64_t a_row;
        mptorch::gemm::ConvCol col;
        int64_t res_col;
    };

    struct NoConvThread
    {
    };

    template <bool GATHERED, class Accumulator>
    __device__ __forceinline__ std::conditional_t<GATHERED, ConvThread, NoConvThread>
    conv_thread(const Accumulator &acc_proto, int64_t bId, int64_t rId, int64_t cId)
    {
        if constexpr (!GATHERED)
        {
            return NoConvThread{};
        }
        else
        {
            const mptorch::gemm::ConvGeom &g = acc_proto.geom;
            ConvThread ct;
            mptorch::gemm::conv_batch(g, bId, ct.b, ct.grp);
            ct.a_row = mptorch::gemm::conv_a_row(g, ct.grp, rId);
            ct.col = mptorch::gemm::conv_b_col(g, ct.b, ct.grp, static_cast<int32_t>(cId));
            ct.res_col = mptorch::gemm::conv_result_col(g, static_cast<int32_t>(cId));
            return ct;
        }
    }

    // A gathered policy's element (row, k) of A and (k, col) of B. The
    // element of B is zero where it falls in the convolution's padding.
    template <class T>
    __device__ __forceinline__ T gather_a(const void *A, mptorch::GemmDtype dt,
                                          const mptorch::gemm::ConvGeom &g, const ConvThread &ct,
                                          int64_t k)
    {
        return load_elem<T>(A, ct.a_row + mptorch::gemm::conv_a_k(g, static_cast<int32_t>(k)), dt);
    }

    template <class T>
    __device__ __forceinline__ T gather_b(const void *B, mptorch::GemmDtype dt,
                                          const mptorch::gemm::ConvGeom &g, const ConvThread &ct,
                                          int64_t k)
    {
        const int64_t off = mptorch::gemm::conv_b_offset(
            g, ct.col, mptorch::gemm::conv_b_k(g, static_cast<int32_t>(k)));
        return off < 0 ? T(0) : load_elem<T>(B, off, dt);
    }

    // A blocked policy's decode tables (the block GEMM; filled by
    // common/gemm_block.h's fill_block_table), A's and then B's, in shared
    // memory, which the thread block fills before its first tile load. Any
    // other policy has none (NoBlockTables) and compiles to the code it would
    // without them.
    struct NoBlockTables
    {
    };

    template <bool BLOCKED, class Accumulator>
    __device__ __forceinline__ std::conditional_t<BLOCKED, const typename Accumulator::value_t *, NoBlockTables>
    block_tables(const Accumulator &acc_proto)
    {
        if constexpr (!BLOCKED)
        {
            return NoBlockTables{};
        }
        else
        {
            using T = typename Accumulator::value_t;
            constexpr int N = mptorch::gemm::BLOCK_TABLE;
            __shared__ T tables[4 * N];
            for (int i = threadIdx.x; i < N; i += blockDim.x)
            {
                mptorch::gemm::fill_block_table<T>(tables, i, acc_proto.block_a.p);
                mptorch::gemm::fill_block_table<T>(tables + 2 * N, i, acc_proto.block_b.p);
            }
            __syncthreads();
            return tables;
        }
    }

    // The kernel. A is logically [M, K] and B [K, N] (a trans flag says the
    // operand is stored transposed); C is written densely as [batch, M, N].
    // The batch element is batch_base + blockIdx.z, and stride_a and stride_b
    // are the element offsets between consecutive batch elements, 0 meaning
    // the operand is broadcast. acc_proto is the fully built Accumulator that
    // every thread copies; rng_args is read only under use_rng (RoundMode::SR).
    // On the palette path, pal holds up to MAX_GEMM_FORMATS Mac policies and
    // prec_idx picks one per output element through the three strides (0
    // along a dimension the index does not vary along).
    //
    // MIXED selects whether this instantiation has the palette lookup before
    // the K-loop. It is a template parameter rather than a runtime
    // `pal.n > 0` test because a merely possible write to acc.mac forces the
    // Mac policy's format constants into registers for the whole K-loop,
    // where they would otherwise be read again from the constant bank that
    // holds acc_proto. That cost the split-mac kernels about 29 extra
    // registers (109 against 80 at BLOCKSIZE^2 threads) and roughly halved
    // their throughput (68.3 ms against 34.7 ms at 1024^3 on sm_89) on every
    // single-format launch, which is nearly all of them. Two instantiations
    // double the kernel count, and the single-format path keeps its speed.
    // Measured on an RTX 4060 Laptop GPU, with a copy of this kernel with and
    // without the lookup compiled in.
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
        // The carrier: every operand is loaded into it, every step is
        // computed in it, and the output is stored from it. The two double
        // buffers take 4 KB of shared memory in binary32 and 8 KB in binary64.
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

        // The batch element is blockIdx.z, offset by batch_base because the
        // launcher splits a batch larger than the 65,535 limit on gridDim.z
        // into several launches. The two operand offsets are computed once and
        // added to every index: stride 0 broadcasts an operand across the
        // batch, and a 2D call has batch element 0 and both strides 0, so it
        // computes exactly the addresses an unbatched kernel would. The
        // offsets stay live across the K-loop and cost 2 to 11 registers; a
        // separate unbatched instantiation would gain back at most 3.6%, for
        // 86 more kernels and 4.7 MB more fatbin, so there is none.
        const int64_t bId = batch_base + static_cast<int64_t>(blockIdx.z);
        const int64_t aOff = bId * stride_a;
        const int64_t bOff = bId * stride_b;

        // A gathered policy (the conv ops; common/gemm_gather.h) reads its
        // operands through the convolution's geometry instead of the two
        // offsets above (it is launched with both strides 0), and keeps its
        // thread's part of the addresses here. Every use is behind
        // `if constexpr`, and for any other policy this is an empty object,
        // so the other ops' device functions compile to the bytes they would
        // without it.
        constexpr bool GATHERED = mptorch::gemm::is_gathered_v<Accumulator>;
        [[maybe_unused]] const auto ct = conv_thread<GATHERED>(acc_proto, bId, rId, cId);

        // A blocked policy (the block GEMM; common/gemm_block.h) decodes its
        // block-format operands in the tile loads, at the two offsets above,
        // which are in bytes for it. Its branches come after the dense ones,
        // behind `if constexpr`, for the same reason as the gathered ones.
        constexpr bool BLOCKED = mptorch::gemm::is_blocked_v<Accumulator>;
        [[maybe_unused]] const auto blk_tables = block_tables<BLOCKED>(acc_proto);

        Accumulator acc = acc_proto;

        // RoundMode::SR: seed this output element's Philox stream once,
        // before the K-loop. The subsequence is the element's linear index
        // into the whole [batch, M, N] result, so the draws it sees depend
        // only on which element it is: not on BLOCKSIZE, the grid, how the
        // batch was split into launches, or whether the product was written
        // as a batched call or as a 2D call with the batch merged into M.
        // Batch element 0 gets the subsequence the 2D call gives, which makes
        // the two calls bit-identical. A stream shared by a block's threads
        // would make the results depend on the launch geometry.
        if (use_rng && rId < M && cId < N)
        {
            auto seeds = at::cuda::philox::unpack(rng_args);
            acc.seed_rng(std::get<0>(seeds),
                         (static_cast<uint64_t>(bId) * M + rId) * N + cId, std::get<1>(seeds));
        }

        // Palette: set this output element's Mac policy from the palette
        // before the K-loop, so that the loop itself is the same code as on
        // the single-format path. When MIXED is false this is compiled out,
        // and the instantiation does not even take the palette argument.
        // There is no test for an empty palette: the palette packers require
        // at least one format.
        if constexpr (MIXED)
        {
            if constexpr (!GATHERED)
            {
                if (rId < M && cId < N)
                    acc.mac = pal.slot(prec_idx[bId * idx_batch_stride + rId * idx_row_stride +
                                                cId * idx_col_stride]);
            }
            else
            {
                // A conv op's format map is indexed by the result tensor's
                // (sample, channel, position), and this thread's row is a
                // channel of one group.
                if (rId < M && cId < N)
                    acc.mac = pal.slot(prec_idx[ct.b * idx_batch_stride +
                                                mptorch::gemm::conv_result_row(ct.grp, M, rId) * idx_row_stride +
                                                ct.res_col * idx_col_stride]);
            }
        }

        // Stage the first tiles of A and B in buffer 0; a thread whose
        // element lies outside the operand loads 0.
        if constexpr (!GATHERED && !BLOCKED)
        {
            As[0][threadRow * BLOCKSIZE + threadCol] =
                (rId < M && threadCol < K) ? load_a<T>(A, dt, M, K, trans_a, rId, threadCol, aOff) : 0.0f;
            Bs[0][threadRow * BLOCKSIZE + threadCol] =
                (threadRow < K && cId < N) ? load_b<T>(B, dt, K, N, trans_b, threadRow, cId, bOff) : 0.0f;
        }
        else if constexpr (BLOCKED)
        {
            As[0][threadRow * BLOCKSIZE + threadCol] =
                (rId < M && threadCol < K)
                    ? mptorch::gemm::block_load_table<T>(A, acc_proto.block_a, blk_tables, bId, aOff, rId, threadCol)
                    : T(0);
            Bs[0][threadRow * BLOCKSIZE + threadCol] =
                (threadRow < K && cId < N)
                    ? mptorch::gemm::block_load_table<T>(B, acc_proto.block_b, blk_tables + 2 * mptorch::gemm::BLOCK_TABLE,
                                                         bId, bOff, cId, threadRow)
                    : T(0);
        }
        else
        {
            const mptorch::gemm::ConvGeom &g = acc_proto.geom;
            As[0][threadRow * BLOCKSIZE + threadCol] =
                (rId < M && threadCol < K) ? gather_a<T>(A, dt, g, ct, threadCol) : T(0);
            Bs[0][threadRow * BLOCKSIZE + threadCol] =
                (threadRow < K && cId < N) ? gather_b<T>(B, dt, g, ct, threadRow) : T(0);
        }

        for (int64_t t = 0; t < numTiles; ++t)
        {
            __syncthreads();

            T *curAs = As[t % 2];
            T *curBs = Bs[t % 2];

            // The unroll factor trades the cast bodies' size against the
            // instruction cache and the register file. With the rounding mode
            // a template parameter, only one cast body is live per step, so
            // the full unroll (16) fits, and it is 3-8% faster on every
            // binary32 single-format kernel (binaryK fma RNE at 1024^3:
            // 4.31 ms at 4, 3.97 ms at 16). The palette kernels hold an extra
            // Mac per output element plus the palette, run out of registers
            // sooner, and are 1.3-1.4x slower at 16, so they stay at 4. The
            // full unroll costs about 2.8 MB of fatbin over the extension.
            // The binary64 kernels are limited by the device's FP64 rate
            // rather than by instruction fetch, so their time is flat within
            // 1% for factors from 2 to 8; 2 has the fewest spills and the
            // smallest binary. Timed on sm_89; binary64 with
            // dev/benchmarks/timing/gemm_f64_unroll.cu.
            //
            // A positional accumulator (KAHAN, BLOCK or TREE;
            // common/gemm_accumulate.h) names its own factor, is given each
            // step's position in its slab (16 consecutive steps of K, one
            // tile), and is told when a slab ends. It does not run the steps
            // past K that the zero-loaded elements pad the last slab with:
            // those are free for a NAIVE sum, but not for a compensated one,
            // and under SR each would spend a draw, so the device's results
            // would no longer match the CPU's. All of it is behind
            // `if constexpr`, so a NaiveAccumulator's kernel compiles to the
            // code it would without it.
            constexpr bool POSITIONAL = is_positional_accumulator_v<Accumulator>;
            constexpr int UNROLL = std::is_same_v<T, double> ? MPTORCH_GEMM_F64_UNROLL
                                   : MIXED                   ? 4
                                                             : unroll_of<Accumulator>();
            if constexpr (POSITIONAL)
            {
                // A whole slab runs the unrolled loop, with no test per step;
                // the last, partial slab runs a plain loop over the steps it
                // has. A `k < K` test inside one loop instead cost 5-15% on
                // every kernel (binaryK RNE on sm_89).
                const int64_t left = K - t * BLOCKSIZE;
                if (left >= BLOCKSIZE)
                {
#pragma unroll UNROLL
                    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
                        acc.accumulate(curAs[threadRow * BLOCKSIZE + dotIdx],
                                       curBs[dotIdx * BLOCKSIZE + threadCol], dotIdx);
                }
                else
                {
#pragma unroll 1
                    for (int dotIdx = 0; dotIdx < static_cast<int>(left); ++dotIdx)
                        acc.accumulate(curAs[threadRow * BLOCKSIZE + dotIdx],
                                       curBs[dotIdx * BLOCKSIZE + threadCol], dotIdx);
                }
                acc.end_slab(left >= BLOCKSIZE ? (t + 1) * BLOCKSIZE : K, K);
            }
            else
            {
#pragma unroll UNROLL
                for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
                {
                    acc.accumulate(curAs[threadRow * BLOCKSIZE + dotIdx], curBs[dotIdx * BLOCKSIZE + threadCol]);
                }
            }

            // Load the next tiles into the other buffer; the barrier below
            // makes them visible to every thread before the next iteration
            // reads them.
            if (t + 1 < numTiles)
            {
                int64_t nextK = (t + 1) * BLOCKSIZE;
                if constexpr (!GATHERED && !BLOCKED)
                {
                    As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                        (rId < M && nextK + threadCol < K) ? load_a<T>(A, dt, M, K, trans_a, rId, nextK + threadCol, aOff) : 0.0f;
                    Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                        (nextK + threadRow < K && cId < N) ? load_b<T>(B, dt, K, N, trans_b, nextK + threadRow, cId, bOff) : 0.0f;
                }
                else if constexpr (BLOCKED)
                {
                    As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                        (rId < M && nextK + threadCol < K)
                            ? mptorch::gemm::block_load_table<T>(A, acc_proto.block_a, blk_tables, bId, aOff, rId,
                                                                 nextK + threadCol)
                            : T(0);
                    Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                        (nextK + threadRow < K && cId < N)
                            ? mptorch::gemm::block_load_table<T>(B, acc_proto.block_b,
                                                                 blk_tables + 2 * mptorch::gemm::BLOCK_TABLE, bId,
                                                                 bOff, cId, nextK + threadRow)
                            : T(0);
                }
                else
                {
                    const mptorch::gemm::ConvGeom &g = acc_proto.geom;
                    As[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                        (rId < M && nextK + threadCol < K) ? gather_a<T>(A, dt, g, ct, nextK + threadCol) : T(0);
                    Bs[(t + 1) % 2][threadRow * BLOCKSIZE + threadCol] =
                        (nextK + threadRow < K && cId < N) ? gather_b<T>(B, dt, g, ct, nextK + threadRow) : T(0);
                }
            }
            __syncthreads();
        }

        // C is a dense [batch, M, N], so its strides need no argument. A
        // gathered policy's result is [batch, M, res_N]: this GEMM's columns
        // are all of its columns (res_N = N), or, in the input gradient, the
        // columns of one residue class of the stride.
        //
        // The fixed-point GEMM's integer accumulator (common/fixedpoint_gemm.h)
        // stores float64 when its format is wider than the operands' dtype
        // (`wide`), whatever the carrier. The condition is has_int_result_v,
        // spelled here so that this header needs no include for it. Every
        // other accumulator takes the first branch, which is the code from
        // before.
        if constexpr (!requires { Accumulator::int_result; })
        {
            if constexpr (!GATHERED)
            {
                if (rId < M && cId < N)
                    store_elem<T>(C, (bId * M + rId) * N + cId, dt, acc.finalize());
            }
            else
            {
                if (rId < M && cId < N)
                    store_elem<T>(C, (bId * M + rId) * acc_proto.geom.res_N + ct.res_col, dt,
                                  acc.finalize());
            }
        }
        else
        {
            // a conv op's result is gathered, as the first branch stores it
            static_assert(!MIXED, "the fixed-point GEMM has no palette ops");
            if (rId < M && cId < N)
            {
                int64_t out;
                if constexpr (!GATHERED)
                    out = (bId * M + rId) * N + cId;
                else
                    out = (bId * M + rId) * acc_proto.geom.res_N + ct.res_col;
                if (acc.mac.wide)
                    static_cast<double *>(C)[out] = acc.finalize_wide();
                else
                    store_elem<T>(C, out, dt, acc.finalize());
            }
        }
    }

    // Launch the kernel over a grid of output tiles for each batch element.
    // The stream and the Philox state come from the entry point:
    // getCurrentCUDAStream() is declared in <ATen/cuda/CUDAContext.h>, which
    // this header must not include (see the top of the file).
    //
    // The batch is gridDim.z, whose extent is capped at 65,535, far below the
    // 2^31 - 1 of gridDim.x and reachable in practice (a per-head attention
    // call has batch * heads elements). A larger batch is split into
    // successive launches, each with a larger batch_base. Splitting here
    // rather than merging the batch into gridDim.y leaves each block's
    // (row, col) mapping and shared-memory tiling unchanged, and the split is
    // invisible in the output: each block writes its own tile of its own
    // batch element, and RoundMode::SR keys each stream on the element's
    // index, not on the launch.
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

    // One host body for every NAIVE GEMM op, in carrier T. Args names the
    // policy to build (common/gemm_args.h) and whether the op takes a
    // palette. dispatch_round_mode turns the round mode from a runtime value
    // into a template parameter here: a runtime switch inside accumulate()
    // kept seven cast bodies live in the unrolled K-loop and cost 1.2-2.3x of
    // kernel time. Explicitly instantiated in cuda/inst/.
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

    // The same body for the GEMM ops under KAHAN, BLOCK and TREE: the
    // algorithm ALG is a template parameter, and args builds the accumulator
    // for it. Instantiated by the cuda/inst/custom_matmul_*_{kahan,block,tree}*
    // units.
    template <class T, AccumulateAlgorithm ALG, class Base>
    void CudaBackend::launch_accumulated_as(const GemmShape &s,
                                            const mptorch::gemm::AccumulateArgs<Base> &args,
                                            const LaunchContext &ctx)
    {
        mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
        {
            constexpr RoundMode RM = decltype(rm_c)::value;
            args.template with_accumulator<T, RM, ALG>([&](auto acc)
            {
                launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                     s.batch, s.stride_a, s.stride_b,
                                     acc, s.use_rng, ctx.rng, ctx.stream);
            });
        });
    }

    // The conv ops (common/gemm_gather.h): the accumulator that `Inner`
    // builds, as its GEMM op builds it, wrapped in a Gathered that carries the
    // geometry, and launched over the pass's [batch, M, N] with no transposes
    // and no batch strides, since the gathered loads address the operands
    // themselves. Instantiated by the cuda/inst/custom_conv_* units.
    template <class T, AccumulateAlgorithm ALG, class Inner>
    void CudaBackend::launch_conv_as(const GemmShape &s, const mptorch::gemm::ConvArgs<Inner> &args,
                                     const LaunchContext &ctx)
    {
        using mptorch::gemm::Gathered;
        mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
        {
            constexpr RoundMode RM = decltype(rm_c)::value;
            if constexpr (Inner::mixed)
            {
                args.inner.template with_palette<T, RM>([&](auto acc, const auto &pal)
                {
                    launch_custom_matmul<true>(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, false, false,
                                               s.batch, 0, 0, Gathered<decltype(acc)>{acc, args.geom},
                                               s.use_rng, ctx.rng, ctx.stream, pal, s.prec_idx,
                                               s.idx_row_stride, s.idx_col_stride, s.idx_batch_stride);
                });
            }
            else if constexpr (ALG == AccumulateAlgorithm::NAIVE)
            {
                args.inner.template with_accumulator<T, RM>([&](auto acc)
                {
                    launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, false, false, s.batch, 0, 0,
                                         Gathered<decltype(acc)>{acc, args.geom}, s.use_rng, ctx.rng,
                                         ctx.stream);
                });
            }
            else
            {
                args.inner.template with_accumulator<T, RM, ALG>([&](auto acc)
                {
                    launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, false, false, s.batch, 0, 0,
                                         Gathered<decltype(acc)>{acc, args.geom}, s.use_rng, ctx.rng,
                                         ctx.stream);
                });
            }
        });
    }

    // The block GEMM (common/gemm_block.h): the accumulator that `Inner`
    // builds, wrapped in a Blocked that carries the two block-format
    // operands, and launched over [batch, M, N] with the operands' own trans
    // flags and their codes' batch strides in bytes. Instantiated by the
    // cuda/inst/custom_matmul_block* units.
    template <class T, AccumulateAlgorithm ALG, class Inner>
    void CudaBackend::launch_block_as(const GemmShape &s, const mptorch::gemm::BlockGemmArgs<Inner> &args,
                                      const LaunchContext &ctx)
    {
        using mptorch::gemm::Blocked;
        mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
        {
            constexpr RoundMode RM = decltype(rm_c)::value;
            if constexpr (ALG == AccumulateAlgorithm::NAIVE)
            {
                args.inner.template with_accumulator<T, RM>([&](auto acc)
                {
                    launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b, s.batch,
                                         s.stride_a, s.stride_b,
                                         Blocked<decltype(acc)>{acc, args.template operand_a<T>(), args.template operand_b<T>()},
                                         s.use_rng, ctx.rng, ctx.stream);
                });
            }
            else
            {
                args.inner.template with_accumulator<T, RM, ALG>([&](auto acc)
                {
                    launch_custom_matmul(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b, s.batch,
                                         s.stride_a, s.stride_b,
                                         Blocked<decltype(acc)>{acc, args.template operand_a<T>(), args.template operand_b<T>()},
                                         s.use_rng, ctx.rng, ctx.stream);
                });
            }
        });
    }
} // namespace mptorch::gemm_cuda
