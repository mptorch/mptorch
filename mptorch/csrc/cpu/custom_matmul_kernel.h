#pragma once

// The CPU GEMM kernel, matmul_cpu_kernel_impl, and the four CpuBackend
// templates that run it, all declared in cpu/gemm_backend.h: launch_as for
// the NAIVE GEMM ops, launch_accumulated_as for the GEMM ops under KAHAN,
// BLOCK and TREE (common/gemm_accumulate.h), launch_conv_as for the conv ops
// and launch_block_as for the block GEMM. Only the instantiation units in
// cpu/inst/ include this header, the odd ones through
// cpu/custom_matmul_odd_kernel.h. The entry points (cpu/custom_*_entry.cpp)
// see only the declarations, so they compile no kernel.
//
// How the instantiations are divided among the units. Each file in cpu/inst/
// holds nothing but explicit instantiations of these templates, for a few
// closely related ops, under one accumulate algorithm and in one carrier.
// There are three reasons:
//   - The kernel is expensive to compile, since every policy is instantiated
//     once per round mode, and separate units compile in parallel. When this
//     was measured, all of the instantiations in one object took about 67 s
//     and were the build's critical path; spread over eight objects, the
//     largest took about 31 s.
//   - setup.py leaves a group of units out of the build by its file stem:
//     *_f64 under MPTORCH_NO_FP64=1, *_odd and *_odd_f64 under
//     MPTORCH_NO_ODD=1, and *_{kahan,block,tree}_odd{,_f64} unless
//     MPTORCH_ODD_ACCUMULATED=1 is set.
//   - A new op gets units of its own rather than lines in an existing one, so
//     that the existing units' objects stay byte-identical.
// The units of the odd ops (carrier_rounding=RoundMode.RO) include
// cpu/custom_matmul_odd_kernel.h instead of this header, for the reason that
// header gives. Those under KAHAN, BLOCK and TREE also include
// common/carrier_odd_accumulate.h, which defines AccumulateFamily for their
// Args and without which they do not compile. tests/test_accumulators.py
// checks both includes in those units.

#include "../common/gemm_accumulate.h"
#include "../common/gemm_args.h"
#include "../common/gemm_block.h"
#include "../common/gemm_gather.h"
#include "../common/gemm_policy.h"
#include "../common/modes.h"
#include "gemm_backend.h"
#include "gemm_rne_tile.h"
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h> // defines at::internal::GRAIN_SIZE
#include <algorithm>

namespace mptorch::gemm_cpu
{
  using mptorch::gemm::GemmShape;

  // Tile packing. The operands' storage dtype is a runtime tag
  // (mptorch::GemmDtype, common/gemm_dtype.h) rather than a template
  // parameter, so the kernel body is instantiated once per carrier (the
  // float type the arithmetic runs in) instead of once per dtype. On the GPU
  // each load switches on the tag; here a switch in the innermost loop would
  // cost a branch per multiply-add, so the kernel does what a BLAS does: it
  // converts each 32 x 32 operand tile into a buffer of the carrier type once
  // (it "packs" the tile), and the K-loop reads only those buffers. This has
  // three further effects, all intended. Each element is converted once per
  // tile rather than once per K-step of every row (on the host a c10::Half
  // to float conversion is a function call, not an instruction). Both
  // operands are read with stride 1 in the innermost loop whatever trans_a
  // and trans_b say, since the pack's index arithmetic applies the
  // transposition. And only these small routines are templated on the
  // storage dtype. Packing changes neither the values nor the order of the
  // Accumulator::accumulate calls, so the results are bit-identical to those
  // of an unpacked loop.
  //
  // pack_a_impl copies the ti x tk block of op(A) at (i0, k0) into dst, row
  // major with row stride tk; pack_b_impl copies the tk x tj block of op(B)
  // at (k0, j0) into dst with row stride tj; store_tile_impl writes the
  // ti x tj block of C at (i0, j0) from src, converting back to the storage
  // dtype. `off` is the batch element's offset into the operand, in
  // elements: 0 for a 2D call and for an operand broadcast across the batch.
  // It is added to the base pointer once per tile, not once per element. T
  // is the carrier the tile is packed in (common/gemm_policy.h): float for
  // float32, float16 and bfloat16 operands, double for float64, whose pack
  // is a plain copy.
  template <typename scalar_t, class T>
  void pack_a_impl(const void *A, T *__restrict__ dst, int64_t M, int64_t K, bool trans_a,
                   int64_t i0, int64_t k0, int64_t ti, int64_t tk, int64_t off)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(A) + off;
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t k = 0; k < tk; ++k)
        dst[i * tk + k] = trans_a ? static_cast<T>(src[(k0 + k) * M + (i0 + i)])
                                  : static_cast<T>(src[(i0 + i) * K + (k0 + k)]);
  }

  template <typename scalar_t, class T>
  void pack_b_impl(const void *B, T *__restrict__ dst, int64_t K, int64_t N, bool trans_b,
                   int64_t k0, int64_t j0, int64_t tk, int64_t tj, int64_t off)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(B) + off;
    for (int64_t k = 0; k < tk; ++k)
      for (int64_t j = 0; j < tj; ++j)
        dst[k * tj + j] = trans_b ? static_cast<T>(src[(j0 + j) * K + (k0 + k)])
                                  : static_cast<T>(src[(k0 + k) * N + (j0 + j)]);
  }

  template <typename scalar_t, class T>
  void store_tile_impl(void *C, const T *__restrict__ src, int64_t N,
                       int64_t i0, int64_t j0, int64_t ti, int64_t tj, int64_t off)
  {
    scalar_t *__restrict__ dst = static_cast<scalar_t *>(C) + off;
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t j = 0; j < tj; ++j)
        dst[(i0 + i) * N + (j0 + j)] = static_cast<scalar_t>(src[i * tj + j]);
  }

  // Expands CALL(scalar_t) for the storage dtype that DT names. It runs once
  // per tile per operand and never inside the K-loop, so the switch costs
  // nothing per multiply-add.
#define MPTORCH_GEMM_BY_DTYPE(DT, CALL)      \
  switch (DT)                                \
  {                                          \
  case mptorch::GemmDtype::Half:             \
    CALL(at::Half);                          \
    break;                                   \
  case mptorch::GemmDtype::BFloat16:         \
    CALL(at::BFloat16);                      \
    break;                                   \
  default:                                   \
    CALL(float);                             \
    break;                                   \
  }

  // Pick the packing template for the runtime dtype tag. These primary
  // templates serve the binary32 carrier, whose operands may be float32,
  // float16 or bfloat16. The double specializations further down skip the
  // switch, since the binary64 carrier has one storage dtype, float64.
  template <class T>
  inline void pack_a(const void *A, mptorch::GemmDtype dt, T *dst, int64_t M, int64_t K,
                     bool trans_a, int64_t i0, int64_t k0, int64_t ti, int64_t tk, int64_t off)
  {
#define MPTORCH_PACK_A(S) pack_a_impl<S, T>(A, dst, M, K, trans_a, i0, k0, ti, tk, off)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_A)
#undef MPTORCH_PACK_A
  }

  template <class T>
  inline void pack_b(const void *B, mptorch::GemmDtype dt, T *dst, int64_t K, int64_t N,
                     bool trans_b, int64_t k0, int64_t j0, int64_t tk, int64_t tj, int64_t off)
  {
#define MPTORCH_PACK_B(S) pack_b_impl<S, T>(B, dst, K, N, trans_b, k0, j0, tk, tj, off)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_B)
#undef MPTORCH_PACK_B
  }

  template <class T>
  inline void store_tile(void *C, mptorch::GemmDtype dt, const T *src, int64_t N,
                         int64_t i0, int64_t j0, int64_t ti, int64_t tj, int64_t off)
  {
#define MPTORCH_STORE_TILE(S) store_tile_impl<S, T>(C, src, N, i0, j0, ti, tj, off)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_STORE_TILE)
#undef MPTORCH_STORE_TILE
  }

  // The conv ops' packs (common/gemm_gather.h): the same tiles, read
  // straight from the convolution's tensors through its geometry, so nothing
  // is unfolded in memory. Each pack decomposes its tile's K indices and its
  // rows or columns once, into local arrays, after which an element costs
  // only additions and, in pack_gather_b_impl, the padding test. A tile is
  // at most GATHER_TILE on a side, which sizes those arrays; the kernel's TK
  // and TJ must not exceed it.
  constexpr int64_t GATHER_TILE = 32;

  template <typename scalar_t, class T>
  void pack_gather_a_impl(const void *A, T *__restrict__ dst, const mptorch::gemm::ConvGeom &g,
                          int64_t bId, int64_t i0, int64_t k0, int64_t ti, int64_t tk)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(A);
    int64_t b;
    int32_t grp;
    mptorch::gemm::conv_batch(g, bId, b, grp);
    int64_t k_part[GATHER_TILE];
    for (int64_t k = 0; k < tk; ++k)
      k_part[k] = mptorch::gemm::conv_a_k(g, static_cast<int32_t>(k0 + k));
    for (int64_t i = 0; i < ti; ++i)
    {
      const int64_t row = mptorch::gemm::conv_a_row(g, grp, i0 + i);
      for (int64_t k = 0; k < tk; ++k)
        dst[i * tk + k] = static_cast<T>(src[row + k_part[k]]);
    }
  }

  template <typename scalar_t, class T>
  void pack_gather_b_impl(const void *B, T *__restrict__ dst, const mptorch::gemm::ConvGeom &g,
                          int64_t bId, int64_t k0, int64_t j0, int64_t tk, int64_t tj)
  {
    const scalar_t *__restrict__ src = static_cast<const scalar_t *>(B);
    int64_t b;
    int32_t grp;
    mptorch::gemm::conv_batch(g, bId, b, grp);
    mptorch::gemm::ConvCol cols[GATHER_TILE];
    for (int64_t j = 0; j < tj; ++j)
      cols[j] = mptorch::gemm::conv_b_col(g, b, grp, static_cast<int32_t>(j0 + j));
    for (int64_t k = 0; k < tk; ++k)
    {
      const mptorch::gemm::ConvK kk = mptorch::gemm::conv_b_k(g, static_cast<int32_t>(k0 + k));
      for (int64_t j = 0; j < tj; ++j)
      {
        const int64_t off = mptorch::gemm::conv_b_offset(g, cols[j], kk);
        dst[k * tj + j] = off < 0 ? T(0) : static_cast<T>(src[off]);
      }
    }
  }

  // Store a gathered policy's result tile: row i0 + i of batch element bId
  // goes to row [bId, i0 + i] of a [batch, M, res_N] result, and column
  // j0 + j to the column conv_result_col maps it to, which is j0 + j itself
  // except in the GEMMs the input gradient runs per residue class of the
  // stride (common/gemm_gather.h).
  template <typename scalar_t, class T>
  void store_gather_tile_impl(void *C, const T *__restrict__ src, const mptorch::gemm::ConvGeom &g,
                              int64_t bId, int64_t M, int64_t i0, int64_t j0, int64_t ti, int64_t tj)
  {
    scalar_t *__restrict__ dst = static_cast<scalar_t *>(C);
    int64_t cols[GATHER_TILE];
    for (int64_t j = 0; j < tj; ++j)
      cols[j] = mptorch::gemm::conv_result_col(g, static_cast<int32_t>(j0 + j));
    for (int64_t i = 0; i < ti; ++i)
    {
      scalar_t *__restrict__ row = dst + (bId * M + i0 + i) * g.res_N;
      for (int64_t j = 0; j < tj; ++j)
        row[cols[j]] = static_cast<scalar_t>(src[i * tj + j]);
    }
  }

  template <class T>
  inline void store_gather_tile(void *C, mptorch::GemmDtype dt, const T *src,
                                const mptorch::gemm::ConvGeom &g, int64_t bId, int64_t M,
                                int64_t i0, int64_t j0, int64_t ti, int64_t tj)
  {
#define MPTORCH_STORE_GATHER_TILE(S) store_gather_tile_impl<S, T>(C, src, g, bId, M, i0, j0, ti, tj)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_STORE_GATHER_TILE)
#undef MPTORCH_STORE_GATHER_TILE
  }

  template <class T>
  inline void pack_gather_a(const void *A, mptorch::GemmDtype dt, T *dst,
                            const mptorch::gemm::ConvGeom &g, int64_t bId, int64_t i0, int64_t k0,
                            int64_t ti, int64_t tk)
  {
#define MPTORCH_PACK_GATHER_A(S) pack_gather_a_impl<S, T>(A, dst, g, bId, i0, k0, ti, tk)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_GATHER_A)
#undef MPTORCH_PACK_GATHER_A
  }

  template <class T>
  inline void pack_gather_b(const void *B, mptorch::GemmDtype dt, T *dst,
                            const mptorch::gemm::ConvGeom &g, int64_t bId, int64_t k0, int64_t j0,
                            int64_t tk, int64_t tj)
  {
#define MPTORCH_PACK_GATHER_B(S) pack_gather_b_impl<S, T>(B, dst, g, bId, k0, j0, tk, tj)
    MPTORCH_GEMM_BY_DTYPE(dt, MPTORCH_PACK_GATHER_B)
#undef MPTORCH_PACK_GATHER_B
  }

#undef MPTORCH_GEMM_BY_DTYPE

  // The block GEMM's packs (common/gemm_block.h): the same tiles, decoded
  // into the carrier from a block-format operand (uint8 codes and their
  // scales). Like every pack, each decodes an element once per tile. `off`
  // is the batch element's offset into the codes, in bytes.
  template <class T>
  void pack_block_a(const void *A, T *__restrict__ dst, const mptorch::gemm::BlockOperandT<T> &o,
                    int64_t bId, int64_t off, int64_t i0, int64_t k0, int64_t ti, int64_t tk)
  {
    for (int64_t i = 0; i < ti; ++i)
      for (int64_t k = 0; k < tk; ++k)
        dst[i * tk + k] = mptorch::gemm::block_load<T>(A, o, bId, off, i0 + i, k0 + k);
  }

  template <class T>
  void pack_block_b(const void *B, T *__restrict__ dst, const mptorch::gemm::BlockOperandT<T> &o,
                    int64_t bId, int64_t off, int64_t k0, int64_t j0, int64_t tk, int64_t tj)
  {
    for (int64_t k = 0; k < tk; ++k)
      for (int64_t j = 0; j < tj; ++j)
        dst[k * tj + j] = mptorch::gemm::block_load<T>(B, o, bId, off, j0 + j, k0 + k);
  }

  template <>
  inline void pack_a<double>(const void *A, mptorch::GemmDtype, double *dst, int64_t M, int64_t K,
                             bool trans_a, int64_t i0, int64_t k0, int64_t ti, int64_t tk,
                             int64_t off)
  {
    pack_a_impl<double, double>(A, dst, M, K, trans_a, i0, k0, ti, tk, off);
  }

  template <>
  inline void pack_b<double>(const void *B, mptorch::GemmDtype, double *dst, int64_t K, int64_t N,
                             bool trans_b, int64_t k0, int64_t j0, int64_t tk, int64_t tj,
                             int64_t off)
  {
    pack_b_impl<double, double>(B, dst, K, N, trans_b, k0, j0, tk, tj, off);
  }

  template <>
  inline void store_tile<double>(void *C, mptorch::GemmDtype, const double *src, int64_t N,
                                 int64_t i0, int64_t j0, int64_t ti, int64_t tj, int64_t off)
  {
    store_tile_impl<double, double>(C, src, N, i0, j0, ti, tj, off);
  }

  // The same double specializations for the conv ops' packs and result
  // tile. Without them the primary templates' switch would take its default
  // arm and read a float64 operand as float.
  template <>
  inline void pack_gather_a<double>(const void *A, mptorch::GemmDtype, double *dst,
                                    const mptorch::gemm::ConvGeom &g, int64_t bId, int64_t i0,
                                    int64_t k0, int64_t ti, int64_t tk)
  {
    pack_gather_a_impl<double, double>(A, dst, g, bId, i0, k0, ti, tk);
  }

  template <>
  inline void pack_gather_b<double>(const void *B, mptorch::GemmDtype, double *dst,
                                    const mptorch::gemm::ConvGeom &g, int64_t bId, int64_t k0,
                                    int64_t j0, int64_t tk, int64_t tj)
  {
    pack_gather_b_impl<double, double>(B, dst, g, bId, k0, j0, tk, tj);
  }

  template <>
  inline void store_gather_tile<double>(void *C, mptorch::GemmDtype, const double *src,
                                        const mptorch::gemm::ConvGeom &g, int64_t bId, int64_t M,
                                        int64_t i0, int64_t j0, int64_t ti, int64_t tj)
  {
    store_gather_tile_impl<double, double>(C, src, g, bId, M, i0, j0, ti, tj);
  }

  // The cache-tiled batched GEMM: C[b] = op(A[b]) @ op(B[b]) for each batch
  // element b, where op(X) is X transposed when the matching trans flag is
  // set. A is [M, K] ([K, M] under trans_a), B is [K, N] ([N, K] under
  // trans_b), and C is written densely as [batch, M, N]. stride_a and
  // stride_b are the element strides from one batch element to the next in
  // A and B; a stride of 0 broadcasts that operand across the batch. A and B
  // are read in their own storage layout: the trans flags change only the
  // index arithmetic, and no transposed copy is made.
  //
  // The Accumulator policy performs both the multiply and the accumulate
  // step, through its Mac policy (common/gemm_policy.h), so this kernel only
  // calls seed_rng(), accumulate() and finalize() and never multiplies
  // operands itself. Each output element's reduction state (a running sum
  // and, under SR, a Philox random stream) is kept in the Accumulator's tile
  // type as dense arrays, not as one Accumulator object per element, so the
  // K-loop steps through a plain array of carrier values. use_rng and seed
  // serve RoundMode::SR: each output element's stream is seeded before its
  // reduction starts, keyed by the element's linear index into
  // [batch, M, N], so the draws it sees depend on neither the tiling nor the
  // thread count.
  //
  // MIXED selects the palette instantiation, in which each output element
  // has its own format: its Mac policy is looked up in a FormatPalette
  // through prec_idx[b, i, j] (read with the three idx_* element strides).
  // It is a template parameter rather than a runtime test because the
  // single-format path keeps its one Mac in registers for the whole call,
  // and a runtime test would turn that into a pointer load per element in
  // the innermost loop. The single-format instantiation takes no palette:
  // PaletteArg<false, Mac> is the empty NoPalette.
  template <bool MIXED = false, class Accumulator>
  void matmul_cpu_kernel_impl(const void *A, const void *B, void *C, mptorch::GemmDtype dt,
                              int64_t M, int64_t K, int64_t N,
                              bool trans_a, bool trans_b,
                              int64_t batch, int64_t stride_a, int64_t stride_b,
                              Accumulator acc_proto,
                              bool use_rng, uint64_t seed,
                              PaletteArg<MIXED, typename Accumulator::mac_type> pal = {},
                              const int32_t *__restrict__ prec_idx = nullptr,
                              int64_t idx_row_stride = 0, int64_t idx_col_stride = 0,
                              int64_t idx_batch_stride = 0)
  {
    constexpr int64_t TI = 32, TJ = 32, TK = 32;
    using Mac = typename Accumulator::mac_type;
    using T = typename Accumulator::value_t; // the carrier (common/gemm_policy.h)

    const int64_t n_tiles_i = (M + TI - 1) / TI;
    const int64_t n_tiles_j = (N + TJ - 1) / TJ;
    const int64_t n_tiles = n_tiles_i * n_tiles_j;

    // The parallel axis is the whole grid of (i0, j0) output tiles, not only
    // its rows: splitting on M alone would give ceil(M / TI) tasks, so a
    // typical QLinear batch (M = 32) would run on one thread however many are
    // available. Output tiles are disjoint and each element's reduction runs
    // in order inside one task, so the results do not depend on the thread
    // count, under RoundMode::SR too, since each stream is keyed by the
    // output element's linear index (NaiveTile::seed_rng) and not by the
    // tiling. The grain is the number of tiles that make up one of ATen's
    // default grains of elementwise work, so a small GEMM is not split finer
    // than the thread pool's dispatch cost justifies.
    //
    // at::parallel_for rather than a raw `#pragma omp parallel for`: it runs
    // on ATen's own thread pool, so torch.set_num_threads applies, and inside
    // an enclosing parallel region it runs serially instead of nesting. Under
    // the AT_PARALLEL_OPENMP backend it still needs setup.py's -fopenmp,
    // because parallel_for is a header template whose `#pragma omp parallel`
    // is compiled in this translation unit; without the flag the loop is
    // silently serial.
    const int64_t work_per_tile = TI * TJ * std::max<int64_t>(K, 1);
    const int64_t grain = std::max<int64_t>(1, at::internal::GRAIN_SIZE / work_per_tile);

    // The batch shares the parallel axis with the tiles instead of being a
    // loop around them: each task is a (batch element, output tile) pair, so
    // a batched call with a small M x N per element (an attention head's
    // 128 x 128 output is 16 tiles) still has work for every thread, where a
    // loop of 2D calls would leave most threads idle in each call. batch = 1
    // is the plain 2D decomposition.
    at::parallel_for(0, batch * n_tiles, grain, [&](int64_t task_begin, int64_t task_end)
    {
      // One tile's worth of reduction state per worker task, reused for
      // every tile the task takes rather than rebuilt per tile. For NAIVE
      // (gemm_policy.h's NaiveTile) that is TI x TJ running sums and
      // TI x TJ Philox streams; the streams exist under every round mode, so
      // that the tile type is one type, but are seeded and drawn only under
      // SR. The Mac policy is not part of the tile: the call has one copy of
      // it (mac_single, below) or, on the palette path, a pointer per element
      // into the palette.
      typename Accumulator::tile_type tile;
      tile.resize(TI * TJ);
      const auto tv = tile.view();

      // The single-format path's one Mac, copied into the task's frame
      // rather than read again per output element. A copy and not a
      // reference to the caller's acc_proto: the reference measured 5-8%
      // slower on the split macs, because the compiler could not keep the
      // policy's fields in registers across the loop.
      const Mac mac_single = acc_proto.mac;
      // A positional accumulator (KAHAN, BLOCK or TREE;
      // common/gemm_accumulate.h) is configured by more than its Mac (also by
      // a block size and an outer format), so its tile functions take the
      // prototype itself. Every use of it is behind `if constexpr`, and it is
      // read through the caller's acc_proto rather than through a second
      // local: even an empty local of a NaiveAccumulator's type moved the
      // register allocation of two existing NAIVE instantiations, and the
      // NAIVE objects must stay byte-identical to what they were before the
      // positional path existed.
      constexpr bool POSITIONAL = is_positional_accumulator_v<Accumulator>;
      const Mac *slot_of[MIXED ? TI * TJ : 1]; // the palette path's Mac per element

      // The packed operand tiles and the finalized output tile, one set per
      // worker task, reused for every tile it takes: 12 KB of stack frame in
      // binary32 (24 KB in binary64), in exchange for converting each element
      // once per tile instead of once per K-step. They are 64-byte aligned so
      // that the innermost loop's reads of b_pack start on a cache line.
      alignas(64) T a_pack[TI * TK];
      alignas(64) T b_pack[TK * TJ];
      alignas(64) T c_pack[TI * TJ];

      for (int64_t task_id = task_begin; task_id < task_end; ++task_id)
      {
        const int64_t bId = task_id / n_tiles;    // batch element of this task
        const int64_t tile_id = task_id % n_tiles; // output tile within it
        const int64_t a_off = bId * stride_a;
        const int64_t b_off = bId * stride_b;
        const int64_t c_off = bId * M * N;
        const int64_t i0 = (tile_id / n_tiles_j) * TI;
        const int64_t j0 = (tile_id % n_tiles_j) * TJ;
        const int64_t ti = std::min<int64_t>(TI, M - i0);
        const int64_t tj = std::min<int64_t>(TJ, N - j0);
        tile.begin(ti * tj);

        // Each element's SR stream is keyed by its linear index into the
        // whole [batch, M, N] result, so batch element 0 of a batched call
        // draws exactly what the 2D call draws, and no two elements draw from
        // the same stream.
        if (use_rng)
        {
          for (int64_t i = 0; i < ti; ++i)
            for (int64_t j = 0; j < tj; ++j)
              tile.seed_rng(i * tj + j, seed,
                            static_cast<uint64_t>(c_off + (i0 + i) * N + (j0 + j)));
        }

        // Palette: look up each output element's Mac policy in the palette
        // (gemm_policy.h's FormatPalette) before its reduction starts. The
        // task keeps pointers into the palette, not copies: the palette
        // outlives the call and an element's slot does not change during its
        // reduction, so a copy would only take frame space.
        if constexpr (MIXED)
        {
          // The dense branch comes first, here and at the packs below. The
          // parallel_for lambda captures by reference, and GCC lays out its
          // closure in the order the body first names each variable,
          // discarded branches included, so a gathered branch ahead of the
          // dense one would reorder the captures and move the stack slots of
          // every existing instantiation.
          if constexpr (!mptorch::gemm::is_gathered_v<Accumulator>)
          {
            for (int64_t i = 0; i < ti; ++i)
              for (int64_t j = 0; j < tj; ++j)
                slot_of[i * tj + j] =
                    &pal.slot(prec_idx[bId * idx_batch_stride + (i0 + i) * idx_row_stride +
                                       (j0 + j) * idx_col_stride]);
          }
          else
          {
            // A conv op's format map is indexed by the result tensor's
            // (sample, channel, position), and this tile's rows are channels
            // of one group (common/gemm_gather.h).
            int64_t b;
            int32_t grp;
            mptorch::gemm::conv_batch(acc_proto.geom, bId, b, grp);
            for (int64_t i = 0; i < ti; ++i)
              for (int64_t j = 0; j < tj; ++j)
                slot_of[i * tj + j] =
                    &pal.slot(prec_idx[b * idx_batch_stride +
                                       mptorch::gemm::conv_result_row(grp, M, i0 + i) * idx_row_stride +
                                       mptorch::gemm::conv_result_col(acc_proto.geom,
                                                                      static_cast<int32_t>(j0 + j)) *
                                           idx_col_stride]);
          }
        }

        for (int64_t k0 = 0; k0 < K; k0 += TK)
        {
          int64_t tk = std::min<int64_t>(TK, K - k0);
          // A gathered policy (the conv ops) packs the same tiles through its
          // geometry, and nothing else in the loop changes for it. Every other
          // policy compiles to the code it would without this branch; the
          // branches are in the palette lookup's order, above.
          if constexpr (!mptorch::gemm::is_gathered_v<Accumulator> &&
                        !mptorch::gemm::is_blocked_v<Accumulator>)
          {
            pack_a(A, dt, a_pack, M, K, trans_a, i0, k0, ti, tk, a_off);
            pack_b(B, dt, b_pack, K, N, trans_b, k0, j0, tk, tj, b_off);
          }
          else if constexpr (mptorch::gemm::is_gathered_v<Accumulator>)
          {
            static_assert(TK <= GATHER_TILE && TJ <= GATHER_TILE);
            pack_gather_a(A, dt, a_pack, acc_proto.geom, bId, i0, k0, ti, tk);
            pack_gather_b(B, dt, b_pack, acc_proto.geom, bId, k0, j0, tk, tj);
          }
          else
          {
            // A blocked policy (the block GEMM) decodes its block-format
            // operands; for it, a_off and b_off are in bytes.
            pack_block_a(A, a_pack, acc_proto.block_a, bId, a_off, i0, k0, ti, tk);
            pack_block_b(B, b_pack, acc_proto.block_b, bId, b_off, k0, j0, tk, tj);
          }
          // A single-format binaryK GEMM in binary32 under RNE whose formats
          // admit it (rne_tile_kind) runs this K-block, the tk steps of the
          // current tile, in vector code (cpu/gemm_rne_tile.h, and
          // cpu/gemm_odd_tile.h for the odd macs), which computes word for
          // word what the loop below computes. When run_rne_tile returns
          // false, the loop below runs the K-block instead. The branch names
          // nothing the lambda does not already capture or declare, so the
          // instantiations it does not cover compile to what they did
          // without it.
          if constexpr (!MIXED && rne_tile_kind<Accumulator>::value != 0)
          {
            static_assert(TI * TJ <= RNE_TILE_MAX);
            if (run_rne_tile<rne_tile_kind<Accumulator>::value>(mac_single, a_pack, b_pack,
                                                                  tv.sums, ti, tk, tj))
              continue;
          }
          for (int64_t i = 0; i < ti; ++i)
          {
            for (int64_t k = 0; k < tk; ++k)
            {
              const T aVal = a_pack[i * tk + k];
              const T *__restrict__ b_row = b_pack + k * tj;
              if constexpr (POSITIONAL)
              {
                // The step's position in its slab, the 16 consecutive K
                // steps the algorithms count in (the CUDA kernel's tile
                // depth, used on every backend so that they agree), and the
                // end of the slab after its 16th step or after the last
                // step of K.
                const int64_t k_end = k0 + k + 1;
                const int pos = static_cast<int>((k0 + k) & 15);
                for (int64_t j = 0; j < tj; ++j)
                  Accumulator::accumulate(tv, i * tj + j, acc_proto, aVal, b_row[j], pos);
                if ((k_end & 15) == 0 || k_end == K)
                  for (int64_t j = 0; j < tj; ++j)
                    Accumulator::end_slab(tv, i * tj + j, acc_proto, k_end, K);
              }
              else
              {
                for (int64_t j = 0; j < tj; ++j)
                {
                  const int64_t idx = i * tj + j;
                  Accumulator::accumulate(tv, idx, MIXED ? *slot_of[MIXED ? idx : 0] : mac_single,
                                          aVal, b_row[j]);
                }
              }
            }
          }
        }

        // The fixed-point GEMM's integer accumulator (common/fixedpoint_gemm.h)
        // keeps int64 codes, which only its Mac can scale, and its result is
        // float64 when the format is wider than the operands' dtype (`wide`),
        // whatever the carrier. The condition is has_int_result_v, spelled
        // here so that this header needs no include for it. Every other
        // accumulator takes the first branch, which is the code from before.
        if constexpr (!requires { Accumulator::int_result; })
        {
          for (int64_t i = 0; i < ti; ++i)
            for (int64_t j = 0; j < tj; ++j)
              c_pack[i * tj + j] = tile.finalize(i * tj + j);
          if constexpr (!mptorch::gemm::is_gathered_v<Accumulator>)
            store_tile(C, dt, c_pack, N, i0, j0, ti, tj, c_off);
          else
            store_gather_tile(C, dt, c_pack, acc_proto.geom, bId, M, i0, j0, ti, tj);
        }
        else
        {
          // a conv op's result is gathered, as the first branch stores it; a
          // wide one through store_gather_tile's double specialization
          static_assert(!MIXED, "the fixed-point GEMM has no palette ops");
          if (mac_single.wide)
          {
            if constexpr (!mptorch::gemm::is_gathered_v<Accumulator>)
            {
              double *__restrict__ dst = static_cast<double *>(C) + c_off;
              for (int64_t i = 0; i < ti; ++i)
                for (int64_t j = 0; j < tj; ++j)
                  dst[(i0 + i) * N + (j0 + j)] =
                      Accumulator::result_wide(tile, i * tj + j, mac_single);
            }
            else
            {
              alignas(64) double w_pack[TI * TJ];
              for (int64_t i = 0; i < ti; ++i)
                for (int64_t j = 0; j < tj; ++j)
                  w_pack[i * tj + j] = Accumulator::result_wide(tile, i * tj + j, mac_single);
              store_gather_tile(C, dt, w_pack, acc_proto.geom, bId, M, i0, j0, ti, tj);
            }
          }
          else
          {
            for (int64_t i = 0; i < ti; ++i)
              for (int64_t j = 0; j < tj; ++j)
                c_pack[i * tj + j] = Accumulator::result(tile, i * tj + j, mac_single);
            if constexpr (!mptorch::gemm::is_gathered_v<Accumulator>)
              store_tile(C, dt, c_pack, N, i0, j0, ti, tj, c_off);
            else
              store_gather_tile(C, dt, c_pack, acc_proto.geom, bId, M, i0, j0, ti, tj);
          }
        }
      }
    });
  }

  // One body for every NAIVE GEMM op, in either carrier T (float or
  // double). The Args type names the Accumulator policy, which is the only
  // thing that differs between the ops, and dispatch_round_mode turns the
  // round mode into a template parameter here, so the accumulate step
  // compiles to straight-line code instead of a switch per multiply-add.
  // Explicitly instantiated in cpu/inst/.
  template <class T, class Args>
  void CpuBackend::launch_as(const GemmShape &s, const Args &args, const LaunchContext &ctx)
  {
    mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      if constexpr (Args::mixed)
      {
        args.template with_palette<T, RM>([&](auto acc, const auto &pal)
        {
          matmul_cpu_kernel_impl<true>(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                       s.batch, s.stride_a, s.stride_b,
                                       acc, s.use_rng, ctx.seed, pal, s.prec_idx,
                                       s.idx_row_stride, s.idx_col_stride, s.idx_batch_stride);
        });
      }
      else
      {
        args.template with_accumulator<T, RM>([&](auto acc)
        {
          matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                                 s.batch, s.stride_a, s.stride_b, acc, s.use_rng, ctx.seed);
        });
      }
    });
  }

  // The same body for the GEMM ops under KAHAN, BLOCK and TREE: the
  // algorithm ALG is a template parameter, and args builds the accumulator
  // for it. Instantiated by the cpu/inst/custom_matmul_*_{kahan,block,tree}*
  // units.
  template <class T, AccumulateAlgorithm ALG, class Base>
  void CpuBackend::launch_accumulated_as(const GemmShape &s,
                                         const mptorch::gemm::AccumulateArgs<Base> &args,
                                         const LaunchContext &ctx)
  {
    mptorch::dispatch_round_mode(s.rm, [&](auto rm_c)
    {
      constexpr RoundMode RM = decltype(rm_c)::value;
      args.template with_accumulator<T, RM, ALG>([&](auto acc)
      {
        matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b,
                               s.batch, s.stride_a, s.stride_b, acc, s.use_rng, ctx.seed);
      });
    });
  }

  // The conv ops (common/gemm_gather.h): the accumulator that `Inner`
  // builds, as its GEMM op builds it, wrapped in a Gathered that carries the
  // geometry, and run over the pass's [batch, M, N] with no transposes and
  // no batch strides, since the gathered packs address the operands
  // themselves. Instantiated by the cpu/inst/custom_conv_* units.
  template <class T, AccumulateAlgorithm ALG, class Inner>
  void CpuBackend::launch_conv_as(const GemmShape &s, const mptorch::gemm::ConvArgs<Inner> &args,
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
          matmul_cpu_kernel_impl<true>(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, false, false, s.batch, 0, 0,
                                       Gathered<decltype(acc)>{acc, args.geom}, s.use_rng, ctx.seed,
                                       pal, s.prec_idx, s.idx_row_stride, s.idx_col_stride,
                                       s.idx_batch_stride);
        });
      }
      else if constexpr (ALG == AccumulateAlgorithm::NAIVE)
      {
        args.inner.template with_accumulator<T, RM>([&](auto acc)
        {
          matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, false, false, s.batch, 0, 0,
                                 Gathered<decltype(acc)>{acc, args.geom}, s.use_rng, ctx.seed);
        });
      }
      else
      {
        args.inner.template with_accumulator<T, RM, ALG>([&](auto acc)
        {
          matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, false, false, s.batch, 0, 0,
                                 Gathered<decltype(acc)>{acc, args.geom}, s.use_rng, ctx.seed);
        });
      }
    });
  }

  // The block GEMM (common/gemm_block.h): the accumulator that `Inner`
  // builds, wrapped in a Blocked that carries the two block-format operands,
  // and run over [batch, M, N] with the operands' own trans flags and their
  // codes' batch strides in bytes. Instantiated by the
  // cpu/inst/custom_matmul_block* units.
  template <class T, AccumulateAlgorithm ALG, class Inner>
  void CpuBackend::launch_block_as(const GemmShape &s, const mptorch::gemm::BlockGemmArgs<Inner> &args,
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
          matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b, s.batch,
                                 s.stride_a, s.stride_b,
                                 Blocked<decltype(acc)>{acc, args.template operand_a<T>(), args.template operand_b<T>()},
                                 s.use_rng, ctx.seed);
        });
      }
      else
      {
        args.inner.template with_accumulator<T, RM, ALG>([&](auto acc)
        {
          matmul_cpu_kernel_impl(s.a, s.b, s.c, s.dt, s.M, s.K, s.N, s.trans_a, s.trans_b, s.batch,
                                 s.stride_a, s.stride_b,
                                 Blocked<decltype(acc)>{acc, args.template operand_a<T>(), args.template operand_b<T>()},
                                 s.use_rng, ctx.seed);
        });
      }
    });
  }
} // namespace mptorch::gemm_cpu
