#pragma once

// The host-side code that the GEMM ops' entry points share, except the conv
// ops' (gemm_conv_host.h) and the block GEMM's (gemm_block_host.h). That is
// 32 ops (the NAIVE, *_accumulated, cross-family and round-to-odd GEMMs), each
// with an entry point on the CPU, CUDA and MPS backends. The code here checks
// the inputs, derives the output's shape, makes the operands contiguous,
// allocates the output, returns early when the output is empty, validates a
// palette op's precision index, tags the operands' dtype and draws the
// random-number state. An entry point only packs its schema's arguments into
// an Args struct (the pack_* functions at the bottom of this file; the
// structs are in gemm_args.h) and calls one of the drivers below with its
// backend. Because every backend runs this one copy, their checks, error
// messages and order of operations cannot drift apart. The Meta kernels run
// the same checks on symbolic sizes, through common/meta_host.h.
//
// The *_accumulated ops (KAHAN, BLOCK and TREE accumulation; see
// gemm_accumulate.h) go through the same driver: run_custom_matmul_accumulated
// wraps the Args of their NAIVE twin, the op without the suffix, in an
// AccumulateArgs. The NAIVE ops pass their own Args.
//
// A Backend provides three things:
//   - `LaunchContext`, whatever its kernel needs besides the shape, such as a
//     random-number state or a stream.
//   - `make_context(use_rng, draws)`, which draws that state on the host.
//     `draws` is the largest number of Philox words one output element may
//     consume.
//   - `launch(shape, args, ctx)`, one overload per kind of Args, which picks
//     the carrier (the floating-point type the kernel computes in, binary32
//     or binary64) from shape.dt and calls `launch_as<T, Args>` or one of its
//     siblings. On the CPU and CUDA those templates are only declared where
//     this header is included: the kernel headers define them, and the units
//     in cpu/inst/ and cuda/inst/ instantiate them, one carrier per file. So
//     this header, which includes ATen, never reaches nvcc, and no object
//     file holds both carriers' kernels.
// A context may also have a `bind(a, b, c, prec_idx)`, which the drivers call
// with the tensors just before the launch (bind_tensors below). The MPS
// backend's context has one, because its kernels take each tensor as a Metal
// buffer plus a byte offset, which GemmShape's raw pointers cannot carry.

#include "dispatch.h"
#include "gemm_accumulate.h"
#include "gemm_args.h"
#include "gemm_policy.h"
#include "stochastic_host.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/stack.h>
#include <algorithm>
#include <initializer_list>
#include <mutex>
#include <vector>

namespace mptorch::gemm
{
  using at::Tensor;

  // The one place where a GEMM's operand ranks and devices, its round_mode
  // and its accumulate_algorithm are validated. Every driver calls this before
  // anything casts those integers to their enums, so dispatch_round_mode never
  // sees an integer that names no mode. Such an integer would fall through to
  // its `default:` case and round to nearest-even, with no error reaching
  // Python. The valid values are listed in gemm_dtype.h's is_round_mode and
  // is_accumulate_algorithm. Whether the op being called implements the
  // algorithm is a separate check, check_accumulate_algorithm below.
  inline void check_matmul_inputs(const Tensor &a, const Tensor &b, const char *op_name,
                                  int64_t round_mode, int64_t accumulate_algorithm)
  {
    TORCH_CHECK((a.dim() == 2 || a.dim() == 3) && (b.dim() == 2 || b.dim() == 3), op_name,
                " expects 2D or 3D tensors, got ", a.dim(), "D and ", b.dim(),
                "D (1D promotion and rank>3 broadcasting live in mptorch.quant.qmatmul)");
    TORCH_CHECK(a.device() == b.device(), op_name, ": both operands must be on one device, got ",
                a.device(), " and ", b.device());
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode,
                " is not a RoundMode");
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
  }

  // Which values of accumulate_algorithm an Args type accepts. A plain Args
  // sums with NAIVE accumulation only. The other algorithms belong to the
  // *_accumulated ops, which give the driver an AccumulateArgs (below). The
  // palette ops have no *_accumulated version: BLOCK and TREE would need a
  // palette of outer formats, which is a design of its own.
  inline void check_naive_only(const char *op_name, int64_t accumulate_algorithm)
  {
    TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) == AccumulateAlgorithm::NAIVE,
                op_name, ": this op implements AccumulateAlgorithm.NAIVE only; KAHAN, BLOCK and "
                         "TREE are the single-format ops' *_accumulated twins (the palette ops "
                         "have none)");
  }

  template <class Args>
  void check_accumulate_algorithm(const Args &, const char *op_name, int64_t accumulate_algorithm)
  {
    check_naive_only(op_name, accumulate_algorithm);
  }

  // An AccumulateArgs carries the algorithm the schema named. This checks
  // block_size against that algorithm, and refuses TREE over a fused mac (a
  // multiply-add rounded once), which has no separate product to sum
  // pairwise.
  template <class Base>
  void check_accumulate_algorithm(const AccumulateArgs<Base> &args, const char *op_name,
                                  int64_t /*accumulate_algorithm*/)
  {
    const int64_t bs = args.block_size;
    switch (args.alg)
    {
    case AccumulateAlgorithm::KAHAN:
      TORCH_CHECK(bs == 0 && !args.outer_quant, op_name,
                  ": AccumulateAlgorithm.KAHAN takes no block_size and no outer format");
      break;
    case AccumulateAlgorithm::BLOCK:
      TORCH_CHECK(bs >= 1 && bs <= (int64_t(1) << 30) && (16 % bs == 0 || bs % 16 == 0), op_name,
                  ": AccumulateAlgorithm.BLOCK needs a block_size that divides 16 or is a "
                  "multiple of 16, got ", bs);
      break;
    case AccumulateAlgorithm::TREE:
      TORCH_CHECK(AccumulateArgs<Base>::has_product, op_name,
                  ": AccumulateAlgorithm.TREE sums products pairwise, and a fused "
                  "multiply-add has no product term; use the split op");
      TORCH_CHECK(bs >= 2 && bs <= 256 && (bs & (bs - 1)) == 0, op_name,
                  ": AccumulateAlgorithm.TREE needs a block_size that is a power of two in "
                  "[2, 256], got ", bs);
      break;
    default:
      TORCH_CHECK(false, op_name, ": an AccumulateArgs is never a NAIVE reduction");
    }
  }

  // The largest number of random draws that one K-step (one multiply-add
  // into one output element) of this Args may take under RoundMode::SR: a
  // constant of the type for NAIVE accumulation, and a function of the
  // algorithm for the others.
  template <class Args>
  uint64_t draws_per_k_step_of(const Args &) { return Args::draws_per_k_step; }

  template <class Base>
  uint64_t draws_per_k_step_of(const AccumulateArgs<Base> &args) { return args.draws_per_k_step(); }

  // Derives (M, K, N), the batch size, the two operands' batch strides and
  // whether the output has a batch dimension, and rejects mismatched inner or
  // batch dimensions.
  //
  // The ops take operands of rank 2 or 3 only. Broadcasting, the promotion of
  // 1D operands and reshaping all happen in Python
  // (mptorch/quant/ops/_operands.py's `_matmul_operands`), where torch.matmul's
  // rules can be expressed as views rather than reimplemented here. What is
  // left is one batch dimension, which each operand either has or does not. A
  // rank-3 operand's leading dimension is the batch size or 1, and a leading
  // 1, like a rank-2 operand, becomes a batch stride of 0, so an operand that
  // the whole batch shares is read in place rather than expanded. `batched`
  // says whether the *output* has that dimension, which depends on the
  // caller's ranks, not on `batch`: `[1, M, K] @ [K, N]` returns [1, M, N].
  inline void matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a, bool trans_b,
                                  const char *op_name, int64_t &M, int64_t &K, int64_t &N,
                                  int64_t &batch, int64_t &stride_a, int64_t &stride_b,
                                  bool &batched)
  {
    const int64_t ba = a.dim() == 3 ? a.size(0) : 1;
    const int64_t bb = b.dim() == 3 ? b.size(0) : 1;
    batched = (a.dim() == 3 || b.dim() == 3);
    if (ba == bb)
      batch = ba;
    else if (ba == 1)
      batch = bb;
    else if (bb == 1)
      batch = ba;
    else
      TORCH_CHECK(false, op_name, ": batch dimensions must match or be 1 (got ", ba, " and ", bb,
                  ")");

    const int64_t a_r = a.size(a.dim() - 2), a_c = a.size(a.dim() - 1);
    const int64_t b_r = b.size(b.dim() - 2), b_c = b.size(b.dim() - 1);
    M = trans_a ? a_c : a_r;
    K = trans_a ? a_r : a_c;
    const int64_t K_b = trans_b ? b_c : b_r;
    N = trans_b ? b_r : b_c;
    TORCH_CHECK(K == K_b, op_name, ": inner dimensions must match (got ", K, " vs ", K_b, ")");

    stride_a = ba == 1 ? 0 : a_r * a_c;
    stride_b = bb == 1 ? 0 : b_r * b_c;
  }

  // The output's shape, as the drivers allocate it. It is returned by value
  // and used within one full-expression, so the vector outlives the
  // IntArrayRef that at::empty borrows from it.
  inline std::vector<int64_t> matmul_output_sizes(int64_t batch, int64_t M, int64_t N,
                                                  bool batched)
  {
    if (batched)
      return {batch, M, N};
    return {M, N};
  }

  // A palette op (a "mixed" op, which computes each output element in one of
  // several formats) has `n` formats, between 1 and MAX_GEMM_FORMATS, and each
  // of its per-format parameter lists must have length `n`. The mixed packers
  // call this before they index any list.
  inline void check_palette_lengths(int64_t n, const char *op_name,
                                    std::initializer_list<int64_t> other_lens)
  {
    TORCH_CHECK(n >= 1 && n <= MAX_GEMM_FORMATS, op_name, ": expected 1..", MAX_GEMM_FORMATS,
                " palette formats, got ", n);
    for (int64_t l : other_lens)
      TORCH_CHECK(l == n, op_name, ": every palette parameter list must have length ", n,
                  " (got one of length ", l, ")");
  }

  // One entry of the memo of precision maps that have already passed
  // resolve_prec_idx's bounds check, so that a map reused across calls is
  // checked once rather than on every call.
  //
  // The check reads the index values on the host. For a map on a device
  // (CUDA or MPS), that takes a device-to-host copy, which waits for the
  // stream to drain before the GEMM is even launched: 0.23 ms per call on an
  // RTX 4060 laptop GPU, against 0.27 ms for a whole 64^3 palette GEMM. A map
  // is normally built once and reused, so the memo skips the copy for as long
  // as the same tensor comes back unchanged. A miss runs the full check, with
  // the same error message.
  //
  // The key is the TensorImpl's address plus its version counter, and the
  // entry holds a weak reference to that TensorImpl. The weak reference is
  // what makes comparing addresses sound: it keeps the TensorImpl's control
  // block alive (but not its storage, so the map's memory is not held), so no
  // other tensor can be created at that address while an entry names it. The
  // version counter changes with every in-place write that goes through ATen.
  //
  // The key is the caller's map, not the contiguous int32 copy that the
  // kernel reads. The two are the same tensor only when the caller passes a
  // contiguous int32 tensor on the operands' device. For anything else, such
  // as an int64 map (what torch.zeros, torch.randint and torch.arange return
  // without an explicit dtype) or a strided view, `to(...).contiguous()`
  // allocates a new tensor on every call, so a key on the copy would miss
  // every time and pay the very sync the memo exists to avoid. Keying on the
  // caller's map is sound because the copy's contents depend only on it, and
  // a view shares its base's version counter.
  //
  // Two cases are never memoized, and take the full check on every call: a
  // map on the host, whose check runs on the host and so has no sync to save
  // (when the operands are on a device, the map is copied there on every call
  // whatever the memo says), and a tensor without a version counter (one
  // created under torch.inference_mode), whose writes cannot be detected.
  //
  // FormatPalette::slot masks the index into range anyway, so a stale entry
  // can never cause an out-of-bounds read.
  struct ValidatedPrecIdx
  {
    c10::weak_intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl> impl;
    const c10::TensorImpl *raw;
    uint32_t version;
    int64_t n_formats;
  };

  // The memo holds the last few distinct maps. A model rarely uses more than a
  // handful of per-layer maps at once, which does not justify a hash table.
  constexpr size_t PREC_IDX_MEMO_SLOTS = 4;
  inline std::mutex g_prec_idx_memo_mutex;
  inline std::vector<ValidatedPrecIdx> g_prec_idx_memo;

  // Fills in the key (the TensorImpl's address and version) and returns true
  // if this tensor can be memoized. `on_device` says the bounds check would
  // run on a device, so there is a sync worth skipping; the map itself must
  // be on the device too, for the reasons given above.
  inline bool prec_idx_memo_key(const Tensor &prec_idx, bool on_device,
                                const c10::TensorImpl *&raw, uint32_t &version)
  {
    if (!on_device || prec_idx.is_cpu())
      return false;
    c10::TensorImpl *impl = prec_idx.unsafeGetTensorImpl();
    if (!impl->version_counter().enabled())
      return false;
    raw = impl;
    version = impl->version_counter().current_version();
    return true;
  }

  // Whether this tensor, at its current version, has passed the bounds check
  // against a palette of `n_formats` formats.
  inline bool prec_idx_already_validated(const Tensor &prec_idx, bool on_device, int64_t n_formats)
  {
    const c10::TensorImpl *raw = nullptr;
    uint32_t version = 0;
    if (!prec_idx_memo_key(prec_idx, on_device, raw, version))
      return false;
    std::lock_guard<std::mutex> lock(g_prec_idx_memo_mutex);
    for (const ValidatedPrecIdx &e : g_prec_idx_memo)
      if (e.raw == raw && e.version == version && e.n_formats == n_formats)
        return true;
    return false;
  }

  // Records a map that has just passed the check, evicting the oldest entry
  // when the memo is full.
  inline void remember_validated_prec_idx(const Tensor &prec_idx, bool on_device,
                                          int64_t n_formats)
  {
    const c10::TensorImpl *raw = nullptr;
    uint32_t version = 0;
    if (!prec_idx_memo_key(prec_idx, on_device, raw, version))
      return;
    std::lock_guard<std::mutex> lock(g_prec_idx_memo_mutex);
    // Drop the entries whose tensor no longer exists, and any older entry for
    // this one.
    auto dead = std::remove_if(g_prec_idx_memo.begin(), g_prec_idx_memo.end(),
                               [&](const ValidatedPrecIdx &e)
                               { return e.impl.expired() || e.raw == raw; });
    g_prec_idx_memo.erase(dead, g_prec_idx_memo.end());
    if (g_prec_idx_memo.size() >= PREC_IDX_MEMO_SLOTS)
      g_prec_idx_memo.erase(g_prec_idx_memo.begin());
    g_prec_idx_memo.push_back(ValidatedPrecIdx{
        c10::weak_intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>(
            prec_idx.getIntrusivePtr()),
        raw, version, n_formats});
  }

  // Validates a palette op's precision index, the map that gives each output
  // element the index of the palette format it is computed in, and derives
  // the batch, row and column strides that the kernel reads it with. The map
  // is dense, [M, N]; per row, [M, 1]; or per column, [1, N] (gemm_policy.h's
  // FormatPalette). Each may have a leading batch dimension of size `batch`
  // (one map per batch element) or 1. A 2D map on a batched call is shared by
  // the whole batch, which is the common case, since the palette belongs to
  // the layer rather than to the sample. The map is converted to int32 on the
  // operands' device, every entry is checked against the palette size (the
  // memo above says when that costs a device sync), and the contiguous tensor
  // is returned in `pidx_out`, which the caller keeps alive across the
  // launch.
  inline void resolve_prec_idx(const Tensor &prec_idx, const Tensor &ref, int64_t batch, int64_t M,
                               int64_t N, const char *op_name, int64_t n_formats, Tensor &pidx_out,
                               int64_t &idx_row_stride, int64_t &idx_col_stride,
                               int64_t &idx_batch_stride)
  {
    TORCH_CHECK(prec_idx.dim() == 2 || prec_idx.dim() == 3, op_name,
                ": prec_idx must be 2D or 3D, got ", prec_idx.dim(), "D");
    Tensor pidx = prec_idx.to(ref.device(), at::kInt).contiguous();
    const int64_t bsz = pidx.dim() == 3 ? pidx.size(0) : 1;
    TORCH_CHECK(bsz == batch || bsz == 1, op_name, ": prec_idx batch dimension must be ", batch,
                " or 1, got ", bsz);
    int64_t r = pidx.size(pidx.dim() - 2), c = pidx.size(pidx.dim() - 1);
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
    idx_batch_stride = bsz == 1 ? 0 : r * c;
    // Check whichever of the two tensors is already on the host. For a map on
    // the host feeding a GEMM on a device, that is the caller's tensor:
    // reading its bounds there costs nothing, whereas reading them from the
    // device copy costs the sync the memo exists to avoid, and could not be
    // memoized, since the copy is a new tensor on every call. `prec_idx` may
    // still be int64 here, which only makes the check stricter: an index
    // that would wrap around when narrowed to int32 is rejected rather than
    // read as a different index.
    const Tensor &to_check = prec_idx.is_cpu() ? prec_idx : pidx;
    const bool memo = !pidx.is_cpu();
    if (!prec_idx_already_validated(prec_idx, memo, n_formats))
    {
      // aminmax is one reduction, where min() and max() would be two. On a
      // device, its two results are stacked so that reading them back is one
      // two-element copy, with one sync, rather than two; on the CPU there is
      // no sync to save, and .item() avoids the extra allocation.
      auto mm = at::aminmax(to_check);
      int64_t lo, hi;
      if (!to_check.is_cpu())
      {
        auto bounds = at::stack({std::get<0>(mm), std::get<1>(mm)}).cpu();
        const int32_t *b = bounds.data_ptr<int32_t>();
        lo = b[0];
        hi = b[1];
      }
      else
      {
        lo = std::get<0>(mm).item<int64_t>();
        hi = std::get<1>(mm).item<int64_t>();
      }
      TORCH_CHECK(lo >= 0 && hi < n_formats, op_name, ": prec_idx entries must be in [0, ",
                  n_formats, "), got range [", lo, ", ", hi, "]");
      remember_validated_prec_idx(prec_idx, memo, n_formats);
    }
    pidx_out = pidx;
  }

  // The default precheck of the palette drivers (run_custom_matmul_mixed,
  // and run_custom_conv_mixed in gemm_conv_host.h), which does nothing. The
  // fused palette ops (the *_fma_mixed ops) pass one that rejects
  // fma_quant=false. In run_custom_matmul_mixed it runs between
  // check_matmul_inputs and matmul_output_shape, so that a call that is
  // invalid in two ways reports the same error on every backend.
  struct NoPrecheck
  {
    void operator()() const {}
  };

  // How many Philox words one draw counted by Args::draws_per_k_step takes:
  // one in binary32 and two in binary64 (philox.h's PhiloxEngine::next64).
  inline uint64_t words_per_draw(mptorch::GemmDtype dt)
  {
    return dt == mptorch::GemmDtype::Double ? 2 : 1;
  }

  // Hands the tensors to a launch context that takes them (see the top of
  // this file). For a context that does not, which is the CPU's and CUDA's,
  // it does nothing. `prec_idx` is null for the single-format ops.
  template <class Ctx>
  void bind_tensors(Ctx &ctx, const Tensor &a, const Tensor &b, const Tensor &c,
                    const Tensor *prec_idx)
  {
    if constexpr (requires { ctx.bind(a, b, c, prec_idx); })
      ctx.bind(a, b, c, prec_idx);
  }

  // ----------------------------------------------------------------------
  // The drivers
  // ----------------------------------------------------------------------
  //
  // run_custom_matmul runs a single-format op: it checks the inputs, derives
  // the shapes, allocates the output, fills a GemmShape, gets the backend's
  // launch context and launches. The order of these steps can be observed
  // from Python, and is deliberate. The early return for an empty output
  // comes before a precision index is validated and before the generator is
  // advanced, so an empty call raises no index error and consumes no random
  // state. The dtype tag is derived before the random state is drawn, for
  // two reasons: a binary64 draw takes two words, so the size of the
  // reservation depends on the carrier; and an operand pair that the tag
  // rejects must not advance the generator first.
  template <class Backend, class Args>
  Tensor run_custom_matmul(const char *op_name, const Args &call_args, Tensor a, Tensor b,
                           bool trans_a, bool trans_b, int64_t accumulate_algorithm,
                           int64_t round_mode)
  {
    static_assert(!Args::mixed, "use run_custom_matmul_mixed for a palette op");
    check_matmul_inputs(a, b, op_name, round_mode, accumulate_algorithm);
    // SR_B and SR_C run as RoundMode::SR, with the variant recorded in each
    // format's prng_bits (stochastic_host.h's sr_canonical_args).
    const Args args = mptorch::sr_canonical_args(call_args, round_mode, op_name,
                                                 mptorch::carrier_bits(a.scalar_type() == at::kDouble));
    check_accumulate_algorithm(args, op_name, accumulate_algorithm);

    int64_t M, K, N, batch, stride_a, stride_b;
    bool batched;
    matmul_output_shape(a, b, trans_a, trans_b, op_name, M, K, N, batch, stride_a, stride_b,
                        batched);

    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c = at::empty(matmul_output_sizes(batch, M, N, batched), a.options());
    if (batch == 0 || M == 0 || N == 0)
      return c;

    GemmShape s;
    s.M = M;
    s.K = K;
    s.N = N;
    s.batch = batch;
    s.stride_a = stride_a;
    s.stride_b = stride_b;
    s.trans_a = trans_a;
    s.trans_b = trans_b;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);

    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    typename Backend::LaunchContext ctx = Backend::make_context(
        s.use_rng, draws_per_k_step_of(args) * words_per_draw(s.dt) * static_cast<uint64_t>(K));

    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    bind_tensors(ctx, a_c, b_c, c, nullptr);
    Backend::launch(s, args, ctx);
    return c;
  }

  // run_custom_matmul_mixed runs a palette op: the single-format steps plus
  // the precision index, which is resolved after the output allocation and
  // the early return for an empty output. `make_args` is a factory rather
  // than a ready-made Args because the palette-length check it runs must come
  // after check_matmul_inputs and the shape check, and before anything
  // indexes the parameter lists, so that a call that is invalid in several
  // ways always reports the same error. It returns the packed Args with n_fmt
  // set.
  template <class Backend, class ArgsFactory, class Precheck = NoPrecheck>
  Tensor run_custom_matmul_mixed(const char *op_name, ArgsFactory &&make_args, Tensor a, Tensor b,
                                 Tensor prec_idx, bool trans_a, bool trans_b,
                                 int64_t accumulate_algorithm, int64_t round_mode,
                                 Precheck &&precheck = Precheck{})
  {
    using Args = decltype(make_args());
    static_assert(Args::mixed, "use run_custom_matmul for a single-format op");
    check_matmul_inputs(a, b, op_name, round_mode, accumulate_algorithm);
    check_naive_only(op_name, accumulate_algorithm);
    precheck();

    int64_t M, K, N, batch, stride_a, stride_b;
    bool batched;
    matmul_output_shape(a, b, trans_a, trans_b, op_name, M, K, N, batch, stride_a, stride_b,
                        batched);

    // SR_B and SR_C run as RoundMode::SR, with the variant recorded in each
    // format's prng_bits (stochastic_host.h's sr_canonical_args).
    const Args args = mptorch::sr_canonical_args(make_args(), round_mode, op_name,
                                                 mptorch::carrier_bits(a.scalar_type() == at::kDouble));

    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c = at::empty(matmul_output_sizes(batch, M, N, batched), a.options());
    if (batch == 0 || M == 0 || N == 0)
      return c;

    Tensor pidx;
    GemmShape s;
    resolve_prec_idx(prec_idx, a, batch, M, N, op_name, args.n_fmt, pidx, s.idx_row_stride,
                     s.idx_col_stride, s.idx_batch_stride);
    s.prec_idx = pidx.data_ptr<int32_t>();

    s.M = M;
    s.K = K;
    s.N = N;
    s.batch = batch;
    s.stride_a = stride_a;
    s.stride_b = stride_b;
    s.trans_a = trans_a;
    s.trans_b = trans_b;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);

    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    typename Backend::LaunchContext ctx = Backend::make_context(
        s.use_rng, Args::draws_per_k_step * words_per_draw(s.dt) * static_cast<uint64_t>(K));

    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    bind_tensors(ctx, a_c, b_c, c, &pidx);
    Backend::launch(s, args, ctx);
    return c;
  }

  // What an *_accumulated schema says about the accumulation besides the
  // algorithm: the block size, and the outer format, in which BLOCK and TREE
  // add up their blocks' sums when outer_quant is set (gemm_accumulate.h).
  template <class Widths, class Common>
  struct AccumulateTail
  {
    int64_t block_size = 0;
    bool outer_quant = false;
    Widths outer{};
    Common outer_c{};
  };

  // run_custom_matmul_accumulated runs an *_accumulated op: it wraps the
  // twin's Args and the accumulation arguments in an AccumulateArgs, which the
  // backend launches in the instantiation for its algorithm. NAIVE is refused
  // rather than forwarded, because NAIVE belongs to the twin op. Giving each
  // algorithm exactly one op keeps the twin's schema, entry point, Args and
  // launch independent of the *_accumulated ops.
  template <class Backend, class Base, class Widths, class Common>
  Tensor run_custom_matmul_accumulated(const char *op_name, const Base &base,
                                       const AccumulateTail<Widths, Common> &tail, Tensor a,
                                       Tensor b, bool trans_a, bool trans_b,
                                       int64_t accumulate_algorithm, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) != AccumulateAlgorithm::NAIVE,
                op_name, ": AccumulateAlgorithm.NAIVE is the op without the _accumulated suffix");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    AccumulateArgs<Base> args;
    args.base = base;
    args.alg = static_cast<AccumulateAlgorithm>(accumulate_algorithm);
    args.block_size = static_cast<int>(tail.block_size);
    args.outer_quant = tail.outer_quant;
    args.outer = tail.outer;
    args.outer_c = tail.outer_c;
    return run_custom_matmul<Backend>(op_name, args, a, b, trans_a, trans_b,
                                      accumulate_algorithm, round_mode);
  }


  // ----------------------------------------------------------------------
  // Schema -> Args
  // ----------------------------------------------------------------------
  //
  // Each packer turns an op's flat TORCH_LIBRARY parameter list into the
  // struct its policies are built from, once for every backend. The mixed
  // packers run check_palette_lengths before they index any list, which is
  // why the mixed driver takes a factory rather than a ready-made Args: that
  // check has to run after the shape check and before the first subscript.

  // The saturation and subnormal modes belong to each format slot, not to
  // the op: the multiply and the accumulate each get their own BinaryKCommon.
  // So a split mac whose multiply saturates to the format's largest value
  // while its accumulate overflows to infinity costs two schema integers and
  // no kernel change. A fused op takes one set of modes, since it has one
  // format.
  inline BinaryKSplitArgs pack_binaryK_split(
      int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
      bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
      int64_t mul_saturation_mode, int64_t mul_subnormals_mode, int64_t acc_saturation_mode,
      int64_t acc_subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    BinaryKSplitArgs args;
    args.mul = binaryK_widths(mul_K, mul_P, mul_bias, mul_is_signed);
    args.mul_c = BinaryKCommon{mul_is_signed, static_cast<SaturationMode>(narrow_to_int(mul_saturation_mode)),
                               static_cast<SubnormalsMode>(narrow_to_int(mul_subnormals_mode)),
                               narrow_to_int(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc = binaryK_widths(acc_K, acc_P, acc_bias, acc_is_signed);
    args.acc_c = BinaryKCommon{acc_is_signed, static_cast<SaturationMode>(narrow_to_int(acc_saturation_mode)),
                               static_cast<SubnormalsMode>(narrow_to_int(acc_subnormals_mode)),
                               narrow_to_int(acc_prng_bits)};
    return args;
  }

  inline BinaryKSplitMixedArgs pack_binaryK_split_mixed(
      const char *op_name,
      c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed,
      bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P,
      c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t mul_saturation_mode,
      int64_t mul_subnormals_mode, int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
      int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(mul_K.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(mul_P.size()), static_cast<int64_t>(mul_bias.size()),
                           static_cast<int64_t>(acc_K.size()), static_cast<int64_t>(acc_P.size()),
                           static_cast<int64_t>(acc_bias.size())});
    BinaryKSplitMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.mul_c = BinaryKCommon{mul_is_signed, static_cast<SaturationMode>(narrow_to_int(mul_saturation_mode)),
                               static_cast<SubnormalsMode>(narrow_to_int(mul_subnormals_mode)),
                               narrow_to_int(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc_c = BinaryKCommon{acc_is_signed, static_cast<SaturationMode>(narrow_to_int(acc_saturation_mode)),
                               static_cast<SubnormalsMode>(narrow_to_int(acc_subnormals_mode)),
                               narrow_to_int(acc_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
    {
      args.mul[i] = binaryK_widths(mul_K[i], mul_P[i], mul_bias[i], mul_is_signed);
      args.acc[i] = binaryK_widths(acc_K[i], acc_P[i], acc_bias[i], acc_is_signed);
    }
    return args;
  }

  inline BinaryKFusedArgs pack_binaryK_fused(
      bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
      int64_t fma_saturation_mode, int64_t fma_subnormals_mode, int64_t fma_prng_bits)
  {
    BinaryKFusedArgs args;
    args.fma_quant = fma_quant;
    args.fma = binaryK_widths(fma_K, fma_P, fma_bias, fma_is_signed);
    args.fma_c = BinaryKCommon{fma_is_signed, static_cast<SaturationMode>(narrow_to_int(fma_saturation_mode)),
                               static_cast<SubnormalsMode>(narrow_to_int(fma_subnormals_mode)),
                               narrow_to_int(fma_prng_bits)};
    return args;
  }

  inline BinaryKFusedMixedArgs pack_binaryK_fused_mixed(
      const char *op_name, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
      c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t fma_saturation_mode,
      int64_t fma_subnormals_mode, int64_t fma_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(fma_K.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(fma_P.size()),
                           static_cast<int64_t>(fma_bias.size())});
    BinaryKFusedMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.fma_c = BinaryKCommon{fma_is_signed, static_cast<SaturationMode>(narrow_to_int(fma_saturation_mode)),
                               static_cast<SubnormalsMode>(narrow_to_int(fma_subnormals_mode)),
                               narrow_to_int(fma_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
      args.fma[i] = binaryK_widths(fma_K[i], fma_P[i], fma_bias[i], fma_is_signed);
    return args;
  }

  inline SuperfpSplitArgs pack_superfp_split(
      int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias,
      bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
      int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed,
      int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits,
      int64_t acc_prng_bits)
  {
    SuperfpSplitArgs args;
    args.mul = superfp_widths(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias);
    args.mul_c = SuperfpCommon{mul_is_signed, static_cast<SaturationMode>(narrow_to_int(mul_saturation_mode)),
                               narrow_to_int(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc = superfp_widths(acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias);
    args.acc_c = SuperfpCommon{acc_is_signed, static_cast<SaturationMode>(narrow_to_int(acc_saturation_mode)),
                               narrow_to_int(acc_prng_bits)};
    return args;
  }

  inline SuperfpSplitMixedArgs pack_superfp_split_mixed(
      const char *op_name, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
      c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed,
      bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
      c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed,
      int64_t mul_saturation_mode, int64_t acc_saturation_mode, int64_t mul_prng_bits,
      int64_t acc_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(mul_man_bits.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(mul_exp_bits.size()),
                           static_cast<int64_t>(mul_normal_binades.size()),
                           static_cast<int64_t>(mul_bias.size()),
                           static_cast<int64_t>(acc_man_bits.size()),
                           static_cast<int64_t>(acc_exp_bits.size()),
                           static_cast<int64_t>(acc_normal_binades.size()),
                           static_cast<int64_t>(acc_bias.size())});
    SuperfpSplitMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.mul_c = SuperfpCommon{mul_is_signed, static_cast<SaturationMode>(narrow_to_int(mul_saturation_mode)),
                               narrow_to_int(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc_c = SuperfpCommon{acc_is_signed, static_cast<SaturationMode>(narrow_to_int(acc_saturation_mode)),
                               narrow_to_int(acc_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
    {
      args.mul[i] = superfp_widths(mul_man_bits[i], mul_exp_bits[i], mul_normal_binades[i],
                                   mul_bias[i]);
      args.acc[i] = superfp_widths(acc_man_bits[i], acc_exp_bits[i], acc_normal_binades[i],
                                   acc_bias[i]);
    }
    return args;
  }

  inline SuperfpFusedArgs pack_superfp_fused(
      bool fma_quant, int64_t fma_man_bits, int64_t fma_exp_bits, int64_t fma_normal_binades,
      int64_t fma_bias, bool fma_is_signed, int64_t fma_saturation_mode, int64_t fma_prng_bits)
  {
    SuperfpFusedArgs args;
    args.fma_quant = fma_quant;
    args.fma = superfp_widths(fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias);
    args.fma_c = SuperfpCommon{fma_is_signed, static_cast<SaturationMode>(narrow_to_int(fma_saturation_mode)),
                               narrow_to_int(fma_prng_bits)};
    return args;
  }

  inline SuperfpFusedMixedArgs pack_superfp_fused_mixed(
      const char *op_name, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
      c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed,
      int64_t fma_saturation_mode, int64_t fma_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(fma_man_bits.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(fma_exp_bits.size()),
                           static_cast<int64_t>(fma_normal_binades.size()),
                           static_cast<int64_t>(fma_bias.size())});
    SuperfpFusedMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.fma_c = SuperfpCommon{fma_is_signed, static_cast<SaturationMode>(narrow_to_int(fma_saturation_mode)),
                               narrow_to_int(fma_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
      args.fma[i] = superfp_widths(fma_man_bits[i], fma_exp_bits[i], fma_normal_binades[i],
                                   fma_bias[i]);
    return args;
  }

  // Packs the accumulation arguments at the end of a binaryK *_accumulated
  // schema.
  inline AccumulateTail<BinaryKWidths, BinaryKCommon> pack_binaryK_outer(
      int64_t block_size, bool outer_quant, int64_t outer_K, int64_t outer_P, int64_t outer_bias,
      bool outer_is_signed, int64_t outer_saturation_mode, int64_t outer_subnormals_mode,
      int64_t outer_prng_bits)
  {
    AccumulateTail<BinaryKWidths, BinaryKCommon> tail;
    tail.block_size = block_size;
    tail.outer_quant = outer_quant;
    tail.outer = binaryK_widths(outer_K, outer_P, outer_bias, outer_is_signed);
    tail.outer_c = BinaryKCommon{outer_is_signed, static_cast<SaturationMode>(narrow_to_int(outer_saturation_mode)),
                                 static_cast<SubnormalsMode>(narrow_to_int(outer_subnormals_mode)),
                                 narrow_to_int(outer_prng_bits)};
    return tail;
  }

  // Packs the accumulation arguments at the end of a superfp *_accumulated
  // schema.
  inline AccumulateTail<SuperfpWidths, SuperfpCommon> pack_superfp_outer(
      int64_t block_size, bool outer_quant, int64_t outer_man_bits, int64_t outer_exp_bits,
      int64_t outer_normal_binades, int64_t outer_bias, bool outer_is_signed,
      int64_t outer_saturation_mode, int64_t outer_prng_bits)
  {
    AccumulateTail<SuperfpWidths, SuperfpCommon> tail;
    tail.block_size = block_size;
    tail.outer_quant = outer_quant;
    tail.outer = superfp_widths(outer_man_bits, outer_exp_bits, outer_normal_binades, outer_bias);
    tail.outer_c = SuperfpCommon{outer_is_signed, static_cast<SaturationMode>(narrow_to_int(outer_saturation_mode)),
                                 narrow_to_int(outer_prng_bits)};
    return tail;
  }

} // namespace mptorch::gemm
