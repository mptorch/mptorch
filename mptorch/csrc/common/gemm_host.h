#pragma once

// The host-side prologue and epilogue every GEMM entry point repeats, written
// once (finding H1).
//
// B2 moved the kernels and their helpers into cuda/custom_matmul_kernel.cuh /
// cpu/custom_matmul_kernel.h and spread the sixteen entry points over eight
// files, but left the entry points themselves ~1,100 lines that differed only
// in which policy they constructed. Every one repeated the input checks, the
// shape derivation, the float64 narrowing, contiguous(), the output
// allocation, the empty early-return, the enum casts, the RNG-state draw, the
// dtype tag and the widen -- and check_matmul_inputs, matmul_output_shape,
// check_palette_lengths and resolve_prec_idx were copied verbatim between the
// two backends' headers besides.
//
// What is left at each of the sixteen call sites is the schema-to-Args
// packing (common/gemm_args.h) and the name of the backend. X1's batch
// handling lands here, once, rather than in twenty places.
//
// A Backend supplies:
//   using LaunchContext            -- whatever its kernel needs beyond the
//                                     shape: an RNG state, a stream.
//   make_context(use_rng, draws)   -- draws that state on the host.
//   launch(shape, args, ctx)       -- one call per Args type. On CUDA this is
//                                     declared here and defined in a .cu (so
//                                     this ATen-carrying header never reaches
//                                     nvcc); on CPU it is defined inline.

#include "dispatch.h"
#include "gemm_args.h"
#include "gemm_policy.h"
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

  inline void check_palette_lengths(int64_t n, const char *op_name,
                                    std::initializer_list<int64_t> other_lens)
  {
    TORCH_CHECK(n >= 1 && n <= MAX_GEMM_FORMATS, op_name, ": expected 1..", MAX_GEMM_FORMATS,
                " palette formats, got ", n);
    for (int64_t l : other_lens)
      TORCH_CHECK(l == n, op_name, ": every palette parameter list must have length ", n,
                  " (got one of length ", l, ")");
  }

  // Remembers precision maps that have already passed the bounds check below,
  // so a map reused across calls is checked once rather than every time.
  //
  // The check needs the index *values* on the host, and on CUDA that means a
  // device-to-host copy, which drains the stream before the GEMM is even
  // launched: 0.23 ms per call on an RTX 4060 laptop under WSL2, against
  // 0.27 ms for an entire 64^3 mixed GEMM. A map is normally built once and
  // reused, so this skips the copy while the same tensor comes back
  // unchanged. A miss runs exactly the check it always ran, with the same
  // error and the same message.
  //
  // The key is the TensorImpl's address plus its version counter, and the
  // entry holds a weak reference to that impl. The weak reference is what
  // makes comparing raw addresses sound: it keeps the impl's control block
  // (not its storage -- the map itself is not pinned) alive, so no other
  // tensor can be constructed at that address while an entry still names it.
  // The version counter catches every in-place write that goes through ATen.
  //
  // The keyed tensor is the caller's map, not the int32 contiguous copy the
  // kernel reads. Those two are the same object only when the caller already
  // holds an int32 contiguous tensor on the operand's device; for anything
  // else -- an int64 map, which is what torch.zeros/randint/arange hand back
  // without an explicit dtype, or a strided view -- `to(...).contiguous()`
  // allocates a fresh tensor on every call, so keying on it missed every
  // time and paid back the very sync this memo exists to remove (finding
  // G5b). Keying on the input is sound because the converted contents are a
  // pure function of it, and a view shares its base's version counter.
  //
  // Two cases are deliberately never memoized: a check that will run on the
  // host, which has no sync to save, and a tensor with no version counter
  // (created under torch.inference_mode), which has nothing to invalidate
  // against. Both take the full check on every call. A map that lives on the
  // host while the operands are on the device is also skipped: the copy makes
  // a new tensor each call, so there is no stable key to hold.
  //
  // FormatPalette::slot masks the index into range regardless, so nothing
  // here can turn a stale entry into an out-of-bounds read.
  // See dev/gemm_perf_audit.md (findings G5, G5b).
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
  // `on_device` says the check would run on CUDA, i.e. that there is a sync
  // worth skipping; the tensor itself must be on the device too, or the copy
  // that puts it there makes the key useless.
  inline bool prec_idx_memo_key(const Tensor &prec_idx, bool on_device,
                                const c10::TensorImpl *&raw, uint32_t &version)
  {
    if (!on_device || !prec_idx.is_cuda())
      return false;
    c10::TensorImpl *impl = prec_idx.unsafeGetTensorImpl();
    if (!impl->version_counter().enabled())
      return false;
    raw = impl;
    version = impl->version_counter().current_version();
    return true;
  }

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

  inline void remember_validated_prec_idx(const Tensor &prec_idx, bool on_device,
                                          int64_t n_formats)
  {
    const c10::TensorImpl *raw = nullptr;
    uint32_t version = 0;
    if (!prec_idx_memo_key(prec_idx, on_device, raw, version))
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
        c10::weak_intrusive_ptr<c10::TensorImpl, at::UndefinedTensorImpl>(
            prec_idx.getIntrusivePtr()),
        raw, version, n_formats});
  }

  // Validates a spatially-varying mixed-format op's per-output-element
  // precision index and derives the (row_stride, col_stride) pair the kernel
  // reads it with -- accepting a dense [M, N] map, a per-row [M, 1] map, or a
  // per-column [1, N] map (see gemm_policy.h's FormatPalette). Casts to int32
  // on the operand's device, bounds-checks every entry against the palette
  // size (see the memo above for when that costs a device sync), and hands
  // back the contiguous tensor to keep alive across the launch.
  inline void resolve_prec_idx(const Tensor &prec_idx, const Tensor &ref, int64_t M, int64_t N,
                               const char *op_name, int64_t n_formats, Tensor &pidx_out,
                               int64_t &idx_row_stride, int64_t &idx_col_stride)
  {
    TORCH_CHECK(prec_idx.dim() == 2, op_name, ": prec_idx must be 2D, got ", prec_idx.dim(), "D");
    Tensor pidx = prec_idx.to(ref.device(), at::kInt).contiguous();
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
    // Check whichever of the two copies is already on the host: for a host
    // map feeding a device GEMM that is the caller's tensor, and reading its
    // bounds there costs nothing, where reading them from the device copy is
    // the same sync the memo exists to avoid -- and cannot be memoized, since
    // the copy is a new tensor every call. `prec_idx` may still be int64 here,
    // which only makes the check stricter: an index that would wrap on the
    // narrowing to int32 is rejected rather than silently aliased.
    const Tensor &to_check = prec_idx.is_cuda() ? pidx : prec_idx;
    const bool memo = pidx.is_cuda();
    if (!prec_idx_already_validated(prec_idx, memo, n_formats))
    {
      // aminmax is one reduction where min() and max() were two. Reading the
      // pair back differs by device on purpose: on CUDA the two scalars are
      // stacked so the sync is one 2-element copy rather than two; on CPU
      // there is no sync to amortize and .item() avoids the extra allocation.
      auto mm = at::aminmax(to_check);
      int64_t lo, hi;
      if (to_check.is_cuda())
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

  // The two fma_mixed ops reject fma_quant=false, and they do it between
  // check_matmul_inputs and matmul_output_shape. Rather than move that check
  // (which would change which error a doubly-invalid call reports), the
  // driver takes an optional hook that runs exactly where it ran.
  struct NoPrecheck
  {
    void operator()() const {}
  };

  // ----------------------------------------------------------------------
  // The driver
  // ----------------------------------------------------------------------
  //
  // The order below is the order the sixteen entry points ran in, and is kept
  // deliberately: the float64 narrowing decides what dtype the output is
  // allocated as, the empty early-return happens before prec_idx is validated
  // and before the generator is advanced, and the dtype tag is derived after
  // the RNG state is drawn. Each of those is observable.
  template <class Backend, class Args>
  Tensor run_custom_matmul(const char *op_name, const Args &args, Tensor a, Tensor b,
                           bool trans_a, bool trans_b, int64_t accumulate_algorithm,
                           int64_t round_mode)
  {
    static_assert(!Args::mixed, "use run_custom_matmul_mixed for a palette op");
    check_matmul_inputs(a, b, op_name, round_mode, accumulate_algorithm);

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, op_name, M, K, N);

    // float64 in, float64 out, narrowed here instead of on every load so the
    // kernel need not instantiate for double -- same values, see
    // common/dispatch.h and dev/gemm_perf_audit.md (finding G6).
    const bool widen_f64 = mptorch::narrow_float64(a, b);
    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
      return mptorch::widen_float64(c, widen_f64);

    GemmShape s;
    s.M = M;
    s.K = K;
    s.N = N;
    s.trans_a = trans_a;
    s.trans_b = trans_b;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);

    typename Backend::LaunchContext ctx =
        Backend::make_context(s.use_rng, Args::draws_per_k_step * static_cast<uint64_t>(K));

    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    Backend::launch(s, args, ctx);
    return mptorch::widen_float64(c, widen_f64);
  }

  // `make_args` is a factory rather than a ready-made Args because the
  // palette-length check has to happen after the two checks above and before
  // anything indexes the parameter lists -- which is exactly where the
  // sixteen entry points ran it. It returns the packed Args with n_fmt set.
  template <class Backend, class ArgsFactory, class Precheck = NoPrecheck>
  Tensor run_custom_matmul_mixed(const char *op_name, ArgsFactory &&make_args, Tensor a, Tensor b,
                                 Tensor prec_idx, bool trans_a, bool trans_b,
                                 int64_t accumulate_algorithm, int64_t round_mode,
                                 Precheck &&precheck = Precheck{})
  {
    using Args = decltype(make_args());
    static_assert(Args::mixed, "use run_custom_matmul for a single-format op");
    check_matmul_inputs(a, b, op_name, round_mode, accumulate_algorithm);
    precheck();

    int64_t M, K, N;
    matmul_output_shape(a, b, trans_a, trans_b, op_name, M, K, N);

    const Args args = make_args();

    const bool widen_f64 = mptorch::narrow_float64(a, b);
    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c = at::empty({M, N}, a.options());
    if (M == 0 || N == 0)
      return mptorch::widen_float64(c, widen_f64);

    Tensor pidx;
    GemmShape s;
    resolve_prec_idx(prec_idx, a, M, N, op_name, args.n_fmt, pidx, s.idx_row_stride,
                     s.idx_col_stride);
    s.prec_idx = pidx.data_ptr<int32_t>();

    s.M = M;
    s.K = K;
    s.N = N;
    s.trans_a = trans_a;
    s.trans_b = trans_b;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);

    typename Backend::LaunchContext ctx =
        Backend::make_context(s.use_rng, Args::draws_per_k_step * static_cast<uint64_t>(K));

    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    Backend::launch(s, args, ctx);
    return mptorch::widen_float64(c, widen_f64);
  }


  // ----------------------------------------------------------------------
  // Schema -> Args
  // ----------------------------------------------------------------------
  //
  // Each op's flat TORCH_LIBRARY parameter list packed into the struct its
  // policies are built from, once rather than once per backend. The mixed
  // packers run check_palette_lengths before they index any list, which is
  // why the driver calls them through a factory instead of taking a
  // ready-made Args: that check has to land after the shape check and before
  // the first subscript, exactly where it always did.

  inline BinaryKSplitArgs pack_binaryK_split(
      int64_t mul_K, int64_t mul_P, int64_t mul_bias, bool mul_is_signed,
      bool accumulate_quant, int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
      int64_t saturation_mode, int64_t subnormals_mode, int64_t mul_prng_bits,
      int64_t acc_prng_bits)
  {
    const auto sat = static_cast<SaturationMode>(saturation_mode);
    const auto sub = static_cast<SubnormalsMode>(subnormals_mode);
    BinaryKSplitArgs args;
    args.mul = binaryK_widths(mul_K, mul_P, mul_bias, mul_is_signed);
    args.mul_c = BinaryKCommon{mul_is_signed, sat, sub, static_cast<int>(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc = binaryK_widths(acc_K, acc_P, acc_bias, acc_is_signed);
    args.acc_c = BinaryKCommon{acc_is_signed, sat, sub, static_cast<int>(acc_prng_bits)};
    return args;
  }

  inline BinaryKSplitMixedArgs pack_binaryK_split_mixed(
      const char *op_name,
      c10::IntArrayRef mul_K, c10::IntArrayRef mul_P, c10::IntArrayRef mul_bias, bool mul_is_signed,
      bool accumulate_quant, c10::IntArrayRef acc_K, c10::IntArrayRef acc_P,
      c10::IntArrayRef acc_bias, bool acc_is_signed, int64_t saturation_mode,
      int64_t subnormals_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(mul_K.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(mul_P.size()), static_cast<int64_t>(mul_bias.size()),
                           static_cast<int64_t>(acc_K.size()), static_cast<int64_t>(acc_P.size()),
                           static_cast<int64_t>(acc_bias.size())});
    const auto sat = static_cast<SaturationMode>(saturation_mode);
    const auto sub = static_cast<SubnormalsMode>(subnormals_mode);
    BinaryKSplitMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.mul_c = BinaryKCommon{mul_is_signed, sat, sub, static_cast<int>(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc_c = BinaryKCommon{acc_is_signed, sat, sub, static_cast<int>(acc_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
    {
      args.mul[i] = binaryK_widths(mul_K[i], mul_P[i], mul_bias[i], mul_is_signed);
      args.acc[i] = binaryK_widths(acc_K[i], acc_P[i], acc_bias[i], acc_is_signed);
    }
    return args;
  }

  inline BinaryKFusedArgs pack_binaryK_fused(
      bool fma_quant, int64_t fma_K, int64_t fma_P, int64_t fma_bias, bool fma_is_signed,
      int64_t saturation_mode, int64_t subnormals_mode, int64_t fma_prng_bits)
  {
    BinaryKFusedArgs args;
    args.fma_quant = fma_quant;
    args.fma = binaryK_widths(fma_K, fma_P, fma_bias, fma_is_signed);
    args.fma_c = BinaryKCommon{fma_is_signed, static_cast<SaturationMode>(saturation_mode),
                               static_cast<SubnormalsMode>(subnormals_mode),
                               static_cast<int>(fma_prng_bits)};
    return args;
  }

  inline BinaryKFusedMixedArgs pack_binaryK_fused_mixed(
      const char *op_name, c10::IntArrayRef fma_K, c10::IntArrayRef fma_P,
      c10::IntArrayRef fma_bias, bool fma_is_signed, int64_t saturation_mode,
      int64_t subnormals_mode, int64_t fma_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(fma_K.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(fma_P.size()),
                           static_cast<int64_t>(fma_bias.size())});
    BinaryKFusedMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.fma_c = BinaryKCommon{fma_is_signed, static_cast<SaturationMode>(saturation_mode),
                               static_cast<SubnormalsMode>(subnormals_mode),
                               static_cast<int>(fma_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
      args.fma[i] = binaryK_widths(fma_K[i], fma_P[i], fma_bias[i], fma_is_signed);
    return args;
  }

  inline SuperfpSplitArgs pack_superfp_split(
      int64_t mul_man_bits, int64_t mul_exp_bits, int64_t mul_normal_binades, int64_t mul_bias,
      bool mul_is_signed, bool accumulate_quant, int64_t acc_man_bits, int64_t acc_exp_bits,
      int64_t acc_normal_binades, int64_t acc_bias, bool acc_is_signed, int64_t saturation_mode,
      int64_t mul_prng_bits, int64_t acc_prng_bits)
  {
    const auto sat = static_cast<SaturationMode>(saturation_mode);
    SuperfpSplitArgs args;
    args.mul = superfp_widths(mul_man_bits, mul_exp_bits, mul_normal_binades, mul_bias);
    args.mul_c = SuperfpCommon{mul_is_signed, sat, static_cast<int>(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc = superfp_widths(acc_man_bits, acc_exp_bits, acc_normal_binades, acc_bias);
    args.acc_c = SuperfpCommon{acc_is_signed, sat, static_cast<int>(acc_prng_bits)};
    return args;
  }

  inline SuperfpSplitMixedArgs pack_superfp_split_mixed(
      const char *op_name, c10::IntArrayRef mul_man_bits, c10::IntArrayRef mul_exp_bits,
      c10::IntArrayRef mul_normal_binades, c10::IntArrayRef mul_bias, bool mul_is_signed,
      bool accumulate_quant, c10::IntArrayRef acc_man_bits, c10::IntArrayRef acc_exp_bits,
      c10::IntArrayRef acc_normal_binades, c10::IntArrayRef acc_bias, bool acc_is_signed,
      int64_t saturation_mode, int64_t mul_prng_bits, int64_t acc_prng_bits)
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
    const auto sat = static_cast<SaturationMode>(saturation_mode);
    SuperfpSplitMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.mul_c = SuperfpCommon{mul_is_signed, sat, static_cast<int>(mul_prng_bits)};
    args.accumulate_quant = accumulate_quant;
    args.acc_c = SuperfpCommon{acc_is_signed, sat, static_cast<int>(acc_prng_bits)};
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
      int64_t fma_bias, bool fma_is_signed, int64_t saturation_mode, int64_t fma_prng_bits)
  {
    SuperfpFusedArgs args;
    args.fma_quant = fma_quant;
    args.fma = superfp_widths(fma_man_bits, fma_exp_bits, fma_normal_binades, fma_bias);
    args.fma_c = SuperfpCommon{fma_is_signed, static_cast<SaturationMode>(saturation_mode),
                               static_cast<int>(fma_prng_bits)};
    return args;
  }

  inline SuperfpFusedMixedArgs pack_superfp_fused_mixed(
      const char *op_name, c10::IntArrayRef fma_man_bits, c10::IntArrayRef fma_exp_bits,
      c10::IntArrayRef fma_normal_binades, c10::IntArrayRef fma_bias, bool fma_is_signed,
      int64_t saturation_mode, int64_t fma_prng_bits)
  {
    const int64_t n = static_cast<int64_t>(fma_man_bits.size());
    check_palette_lengths(n, op_name,
                          {static_cast<int64_t>(fma_exp_bits.size()),
                           static_cast<int64_t>(fma_normal_binades.size()),
                           static_cast<int64_t>(fma_bias.size())});
    SuperfpFusedMixedArgs args;
    args.n_fmt = static_cast<int>(n);
    args.fma_c = SuperfpCommon{fma_is_signed, static_cast<SaturationMode>(saturation_mode),
                               static_cast<int>(fma_prng_bits)};
    for (int64_t i = 0; i < n; ++i)
      args.fma[i] = superfp_widths(fma_man_bits[i], fma_exp_bits[i], fma_normal_binades[i],
                                   fma_bias[i]);
    return args;
  }

} // namespace mptorch::gemm
