#pragma once

// The Meta ("fake") side of every op (phase O of dev/continuation_plan.md):
// what the meta device, FakeTensorMode, torch.export and torch.compile run
// in an op's place to learn its result's shape, dtype and strides without
// computing it. Only meta_gemm.cpp and meta_ops.cpp include this header, so
// no object of the real ops changes.
//
// A meta kernel accepts what the CPU op accepts and raises what it raises,
// with the CPU op's messages and in its order; what it leaves out needs
// values or pointers, and still runs when the real op does:
//
//   - the copies, data_ptr, the RNG draw (a meta SR call leaves every
//     generator as it was) and the launch;
//   - the palette ops' prec_idx bounds, which read its values;
//   - CUDA's 16-byte refusal of an in-place quantizer's misaligned view, and
//     its size cap on _carrier_odd;
//   - the MPS stubs' refusals (the odd ops, _carrier_odd).
//
// FakeTensorMode hands a Meta kernel its FakeTensors, which throw on a
// mutable data_ptr, and torch.compile and torch.export give them symbolic
// sizes, on which sizes(), size(i) and numel() throw too. So the real
// drivers (gemm_host.h, gemm_conv_host.h, gemm_block_host.h) cannot run
// here, and neither can their int64 shape helpers. What follows are the
// drivers' twins, taking the drivers' parameter lists (so a meta entry body
// is the CPU entry's body with the driver's name swapped, as on MPS), and
// SymInt twins of the shape helpers, each its original's checks in its
// order with its messages. The checks that read only ranks, dtypes and
// integers are the real ones: check_matmul_inputs, check_accumulate_algorithm,
// check_naive_only, the pack_* packers (with check_palette_lengths),
// check_fma_quant, gemm_dtype_of, checked_block_fmt and block_dtype.
//
// A change to a real driver's checks needs the same change here, and
// tests/test_fake_kernels.py holds the two to the same messages.
//
// The symbolic spelling. A refusal is a TORCH_SYM_CHECK: on a backed size
// (every size torch.compile and torch.export give a tensor) it guards, as a
// plain comparison would, and on an unbacked one it defers to a runtime
// assert instead of failing to trace. The empty-result test is
// TORCH_GUARD_OR_FALSE, so an unbacked size is taken to be nonempty. A
// SymBool combines with & and |. Every / and % comes after the checks that
// make its operands non-negative, where C++'s truncating division and the
// symbolic floor division agree. Inside a Meta kernel every tensor's
// device() is meta, so the device checks pass there; FakeTensorMode raises
// its own error for operands on two fake devices.

#include "block_host.h"
#include "carrier_odd_host.h"
#include "gemm_block_host.h"
#include "gemm_conv_host.h"
#include "gemm_cross_host.h"
#include "gemm_host.h"
#include "stochastic_host.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <vector>

namespace mptorch::meta
{
  using at::Tensor;
  using c10::SymBool;
  using c10::SymInt;
  using SymSizes = std::vector<SymInt>;
  namespace g = mptorch::gemm;
  namespace blk = mptorch::block;

  constexpr int64_t LIM31 = (int64_t(1) << 31) - 1;

  inline Tensor sym_empty(const SymSizes &sizes, const at::TensorOptions &options)
  {
    return at::empty_symint(c10::SymIntArrayRef(sizes), options);
  }

  // The drivers' empty early-return: a result with no elements is returned
  // before the palette map and the dtype are checked.
  inline bool sym_is_empty(const SymInt &batch, const SymInt &M, const SymInt &N)
  {
    return TORCH_GUARD_OR_FALSE(batch.sym_eq(0) | M.sym_eq(0) | N.sym_eq(0));
  }

  // [batch,] M, N, as the drivers allocate it (gemm_host.h's
  // matmul_output_sizes).
  inline SymSizes sym_output_sizes(const SymInt &batch, const SymInt &M, const SymInt &N,
                                   bool batched)
  {
    if (batched)
      return {batch, M, N};
    return {M, N};
  }

  // ----------------------------------------------------------------------
  // The GEMM
  // ----------------------------------------------------------------------

  struct SymMatmulShape
  {
    SymInt M, K, N, batch;
    bool batched = false;
  };

  // matmul_output_shape (gemm_host.h).
  inline SymMatmulShape sym_matmul_output_shape(const Tensor &a, const Tensor &b, bool trans_a,
                                                bool trans_b, const char *op_name)
  {
    SymMatmulShape s;
    const SymInt ba = a.dim() == 3 ? a.sym_size(0) : SymInt(1);
    const SymInt bb = b.dim() == 3 ? b.sym_size(0) : SymInt(1);
    s.batched = (a.dim() == 3 || b.dim() == 3);
    if (ba == bb)
      s.batch = ba;
    else if (ba == 1)
      s.batch = bb;
    else if (bb == 1)
      s.batch = ba;
    else
      TORCH_CHECK(false, op_name, ": batch dimensions must match or be 1 (got ", ba, " and ", bb,
                  ")");

    const SymInt a_r = a.sym_size(-2), a_c = a.sym_size(-1);
    const SymInt b_r = b.sym_size(-2), b_c = b.sym_size(-1);
    s.M = trans_a ? a_c : a_r;
    s.K = trans_a ? a_r : a_c;
    const SymInt K_b = trans_b ? b_c : b_r;
    s.N = trans_b ? b_r : b_c;
    TORCH_SYM_CHECK(s.K.sym_eq(K_b), op_name, ": inner dimensions must match (got ", s.K, " vs ",
                    K_b, ")");
    return s;
  }

  // resolve_prec_idx's shape checks (gemm_host.h), without the bounds check
  // on its entries, which reads them.
  inline void sym_prec_idx_shape(const Tensor &prec_idx, const SymInt &batch, const SymInt &M,
                                 const SymInt &N, const char *op_name)
  {
    TORCH_CHECK(prec_idx.dim() == 2 || prec_idx.dim() == 3, op_name,
                ": prec_idx must be 2D or 3D, got ", prec_idx.dim(), "D");
    const SymInt bsz = prec_idx.dim() == 3 ? prec_idx.sym_size(0) : SymInt(1);
    TORCH_SYM_CHECK(bsz.sym_eq(batch) | bsz.sym_eq(1), op_name,
                    ": prec_idx batch dimension must be ", batch, " or 1, got ", bsz);
    const SymInt r = prec_idx.sym_size(-2), c = prec_idx.sym_size(-1);
    TORCH_SYM_CHECK((r.sym_eq(M) & c.sym_eq(N)) | (r.sym_eq(M) & c.sym_eq(1)) |
                        (r.sym_eq(1) & c.sym_eq(N)),
                    op_name, ": prec_idx shape must be [M, N], [M, 1] or [1, N] (M=", M,
                    ", N=", N, "), got [", r, ", ", c, "]");
  }

  // run_custom_matmul (gemm_host.h).
  template <class Args>
  Tensor meta_custom_matmul(const char *op_name, const Args &args, const Tensor &a,
                            const Tensor &b, bool trans_a, bool trans_b,
                            int64_t accumulate_algorithm, int64_t round_mode)
  {
    static_assert(!Args::mixed, "use meta_custom_matmul_mixed for a palette op");
    g::check_matmul_inputs(a, b, op_name, round_mode, accumulate_algorithm);
    mptorch::sr_canonical_args(args, round_mode, op_name); // the prng_bits refusal
    g::check_accumulate_algorithm(args, op_name, accumulate_algorithm);
    const SymMatmulShape s = sym_matmul_output_shape(a, b, trans_a, trans_b, op_name);
    Tensor c = sym_empty(sym_output_sizes(s.batch, s.M, s.N, s.batched), a.options());
    if (sym_is_empty(s.batch, s.M, s.N))
      return c;
    mptorch::gemm_dtype_of(a, b, op_name);
    return c;
  }

  // run_custom_matmul_mixed (gemm_host.h).
  template <class ArgsFactory, class Precheck = g::NoPrecheck>
  Tensor meta_custom_matmul_mixed(const char *op_name, ArgsFactory &&make_args, const Tensor &a,
                                  const Tensor &b, const Tensor &prec_idx, bool trans_a,
                                  bool trans_b, int64_t accumulate_algorithm, int64_t round_mode,
                                  Precheck &&precheck = Precheck{})
  {
    using Args = decltype(make_args());
    static_assert(Args::mixed, "use meta_custom_matmul for a single-format op");
    g::check_matmul_inputs(a, b, op_name, round_mode, accumulate_algorithm);
    g::check_naive_only(op_name, accumulate_algorithm);
    precheck();
    const SymMatmulShape s = sym_matmul_output_shape(a, b, trans_a, trans_b, op_name);
    mptorch::sr_canonical_args(make_args(), round_mode, op_name); // palette lengths, prng_bits
    Tensor c = sym_empty(sym_output_sizes(s.batch, s.M, s.N, s.batched), a.options());
    if (sym_is_empty(s.batch, s.M, s.N))
      return c;
    sym_prec_idx_shape(prec_idx, s.batch, s.M, s.N, op_name);
    mptorch::gemm_dtype_of(a, b, op_name);
    return c;
  }

  // run_custom_matmul_accumulated (gemm_host.h).
  template <class Base, class Widths, class Common>
  Tensor meta_custom_matmul_accumulated(const char *op_name, const Base &base,
                                        const g::AccumulateTail<Widths, Common> &tail,
                                        const Tensor &a, const Tensor &b, bool trans_a,
                                        bool trans_b, int64_t accumulate_algorithm,
                                        int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(static_cast<AccumulateAlgorithm>(accumulate_algorithm) != AccumulateAlgorithm::NAIVE,
                op_name, ": AccumulateAlgorithm.NAIVE is the op without the _accumulated suffix");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    g::AccumulateArgs<Base> args;
    args.base = base;
    args.alg = static_cast<AccumulateAlgorithm>(accumulate_algorithm);
    args.block_size = static_cast<int>(tail.block_size);
    args.outer_quant = tail.outer_quant;
    args.outer = tail.outer;
    args.outer_c = tail.outer_c;
    return meta_custom_matmul(op_name, args, a, b, trans_a, trans_b, accumulate_algorithm,
                              round_mode);
  }

  // ----------------------------------------------------------------------
  // The convolutions
  // ----------------------------------------------------------------------

  struct SymConvShape
  {
    SymInt M, K, N, batch, map_batch;
    SymSizes out_sizes;
  };

  // conv_shape (gemm_conv_host.h): its checks and the extents of the pass's
  // GEMM, without the kernels' ConvGeom.
  inline SymConvShape sym_conv_shape(const Tensor &a, const Tensor &b, int64_t conv_pass,
                                     c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                     c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                     int64_t groups, const char *op_name)
  {
    TORCH_CHECK(conv_pass >= 0 && conv_pass <= 2, op_name, ": conv_pass must be 0 (forward), "
                "1 (input gradient) or 2 (weight gradient), got ", conv_pass);
    const int64_t nd = static_cast<int64_t>(out_size.size());
    TORCH_CHECK(nd >= 1 && nd <= 3, op_name, ": out_size must have 1, 2 or 3 entries, got ", nd);
    TORCH_CHECK(static_cast<int64_t>(stride.size()) == nd &&
                    static_cast<int64_t>(padding.size()) == nd &&
                    static_cast<int64_t>(dilation.size()) == nd,
                op_name, ": stride, padding and dilation must each have ", nd,
                " entries, like out_size");
    TORCH_CHECK(a.dim() == nd + 2 && b.dim() == nd + 2, op_name, ": a ", nd,
                "-dimensional convolution takes operands of rank ", nd + 2, ", got ", a.dim(),
                " and ", b.dim());
    TORCH_CHECK(a.device() == b.device(), op_name, ": both operands must be on one device, got ",
                a.device(), " and ", b.device());
    TORCH_CHECK(groups >= 1, op_name, ": groups must be at least 1, got ", groups);

    const auto pass = static_cast<g::ConvPass>(conv_pass);
    SymInt B, C, Cout, Cg;
    SymSizes in(nd), out(nd), k(nd);
    switch (pass)
    {
    case g::ConvPass::FWD: // a = W, b = x
      Cout = a.sym_size(0);
      Cg = a.sym_size(1);
      B = b.sym_size(0);
      C = b.sym_size(1);
      for (int64_t i = 0; i < nd; ++i)
      {
        k[i] = a.sym_size(2 + i);
        in[i] = b.sym_size(2 + i);
        out[i] = out_size[i];
      }
      break;
    case g::ConvPass::IGRAD: // a = W, b = dy
      Cout = a.sym_size(0);
      Cg = a.sym_size(1);
      B = b.sym_size(0);
      C = Cg * groups;
      TORCH_SYM_CHECK(b.sym_size(1).sym_eq(Cout), op_name, ": the output gradient has ",
                      b.sym_size(1), " channels and the weight ", Cout);
      for (int64_t i = 0; i < nd; ++i)
      {
        k[i] = a.sym_size(2 + i);
        out[i] = b.sym_size(2 + i);
        in[i] = out_size[i];
      }
      break;
    default: // WGRAD: a = dy, b = x
      B = a.sym_size(0);
      Cout = a.sym_size(1);
      C = b.sym_size(1);
      TORCH_SYM_CHECK(b.sym_size(0).sym_eq(B), op_name, ": the output gradient has ", B,
                      " samples and the input ", b.sym_size(0));
      TORCH_SYM_CHECK((C % groups).sym_eq(0), op_name, ": ", C,
                      " input channels do not split into ", groups, " groups");
      Cg = C / groups;
      for (int64_t i = 0; i < nd; ++i)
      {
        out[i] = a.sym_size(2 + i);
        in[i] = b.sym_size(2 + i);
        k[i] = out_size[i];
      }
      break;
    }
    TORCH_SYM_CHECK(C.sym_eq(Cg * groups), op_name, ": the input has ", C,
                    " channels, and a weight of ", Cg, " channels per group over ", groups,
                    " groups takes ", Cg * groups);
    TORCH_SYM_CHECK((Cout % groups).sym_eq(0), op_name, ": ", Cout,
                    " output channels do not split into ", groups, " groups");

    for (int64_t i = 0; i < nd; ++i)
    {
      TORCH_CHECK(stride[i] >= 1 && dilation[i] >= 1 && padding[i] >= 0, op_name,
                  ": stride and dilation must be at least 1 and padding at least 0, got ",
                  stride, ", ", dilation, " and ", padding);
      TORCH_CHECK(out_size[i] >= 0, op_name, ": out_size must not be negative, got ", out_size);
      const SymInt reach = out[i].max(SymInt(1)) * stride[i] + k[i].max(SymInt(1)) * dilation[i] +
                           padding[i] + in[i];
      TORCH_SYM_CHECK(reach.sym_le(LIM31), op_name,
                      ": the geometry's coordinates along dimension ", i, " pass 2^31");
    }

    SymInt KK = 1, IN = 1, OUT = 1;
    for (int64_t i = 0; i < nd; ++i)
    {
      KK = KK * k[i];
      IN = IN * in[i];
      OUT = OUT * out[i];
    }
    TORCH_SYM_CHECK(KK.sym_le(LIM31) & IN.sym_le(LIM31) & OUT.sym_le(LIM31), op_name,
                    ": a kernel, input or output of more than 2^31 - 1 positions is not supported");

    SymConvShape cs;
    const SymInt Coutg = Cout / groups;
    switch (pass)
    {
    case g::ConvPass::FWD:
      cs.M = Coutg;
      cs.K = Cg * KK;
      cs.N = OUT;
      cs.batch = B * groups;
      cs.map_batch = B;
      cs.out_sizes = {B, Cout};
      cs.out_sizes.insert(cs.out_sizes.end(), out.begin(), out.end());
      break;
    case g::ConvPass::IGRAD:
      cs.M = Cg;
      cs.K = Coutg * KK;
      cs.N = IN;
      cs.batch = B * groups;
      cs.map_batch = B;
      cs.out_sizes = {B, C};
      cs.out_sizes.insert(cs.out_sizes.end(), in.begin(), in.end());
      break;
    default:
      cs.M = Coutg;
      cs.K = B * OUT;
      cs.N = Cg * KK;
      cs.batch = SymInt(groups);
      cs.map_batch = SymInt(1);
      cs.out_sizes = {Cout, Cg};
      cs.out_sizes.insert(cs.out_sizes.end(), k.begin(), k.end());
      break;
    }
    TORCH_SYM_CHECK(cs.K.sym_le(LIM31) & cs.N.sym_le(LIM31), op_name, ": the pass's GEMM has K = ",
                    cs.K, " and N = ", cs.N, "; both must be below 2^31");
    return cs;
  }

  // run_custom_conv (gemm_conv_host.h).
  template <class Inner>
  Tensor meta_custom_conv(const char *op_name, const Inner &inner, const Tensor &a, const Tensor &b,
                          const Tensor *prec_idx, int64_t conv_pass, c10::IntArrayRef out_size,
                          c10::IntArrayRef stride, c10::IntArrayRef padding,
                          c10::IntArrayRef dilation, int64_t groups, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode,
                " is not a RoundMode");
    mptorch::sr_canonical_args(inner, round_mode, op_name); // the prng_bits refusal
    const SymConvShape cs =
        sym_conv_shape(a, b, conv_pass, out_size, stride, padding, dilation, groups, op_name);
    Tensor c = sym_empty(cs.out_sizes, a.options());
    if (sym_is_empty(cs.batch, cs.M, cs.N))
      return c;
    if constexpr (Inner::mixed)
      sym_prec_idx_shape(*prec_idx, cs.map_batch, cs.M * groups, cs.N, op_name);
    mptorch::gemm_dtype_of(a, b, op_name);
    return c;
  }

  // run_custom_conv_single (gemm_conv_host.h).
  template <class Base, class Widths, class Common>
  Tensor meta_custom_conv_single(const char *op_name, const Base &base,
                                 const g::AccumulateTail<Widths, Common> &tail, const Tensor &a,
                                 const Tensor &b, int64_t conv_pass, c10::IntArrayRef out_size,
                                 c10::IntArrayRef stride, c10::IntArrayRef padding,
                                 c10::IntArrayRef dilation, int64_t groups,
                                 int64_t accumulate_algorithm, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op_name,
                ": block_size ", tail.block_size, " is out of range");
    const auto alg = static_cast<AccumulateAlgorithm>(accumulate_algorithm);
    if (alg == AccumulateAlgorithm::NAIVE)
    {
      TORCH_CHECK(tail.block_size == 0 && !tail.outer_quant, op_name,
                  ": AccumulateAlgorithm.NAIVE takes no block_size and no outer format");
      return meta_custom_conv(op_name, base, a, b, nullptr, conv_pass, out_size, stride, padding,
                              dilation, groups, round_mode);
    }
    g::AccumulateArgs<Base> args;
    args.base = base;
    args.alg = alg;
    args.block_size = static_cast<int>(tail.block_size);
    args.outer_quant = tail.outer_quant;
    args.outer = tail.outer;
    args.outer_c = tail.outer_c;
    g::check_accumulate_algorithm(args, op_name, accumulate_algorithm);
    return meta_custom_conv(op_name, args, a, b, nullptr, conv_pass, out_size, stride, padding,
                            dilation, groups, round_mode);
  }

  // run_custom_conv_mixed (gemm_conv_host.h).
  template <class ArgsFactory, class Precheck = g::NoPrecheck>
  Tensor meta_custom_conv_mixed(const char *op_name, ArgsFactory &&make_args, const Tensor &a,
                                const Tensor &b, const Tensor &prec_idx, int64_t conv_pass,
                                c10::IntArrayRef out_size, c10::IntArrayRef stride,
                                c10::IntArrayRef padding, c10::IntArrayRef dilation,
                                int64_t groups, int64_t accumulate_algorithm, int64_t round_mode,
                                Precheck &&precheck = Precheck{})
  {
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    g::check_naive_only(op_name, accumulate_algorithm);
    precheck();
    const auto args = make_args();
    static_assert(std::remove_cvref_t<decltype(args)>::mixed,
                  "use meta_custom_conv_single for a single-format op");
    return meta_custom_conv(op_name, args, a, b, &prec_idx, conv_pass, out_size, stride, padding,
                            dilation, groups, round_mode);
  }

  // ----------------------------------------------------------------------
  // Block formats
  // ----------------------------------------------------------------------

  // block_extents (block_host.h).
  struct SymBlockExtents
  {
    SymInt batch = 1, rows = 0, cols = 0;
    bool batched = false;
  };

  inline SymBlockExtents sym_block_extents(const Tensor &x, const char *op)
  {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, op, " expects a 2D or 3D tensor with the packed axis "
                                                  "last, got ", x.dim(), "D (mptorch.quant.block_pack "
                                                  "moves an axis and folds leading dimensions)");
    SymBlockExtents e;
    e.batched = x.dim() == 3;
    e.batch = e.batched ? x.sym_size(0) : SymInt(1);
    e.rows = x.sym_size(-2);
    e.cols = x.sym_size(-1);
    return e;
  }

  // block_sizes (block_host.h).
  inline SymSizes sym_block_sizes(const SymBlockExtents &e, const SymInt &d1, const SymInt &d2)
  {
    if (e.batched)
      return {e.batch, d1, d2};
    return {d1, d2};
  }

  // The extents make_block_format_params (block_decode.h) derives from a
  // checked format and a [rows, cols] operand.
  struct SymBlockLayout
  {
    SymInt n_blocks, row_bytes, row_tiles, scale_cols;
    int64_t elem_bits = 0;
  };

  inline SymBlockLayout sym_block_layout(const blk::Fmt &f, const SymInt &cols, const SymInt &rows)
  {
    auto at = [&f](blk::BlockFmtField k) { return f[static_cast<int>(k)]; };
    const int64_t bs = at(blk::BlockFmtField::BLOCK_SIZE);
    const int64_t br = at(blk::BlockFmtField::BLOCK_ROWS);
    SymBlockLayout l;
    l.elem_bits = at(blk::BlockFmtField::ELEM_BITS);
    const int64_t block_bytes = bs * l.elem_bits / 8;
    l.n_blocks = (cols + (bs - 1)) / bs;
    l.row_bytes = l.n_blocks * block_bytes;
    l.row_tiles = (rows + (br - 1)) / br;
    l.scale_cols = at(blk::BlockFmtField::SCALE_KIND) == static_cast<int64_t>(blk::ScaleKind::NONE)
                       ? SymInt(0)
                       : l.n_blocks;
    return l;
  }

  // block_pack (cpu/block_kernel.cpp).
  inline std::tuple<Tensor, Tensor> meta_block_pack(const Tensor &x, c10::IntArrayRef fmt,
                                                    double elem_max, double scale_max,
                                                    double tensor_scale, int64_t round_mode)
  {
    constexpr const char *op = "block_pack";
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
    const SymBlockExtents e = sym_block_extents(x, op);
    const mptorch::GemmDtype dt = blk::block_dtype(x.scalar_type(), op);
    blk::Fmt f = blk::checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op,
                                        dt == mptorch::GemmDtype::Double);
    mptorch::sr_canonical_fmt(f, static_cast<int>(blk::BlockFmtField::ELEM_PRNG_BITS), round_mode, op);
    const SymBlockLayout l = sym_block_layout(f, e.cols, e.rows);
    Tensor data = sym_empty(sym_block_sizes(e, e.rows, l.row_bytes), x.options().dtype(at::kByte));
    Tensor scales =
        sym_empty(sym_block_sizes(e, l.row_tiles, l.scale_cols), x.options().dtype(at::kByte));
    return {data, scales};
  }

  // quantize_into's checks (cpu/block_kernel.cpp), shared by block_quant
  // and block_quant_.
  inline void meta_block_quantize_checks(const Tensor &x, c10::IntArrayRef fmt, double elem_max,
                                         double scale_max, double tensor_scale, int64_t round_mode,
                                         const char *op)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
    sym_block_extents(x, op);
    const mptorch::GemmDtype dt = blk::block_dtype(x.scalar_type(), op);
    blk::Fmt f = blk::checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op,
                                        dt == mptorch::GemmDtype::Double);
    mptorch::sr_canonical_fmt(f, static_cast<int>(blk::BlockFmtField::ELEM_PRNG_BITS), round_mode, op);
  }

  // unpack_extents (block_host.h).
  inline SymBlockExtents sym_unpack_extents(const Tensor &data, const Tensor &scales, int64_t cols,
                                            const blk::Fmt &f, const char *op)
  {
    TORCH_CHECK(data.scalar_type() == at::kByte && scales.scalar_type() == at::kByte, op,
                ": data and scales are uint8, got ", data.scalar_type(), " and ",
                scales.scalar_type());
    TORCH_CHECK(data.device() == scales.device(), op, ": data and scales must be on one device");
    TORCH_CHECK((data.dim() == 2 || data.dim() == 3) && scales.dim() == data.dim(), op,
                ": data and scales are both 2D or both 3D, got ", data.dim(), "D and ",
                scales.dim(), "D");
    TORCH_CHECK(cols >= 0, op, ": cols must not be negative, got ", cols);
    SymBlockExtents e;
    e.batched = data.dim() == 3;
    e.batch = e.batched ? data.sym_size(0) : SymInt(1);
    e.rows = data.sym_size(-2);
    e.cols = cols;
    if (e.batched)
      TORCH_SYM_CHECK(scales.sym_size(0).sym_eq(e.batch), op, ": data has ", e.batch,
                      " batch elements and scales ", scales.sym_size(0));
    const SymBlockLayout l = sym_block_layout(f, e.cols, e.rows);
    TORCH_SYM_CHECK(data.sym_size(-1).sym_eq(l.row_bytes) & scales.sym_size(-2).sym_eq(l.row_tiles) &
                        scales.sym_size(-1).sym_eq(l.scale_cols),
                    op, ": ", cols, " elements per row of this format pack into data [..., ",
                    e.rows, ", ", l.row_bytes, "] and scales [..., ", l.row_tiles, ", ",
                    l.scale_cols, "], got ", data.sym_sizes(), " and ", scales.sym_sizes());
    return e;
  }

  // block_unpack (cpu/block_kernel.cpp).
  inline Tensor meta_block_unpack(const Tensor &data, const Tensor &scales, int64_t cols,
                                  c10::IntArrayRef fmt, double elem_max, double scale_max,
                                  double tensor_scale, at::ScalarType dtype)
  {
    constexpr const char *op = "block_unpack";
    const mptorch::GemmDtype dt = blk::block_dtype(dtype, op);
    const blk::Fmt f = blk::checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op,
                                              dt == mptorch::GemmDtype::Double);
    const SymBlockExtents e = sym_unpack_extents(data, scales, cols, f, op);
    return sym_empty(sym_block_sizes(e, e.rows, e.cols), data.options().dtype(dtype));
  }

  // One packed operand of the block GEMM: block_gemm_operand
  // (gemm_block_host.h), whose extents it returns.
  struct SymBlockOperand
  {
    SymInt rows, cols, batch = 1;
    bool batched = false;
  };

  inline SymBlockOperand sym_block_operand(const Tensor &data, const Tensor &scales, int64_t cols,
                                           c10::IntArrayRef fmt, double elem_max, double scale_max,
                                           double tensor_scale, const char *op_name,
                                           const char *which, bool wide)
  {
    TORCH_CHECK(data.scalar_type() == at::kByte && scales.scalar_type() == at::kByte, op_name, ": ",
                which, "'s data and scales are uint8, got ", data.scalar_type(), " and ",
                scales.scalar_type());
    TORCH_CHECK((data.dim() == 2 || data.dim() == 3) && scales.dim() == data.dim(), op_name, ": ",
                which, "'s data and scales are both 2D or both 3D, got ", data.dim(), "D and ",
                scales.dim(), "D");
    TORCH_CHECK(cols >= 0, op_name, ": ", which, "_cols must not be negative, got ", cols);
    const blk::Fmt f = blk::checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op_name, wide);
    SymBlockOperand o;
    o.batched = data.dim() == 3;
    o.batch = o.batched ? data.sym_size(0) : SymInt(1);
    if (o.batched)
      TORCH_SYM_CHECK(scales.sym_size(0).sym_eq(o.batch), op_name, ": ", which, "'s data has ",
                      o.batch, " batch elements and its scales ", scales.sym_size(0));
    o.rows = data.sym_size(-2);
    o.cols = cols;
    const SymBlockLayout l = sym_block_layout(f, o.cols, o.rows);
    TORCH_SYM_CHECK(o.rows.sym_le(LIM31) & o.cols.sym_le(LIM31) & l.row_bytes.sym_le(LIM31) &
                        l.scale_cols.sym_le(LIM31),
                    op_name, ": ", which, " has ", o.rows, " rows of ", o.cols,
                    " elements; the tile loads address a batch element in 32-bit coordinates, so "
                    "both must be below 2^31");
    TORCH_CHECK(l.elem_bits <= g::BLOCK_TABLE_BITS, op_name, ": ", which, "'s element codes are ",
                l.elem_bits, " bits, and the block GEMM decodes codes of at most ",
                g::BLOCK_TABLE_BITS, " (every OCP and NVFP4 format's) through tables; unpack a "
                "wider format and use a float GEMM");
    TORCH_SYM_CHECK(data.sym_size(-1).sym_eq(l.row_bytes) & scales.sym_size(-2).sym_eq(l.row_tiles) &
                        scales.sym_size(-1).sym_eq(l.scale_cols),
                    op_name, ": ", which, ": ", cols, " elements per row of its format pack into "
                    "data [..., ", o.rows, ", ", l.row_bytes, "] and scales [..., ", l.row_tiles,
                    ", ", l.scale_cols, "], got ", data.sym_sizes(), " and ", scales.sym_sizes());
    return o;
  }

  // run_custom_matmul_block_in and run_custom_matmul_block
  // (gemm_block_host.h); the mac's Args only decide which checks ran before.
  inline Tensor meta_run_custom_matmul_block(const char *op_name, const g::BlockGemmCall &call,
                                             int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode, " is not a RoundMode");
#if defined(MPTORCH_NO_FP64)
    TORCH_CHECK(!call.binary64, op_name, ": this build has no float64 GEMM kernels (it was built with "
                                         "MPTORCH_NO_FP64=1); use binary32, or rebuild without it");
#endif
    const bool wide = call.binary64;
    const SymBlockOperand A =
        sym_block_operand(call.a_data, call.a_scales, call.a_cols, call.a_fmt, call.a_elem_max,
                          call.a_scale_max, call.a_tensor_scale, op_name, "a", wide);
    const SymBlockOperand B =
        sym_block_operand(call.b_data, call.b_scales, call.b_cols, call.b_fmt, call.b_elem_max,
                          call.b_scale_max, call.b_tensor_scale, op_name, "b", wide);
    TORCH_CHECK(call.a_data.device() == call.a_scales.device() &&
                    call.b_data.device() == call.a_data.device() &&
                    call.b_scales.device() == call.a_data.device(),
                op_name, ": every operand tensor must be on one device");

    const SymInt M = call.trans_a ? A.cols : A.rows;
    const SymInt K = call.trans_a ? A.rows : A.cols;
    const SymInt N = call.trans_b ? B.cols : B.rows;
    const SymInt K_b = call.trans_b ? B.rows : B.cols;
    TORCH_SYM_CHECK(K.sym_eq(K_b), op_name, ": inner dimensions must match (got ", K, " vs ", K_b,
                    ")");

    SymInt batch;
    if (A.batch == B.batch)
      batch = A.batch;
    else if (A.batch == 1)
      batch = B.batch;
    else if (B.batch == 1)
      batch = A.batch;
    else
      TORCH_CHECK(false, op_name, ": batch dimensions must match or be 1 (got ", A.batch, " and ",
                  B.batch, ")");
    const bool batched = A.batched || B.batched;
    return sym_empty(sym_output_sizes(batch, M, N, batched),
                     call.a_data.options().dtype(wide ? at::kDouble : at::kFloat));
  }

  // custom_matmul_block_entry (gemm_block_host.h).
  inline Tensor meta_custom_matmul_block_entry(
      const g::BlockGemmCall &call, bool fused, bool accumulate_quant, int64_t acc_K, int64_t acc_P,
      int64_t acc_bias, bool acc_is_signed, int64_t accumulate_algorithm, int64_t round_mode,
      int64_t acc_saturation_mode, int64_t acc_subnormals_mode, int64_t acc_prng_bits,
      const g::AccumulateTail<g::BinaryKWidths, g::BinaryKCommon> &tail)
  {
    constexpr const char *op = "custom_matmul_block";
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op, ": block_size ",
                tail.block_size, " is out of range");
    const auto alg = static_cast<AccumulateAlgorithm>(accumulate_algorithm);
    auto run = [&](const auto &base) -> Tensor
    {
      using Base = std::remove_cvref_t<decltype(base)>;
      if (alg == AccumulateAlgorithm::NAIVE)
      {
        TORCH_CHECK(tail.block_size == 0 && !tail.outer_quant, op,
                    ": AccumulateAlgorithm.NAIVE takes no block_size and no outer format");
        int64_t rm = round_mode;
        mptorch::sr_canonical_args(base, rm, op); // the prng_bits refusal
        return meta_run_custom_matmul_block(op, call, round_mode);
      }
      g::AccumulateArgs<Base> args;
      args.base = base;
      args.alg = alg;
      args.block_size = static_cast<int>(tail.block_size);
      args.outer_quant = tail.outer_quant;
      args.outer = tail.outer;
      args.outer_c = tail.outer_c;
      g::check_accumulate_algorithm(args, op, accumulate_algorithm);
      int64_t rm = round_mode;
      mptorch::sr_canonical_args(args, rm, op); // the prng_bits refusal
      return meta_run_custom_matmul_block(op, call, round_mode);
    };
    if (fused)
      return run(g::pack_binaryK_fused(accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
                                       acc_saturation_mode, acc_subnormals_mode, acc_prng_bits));
    g::BlockSplitArgs split;
    split.accumulate_quant = accumulate_quant;
    split.acc = g::binaryK_widths(acc_K, acc_P, acc_bias, acc_is_signed);
    split.acc_c = g::BinaryKCommon{acc_is_signed, static_cast<SaturationMode>(acc_saturation_mode),
                                   static_cast<SubnormalsMode>(acc_subnormals_mode),
                                   static_cast<int>(acc_prng_bits)};
    return run(split);
  }

  // ----------------------------------------------------------------------
  // _carrier_odd
  // ----------------------------------------------------------------------

  // check_carrier_odd (carrier_odd_host.h).
  inline void sym_check_carrier_odd(const Tensor &a, const Tensor &b, const Tensor &c, int64_t op)
  {
    TORCH_CHECK(op >= 0 && op <= 2, "_carrier_odd: op is 0 (fma), 1 (add) or 2 (mul), got ",
                op);
    TORCH_CHECK(a.scalar_type() == at::kFloat || a.scalar_type() == at::kDouble,
                "_carrier_odd: float32 or float64 tensors, got ", a.scalar_type());
    for (const Tensor *t : {&b, &c})
    {
      SymBool same(t->scalar_type() == a.scalar_type() && t->device() == a.device() &&
                   t->dim() == a.dim());
      for (int64_t i = 0; i < a.dim() && t->dim() == a.dim(); ++i)
        same = same & t->sym_size(i).sym_eq(a.sym_size(i));
      TORCH_SYM_CHECK(same, "_carrier_odd: a, b and c must share one dtype, device and shape");
    }
  }

} // namespace mptorch::meta
