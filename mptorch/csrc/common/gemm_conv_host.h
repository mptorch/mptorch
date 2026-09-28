#pragma once

// The host side of the eight conv ops (common/gemm_gather.h): the geometry's
// checks and derivation, the output allocation, and the driver that hands a
// ConvArgs to the backend. The GEMM's own pieces are reused as they are:
// the precision-map resolution and its memo, the dtype tag, the RNG draw and
// the launch context all come from common/gemm_host.h.
//
// A header of its own so that the entry points of the other GEMM ops, which
// include gemm_host.h, compile to what they compiled before these ops existed.
// Only {cpu,cuda,mps}/custom_conv_entry.cpp include it.
//
// A conv op is its GEMM twin's schema with the operand pair and the geometry
// in front of the format arguments:
//
//   custom_conv_<twin>(Tensor a, Tensor b, [Tensor prec_idx,] int conv_pass,
//                      int[] out_size, int[] stride, int[] padding,
//                      int[] dilation, int groups, <the twin's arguments>)
//
// where the twin is the single-format op's *_accumulated schema (so the
// single-format conv ops take every AccumulateAlgorithm, NAIVE with a zero
// block size and no outer format) or a palette op's (NAIVE only). Per pass:
//
//   conv_pass  a                    b                  out_size    result
//   FWD   0    W [Cout, Cg, *k]     x [B, C, *in]      *out        y  [B, Cout, *out]
//   IGRAD 1    W [Cout, Cg, *k]     dy [B, Cout, *out] *in         dx [B, C, *in]
//   WGRAD 2    dy [B, Cout, *out]   x [B, C, *in]      *k          dW [Cout, Cg, *k]
//
// with C = groups * Cg. `out_size` is the one extent no operand carries.
// Nothing ties the output extent to the input's: output position o reads
// input position o*s - p + i*d, zero outside the input, so `padding` is the
// leading padding and the trailing one is whatever the extents imply
// (padding="same" with an even kernel is one such case).

#include "dispatch.h"
#include "gemm_accumulate.h"
#include "gemm_args.h"
#include "gemm_gather.h"
#include "gemm_host.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <vector>

namespace mptorch::gemm
{
  using at::Tensor;

  // The geometry a conv call derives, and what the driver needs besides it.
  struct ConvShape
  {
    ConvGeom geom;
    int64_t M = 0, K = 0, N = 0;
    int64_t batch = 0;     // the kernel's: B * G, or G for WGRAD
    int64_t map_batch = 0; // the precision map's batch extent: B, or 1 for WGRAD
    std::vector<int64_t> out_sizes;
  };

  // Validates a conv call's operands and geometry and derives the GEMM each
  // pass is. Every extent the kernels split into coordinates, and every
  // coordinate they form, has to fit in int32 (FastDivmod's range), which is
  // what the bounds below say; offsets into the tensors are int64 throughout.
  inline ConvShape conv_shape(const Tensor &a, const Tensor &b, int64_t conv_pass,
                              c10::IntArrayRef out_size, c10::IntArrayRef stride,
                              c10::IntArrayRef padding, c10::IntArrayRef dilation, int64_t groups,
                              const char *op_name)
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

    const ConvPass pass = static_cast<ConvPass>(conv_pass);
    // The three operands' extents, by role, from whichever tensors carry them.
    int64_t B, C, Cout, Cg;
    std::vector<int64_t> in(nd), out(nd), k(nd);
    switch (pass)
    {
    case ConvPass::FWD: // a = W, b = x
      Cout = a.size(0);
      Cg = a.size(1);
      B = b.size(0);
      C = b.size(1);
      for (int64_t i = 0; i < nd; ++i)
      {
        k[i] = a.size(2 + i);
        in[i] = b.size(2 + i);
        out[i] = out_size[i];
      }
      break;
    case ConvPass::IGRAD: // a = W, b = dy
      Cout = a.size(0);
      Cg = a.size(1);
      B = b.size(0);
      C = Cg * groups;
      TORCH_CHECK(b.size(1) == Cout, op_name, ": the output gradient has ", b.size(1),
                  " channels and the weight ", Cout);
      for (int64_t i = 0; i < nd; ++i)
      {
        k[i] = a.size(2 + i);
        out[i] = b.size(2 + i);
        in[i] = out_size[i];
      }
      break;
    default: // WGRAD: a = dy, b = x
      B = a.size(0);
      Cout = a.size(1);
      C = b.size(1);
      TORCH_CHECK(b.size(0) == B, op_name, ": the output gradient has ", B,
                  " samples and the input ", b.size(0));
      TORCH_CHECK(C % groups == 0, op_name, ": ", C, " input channels do not split into ", groups,
                  " groups");
      Cg = C / groups;
      for (int64_t i = 0; i < nd; ++i)
      {
        out[i] = a.size(2 + i);
        in[i] = b.size(2 + i);
        k[i] = out_size[i];
      }
      break;
    }
    TORCH_CHECK(C == Cg * groups, op_name, ": the input has ", C, " channels, and a weight of ", Cg,
                " channels per group over ", groups, " groups takes ", Cg * groups);
    TORCH_CHECK(Cout % groups == 0, op_name, ": ", Cout, " output channels do not split into ",
                groups, " groups");

    constexpr int64_t LIM = (int64_t(1) << 31) - 1;
    for (int64_t i = 0; i < nd; ++i)
    {
      TORCH_CHECK(stride[i] >= 1 && dilation[i] >= 1 && padding[i] >= 0, op_name,
                  ": stride and dilation must be at least 1 and padding at least 0, got ",
                  stride, ", ", dilation, " and ", padding);
      TORCH_CHECK(out_size[i] >= 0, op_name, ": out_size must not be negative, got ", out_size);
      // The largest coordinate any pass forms: o*s + i*d (FWD, WGRAD) or
      // h + p (IGRAD), and their difference with the padding.
      const int64_t reach = std::max<int64_t>(out[i], 1) * stride[i] +
                            std::max<int64_t>(k[i], 1) * dilation[i] + padding[i] + in[i];
      TORCH_CHECK(reach <= LIM, op_name, ": the geometry's coordinates along dimension ", i,
                  " pass 2^31");
    }

    ConvShape cs;
    ConvGeom &g = cs.geom;
    g.pass = pass;
    g.G = static_cast<int32_t>(groups);
    g.Cg = static_cast<int32_t>(Cg);
    g.Coutg = static_cast<int32_t>(Cout / groups);
    const int64_t lead = 3 - nd;
    int64_t KK = 1, IN = 1, OUT = 1;
    for (int64_t i = 0; i < nd; ++i)
    {
      g.in[lead + i] = static_cast<int32_t>(in[i]);
      g.out[lead + i] = static_cast<int32_t>(out[i]);
      g.k[lead + i] = static_cast<int32_t>(k[i]);
      g.s[lead + i] = static_cast<int32_t>(stride[i]);
      g.p[lead + i] = static_cast<int32_t>(padding[i]);
      g.d[lead + i] = static_cast<int32_t>(dilation[i]);
      KK *= k[i];
      IN *= in[i];
      OUT *= out[i];
    }
    TORCH_CHECK(KK <= LIM && IN <= LIM && OUT <= LIM, op_name,
                ": a kernel, input or output of more than 2^31 - 1 positions is not supported");
    g.KK = static_cast<int32_t>(KK);
    g.IN = IN;
    g.OUT = OUT;

    // A divisor of 0 only arises when the extent it divides is empty too, and
    // an empty call returns before the kernels run.
    auto div = [](int64_t d) { return make_fast_divmod(static_cast<int32_t>(std::max<int64_t>(d, 1))); };
    g.by_KK = div(KK);
    g.by_k12 = div(int64_t(g.k[1]) * g.k[2]);
    g.by_k2 = div(g.k[2]);
    g.by_OUT = div(OUT);
    g.by_out12 = div(int64_t(g.out[1]) * g.out[2]);
    g.by_out2 = div(g.out[2]);

    switch (pass)
    {
    case ConvPass::FWD:
      cs.M = g.Coutg;
      cs.K = Cg * KK;
      cs.N = OUT;
      cs.batch = B * groups;
      cs.map_batch = B;
      cs.out_sizes = {B, Cout};
      cs.out_sizes.insert(cs.out_sizes.end(), out.begin(), out.end());
      break;
    case ConvPass::IGRAD:
      cs.M = Cg;
      cs.K = int64_t(g.Coutg) * KK;
      cs.N = IN;
      cs.batch = B * groups;
      cs.map_batch = B;
      cs.out_sizes = {B, C};
      cs.out_sizes.insert(cs.out_sizes.end(), in.begin(), in.end());
      break;
    default:
      cs.M = g.Coutg;
      cs.K = B * OUT;
      cs.N = Cg * KK;
      cs.batch = groups;
      cs.map_batch = 1;
      cs.out_sizes = {Cout, Cg};
      cs.out_sizes.insert(cs.out_sizes.end(), k.begin(), k.end());
      break;
    }
    TORCH_CHECK(cs.K <= LIM && cs.N <= LIM, op_name, ": the pass's GEMM has K = ", cs.K,
                " and N = ", cs.N, "; both must be below 2^31");
    g.res_N = cs.N;
    return cs;
  }

  // One GEMM of the input gradient: the geometry of one residue class of the
  // stride (common/gemm_gather.h), and its K and N.
  struct ConvClass
  {
    ConvGeom geom;
    int64_t K = 0, N = 0;
  };

  // The input gradient's residue classes, in row-major order of the residue
  // (r0, r1, r2), leaving out any with no input positions. Per dimension, the
  // positions h of class r are those with (h + p) mod s = r, from
  // h0 = (r - p) mod s in steps of s, and its taps the j in [0, k) with
  // j*d = r (mod s): none unless gcd(d, s) divides r, and otherwise every
  // e = s/gcd(d, s)-th from the first. A class with positions and no taps
  // still runs, with K = 0: its positions are empty sums, zeros, which the
  // kernel writes. At s = 1 there is one class: every position, every tap.
  inline std::vector<ConvClass> conv_igrad_classes(const ConvGeom &g)
  {
    struct Dim
    {
      int32_t h0, n, c, jlast, e, o0, ostep;
    };
    std::vector<Dim> per[3];
    for (int i = 0; i < 3; ++i)
    {
      const int64_t s = g.s[i], d = g.d[i], p = g.p[i], k = g.k[i], in = g.in[i];
      int64_t gcd = s, b = d % s;
      while (b != 0)
      {
        const int64_t t = gcd % b;
        gcd = b;
        b = t;
      }
      const int64_t e = s / gcd;
      for (int64_t r = 0; r < s; ++r)
      {
        Dim dm{};
        dm.h0 = static_cast<int32_t>(((r - p) % s + s) % s);
        dm.n = dm.h0 < in ? static_cast<int32_t>((in - 1 - dm.h0) / s + 1) : 0;
        int64_t first = -1;
        for (int64_t j = 0; j < k && j < s; ++j)
          if ((j * d) % s == r)
          {
            first = j;
            break;
          }
        dm.c = first < 0 ? 0 : static_cast<int32_t>((k - 1 - first) / e + 1);
        dm.e = static_cast<int32_t>(e);
        if (dm.c > 0)
        {
          dm.jlast = static_cast<int32_t>(first + (dm.c - 1) * e);
          dm.o0 = static_cast<int32_t>((dm.h0 + p - dm.jlast * d) / s); // exact
          dm.ostep = static_cast<int32_t>(e * d / s);                     // exact
        }
        per[i].push_back(dm);
      }
    }
    auto div = [](int64_t d) { return make_fast_divmod(static_cast<int32_t>(std::max<int64_t>(d, 1))); };
    const int64_t kstride[3] = {int64_t(g.k[1]) * g.k[2], g.k[2], 1};
    std::vector<ConvClass> classes;
    for (const Dim &a : per[0])
      for (const Dim &b : per[1])
        for (const Dim &c : per[2])
        {
          const Dim *dm[3] = {&a, &b, &c};
          ConvClass cl;
          cl.geom = g;
          ConvGeom &cg = cl.geom;
          int64_t n = 1, taps = 1, jlast = 0;
          for (int i = 0; i < 3; ++i)
          {
            cg.cls_h0[i] = dm[i]->h0;
            cg.cls_n[i] = dm[i]->n;
            cg.cls_c[i] = dm[i]->c;
            cg.cls_wstep[i] = static_cast<int32_t>(dm[i]->e * kstride[i]);
            cg.cls_o0[i] = dm[i]->o0;
            cg.cls_ostep[i] = dm[i]->ostep;
            n *= dm[i]->n;
            taps *= dm[i]->c;
            jlast += dm[i]->jlast * kstride[i];
          }
          if (n == 0)
            continue;
          cg.cls_KK = static_cast<int32_t>(taps);
          cg.cls_jlast = static_cast<int32_t>(jlast);
          cg.by_cKK = div(taps);
          cg.by_c12 = div(int64_t(cg.cls_c[1]) * cg.cls_c[2]);
          cg.by_c2 = div(cg.cls_c[2]);
          cg.by_n12 = div(int64_t(cg.cls_n[1]) * cg.cls_n[2]);
          cg.by_n2 = div(cg.cls_n[2]);
          cg.res_N = g.IN;
          cl.K = int64_t(g.Coutg) * taps;
          cl.N = n;
          classes.push_back(cl);
        }
    return classes;
  }

  // Runs a conv op on an Args its GEMM twin would run on (the single-format
  // op's, its AccumulateArgs, or a palette op's). `prec_idx` is null except
  // on the palette ops, whose `n_fmt` it is checked against. The order is
  // the GEMM driver's: shape checks, output, the empty return, the map, the
  // dtype, the RNG draw.
  template <class Backend, class Inner>
  Tensor run_custom_conv(const char *op_name, const Inner &inner, Tensor a, Tensor b,
                         const Tensor *prec_idx, int64_t conv_pass, c10::IntArrayRef out_size,
                         c10::IntArrayRef stride, c10::IntArrayRef padding,
                         c10::IntArrayRef dilation, int64_t groups, int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode,
                " is not a RoundMode");
    ConvShape cs = conv_shape(a, b, conv_pass, out_size, stride, padding, dilation, groups, op_name);

    Tensor a_c = a.contiguous();
    Tensor b_c = b.contiguous();
    Tensor c = at::empty(cs.out_sizes, a.options());
    if (cs.batch == 0 || cs.M == 0 || cs.N == 0)
      return c;

    GemmShape s;
    Tensor pidx;
    if constexpr (Inner::mixed)
    {
      resolve_prec_idx(*prec_idx, a, cs.map_batch, int64_t(cs.geom.G) * cs.M, cs.N, op_name,
                       inner.n_fmt, pidx, s.idx_row_stride, s.idx_col_stride, s.idx_batch_stride);
      s.prec_idx = pidx.data_ptr<int32_t>();
    }

    s.M = cs.M;
    s.K = cs.K;
    s.N = cs.N;
    s.batch = cs.batch;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);

    s.dt = mptorch::gemm_dtype_of(a_c, b_c, op_name);
    TORCH_CHECK(s.dt != mptorch::GemmDtype::Double, op_name,
                ": the conv ops have binary32 kernels only so far, and float64 operands round in "
                "binary64 (dev/continuation_plan.md, phase G); use float32, float16 or bfloat16 "
                "operands");
    s.a = a_c.data_ptr();
    s.b = b_c.data_ptr();
    s.c = c.data_ptr();

    ConvArgs<Inner> args;
    args.inner = inner;
    const uint64_t draws = draws_per_k_step_of(inner) * words_per_draw(s.dt);
    if (cs.geom.pass != ConvPass::IGRAD)
    {
      typename Backend::LaunchContext ctx =
          Backend::make_context(s.use_rng, draws * static_cast<uint64_t>(s.K));
      bind_tensors(ctx, a_c, b_c, c, Inner::mixed ? &pidx : nullptr);
      args.geom = cs.geom;
      Backend::launch(s, args, ctx);
      return c;
    }
    // The input gradient: one GEMM per residue class of the stride, each a
    // GEMM call of its own, RNG draw included (common/gemm_gather.h).
    for (const ConvClass &cl : conv_igrad_classes(cs.geom))
    {
      s.K = cl.K;
      s.N = cl.N;
      typename Backend::LaunchContext ctx =
          Backend::make_context(s.use_rng, draws * static_cast<uint64_t>(s.K));
      bind_tensors(ctx, a_c, b_c, c, Inner::mixed ? &pidx : nullptr);
      args.geom = cl.geom;
      Backend::launch(s, args, ctx);
    }
    return c;
  }

  // A single-format conv op: NAIVE runs on the twin's Args, KAHAN, BLOCK and
  // TREE on its AccumulateArgs, with the accumulation checked as the
  // *_accumulated GEMM ops check it.
  template <class Backend, class Base, class Widths, class Common>
  Tensor run_custom_conv_single(const char *op_name, const Base &base,
                                const AccumulateTail<Widths, Common> &tail, Tensor a, Tensor b,
                                int64_t conv_pass, c10::IntArrayRef out_size,
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
      return run_custom_conv<Backend>(op_name, base, a, b, nullptr, conv_pass, out_size, stride,
                                      padding, dilation, groups, round_mode);
    }
    AccumulateArgs<Base> args;
    args.base = base;
    args.alg = alg;
    args.block_size = static_cast<int>(tail.block_size);
    args.outer_quant = tail.outer_quant;
    args.outer = tail.outer;
    args.outer_c = tail.outer_c;
    check_accumulate_algorithm(args, op_name, accumulate_algorithm);
    return run_custom_conv<Backend>(op_name, args, a, b, nullptr, conv_pass, out_size, stride,
                                    padding, dilation, groups, round_mode);
  }

  // A palette conv op: NAIVE only, like its GEMM twin. `make_args` is the
  // twin's packer, called after the algorithm and the pre-check, as the GEMM
  // driver calls it.
  template <class Backend, class ArgsFactory, class Precheck = NoPrecheck>
  Tensor run_custom_conv_mixed(const char *op_name, ArgsFactory &&make_args, Tensor a, Tensor b,
                               const Tensor &prec_idx, int64_t conv_pass,
                               c10::IntArrayRef out_size, c10::IntArrayRef stride,
                               c10::IntArrayRef padding, c10::IntArrayRef dilation,
                               int64_t groups, int64_t accumulate_algorithm, int64_t round_mode,
                               Precheck &&precheck = Precheck{})
  {
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op_name, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    check_naive_only(op_name, accumulate_algorithm);
    precheck();
    const auto args = make_args();
    static_assert(std::remove_cvref_t<decltype(args)>::mixed, "use run_custom_conv_single for a single-format op");
    return run_custom_conv<Backend>(op_name, args, a, b, &prec_idx, conv_pass, out_size, stride,
                                    padding, dilation, groups, round_mode);
  }

} // namespace mptorch::gemm
