#pragma once

// A convolution's three passes as GEMMs whose operands are gathers of the
// convolution's tensors (dev/gemm_roadmap.md, R-4). Nothing is unfolded: the
// kernels' tile loads compute each operand element's address from the
// geometry below and read it in place, or read zero where the element is
// padding. The K-loop, the Mac and Accumulator policies and the SR keying are
// the GEMM's own, so every format, accumulate algorithm and rounding mode
// applies, and each pass is bit-identical to a GEMM over explicitly gathered
// operands, in the same K order and with the same [batch, M, N] output index
// (tests/test_qconv_gemm.py builds those operands and calls the GEMM op).
//
// The passes, for a convolution of x [B, G*Cg, *in] with W [G*Coutg, Cg, *k]
// into y [B, G*Coutg, *out] (spatial extents flattened: IN, OUT and KK
// elements; nd = 1, 2 or 3 spatial dimensions, padded here to three with
// leading extents of 1):
//
//   pass   result                M      K            N        batch  A[m, k]      B[k, n]
//   FWD    y [B, G, Coutg, OUT]  Coutg  Cg*KK        OUT      B*G    W, dense     im2col(x)
//   IGRAD  dx [B, G, Cg, IN_r]   Cg     Coutg*KK_r   IN_r     B*G    W, the taps  dy at the
//          (one GEMM per class r)                                    of class r   taps' outputs
//   WGRAD  dW [G, Coutg, Cg*KK]  Coutg  B*OUT        Cg*KK    G      dy rows      im2col(x)^T
//
// The groups ride on the kernels' batch dimension, as batch element b*G + g
// (or g), so each result is written in its natural layout, one launch serves
// every group, and the SR key of an element is its index into the kernel's
// [batch, M, N].
//
// The input gradient is one GEMM per *residue class* of the stride. An input
// position h (per dimension) receives a term from kernel tap j exactly when
// t = h + p - j*d is a multiple of s and t/s is an output position. Whether
// s divides t depends only on r = (h + p) mod s and on j, so the positions of
// one class r (h = h0 + m*s) all take the same taps: those with j*d = r
// (mod s), an arithmetic progression of step e = s / gcd(d, s). Summing a
// class over its own taps leaves out the terms a stride makes zero (a share
// 1 - 1/s^nd of the transposed convolution's), which are free for a plain sum
// under a deterministic rounding and nothing else: under SR each spends a
// draw, and under KAHAN each applies the pending compensation. What is left is
// the sum over the terms that exist, and for s = 1 it is the whole sum, one
// class, the transposed convolution itself. The output position of tap j is
// then o = m + (h0 + p - j*d)/s, an exact division the host does once per
// tap, so the device divides nothing. Within a class the K order is (co, j)
// with the taps in *descending* j, which is the order of the
// transposed-convolution-as-convolution reference (correlate with the flipped
// kernel): for s = 1 the result is that reference's. The classes are separate
// GEMMs, launched in row-major order of r with an RNG draw each, as separate
// GEMM calls would be; each writes its positions of dx through a strided
// store (conv_result_col), and the palette maps read the result's columns.
//
// The padding is zeros, and not only the padding the caller names: an output
// position o reads input position o*s - p + i*d along each dimension and a
// position outside [0, in) reads zero, so the right-hand padding is whatever
// the output extent implies. That is what lets padding="same" with an even
// kernel, whose padding is asymmetric, run with no copy of the input. The
// input gradient's terms at the border (an output position outside [0, out))
// are zeros of the same kind, and stay in its sum.
//
// No division runs per element. Everything that is split into coordinates
// (a K index, an output column, the batch element) is divided by one of the
// geometry's extents, and FastDivmod does that with a multiply-high and a
// shift from constants the host derives once per call. On the device a tile
// load decomposes its thread's K index once per 16 K-steps and the output
// column once per thread; on the CPU a pack decomposes each K index and each
// column of its tile once.
//
// This header is ATen-free and includes nothing the policies do not: the .cu
// files that instantiate the gathered kernels see it, and the GEMM kernels
// the other ops run do not change under it (their instantiations never take
// the branches that call it; custom_matmul_kernel.cuh).

#include "bit_helper.h"
#if !defined(__METAL_VERSION__)
#include <cstdint>
#endif

namespace mptorch::gemm
{

  enum class ConvPass : int
  {
    FWD = 0,
    IGRAD = 1,
    WGRAD = 2,
  };

  // n / d and n % d for 0 <= n < 2^31 and 1 <= d < 2^31, by a multiply-high
  // and a shift: Granlund and Montgomery's unsigned division by an invariant
  // integer (PLDI 1994, figure 4.1), with the sum they compute in N + 1 bits
  // done in 32 because n < 2^31. l = ceil(log2 d), mul = floor(2^32 (2^l -
  // d) / d) + 1, and n / d = (mulhi(mul, n) + n) >> l. d = 1 is l = 0 and
  // mul = 1, whose mulhi is 0, so no case is special.
  struct FastDivmod
  {
    uint32_t d = 1;
    uint32_t mul = 1;
    uint32_t shr = 0;

    CUDA_HOST_DEVICE_INLINE int32_t div(int32_t n) const
    {
      const uint32_t un = static_cast<uint32_t>(n);
#if defined(__CUDA_ARCH__)
      const uint32_t hi = __umulhi(un, mul);
#else
      const uint32_t hi = static_cast<uint32_t>((static_cast<uint64_t>(un) * mul) >> 32);
#endif
      return static_cast<int32_t>((hi + un) >> shr);
    }

    CUDA_HOST_DEVICE_INLINE void divmod(int32_t n, MPTORCH_THREAD int32_t &q,
                                        MPTORCH_THREAD int32_t &r) const
    {
      q = div(n);
      r = n - q * static_cast<int32_t>(d);
    }
  };

#if !defined(__METAL_VERSION__)
  inline FastDivmod make_fast_divmod(int32_t d)
  {
    FastDivmod f;
    f.d = static_cast<uint32_t>(d);
    uint32_t l = 0;
    while ((uint64_t(1) << l) < static_cast<uint64_t>(d))
      ++l;
    f.shr = l;
    f.mul = static_cast<uint32_t>(((uint64_t(1) << 32) * ((uint64_t(1) << l) - d)) / d + 1);
    return f;
  }
#endif

  // Everything a gathered tile load needs, flat and by value: it is a kernel
  // parameter on the device. Extents are in elements; the spatial arrays are
  // (depth, height, width) with the leading ones 1 (and stride 1, padding 0,
  // dilation 1) below three dimensions.
  struct ConvGeom
  {
    ConvPass pass = ConvPass::FWD;
    int32_t G = 1;     // groups
    int32_t Cg = 1;    // input channels per group
    int32_t Coutg = 1; // output channels per group
    int32_t in[3] = {1, 1, 1};
    int32_t out[3] = {1, 1, 1};
    int32_t k[3] = {1, 1, 1};
    int32_t s[3] = {1, 1, 1};
    int32_t p[3] = {0, 0, 0};
    int32_t d[3] = {1, 1, 1};
    int32_t KK = 1;  // k[0] * k[1] * k[2]
    int64_t IN = 1;  // in[0] * in[1] * in[2]
    int64_t OUT = 1; // out[0] * out[1] * out[2]
    // The divisors: the kernel's flattened extent and its two trailing
    // partial products, and the same for the output.
    FastDivmod by_KK, by_k12, by_k2, by_OUT, by_out12, by_out2;

    // The input gradient's residue class (IGRAD only; see the top of the
    // file). Per dimension: the class's first input position and its number
    // of positions, its number of taps, the flattened offset in W of each
    // tap step (a tap step is e = s/gcd(d, s) kernel positions, taken
    // downwards from the class's last tap), and the output position of tap
    // step 0 at class position 0, (h0 + p - j_last*d)/s, with the output
    // step per tap step, e*d/s. Then the class's flattened tap count and the
    // flat index in W of its last tap, and the divisors that split a class K
    // index into (co, tap steps) and a class column into positions.
    int32_t cls_h0[3] = {0, 0, 0};
    int32_t cls_n[3] = {1, 1, 1};
    int32_t cls_c[3] = {1, 1, 1};
    int32_t cls_wstep[3] = {0, 0, 0};
    int32_t cls_o0[3] = {0, 0, 0};
    int32_t cls_ostep[3] = {0, 0, 0};
    int32_t cls_KK = 1;
    int32_t cls_jlast = 0;
    FastDivmod by_cKK, by_c12, by_c2, by_n12, by_n2;

    // The column extent of the result the kernel stores into: IN for the
    // input gradient, whose class columns are a strided subset of it, and the
    // GEMM's own N otherwise.
    int64_t res_N = 1;
  };

  // The coordinates a flat index into a [e0, e1, e2] extent names, from the
  // divisors of e1 * e2 and e2.
  CUDA_HOST_DEVICE_INLINE void conv_unflatten(int32_t f, const MPTORCH_THREAD FastDivmod &by12,
                                              const MPTORCH_THREAD FastDivmod &by2,
                                              MPTORCH_THREAD int32_t *c)
  {
    int32_t r;
    by12.divmod(f, c[0], r);
    by2.divmod(r, c[1], c[2]);
  }

  // The batch element the kernel is on, as (sample, group). FWD and IGRAD run
  // one batch element per (sample, group); WGRAD sums over the samples in its
  // K and runs one per group.
  CUDA_HOST_DEVICE_INLINE void conv_batch(const MPTORCH_THREAD ConvGeom &g, int64_t bId,
                                          MPTORCH_THREAD int64_t &b, MPTORCH_THREAD int32_t &grp)
  {
    if (g.pass == ConvPass::WGRAD)
    {
      b = 0;
      grp = static_cast<int32_t>(bId);
    }
    else
    {
      // bId < B * G, which can pass 2^31, so this one division is the
      // 64-bit one; it runs once per thread (or per CPU tile).
      b = bId / g.G;
      grp = static_cast<int32_t>(bId - b * g.G);
    }
  }

  // --- operand A ------------------------------------------------------------
  //
  // A is never padded: every (m, k) in range is an element of W or dy. Its
  // offset is a row part, fixed per output row, plus a K part.

  CUDA_HOST_DEVICE_INLINE int64_t conv_a_row(const MPTORCH_THREAD ConvGeom &g, int32_t grp,
                                             int64_t m)
  {
    switch (g.pass)
    {
    case ConvPass::IGRAD: // W[grp*Coutg + co, ci, tap]: the row is ci
      return (static_cast<int64_t>(grp) * g.Coutg * g.Cg + m) * g.KK;
    case ConvPass::WGRAD: // dy[b, grp*Coutg + m, o]
      return (static_cast<int64_t>(grp) * g.Coutg + m) * g.OUT;
    default: // W[grp*Coutg + m, c, i], dense [Coutg, Cg*KK] per group
      return (static_cast<int64_t>(grp) * g.Coutg + m) * g.Cg * g.KK;
    }
  }

  CUDA_HOST_DEVICE_INLINE int64_t conv_a_k(const MPTORCH_THREAD ConvGeom &g, int32_t k)
  {
    int32_t q, r;
    switch (g.pass)
    {
    case ConvPass::IGRAD:
    {
      // k = (co, t): the class's tap steps t, the taps descending from its
      // last one.
      int32_t t3[3];
      g.by_cKK.divmod(k, q, r);
      conv_unflatten(r, g.by_c12, g.by_c2, t3);
      return static_cast<int64_t>(q) * g.Cg * g.KK + g.cls_jlast -
             (t3[0] * g.cls_wstep[0] + t3[1] * g.cls_wstep[1] + t3[2] * g.cls_wstep[2]);
    }
    case ConvPass::WGRAD: // k = (b, o)
      g.by_OUT.divmod(k, q, r);
      return static_cast<int64_t>(q) * g.G * g.Coutg * g.OUT + r;
    default:
      return k;
    }
  }

  // --- operand B ------------------------------------------------------------
  //
  // B's element is the result tensor's neighbour along the convolution: an
  // input position (FWD, WGRAD) or an output position (IGRAD), at u + v in
  // each dimension, of which u is fixed per output column and v per K index,
  // and which may fall in the padding. The column part and the K part are
  // each a base offset (the channel and sample) and three coordinates.

  struct ConvCol
  {
    int64_t base;
    int32_t u[3];
  };

  struct ConvK
  {
    int64_t base;
    int32_t v[3];
  };

  CUDA_HOST_DEVICE_INLINE ConvCol conv_b_col(const MPTORCH_THREAD ConvGeom &g, int64_t b,
                                             int32_t grp, int32_t n)
  {
    ConvCol col;
    int32_t c3[3];
    switch (g.pass)
    {
    case ConvPass::IGRAD:
      // n is the class's position m, whose output position under tap step t
      // is m + o0 + t*ostep; dy's channel base.
      col.base = (b * g.G + grp) * g.Coutg * g.OUT;
      conv_unflatten(n, g.by_n12, g.by_n2, c3);
      for (int i = 0; i < 3; ++i)
        col.u[i] = c3[i];
      break;
    case ConvPass::WGRAD:
    {
      // n = (c, i): the input channel of this group, and the kernel tap's
      // dilated offset.
      int32_t c, r;
      g.by_KK.divmod(n, c, r);
      col.base = (static_cast<int64_t>(grp) * g.Cg + c) * g.IN;
      conv_unflatten(r, g.by_k12, g.by_k2, c3);
      for (int i = 0; i < 3; ++i)
        col.u[i] = c3[i] * g.d[i];
      break;
    }
    default:
      // n is an output position o; x's group base, and o*s - p.
      col.base = (b * g.G + grp) * g.Cg * g.IN;
      conv_unflatten(n, g.by_out12, g.by_out2, c3);
      for (int i = 0; i < 3; ++i)
        col.u[i] = c3[i] * g.s[i] - g.p[i];
      break;
    }
    return col;
  }

  CUDA_HOST_DEVICE_INLINE ConvK conv_b_k(const MPTORCH_THREAD ConvGeom &g, int32_t k)
  {
    ConvK kk;
    int32_t q, r, c3[3];
    switch (g.pass)
    {
    case ConvPass::IGRAD:
      // k = (co, t): the output channel, and the tap steps' output offset.
      g.by_cKK.divmod(k, q, r);
      kk.base = static_cast<int64_t>(q) * g.OUT;
      conv_unflatten(r, g.by_c12, g.by_c2, c3);
      for (int i = 0; i < 3; ++i)
        kk.v[i] = g.cls_o0[i] + c3[i] * g.cls_ostep[i];
      break;
    case ConvPass::WGRAD:
      // k = (b, o): the sample, and o*s - p.
      g.by_OUT.divmod(k, q, r);
      kk.base = static_cast<int64_t>(q) * g.G * g.Cg * g.IN;
      conv_unflatten(r, g.by_out12, g.by_out2, c3);
      for (int i = 0; i < 3; ++i)
        kk.v[i] = c3[i] * g.s[i] - g.p[i];
      break;
    default:
      // k = (c, i): the channel within the group, and the tap's dilated
      // offset.
      g.by_KK.divmod(k, q, r);
      kk.base = static_cast<int64_t>(q) * g.IN;
      conv_unflatten(r, g.by_k12, g.by_k2, c3);
      for (int i = 0; i < 3; ++i)
        kk.v[i] = c3[i] * g.d[i];
      break;
    }
    return kk;
  }

  // The offset of B's element (k, n) in its tensor, or -1 where it is
  // padding (zero): position u + v in x (FWD, WGRAD) or in dy (IGRAD).
  CUDA_HOST_DEVICE_INLINE int64_t conv_b_offset(const MPTORCH_THREAD ConvGeom &g,
                                                const MPTORCH_THREAD ConvCol &col,
                                                const MPTORCH_THREAD ConvK &kk)
  {
    const MPTORCH_THREAD int32_t *lim = g.pass == ConvPass::IGRAD ? g.out : g.in;
    int32_t c3[3];
    for (int i = 0; i < 3; ++i)
    {
      c3[i] = col.u[i] + kk.v[i];
      if (c3[i] < 0 || c3[i] >= lim[i])
        return -1;
    }
    return col.base + kk.base + (static_cast<int64_t>(c3[0]) * lim[1] + c3[1]) * lim[2] + c3[2];
  }

  // The column of the result tensor the GEMM's column n is: n itself, except
  // for the input gradient, whose class position m is input position
  // h0 + m*s.
  CUDA_HOST_DEVICE_INLINE int64_t conv_result_col(const MPTORCH_THREAD ConvGeom &g, int32_t n)
  {
    if (g.pass != ConvPass::IGRAD)
      return n;
    int32_t m3[3];
    conv_unflatten(n, g.by_n12, g.by_n2, m3);
    int64_t h[3];
    for (int i = 0; i < 3; ++i)
      h[i] = g.cls_h0[i] + static_cast<int64_t>(m3[i]) * g.s[i];
    return (h[0] * g.in[1] + h[1]) * g.in[2] + h[2];
  }

  // The row of the caller's precision-index map an output row is, for the
  // palette ops: the map is indexed by the result tensor's (sample, channel,
  // position), and the kernel's row m of batch element (b, grp) is channel
  // grp*M + m of the result.
  CUDA_HOST_DEVICE_INLINE int64_t conv_result_row(int32_t grp, int64_t M, int64_t m)
  {
    return static_cast<int64_t>(grp) * M + m;
  }

  // An Accumulator policy whose operands are gathered with `geom`. It is the
  // wrapped policy in every other respect (it derives from it), so the
  // kernels drive it through the same calls; what they add for it is behind
  // `if constexpr (is_gathered_v<Accumulator>)`, which no other Accumulator
  // satisfies, so the kernels every other op runs compile to what they
  // compiled before (dev/gemm_roadmap.md, R-4). The geometry rides on the
  // policy because the policy is already the kernels' one by-value parameter
  // that says how an op differs; a new kernel parameter would have changed
  // every existing kernel's signature.
  template <class Acc>
  struct Gathered : Acc
  {
    static constexpr MPTORCH_CONSTANT bool gathered = true;
    ConvGeom geom;
  };

  template <class Accumulator>
  inline constexpr bool is_gathered_v = requires { Accumulator::gathered; };

#if !defined(__METAL_VERSION__)
  // The Args of a conv op: the Args of the GEMM op whose arithmetic it runs
  // (a single-format op's, its AccumulateArgs for KAHAN, BLOCK and TREE, or a
  // palette op's), and the geometry. The backends' launch_conv_as builds
  // `inner`'s policy exactly as the GEMM op does and wraps it in a Gathered.
  template <class Inner>
  struct ConvArgs
  {
    static constexpr bool mixed = Inner::mixed;
    Inner inner{};
    ConvGeom geom{};
  };
#endif

} // namespace mptorch::gemm
