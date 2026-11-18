#pragma once

// A convolution's three passes (the forward pass and the two gradients) as
// GEMMs whose operands are gathered from the convolution's tensors. Nothing
// is unfolded into a buffer: the kernels' tile loads compute each operand
// element's address from the geometry below and read it in place, or read
// zero where the element is padding. The K-loop, the Mac and accumulator
// policies and the keying of SR's random streams are the GEMM's own, so
// every format, accumulate algorithm and rounding mode applies, and each pass
// gives bit for bit the result of a GEMM over explicitly gathered operands,
// with the same K order and the same [batch, M, N] output index
// (tests/test_qconv_gemm.py builds those operands and calls the GEMM op).
//
// The passes, for a convolution of x [B, G*Cg, *in] with W [G*Coutg, Cg, *k]
// into y [B, G*Coutg, *out]. The spatial extents are flattened into IN, OUT
// and KK elements; there are nd = 1, 2 or 3 spatial dimensions, extended here
// to three with leading extents of 1:
//
//   pass   result                M      K            N        batch  A[m, k]      B[k, n]
//   FWD    y [B, G, Coutg, OUT]  Coutg  Cg*KK        OUT      B*G    W, dense     im2col(x)
//   IGRAD  dx [B, G, Cg, IN_r]   Cg     Coutg*KK_r   IN_r     B*G    W, the taps  dy at the
//          (one GEMM per class r)                                    of class r   taps' outputs
//   WGRAD  dW [G, Coutg, Cg*KK]  Coutg  B*OUT        Cg*KK    G      dy rows      im2col(x)^T
//
// The groups are carried in the kernels' batch dimension, as batch element
// b*G + g (or g, for WGRAD). So each result is written in its own layout, one
// launch serves every group, and the key of an element's SR stream is its
// index in the kernel's [batch, M, N].
//
// The input gradient is one GEMM per *residue class* of the stride. An input
// position h (per dimension) receives a term from kernel tap j exactly when
// t = h + p - j*d is a multiple of s and t/s is an output position. Whether
// s divides t depends only on r = (h + p) mod s and on j, so the positions of
// one class r (h = h0 + m*s) all take the same taps: those with
// j*d = r (mod s), an arithmetic progression of step e = s / gcd(d, s).
// Summing a class over its own taps leaves out the terms that the stride
// makes zero (a fraction 1 - 1/s^nd of the transposed convolution's terms).
// Leaving them out changes nothing for a plain sum under a deterministic
// rounding, and only then: under SR each zero term would spend a draw, and
// under KAHAN each would apply the pending compensation. What remains is the
// sum over the terms that exist; for s = 1 it is the whole sum, a single
// class, the transposed convolution itself. The output position of tap j is
// then o = m + (h0 + p - j*d)/s, an exact division that the host does once
// per tap, so the device divides nothing. Within a class, the K order is
// (co, j) with the taps in *descending* j, which is the order of the
// reference that computes a transposed convolution as a convolution (a
// correlation with the flipped kernel): for s = 1 the result is that
// reference's. The classes are separate GEMMs, launched in row-major order
// of r, each with its own draw of random state, as separate GEMM calls would
// be. Each writes its positions of dx through a strided store
// (conv_result_col), and a palette op's map is indexed by the result's
// columns.
//
// The padding is zeros, and not only the padding the caller asks for: an
// output position o reads input position o*s - p + i*d along each dimension,
// and a position outside [0, in) reads zero, so the right-hand padding is
// whatever the output extent implies. That is what lets padding="same" with
// an even kernel, whose padding is asymmetric, run without copying the
// input. The input gradient's terms at the border (an output position
// outside [0, out)) are zeros of the same kind, and stay in its sum.
//
// No division runs per element. Every value that is split into coordinates
// (a K index, an output column, the batch element) is divided by one of the
// geometry's extents, and FastDivmod does that with a multiply-high and a
// shift, from constants the host derives once per call. On the device, a
// tile load splits its thread's K index once per 16 K-steps and its output
// column once per thread; on the CPU, a pack splits each K index and each
// column of its tile once.
//
// This header uses no ATen, and includes nothing that the policies do not.
// The units that instantiate the gathered kernels include it, and the GEMM
// kernels that the other ops run do not change because of it: their
// instantiations never take the branches that call it (see
// custom_matmul_kernel.cuh).

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

  // n / d and n % d for 0 <= n < 2^31 and 1 <= d < 2^31, computed with a
  // multiply-high and a shift: Granlund and Montgomery's unsigned division by
  // an invariant integer (PLDI 1994, figure 4.1). The sum they compute in
  // N + 1 bits fits in 32 bits here, because n < 2^31. With l = ceil(log2 d)
  // and mul = floor(2^32 (2^l - d) / d) + 1, n / d = (mulhi(mul, n) + n) >> l.
  // For d = 1, l = 0 and mul = 1, whose mulhi is 0, so no case is special.
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
#elif defined(__METAL_VERSION__)
      const uint32_t hi = metal::mulhi(un, mul);
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

  // Everything a gathered tile load needs, as a flat struct passed by value,
  // since it is a kernel parameter on the device. Extents are in elements.
  // The spatial arrays are (depth, height, width); for fewer than three
  // dimensions, the leading entries are an extent of 1, stride 1, padding 0
  // and dilation 1.
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
    // The divisors: the convolution kernel's flattened extent and the
    // products of its last two and its last extent, and the same for the
    // output.
    FastDivmod by_KK, by_k12, by_k2, by_OUT, by_out12, by_out2;

    // The input gradient's residue class (IGRAD only; see the top of the
    // file). Per dimension: the class's first input position and its number
    // of positions; its number of taps; the flattened offset in W of one tap
    // step (a tap step is e = s/gcd(d, s) kernel positions, counted down from
    // the class's last tap); and the output position of tap step 0 at class
    // position 0, (h0 + p - j_last*d)/s, with the change in output position
    // per tap step, e*d/s. Then the class's flattened number of taps, the flat
    // index in W of its last tap, and the divisors that split a class's K
    // index into (co, tap steps) and a class's column into positions.
    int32_t cls_h0[3] = {0, 0, 0};
    int32_t cls_n[3] = {1, 1, 1};
    int32_t cls_c[3] = {1, 1, 1};
    int32_t cls_wstep[3] = {0, 0, 0};
    int32_t cls_o0[3] = {0, 0, 0};
    int32_t cls_ostep[3] = {0, 0, 0};
    int32_t cls_KK = 1;
    int32_t cls_jlast = 0;
    FastDivmod by_cKK, by_c12, by_c2, by_n12, by_n2;

    // The number of columns of the result the kernel stores into: IN for the
    // input gradient, whose class columns are a strided subset of it, and the
    // GEMM's own N otherwise.
    int64_t res_N = 1;
  };

  // Splits a flat index into an [e0, e1, e2] extent into its three
  // coordinates, using the divisors of e1 * e2 and of e2.
  CUDA_HOST_DEVICE_INLINE void conv_unflatten(int32_t f, const MPTORCH_THREAD FastDivmod &by12,
                                              const MPTORCH_THREAD FastDivmod &by2,
                                              MPTORCH_THREAD int32_t *c)
  {
    int32_t r;
    by12.divmod(f, c[0], r);
    by2.divmod(r, c[1], c[2]);
  }

  // The kernel's batch element, as (sample, group). FWD and IGRAD have one
  // batch element per (sample, group); WGRAD sums over the samples within its
  // K, and has one batch element per group.
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
      // bId < B * G, which can exceed 2^31, so this division is a 64-bit
      // one; it runs once per thread (or once per CPU tile).
      b = bId / g.G;
      grp = static_cast<int32_t>(bId - b * g.G);
    }
  }

  // --- operand A ------------------------------------------------------------
  //
  // A has no padding: every (m, k) in range is an element of W or of dy. Its
  // offset is a row part, which is fixed for an output row, plus a K part.

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
      // k = (co, t), where t counts the class's tap steps down from its last
      // tap.
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
  // B's element is an element of the tensor that the result is convolved
  // from: an input position (FWD, WGRAD) or an output position (IGRAD), at
  // u + v in each dimension, where u depends only on the output column and v
  // only on the K index. The position may fall in the padding. The column
  // part and the K part are each a base offset (for the channel and the
  // sample) plus three coordinates.

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
      // is m + o0 + t*ostep. The base is dy's, for this sample and group.
      col.base = (b * g.G + grp) * g.Coutg * g.OUT;
      conv_unflatten(n, g.by_n12, g.by_n2, c3);
      for (int i = 0; i < 3; ++i)
        col.u[i] = c3[i];
      break;
    case ConvPass::WGRAD:
    {
      // n = (c, i): the input channel within this group, and the kernel
      // tap, whose offset is dilated.
      int32_t c, r;
      g.by_KK.divmod(n, c, r);
      col.base = (static_cast<int64_t>(grp) * g.Cg + c) * g.IN;
      conv_unflatten(r, g.by_k12, g.by_k2, c3);
      for (int i = 0; i < 3; ++i)
        col.u[i] = c3[i] * g.d[i];
      break;
    }
    default:
      // n is an output position o. The base is x's, for this sample and
      // group, and the coordinates are o*s - p.
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
      // k = (co, t): the output channel, and the output offset of the tap
      // steps.
      g.by_cKK.divmod(k, q, r);
      kk.base = static_cast<int64_t>(q) * g.OUT;
      conv_unflatten(r, g.by_c12, g.by_c2, c3);
      for (int i = 0; i < 3; ++i)
        kk.v[i] = g.cls_o0[i] + c3[i] * g.cls_ostep[i];
      break;
    case ConvPass::WGRAD:
      // k = (b, o): the sample, and the coordinates o*s - p.
      g.by_OUT.divmod(k, q, r);
      kk.base = static_cast<int64_t>(q) * g.G * g.Cg * g.IN;
      conv_unflatten(r, g.by_out12, g.by_out2, c3);
      for (int i = 0; i < 3; ++i)
        kk.v[i] = c3[i] * g.s[i] - g.p[i];
      break;
    default:
      // k = (c, i): the channel within the group, and the tap, whose offset
      // is dilated.
      g.by_KK.divmod(k, q, r);
      kk.base = static_cast<int64_t>(q) * g.IN;
      conv_unflatten(r, g.by_k12, g.by_k2, c3);
      for (int i = 0; i < 3; ++i)
        kk.v[i] = c3[i] * g.d[i];
      break;
    }
    return kk;
  }

  // The offset of B's element (k, n) in its tensor, position u + v in x
  // (FWD, WGRAD) or in dy (IGRAD), or -1 where that is padding, which reads
  // as zero.
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

  // The column of the result tensor that the GEMM's column n stores to: n
  // itself, except for the input gradient, whose class position m is input
  // position h0 + m*s.
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

  // The row of the caller's precision-index map for an output row, for the
  // palette ops. The map is indexed by the result tensor's (sample, channel,
  // position), and the kernel's row m of batch element (b, grp) is channel
  // grp*M + m of the result.
  CUDA_HOST_DEVICE_INLINE int64_t conv_result_row(int32_t grp, int64_t M, int64_t m)
  {
    return static_cast<int64_t>(grp) * M + m;
  }

#if !defined(__METAL_VERSION__)
  // An accumulator policy whose operands are gathered with `geom`. In every
  // other respect it is the policy it wraps (it derives from it), so the
  // kernels drive it through the same calls. What they add for it is behind
  // `if constexpr (is_gathered_v<Accumulator>)`, which no other accumulator
  // satisfies, so the kernels of every other op compile exactly as they would
  // without it. The geometry is stored in the policy because the policy is
  // already the one by-value kernel parameter that says how an op differs; a
  // new kernel parameter would have changed every existing kernel's
  // signature. Only the CPU and CUDA use it: Metal has no derived classes,
  // and the MPS kernel takes the geometry as a buffer of its own
  // (mps/gemm.metal).
  template <class Acc>
  struct Gathered : Acc
  {
    static constexpr MPTORCH_CONSTANT bool gathered = true;
    ConvGeom geom;
  };

  template <class Accumulator>
  inline constexpr bool is_gathered_v = requires { Accumulator::gathered; };

  // The Args of a conv op: the Args of the GEMM op whose arithmetic it runs
  // (a single-format op's, an AccumulateArgs for KAHAN, BLOCK and TREE, or a
  // palette op's), plus the geometry. The backends' launch_conv_as builds
  // `inner`'s policy exactly as the GEMM op does, and wraps it in a
  // Gathered.
  template <class Inner>
  struct ConvArgs
  {
    static constexpr bool mixed = Inner::mixed;
    Inner inner{};
    ConvGeom geom{};
  };
#endif

} // namespace mptorch::gemm
