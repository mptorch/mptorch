#pragma once

// block_project: P3109's BlockProject (arXiv:2606.04028, section VIII), the
// element codes of a tensor under scale codes the caller gives, one per tile,
// laid out as block_pack writes its scales. For element X_i of a tile whose
// scale code decodes (decode_scale, block_decode.h) to S, the first case that
// matches gives Z_i:
//
//     NaN        if S or X_i is NaN;
//     0          if S = 0, an infinite X_i included;
//     X_i / S    otherwise,
//
// and Z_i is then projected once, as block_pack projects an element: the
// element format's cast under SAT_FINITE in the call's rounding mode, with
// element i's own random word under SR, then the clamp to elem_max and the
// encode (encode_elem). P3109's fourth case, S = +-inf, cannot arise: a
// power-of-two scale decodes to a power of two or NaN, and the checks on a
// cast scale keep its product with the tensor scale finite. Nor can S be
// negative, since the decode reads the code's magnitude only. A format
// without a scale has S = 1, so that its elements are block_pack's.
//
// X_i / S is computed as block_pack's scaled_elem computes it, a product by
// 2^-k for a power-of-two scale and a quotient otherwise, so that projecting
// onto the scales block_pack chose gives block_pack's codes word for word,
// SR included under the same seed. The one exception is a tile that
// block_pack gave the scale format's NaN because it held a NaN its element
// format had no code for: block_pack keeps the other elements' codes there,
// and block_project writes code 0 for all of them, Z_i being NaN. Both
// decode to NaN.
//
// A NaN element that its tile's scale does not make NaN needs the element
// format's NaN code. A format without one (E2M1, E2M3, E3M2) cannot store it,
// and the kernels report that through `unstored`, which the entry points
// turn into an error rather than store some other element's code. A format
// without a scale stores elem_max's code instead, as block_pack does.
//
// This header uses no ATen and is valid Metal, like block_decode.h: the CPU,
// CUDA and MPS kernels share the step below. The tensors and checks are in
// block_project_host.h.

#include "block_decode.h"
#if !defined(__METAL_VERSION__)
#include "gemm_dtype.h"
#endif

namespace mptorch::block
{

  // A tile's scale as block_project reads it: whether S is NaN, zero or
  // positive and finite, and in the last case the factor that scaled_elem
  // applies, built as tile_scale builds it.
  enum class ProjectedScale : int32_t
  {
    FINITE = 0,
    ZERO = 1,
    NAN_SCALE = 2,
  };

  template <class T>
  struct ProjectScale
  {
    TileScale<T> s;
    ProjectedScale kind;
  };

  // The scale that code `sc` gives a tile. A power-of-two code k decodes to
  // 2^(k - bias) and scales by 2^(bias - k), which the format checks keep a
  // normal value of the carrier; a cast scale's factor is its decoded value,
  // the minifloat times the tensor scale, as tile_scale forms it.
  template <class T>
  CUDA_HOST_DEVICE_INLINE ProjectScale<T> project_scale(uint32_t sc, const MPTORCH_THREAD BlockFormatParamsT<T> &p)
  {
    using F = FloatTraits<T>;
    ProjectScale<T> r;
    r.s.code = sc;
    r.s.factor = T(1);
    r.s.nan_block = false;
    r.kind = ProjectedScale::FINITE;
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::NONE))
      return r;
    const T S = decode_scale<T>(sc, p);
    if (block_isnan(S))
      r.kind = ProjectedScale::NAN_SCALE;
    else if ((block_word(S) & F::ABS_MASK) == 0)
      r.kind = ProjectedScale::ZERO;
    else if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
      r.s.factor = block_pow2<T>(p.scale_bias - static_cast<int>(sc));
    else
      r.s.factor = S;
    return r;
  }

  // Element v's code under its tile's scale: the cases above, then the
  // projection. `is_signed` and `rbits` are as block_cast takes them.
  // `unstored` is set for a NaN that the element format cannot store.
  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE uint32_t project_code(T v, const MPTORCH_THREAD ProjectScale<T> &s, bool is_signed,
                                                typename FloatTraits<T>::word_t rbits,
                                                const MPTORCH_THREAD BlockFormatParamsT<T> &p,
                                                const MPTORCH_THREAD BlockCastT<T> &c,
                                                MPTORCH_THREAD bool &unstored)
  {
    if (s.kind == ProjectedScale::NAN_SCALE)
      return encode_elem(block_quiet_nan<T>(), T(0), p);
    if (block_isnan(v))
    {
      if (p.elem_nan_code == NO_CODE && p.scale_kind != static_cast<int32_t>(ScaleKind::NONE))
        unstored = true;
      return encode_elem(v, T(0), p);
    }
    if (s.kind == ProjectedScale::ZERO)
      return 0u;
    return encode_elem(v, block_cast<T, RM>(scaled_elem(v, s.s, p), is_signed, rbits, c), p);
  }

  // Whether a format can meet a NaN it cannot store: one with a scale and no
  // element NaN code. Only such a call reads `unstored` back.
  template <class T>
  CUDA_HOST_DEVICE_INLINE bool project_can_refuse(const MPTORCH_THREAD BlockFormatParamsT<T> &p)
  {
    return p.elem_nan_code == NO_CODE && p.scale_kind != static_cast<int32_t>(ScaleKind::NONE);
  }

  // The number of consecutive elements whose codes fill whole bytes:
  // 8 / gcd(elem_bits, 8). A block holds a whole number of them, since its
  // codes fill whole bytes, so the CUDA and MPS kernels give each thread one
  // such group and no two threads write one byte.
  CUDA_HOST_DEVICE_INLINE int32_t project_group(int32_t elem_bits)
  {
    const int32_t low = elem_bits & -elem_bits; // its lowest set bit: gcd(elem_bits, 8) is min(low, 8)
    return low >= 8 ? 1 : 8 / low;
  }

#if !defined(__METAL_VERSION__)
  // block_project as its kernels see it: x [batch, rows, cols] with the
  // packed axis last, read through its element strides; scales
  // [batch, row_tiles, scale_cols] and data [batch, rows, row_bytes], both
  // contiguous. Element (b, r, c)'s random word is block_pack's, at index
  // i = (b * rows + r) * cols + c (block_kernels.h). `unstored` points to an
  // int32 flag that a kernel sets to 1 for a NaN it cannot store, or is null
  // when project_can_refuse is false.
  struct BlockProjectJob
  {
    const void *x = nullptr;
    const uint8_t *scales = nullptr;
    uint8_t *data = nullptr;
    int32_t *unstored = nullptr;
    mptorch::GemmDtype dt = mptorch::GemmDtype::Float;
    int64_t batch = 1, rows = 0, cols = 0;
    int64_t xs0 = 0, xs1 = 0, xs2 = 1;
    RoundMode rm = RoundMode::RNE;
  };
#endif

} // namespace mptorch::block
