#pragma once

// Block formats (OCP MX, NVFP4; mptorch.number.BlockFormat): the layout of a
// packed operand, and the one decode and one encode every block op shares.
// The unpack kernels, the fused quantizers and the block GEMM's tile loads all
// decode through `decode_value` below, and the pack kernels and the fused
// quantizers all encode through `encode_elem` and `tile_scale`, so what
// block_quantize returns is what block_unpack(block_pack(x)) returns and what
// the GEMM multiplies, by construction.
//
// Codes. An element and a scale are each a *minifloat code*: sign-magnitude,
// `exp_bits` exponent and `man_bits` mantissa bits over a bias, of one of the
// two families the casts implement. A binaryK code's exponent field 0 holds
// subnormals m * 2^(1 - bias - man_bits), or, under
// SubnormalsMode::EXTENDED_NORMALS, one more binade of normals whose
// mantissa-zero code is the zero. A superfp code's top `normal_binades`
// binades are normals in the same layout, and the magnitude codes below them,
// 1 to S - 1 with S = (2^exp_bits - normal_binades) * 2^man_bits, are the
// supernormals, the powers of two 2^(supernormal_cutoff + code - 1)
// (cast_superfp.h); code 0 is the zero. So one decode serves both, a
// supernormal branch in front of binaryK's (S is 0 for a binaryK code). The
// element format may spend magnitude codes above its largest value on NaN and
// infinity (OCP's E4M3 and E5M2 do); a scale spends its all-ones magnitude
// code on NaN. A binaryK scale with no mantissa (E8M0) is a power of two,
// 2^(code - bias), set by integer arithmetic; any other scale, binaryK with a
// mantissa (E4M3) or superfp, is cast to nearest even by its family's cast and
// multiplied by a float32 tensor scale.
//
// Layout. A packed operand is `data` [rows, row_bytes] and `scales`
// [row_tiles, scale_cols], uint8, row-major, with any number of leading batch
// dimensions in front of both. A row holds n_blocks = ceil(cols / block_size)
// blocks of block_bytes = block_size * elem_bits / 8 bytes, element i of a
// block at bits [i * elem_bits, (i + 1) * elem_bits) of a little-endian bit
// stream (a 4-bit code in the low nibble first, as torch.float4_e2m1fn_x2 and
// CUTLASS lay them out), the last block's tail zero. A scale is shared by a
// tile of block_rows consecutive rows of one block, so scale (t, j) belongs to
// rows [t * block_rows, (t + 1) * block_rows) of block j, and the last tile
// may be partial. scale_cols is n_blocks, or 0 for a format with no scale.
//
// Numbers are assembled on the carrier's word (FloatTraits), not computed:
// which also keeps an Apple GPU's flush of binary32 subnormals away from them
// when phase H decodes with this header. What the kernels round in is
// binary32 only so far (dev/continuation_plan.md, phase G); the functions are
// written on the carrier T all the same.
//
// ATen-free, and MSL-legal like the other common/ headers, so the GEMM .cu
// files that include it pay no ATen front end and phase H ports rather than
// rewrites it. MPTORCH_BLOCK_DEVICE names the address space of an operand
// pointer on Metal and is empty elsewhere.

#include "bit_helper.h"
#include "cast_binaryK.h"
#include "cast_superfp.h"
#include "modes.h"
#if !defined(__METAL_VERSION__)
#include <cstdint>
#endif

#if defined(__METAL_VERSION__)
#define MPTORCH_BLOCK_DEVICE device
#else
#define MPTORCH_BLOCK_DEVICE
#endif

namespace mptorch::block
{

  // The order of the `int[] fmt` descriptor every block op takes: one builder
  // (mptorch/quant/block.py's _block_format_ints) and one reader
  // (make_block_format_params), which the round trip and the decode sweep
  // hold to each other on every preset.
  enum class BlockFmtField : int
  {
    ELEM_BITS = 0,        // bits per element code, sign included
    ELEM_FAMILY,          // BlockFamily
    ELEM_SIGNED,          // 1 if the element code has a sign bit
    ELEM_EXP_BITS,        // exponent field width
    ELEM_MAN_BITS,        // mantissa field width
    ELEM_BIAS,            // exponent bias
    ELEM_SUBNORMALS,      // a binaryK element's SubnormalsMode (0 for superfp)
    ELEM_NORMAL_BINADES,  // a superfp element's normal binades (0 for binaryK)
    ELEM_NAN_CODE,        // magnitude code spent on NaN, or -1
    ELEM_INF_CODE,        // magnitude code spent on infinity, or -1
    ELEM_PRNG_BITS,       // RoundMode::SR's random bits for the element
    SCALE_KIND,           // ScaleKind
    SCALE_FAMILY,         // BlockFamily
    SCALE_SIGNED,         // the scale code's sign bit (never set in a stored scale)
    SCALE_EXP_BITS,
    SCALE_MAN_BITS,
    SCALE_BIAS,
    SCALE_SUBNORMALS,     // a binaryK scale's SubnormalsMode (0 for superfp)
    SCALE_NORMAL_BINADES, // a superfp scale's normal binades (0 for binaryK)
    SCALE_ROUNDING,       // ScaleRounding: how a tile's scale is chosen
    BLOCK_SIZE,           // elements per block along the packed axis
    BLOCK_ROWS,           // rows per scale tile
    COUNT
  };

  // The format family an element or a scale code belongs to, which selects
  // its cast (cast_binaryK.h or cast_superfp.h) and its supernormal codes.
  enum class BlockFamily : int
  {
    BINARYK = 0,
    SUPERFP = 1,
  };

  enum class ScaleKind : int
  {
    NONE = 0,      // unscaled codes
    POW2 = 1,      // a power of two, 2^(code - bias): E8M0
    MINIFLOAT = 2, // a cast minifloat times a tensor scale: E4M3, or a superfp
  };

  // How a tile's scale is chosen from its largest magnitude `amax`, mirroring
  // mptorch.ScaleRounding by name and value. With r = amax / elem_max (over
  // the tensor scale for a cast scale) and T = elem_max plus half the element
  // format's step above it (BlockCastT::elem_fit):
  //
  //   OCP        a power-of-two scale only: 2^(floor(log2(amax)) - emax).
  //   NEAREST    r rounded to nearest even in the scale format.
  //   UP         the smallest scale >= r, so no element passes elem_max.
  //   SELECTIVE  NEAREST, or the next scale up if the tile's largest element
  //              would then land above T: the one case where the clamp at
  //              elem_max moves it further than rounding to nearest would.
  //
  // Every rule is then clamped to the scale's codes. The rule does not depend
  // on the elements' rounding mode.
  enum class ScaleRounding : int
  {
    OCP = 0,
    NEAREST = 1,
    UP = 2,
    SELECTIVE = 3,
  };

  // What `elem_nan_code` and `elem_inf_code` hold for a format without one.
  constexpr MPTORCH_CONSTANT uint32_t NO_CODE = 0xFFFFFFFFu;

  // Everything decoding and encoding one operand needs, flat (a nested
  // aggregate replicated across the GEMM's instantiations is what once took
  // nvcc's cicc from 18 s to ten minutes; cast_binaryK.h) and by value: the
  // block GEMM carries one per operand in its policy, where the kernel reads
  // it from the parameter bank in the tile load and nowhere else.
  struct BlockFormatParams
  {
    // the element code
    int32_t elem_bits;
    int32_t elem_man_bits;
    int32_t elem_bias;
    uint32_t elem_mag_mask;    // the magnitude's bits
    uint32_t elem_sign_bit;    // 0 for an unsigned element
    uint32_t elem_special_min; // magnitude codes from here up are NaN or infinity
    uint32_t elem_inf_code;    // NO_CODE without one
    uint32_t elem_nan_code;    // NO_CODE without one
    bool elem_extended;
    uint32_t elem_super_codes;  // superfp: magnitude codes below it are supernormals (0: binaryK)
    int32_t elem_super_cutoff;  // superfp: the exponent of the smallest supernormal
    float elem_max;
    int32_t emax; // floor(log2(elem_max))
    // the scale code
    int32_t scale_kind;
    int32_t scale_man_bits;
    int32_t scale_bias;
    uint32_t scale_mag_mask;
    uint32_t scale_nan_code;
    int32_t scale_code_max; // POW2: the largest code that is a power of two
    int32_t scale_subnormals; // MINIFLOAT binaryK: its SubnormalsMode, for the cast
    int32_t scale_family;     // MINIFLOAT: which cast
    uint32_t scale_super_codes;
    int32_t scale_super_cutoff;
    bool scale_extended;
    bool scale_signed;
    float scale_max;
    float scale_min; // MINIFLOAT: the smallest positive scale
    float tensor_scale;
    // the layout
    int32_t block_size;
    int32_t log2_block_size;
    int32_t block_rows;
    int32_t log2_block_rows;
    int32_t block_bytes;
    int64_t cols;
    int64_t rows;
    int64_t n_blocks;
    int64_t row_bytes;
    int64_t row_tiles;
    int64_t scale_cols;
    // the block GEMM's: whether the operand is stored with its K dimension
    // as the rows (read transposed)
    bool trans;
  };

  // --- words and powers of two -------------------------------------------

  template <class T>
  CUDA_HOST_DEVICE_INLINE typename FloatTraits<T>::word_t block_word(T x)
  {
    return reinterpret_cast<const MPTORCH_THREAD typename FloatTraits<T>::word_t &>(x);
  }

  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_float(typename FloatTraits<T>::word_t w)
  {
    return reinterpret_cast<const MPTORCH_THREAD T &>(w);
  }

  // 2^e for e from the carrier's smallest subnormal exponent to its largest,
  // [-149, 127] in binary32, built on the word: FloatTraits::pow2 stops at the
  // smallest normal, and an E8M0 scale reaches 2^-127.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_pow2(int e)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    const W bits = e >= F::MIN_NORMAL_EXP ? static_cast<W>(e + F::BIAS) << F::MAN_BITS
                                          : W(1) << (e - F::MIN_NORMAL_EXP + F::MAN_BITS);
    return block_float<T>(bits);
  }

  // floor(log2(w)) of a nonzero word.
  CUDA_HOST_DEVICE_INLINE int block_leading_bit(uint32_t w)
  {
#if defined(__CUDA_ARCH__)
    return 31 - __clz(w);
#elif defined(__METAL_VERSION__)
    return 31 - static_cast<int>(clz(w));
#else
    return 31 - __builtin_clz(w);
#endif
  }

#if !defined(__METAL_VERSION__)
  CUDA_HOST_DEVICE_INLINE int block_leading_bit(uint64_t w)
  {
#if defined(__CUDA_ARCH__)
    return 63 - __clzll(static_cast<long long>(w));
#else
    return 63 - __builtin_clzll(w);
#endif
  }
#endif

  // x / 2^floor(log2(x)), in [1, 2), of a positive finite carrier value,
  // subnormals included, on the word: exact.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_significand(T x)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    W w = block_word(x) & F::ABS_MASK;
    if ((w >> F::MAN_BITS) == 0)
      w <<= F::MAN_BITS - block_leading_bit(w);
    return block_float<T>((static_cast<W>(F::BIAS) << F::MAN_BITS) | (w & F::MAN_MASK));
  }

  // floor(log2(x)) of a positive finite carrier value, subnormals included,
  // and a value past every exponent for an infinity (which the POW2 scale
  // clamps to its largest code).
  template <class T>
  CUDA_HOST_DEVICE_INLINE int block_floor_log2(T x)
  {
    using F = FloatTraits<T>;
    const auto w = block_word(x) & F::ABS_MASK;
    const int field = static_cast<int>(w >> F::MAN_BITS);
    if (field == static_cast<int>(F::FIELD_MASK))
      return 1 << 20;
    if (field != 0)
      return field - F::BIAS;
    return block_leading_bit(w) + F::MIN_NORMAL_EXP - F::MAN_BITS;
  }

  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_quiet_nan()
  {
    using F = FloatTraits<T>;
    return block_float<T>(F::INF_BITS | F::TIE);
  }

  template <class T>
  CUDA_HOST_DEVICE_INLINE bool block_isnan(T x)
  {
    using F = FloatTraits<T>;
    return (block_word(x) & F::ABS_MASK) > F::INF_BITS;
  }

  // --- codes ---------------------------------------------------------------

  // The magnitude a minifloat magnitude code encodes (specials aside), on the
  // word. A superfp supernormal (mag below super_codes) is 2^(super_cutoff +
  // mag - 1); a binaryK subnormal is m * 2^(1 - bias - man_bits), an integer
  // below 2^16 times a normal power of two, so the product is exact.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_minifloat(uint32_t mag, int man_bits, int bias, bool extended,
                                             uint32_t super_codes, int super_cutoff)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    if (mag < super_codes)
      return mag == 0 ? T(0) : block_pow2<T>(super_cutoff + static_cast<int>(mag) - 1);
    const uint32_t e = mag >> man_bits;
    const uint32_t m = mag & ((1u << man_bits) - 1u);
    if (e == 0 && !extended)
      return static_cast<T>(m) * block_pow2<T>(1 - bias - man_bits);
    if (e == 0 && m == 0)
      return T(0);
    const W bits = (static_cast<W>(static_cast<int>(e) - bias + F::BIAS) << F::MAN_BITS) |
                   (static_cast<W>(m) << (F::MAN_BITS - man_bits));
    return block_float<T>(bits);
  }

  // The magnitude code of `q`, a positive value of the layout (on its grid,
  // finite, and a normal carrier value, which every layout a BlockFormat
  // admits guarantees), or 0 for a zero. The inverse of decode_minifloat: a
  // superfp value below its normal binades is a power of two, whose code is
  // its exponent's distance from the smallest supernormal's, plus one.
  template <class T>
  CUDA_HOST_DEVICE_INLINE uint32_t encode_minifloat(T q, int man_bits, int bias, bool extended,
                                                    uint32_t super_codes, int super_cutoff)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    const W w = block_word(q) & F::ABS_MASK;
    if (w == 0)
      return 0;
    const int exponent = static_cast<int>(w >> F::MAN_BITS) - F::BIAS;
    if (super_codes != 0)
    {
      const int super_code = exponent - super_cutoff + 1;
      if (super_code < static_cast<int>(super_codes))
        return static_cast<uint32_t>(super_code);
    }
    const int field = exponent + bias;
    const W mant = w & F::MAN_MASK;
    if (field >= 1 || (extended && field == 0))
      return (static_cast<uint32_t>(field) << man_bits) |
             static_cast<uint32_t>(mant >> (F::MAN_BITS - man_bits));
    // a subnormal: the significand, implicit bit included, shifted onto the
    // subnormal grid's step 2^(1 - bias - man_bits)
    const W sig = mant | (W(1) << F::MAN_BITS);
    return static_cast<uint32_t>(sig >> (F::MAN_BITS - man_bits + 1 - field));
  }

  // An element code's value: NaN and infinity first, then the magnitude,
  // then the sign, on the word. The sign-only code decodes to +0.0 (no cast
  // in the library returns -0.0), and a NaN code to the canonical quiet NaN.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_elem(uint32_t code, const MPTORCH_THREAD BlockFormatParams &p)
  {
    using F = FloatTraits<T>;
    const uint32_t mag = code & p.elem_mag_mask;
    typename F::word_t w;
    if (mag >= p.elem_special_min)
    {
      if (mag != p.elem_inf_code)
        return block_quiet_nan<T>();
      w = F::INF_BITS;
    }
    else
    {
      w = block_word(decode_minifloat<T>(mag, p.elem_man_bits, p.elem_bias, p.elem_extended,
                                         p.elem_super_codes, p.elem_super_cutoff));
    }
    if ((code & p.elem_sign_bit) != 0 && w != 0)
      w |= F::SIGN_MASK;
    return block_float<T>(w);
  }

  // A stored scale code's factor: the power of two, or the minifloat times
  // the tensor scale, in that order. The scale's NaN code decodes to NaN, and
  // so does any POW2 code past its largest.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_scale(uint32_t sc, const MPTORCH_THREAD BlockFormatParams &p)
  {
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
      return static_cast<int32_t>(sc) > p.scale_code_max ? block_quiet_nan<T>()
                                                          : block_pow2<T>(static_cast<int>(sc) - p.scale_bias);
    const uint32_t mag = sc & p.scale_mag_mask;
    if (mag == p.scale_nan_code)
      return block_quiet_nan<T>();
    return FloatTraits<T>::rn_mul(decode_minifloat<T>(mag, p.scale_man_bits, p.scale_bias, p.scale_extended,
                                                      p.scale_super_codes, p.scale_super_cutoff),
                                  static_cast<T>(p.tensor_scale));
  }

  // The value an element code and its tile's scale code decode to:
  // decode_elem(code) * decode_scale(scale), one binary32 product, which is
  // exact for every OCP format. A product that underflows is +0.0, never
  // -0.0.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_value(uint32_t code, uint32_t sc,
                                         const MPTORCH_THREAD BlockFormatParams &p)
  {
    using F = FloatTraits<T>;
    T v = decode_elem<T>(code, p);
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::NONE))
      return v;
    v = F::rn_mul(v, decode_scale<T>(sc, p));
    return (block_word(v) & F::ABS_MASK) == 0 ? T(0) : v;
  }

  // --- the bit stream ------------------------------------------------------

  // Element i's code of a block at `blk`: bits [i * bits, (i + 1) * bits) of
  // the block's little-endian stream. Codes are at most 16 bits, so a code
  // spans at most three bytes, and only the bytes it spans are read: never
  // past the block.
  CUDA_HOST_DEVICE_INLINE uint32_t extract_code(const MPTORCH_BLOCK_DEVICE uint8_t *blk, int64_t i,
                                                int bits)
  {
    const int64_t off = i * bits;
    const MPTORCH_BLOCK_DEVICE uint8_t *b = blk + (off >> 3);
    const int sh = static_cast<int>(off & 7);
    uint32_t w = b[0];
    if (sh + bits > 8)
      w |= static_cast<uint32_t>(b[1]) << 8;
    if (sh + bits > 16)
      w |= static_cast<uint32_t>(b[2]) << 16;
    return (w >> sh) & ((1u << bits) - 1u);
  }

  // ORs element i's code into a zeroed block buffer, the inverse of
  // extract_code. Host code (the CPU pack); the device kernels assemble their
  // bytes by lanes instead.
  CUDA_HOST_DEVICE_INLINE void put_code(MPTORCH_THREAD uint8_t *blk, int64_t i, uint32_t code, int bits)
  {
    const int64_t off = i * bits;
    MPTORCH_THREAD uint8_t *b = blk + (off >> 3);
    const int sh = static_cast<int>(off & 7);
    const uint32_t w = code << sh;
    b[0] |= static_cast<uint8_t>(w);
    if (sh + bits > 8)
      b[1] |= static_cast<uint8_t>(w >> 8);
    if (sh + bits > 16)
      b[2] |= static_cast<uint8_t>(w >> 16);
  }

  // Byte j of a block from its codes, for a kernel that holds the codes in
  // an array: the OR of every code whose bit range meets bits [8j, 8j + 8).
  CUDA_HOST_DEVICE_INLINE uint8_t assemble_byte(const MPTORCH_THREAD uint16_t *codes, int j, int bits)
  {
    const int lo = 8 * j;
    uint32_t byte = 0;
    for (int i = lo / bits; i * bits < lo + 8; ++i)
    {
      const int sh = i * bits - lo;
      const uint32_t c = codes[i];
      byte |= sh >= 0 ? (c << sh) : (c >> -sh);
    }
    return static_cast<uint8_t>(byte);
  }

  // --- reading a packed operand ---------------------------------------------

  // Element (r, c) of a packed operand, in *storage* coordinates: row r of
  // the data, element c along the packed axis. `data` and `scales` point at
  // this batch element's arrays.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T load_block_elem(const MPTORCH_BLOCK_DEVICE uint8_t *data,
                                            const MPTORCH_BLOCK_DEVICE uint8_t *scales, int64_t r,
                                            int64_t c, const MPTORCH_THREAD BlockFormatParams &p)
  {
    const int64_t blk = c >> p.log2_block_size;
    const uint32_t code = extract_code(data + r * p.row_bytes + blk * p.block_bytes,
                                       c & (p.block_size - 1), p.elem_bits);
    const uint32_t sc =
        p.scale_kind == static_cast<int32_t>(ScaleKind::NONE)
            ? 0u
            : static_cast<uint32_t>(scales[(r >> p.log2_block_rows) * p.scale_cols + blk]);
    return decode_value<T>(code, sc, p);
  }

  // Element (i, k) of a GEMM operand, logically [rows of the result, K]: the
  // data's row i and its element k, or, for an operand stored with K as its
  // rows (`trans`), row k and element i. The same function reads A (i = m)
  // and B (i = n), since the block GEMM reads both with K along their packed
  // axis unless told otherwise.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T load_block_operand(const MPTORCH_BLOCK_DEVICE uint8_t *data,
                                               const MPTORCH_BLOCK_DEVICE uint8_t *scales, int64_t i,
                                               int64_t k, const MPTORCH_THREAD BlockFormatParams &p)
  {
    return p.trans ? load_block_elem<T>(data, scales, k, i, p) : load_block_elem<T>(data, scales, i, k, p);
  }

  // --- writing one -----------------------------------------------------------

  // The casts a pack runs: the element format's (SAT_FINITE; the clamp to
  // elem_max follows it) and a MINIFLOAT scale's, each of the family its
  // descriptor names, built once per launch. The other family's constants are
  // carried unread, so the kernels choose a cast with one uniform branch
  // rather than one instantiation per family. The scale rule and what it
  // compares with are here too rather than in BlockFormatParams, which the
  // block GEMM carries per operand and which a pack-only field would change.
  template <class T>
  struct BlockCastT
  {
    int32_t elem_family;
    bool elem_signed;
    SubnormalsMode elem_sub;
    int prng_bits;
    BinaryKParamsT<T> elem_bk;
    SuperfpParamsT<T> elem_sfp;
    int32_t scale_family;
    BinaryKParamsT<T> scale_bk;
    SuperfpParamsT<T> scale_sfp;
    int32_t scale_rounding; // ScaleRounding
    // T: elem_max plus half the element format's step above it, the largest
    // scaled element that rounding to nearest would still take to elem_max
    T elem_fit;
    // POW2: elem_max's significand, and elem_fit over elem_max's power of
    // two, which the rules compare an amax's significand with
    T max_sig;
    T fit_sig;
  };

  // A tile's scale: the code stored, and what an element is taken onto the
  // element grid with: multiplied by `factor` (POW2, 2^-shared_exp: exact),
  // or divided by it (MINIFLOAT, the decoded scale times the tensor scale).
  // `nan_block` is a tile holding a NaN its element format has no code for,
  // whose scale is then the scale's NaN.
  template <class T>
  struct TileScale
  {
    uint32_t code;
    T factor;
    bool nan_block;
  };

  // The step a POW2 rule other than OCP takes from OCP's shared exponent e =
  // floor(log2(amax)) - emax: -1, 0 or 1. With amax = sig * 2^(e + emax) and
  // elem_max = max_sig * 2^emax, amax / elem_max = (sig / max_sig) * 2^e, so
  // the rules compare sig with multiples of max_sig, all exact: NEAREST's
  // midpoints are 0.75 and 1.5 times it (a tie between two powers of two goes
  // to the even exponent, as every cast in the library breaks one), UP's
  // bound is max_sig itself, and SELECTIVE bumps NEAREST's choice if amax at
  // that scale, sig * 2^(emax - d), is above elem_fit.
  template <class T>
  CUDA_HOST_DEVICE_INLINE int pow2_scale_step(T amax, int e, const MPTORCH_THREAD BlockCastT<T> &c)
  {
    using F = FloatTraits<T>;
    const T sig = block_significand(amax);
    if (c.scale_rounding == static_cast<int32_t>(ScaleRounding::UP))
      return sig <= c.max_sig ? 0 : 1;
    int d = 0;
    if (sig < c.max_sig)
    {
      const T mid = F::rn_mul(T(0.75), c.max_sig);
      if (sig < mid || (sig == mid && ((e - 1) & 1) == 0))
        d = -1;
    }
    else if (sig > c.max_sig)
    {
      const T mid = F::rn_mul(T(1.5), c.max_sig);
      if (sig > mid || (sig == mid && ((e + 1) & 1) == 0))
        d = 1;
    }
    if (c.scale_rounding == static_cast<int32_t>(ScaleRounding::SELECTIVE))
    {
      // amax / 2^(e + d) = sig * 2^(emax - d) > elem_fit = fit_sig * 2^emax
      const T at_scale = d < 0 ? F::rn_mul(T(2), sig) : (d > 0 ? F::rn_mul(T(0.5), sig) : sig);
      if (at_scale > c.fit_sig)
        ++d;
    }
    return d;
  }

  // A MINIFLOAT scale's cast of `ratio`, to nearest even or up, then clamped:
  // at most scale_max, and the smallest positive scale for a zero.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T cast_scale(T ratio, bool up, const MPTORCH_THREAD BlockFormatParams &p,
                                       const MPTORCH_THREAD BlockCastT<T> &c)
  {
    const bool superfp = c.scale_family == static_cast<int32_t>(BlockFamily::SUPERFP);
    const SubnormalsMode sub = static_cast<SubnormalsMode>(p.scale_subnormals);
    T sv = up ? (superfp ? cast_superfp_up(ratio, p.scale_signed, c.scale_sfp)
                         : cast_binaryK_up(ratio, p.scale_signed, sub, c.scale_bk))
              : (superfp ? cast_superfp_nearest_even(ratio, p.scale_signed, c.scale_sfp)
                         : cast_binaryK_nearest_even(ratio, p.scale_signed, sub, c.scale_bk));
    const T smax = static_cast<T>(p.scale_max);
    if (block_word(sv) > block_word(smax))
      sv = smax;
    if (block_word(sv) == 0)
      sv = static_cast<T>(p.scale_min);
    return sv;
  }

  // The scale of a tile whose largest magnitude is `amax` (NaNs left out, and
  // reported in `has_nan`), by the rule c.scale_rounding (ScaleRounding
  // above). `c` holds the MINIFLOAT scale's cast, built once per launch under
  // SAT_FINITE; the scale's rounding never depends on the elements'.
  //
  //   POW2       amax == 0: code 0. Otherwise shared = floor(log2(amax)) -
  //              emax, moved by pow2_scale_step for a rule other than OCP,
  //              clamped to the codes, and an element is v * 2^-shared.
  //   MINIFLOAT  s = amax / (elem_max * S_t) cast to nearest even, or up, in
  //              the scale format, then min(s, scale_max), and the smallest
  //              positive scale for a zero; SELECTIVE takes the cast up
  //              instead where the tile's largest element, amax / (s * S_t),
  //              the very quotient scaled_elem computes, is above elem_fit.
  //              An element is v / (s * S_t).
  template <class T>
  CUDA_HOST_DEVICE_INLINE TileScale<T> tile_scale(T amax, bool has_nan,
                                                  const MPTORCH_THREAD BlockFormatParams &p,
                                                  const MPTORCH_THREAD BlockCastT<T> &c)
  {
    using F = FloatTraits<T>;
    TileScale<T> s{0u, T(1), false};
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
    {
      int shared = -p.scale_bias;
      if (block_word(amax) != 0)
      {
        const int lo = -p.scale_bias, hi = p.scale_code_max - p.scale_bias;
        int e = block_floor_log2(amax) - p.emax;
        if (c.scale_rounding != static_cast<int32_t>(ScaleRounding::OCP) &&
            (block_word(amax) & F::ABS_MASK) < F::INF_BITS)
          e += pow2_scale_step(amax, e, c);
        shared = e < lo ? lo : (e > hi ? hi : e);
      }
      s.code = static_cast<uint32_t>(shared + p.scale_bias);
      s.factor = block_pow2<T>(-shared);
    }
    else if (p.scale_kind == static_cast<int32_t>(ScaleKind::MINIFLOAT))
    {
      const T ts = static_cast<T>(p.tensor_scale);
      const T ratio = amax / F::rn_mul(static_cast<T>(p.elem_max), ts);
      T sv = cast_scale(ratio, c.scale_rounding == static_cast<int32_t>(ScaleRounding::UP), p, c);
      T factor = F::rn_mul(sv, ts);
      if (c.scale_rounding == static_cast<int32_t>(ScaleRounding::SELECTIVE) && amax / factor > c.elem_fit)
      {
        sv = cast_scale(ratio, true, p, c);
        factor = F::rn_mul(sv, ts);
      }
      s.code = encode_minifloat(sv, p.scale_man_bits, p.scale_bias, p.scale_extended, p.scale_super_codes,
                                p.scale_super_cutoff);
      s.factor = factor;
    }
    if (has_nan && p.elem_nan_code == NO_CODE && p.scale_kind != static_cast<int32_t>(ScaleKind::NONE))
    {
      s.code = p.scale_nan_code;
      s.nan_block = true;
    }
    return s;
  }

  // An element taken onto its format's grid by the tile's scale. A POW2
  // product is exact unless it leaves binary32's normals, which takes a block
  // spanning more than 2^125; one that underflows to zero keeps its sign and a
  // nonzero magnitude, so that the directed modes still round it away from
  // zero as they would the exact value (every element format's smallest value
  // is at least 2^-125, so every such input is below half of it).
  template <class T>
  CUDA_HOST_DEVICE_INLINE T scaled_elem(T v, const MPTORCH_THREAD TileScale<T> &s,
                                        const MPTORCH_THREAD BlockFormatParams &p)
  {
    using F = FloatTraits<T>;
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
    {
      const T x = F::rn_mul(v, s.factor);
      const auto vw = block_word(v);
      if ((block_word(x) & F::ABS_MASK) == 0 && (vw & F::ABS_MASK) != 0)
        return block_float<T>((vw & F::SIGN_MASK) | 1u);
      return x;
    }
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::MINIFLOAT))
      return v / s.factor;
    return v;
  }

  // The element format's cast in rounding mode RM, a template parameter as in
  // the GEMM's policies (gemm_policy.h), so a kernel carries one cast body per
  // family. `rbits` is RoundMode::SR's random word and is read by no other
  // mode. `is_signed` is a parameter rather than c.elem_signed so that the CPU
  // kernel, which instantiates on the sign, can hand it a constant.
  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE T block_cast_binaryK(T x, bool is_signed, SubnormalsMode sub, int prng_bits,
                                               typename FloatTraits<T>::word_t rbits,
                                               const MPTORCH_THREAD BinaryKParamsT<T> &p)
  {
    if constexpr (RM == RoundMode::RNA)
      return cast_binaryK_nearest_away(x, is_signed, sub, p);
    else if constexpr (RM == RoundMode::RU)
      return cast_binaryK_up(x, is_signed, sub, p);
    else if constexpr (RM == RoundMode::RD)
      return cast_binaryK_down(x, is_signed, sub, p);
    else if constexpr (RM == RoundMode::RZ)
      return cast_binaryK_zero(x, is_signed, sub, p);
    else if constexpr (RM == RoundMode::RO)
      return cast_binaryK_odd(x, is_signed, sub, p);
    else if constexpr (RM == RoundMode::SR)
      return cast_binaryK_stochastic(x, rbits, prng_bits, is_signed, sub, p);
    else
      return cast_binaryK_nearest_even(x, is_signed, sub, p);
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE T block_cast_superfp(T x, bool is_signed, int prng_bits,
                                               typename FloatTraits<T>::word_t rbits,
                                               const MPTORCH_THREAD SuperfpParamsT<T> &p)
  {
    if constexpr (RM == RoundMode::RNA)
      return cast_superfp_nearest_away(x, is_signed, p);
    else if constexpr (RM == RoundMode::RU)
      return cast_superfp_up(x, is_signed, p);
    else if constexpr (RM == RoundMode::RD)
      return cast_superfp_down(x, is_signed, p);
    else if constexpr (RM == RoundMode::RZ)
      return cast_superfp_zero(x, is_signed, p);
    else if constexpr (RM == RoundMode::RO)
      return cast_superfp_odd(x, is_signed, p);
    else if constexpr (RM == RoundMode::SR)
      return cast_superfp_stochastic(x, rbits, prng_bits, is_signed, p);
    else
      return cast_superfp_nearest_even(x, is_signed, p);
  }

  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE T block_cast(T x, bool is_signed, typename FloatTraits<T>::word_t rbits,
                                       const MPTORCH_THREAD BlockCastT<T> &c)
  {
    if (c.elem_family == static_cast<int32_t>(BlockFamily::SUPERFP))
      return block_cast_superfp<T, RM>(x, is_signed, c.prng_bits, rbits, c.elem_sfp);
    return block_cast_binaryK<T, RM>(x, is_signed, c.elem_sub, c.prng_bits, rbits, c.elem_bk);
  }

  // One element's code. `rounded` is the element format's cast of the scaled
  // value (under SAT_FINITE, in the call's rounding mode), which the caller
  // computes so the rounding mode stays a template parameter of its kernel.
  // The clamp folds the codes above elem_max, which OCP spends on NaN and
  // infinity, onto elem_max. A NaN input takes the element's NaN code, or
  // code 0 under a scale that has become NaN, or elem_max without either.
  template <class T>
  CUDA_HOST_DEVICE_INLINE uint32_t encode_elem(T v, T rounded, const MPTORCH_THREAD BlockFormatParams &p)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    const W max_w = block_word(static_cast<T>(p.elem_max));
    if (block_isnan(v))
    {
      if (p.elem_nan_code != NO_CODE)
        return p.elem_nan_code;
      if (p.scale_kind != static_cast<int32_t>(ScaleKind::NONE))
        return 0u;
      return encode_minifloat(block_float<T>(max_w), p.elem_man_bits, p.elem_bias, p.elem_extended,
                              p.elem_super_codes, p.elem_super_cutoff);
    }
    const W w = block_word(rounded);
    const W mag = (w & F::ABS_MASK) > max_w ? max_w : (w & F::ABS_MASK);
    const uint32_t code = encode_minifloat(block_float<T>(mag), p.elem_man_bits, p.elem_bias, p.elem_extended,
                                           p.elem_super_codes, p.elem_super_cutoff);
    return (w & F::SIGN_MASK) != 0 && code != 0 ? (code | p.elem_sign_bit) : code;
  }

  // --- building the parameters ------------------------------------------------

#if !defined(__METAL_VERSION__)
  inline int32_t block_log2(int64_t v)
  {
    int32_t l = 0;
    while ((int64_t(1) << l) < v)
      ++l;
    return l;
  }

  // A superfp code's supernormals: how many magnitude codes they take (the
  // zero's included), and the exponent of the smallest, from the cast's own
  // region cutoffs.
  inline void set_supernormals(uint32_t &codes, int32_t &cutoff, int man_bits, int exp_bits,
                               int normal_binades, int bias)
  {
    codes = static_cast<uint32_t>(((1 << exp_bits) - normal_binades) << man_bits);
    cutoff = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias).supernormal_cutoff;
  }

  // The parameters of one operand from its descriptor (`f`, BlockFmtField's
  // order, validated by the caller), its three floats and its extents: `cols`
  // along the packed axis, `rows` across it.
  inline BlockFormatParams make_block_format_params(const int64_t *f, double elem_max, double scale_max,
                                                    double tensor_scale, int64_t cols, int64_t rows,
                                                    bool trans = false)
  {
    auto at = [f](BlockFmtField k) { return f[static_cast<int>(k)]; };
    BlockFormatParams p{};
    p.elem_bits = static_cast<int32_t>(at(BlockFmtField::ELEM_BITS));
    const bool elem_signed = at(BlockFmtField::ELEM_SIGNED) != 0;
    p.elem_man_bits = static_cast<int32_t>(at(BlockFmtField::ELEM_MAN_BITS));
    p.elem_bias = static_cast<int32_t>(at(BlockFmtField::ELEM_BIAS));
    if (at(BlockFmtField::ELEM_FAMILY) == static_cast<int64_t>(BlockFamily::SUPERFP))
      set_supernormals(p.elem_super_codes, p.elem_super_cutoff, p.elem_man_bits,
                       static_cast<int>(at(BlockFmtField::ELEM_EXP_BITS)),
                       static_cast<int>(at(BlockFmtField::ELEM_NORMAL_BINADES)), p.elem_bias);
    p.elem_mag_mask = (1u << (p.elem_bits - (elem_signed ? 1 : 0))) - 1u;
    p.elem_sign_bit = elem_signed ? (1u << (p.elem_bits - 1)) : 0u;
    p.elem_extended = at(BlockFmtField::ELEM_SUBNORMALS) == static_cast<int64_t>(SubnormalsMode::EXTENDED_NORMALS);
    const int64_t nan = at(BlockFmtField::ELEM_NAN_CODE), inf = at(BlockFmtField::ELEM_INF_CODE);
    p.elem_nan_code = nan < 0 ? NO_CODE : static_cast<uint32_t>(nan);
    p.elem_inf_code = inf < 0 ? NO_CODE : static_cast<uint32_t>(inf);
    p.elem_special_min = p.elem_mag_mask + 1u;
    if (p.elem_nan_code < p.elem_special_min)
      p.elem_special_min = p.elem_nan_code;
    if (p.elem_inf_code < p.elem_special_min)
      p.elem_special_min = p.elem_inf_code;
    p.elem_max = static_cast<float>(elem_max);
    p.emax = block_floor_log2(p.elem_max);

    p.scale_kind = static_cast<int32_t>(at(BlockFmtField::SCALE_KIND));
    const bool scale_signed = at(BlockFmtField::SCALE_SIGNED) != 0;
    const int32_t s_exp = static_cast<int32_t>(at(BlockFmtField::SCALE_EXP_BITS));
    p.scale_man_bits = static_cast<int32_t>(at(BlockFmtField::SCALE_MAN_BITS));
    p.scale_bias = static_cast<int32_t>(at(BlockFmtField::SCALE_BIAS));
    const int64_t s_sub = at(BlockFmtField::SCALE_SUBNORMALS);
    p.scale_extended = s_sub == static_cast<int64_t>(SubnormalsMode::EXTENDED_NORMALS);
    p.scale_subnormals = static_cast<int32_t>(s_sub);
    p.scale_family = static_cast<int32_t>(at(BlockFmtField::SCALE_FAMILY));
    p.scale_signed = scale_signed;
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::MINIFLOAT) &&
        p.scale_family == static_cast<int32_t>(BlockFamily::SUPERFP))
      set_supernormals(p.scale_super_codes, p.scale_super_cutoff, p.scale_man_bits, s_exp,
                       static_cast<int>(at(BlockFmtField::SCALE_NORMAL_BINADES)), p.scale_bias);
    p.scale_max = static_cast<float>(scale_max);
    p.tensor_scale = 1.0f;
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
    {
      p.scale_mag_mask = (1u << s_exp) - 1u;
      p.scale_nan_code = p.scale_mag_mask;
      p.scale_code_max = static_cast<int32_t>(p.scale_nan_code) - 1;
    }
    else if (p.scale_kind == static_cast<int32_t>(ScaleKind::MINIFLOAT))
    {
      p.scale_mag_mask = (1u << (s_exp + p.scale_man_bits)) - 1u;
      p.scale_nan_code = p.scale_mag_mask;
      p.scale_code_max = static_cast<int32_t>(p.scale_nan_code) - 1;
      // The smallest positive scale: the first subnormal or supernormal, the
      // smallest normal under NORMALS, or the first code above the extended
      // binade's foot.
      const uint32_t first = p.scale_super_codes == 0 && s_sub == static_cast<int64_t>(SubnormalsMode::NORMALS)
                                 ? (1u << p.scale_man_bits)
                                 : 1u;
      p.scale_min = decode_minifloat<float>(first, p.scale_man_bits, p.scale_bias, p.scale_extended,
                                            p.scale_super_codes, p.scale_super_cutoff);
      p.tensor_scale = static_cast<float>(tensor_scale);
    }

    p.block_size = static_cast<int32_t>(at(BlockFmtField::BLOCK_SIZE));
    p.log2_block_size = block_log2(p.block_size);
    p.block_rows = static_cast<int32_t>(at(BlockFmtField::BLOCK_ROWS));
    p.log2_block_rows = block_log2(p.block_rows);
    p.block_bytes = p.block_size * p.elem_bits / 8;
    p.cols = cols;
    p.rows = rows;
    p.n_blocks = (cols + p.block_size - 1) / p.block_size;
    p.row_bytes = p.n_blocks * p.block_bytes;
    p.row_tiles = (rows + p.block_rows - 1) / p.block_rows;
    p.scale_cols = p.scale_kind == static_cast<int32_t>(ScaleKind::NONE) ? 0 : p.n_blocks;
    p.trans = trans;
    return p;
  }

  // The casts a pack runs (BlockCastT), from the descriptor and elem_max:
  // the element format's under SAT_FINITE, and a MINIFLOAT scale's. A
  // family's constants are built only for the family named; the other's stay
  // zero, unread. Then the scale rule's constants: elem_fit is elem_max plus
  // half the step to the value its code's successor would have if the format
  // went on, 2 * elem_max where the format steps by powers of two (a superfp
  // supernormal, or no mantissa), the subnormal step below the normals, and
  // an ulp of elem_max's binade otherwise.
  template <class T>
  BlockCastT<T> make_block_cast(const int64_t *f, double elem_max)
  {
    auto at = [f](BlockFmtField k) { return static_cast<int>(f[static_cast<int>(k)]); };
    BlockCastT<T> c{};
    c.elem_family = at(BlockFmtField::ELEM_FAMILY);
    c.elem_signed = at(BlockFmtField::ELEM_SIGNED) != 0;
    c.elem_sub = static_cast<SubnormalsMode>(at(BlockFmtField::ELEM_SUBNORMALS));
    c.prng_bits = at(BlockFmtField::ELEM_PRNG_BITS);
    if (c.elem_family == static_cast<int>(BlockFamily::SUPERFP))
      c.elem_sfp = make_superfp_params<T>(at(BlockFmtField::ELEM_MAN_BITS), at(BlockFmtField::ELEM_EXP_BITS),
                                          at(BlockFmtField::ELEM_NORMAL_BINADES), at(BlockFmtField::ELEM_BIAS),
                                          SaturationMode::SAT_FINITE);
    else
      c.elem_bk = make_binaryK_params<T>(at(BlockFmtField::ELEM_MAN_BITS), at(BlockFmtField::ELEM_EXP_BITS),
                                         at(BlockFmtField::ELEM_BIAS), c.elem_signed, SaturationMode::SAT_FINITE,
                                         c.elem_sub == SubnormalsMode::EXTENDED_NORMALS);
    c.scale_family = at(BlockFmtField::SCALE_FAMILY);
    if (at(BlockFmtField::SCALE_KIND) == static_cast<int>(ScaleKind::MINIFLOAT))
    {
      if (c.scale_family == static_cast<int>(BlockFamily::SUPERFP))
        c.scale_sfp = make_superfp_params<T>(at(BlockFmtField::SCALE_MAN_BITS), at(BlockFmtField::SCALE_EXP_BITS),
                                             at(BlockFmtField::SCALE_NORMAL_BINADES),
                                             at(BlockFmtField::SCALE_BIAS), SaturationMode::SAT_FINITE);
      else
        c.scale_bk = make_binaryK_params<T>(
            at(BlockFmtField::SCALE_MAN_BITS), at(BlockFmtField::SCALE_EXP_BITS), at(BlockFmtField::SCALE_BIAS),
            at(BlockFmtField::SCALE_SIGNED) != 0, SaturationMode::SAT_FINITE,
            at(BlockFmtField::SCALE_SUBNORMALS) == static_cast<int>(SubnormalsMode::EXTENDED_NORMALS));
    }

    c.scale_rounding = at(BlockFmtField::SCALE_ROUNDING);
    const int man = at(BlockFmtField::ELEM_MAN_BITS), bias = at(BlockFmtField::ELEM_BIAS);
    const bool extended = c.elem_sub == SubnormalsMode::EXTENDED_NORMALS;
    uint32_t super_codes = 0;
    int32_t super_cutoff = 0;
    if (c.elem_family == static_cast<int>(BlockFamily::SUPERFP))
      set_supernormals(super_codes, super_cutoff, man, at(BlockFmtField::ELEM_EXP_BITS),
                       at(BlockFmtField::ELEM_NORMAL_BINADES), bias);
    const T em = static_cast<T>(elem_max);
    const uint32_t code = encode_minifloat(em, man, bias, extended, super_codes, super_cutoff);
    const int emax = block_floor_log2(em);
    T step;
    if (code < super_codes || man == 0)
      step = em;
    else if ((code >> man) == 0 && !extended)
      step = block_pow2<T>(1 - bias - man);
    else
      step = block_pow2<T>(emax - man);
    c.elem_fit = em + step / T(2);
    c.max_sig = block_significand(em);
    c.fit_sig = c.elem_fit * block_pow2<T>(-emax);
    return c;
  }
#endif // !__METAL_VERSION__

} // namespace mptorch::block
