#pragma once

// Block formats (OCP MX, NVFP4; mptorch.number.BlockFormat): the layout of a
// packed operand, and the single decode and single encode that every block
// op shares. The unpack kernels, the fused quantizers and the block GEMM's
// tile loads all decode through `decode_value` below, and the pack kernels
// and the fused quantizers all encode through `encode_elem` and `tile_scale`.
// So by construction, block_quantize returns what
// block_unpack(block_pack(x)) returns, which is what the GEMM multiplies.
//
// Codes. An element and a scale are each a *minifloat code*: a sign and a
// magnitude, with `exp_bits` exponent bits and `man_bits` mantissa bits over
// a bias, in one of the two families the casts implement. In a binaryK code,
// exponent field 0 holds the subnormals m * 2^(1 - bias - man_bits), or,
// under SubnormalsMode::EXTENDED_NORMALS, one more binade of normal values,
// whose code with a zero mantissa is the zero. In a superfp code, the top
// `normal_binades` binades are normal values in the same layout, and the
// magnitude codes below them, 1 to S - 1 with
// S = (2^exp_bits - normal_binades) * 2^man_bits, are the supernormals, the
// powers of two 2^(supernormal_cutoff + code - 1) (cast_superfp.h); code 0
// is the zero. So one decode serves both families: a supernormal branch in
// front of binaryK's decode (S is 0 for a binaryK code). The element format
// may spend the magnitude codes above its largest value on NaN and infinity
// (OCP's E4M3 and E5M2 do); a scale spends its all-ones magnitude code on
// NaN. A binaryK scale with no mantissa (E8M0) is a power of two,
// 2^(code - bias), formed with integer arithmetic. Any other scale (a
// binaryK scale with a mantissa, such as E4M3, or a superfp scale) is
// rounded to nearest even by its family's cast and multiplied by a tensor
// scale, one value for the whole tensor.
//
// Layout. A packed operand is `data` [rows, row_bytes] and `scales`
// [row_tiles, scale_cols], both uint8 and row-major, with any number of
// leading batch dimensions in front of both. A row holds
// n_blocks = ceil(cols / block_size) blocks of
// block_bytes = block_size * elem_bits / 8 bytes. Element i of a block is at
// bits [i * elem_bits, (i + 1) * elem_bits) of a little-endian bit stream (a
// 4-bit code in the low nibble first, as torch.float4_e2m1fn_x2 and CUTLASS
// lay them out), and the unused end of the last block is zero. A scale is
// shared by a tile of block_rows consecutive rows of one block, so scale
// (t, j) belongs to rows [t * block_rows, (t + 1) * block_rows) of block j,
// and the last tile may be partial. scale_cols is n_blocks, or 0 for a
// format with no scale.
//
// Values are assembled on the carrier's word (FloatTraits) rather than
// computed, which also keeps an Apple GPU's flush of binary32 subnormals away
// from them; the few float operations that remain go through
// MPTORCH_BLOCK_MUL, _DIV and _SUB below, for the same reason. Every function
// is written over the carrier T, binary32 or binary64, and so is the
// parameter struct (BlockFormatParamsT): a format's largest element, its
// scales and its tensor scale are values of the carrier. So a format that
// only binary64 can hold (an element with nine or ten exponent bits, a
// power-of-two scale above 2^127, a tensor scale that is not a float32 value)
// has binary64 parameters and runs binary64 kernels, and the validation
// (block_host.h) checks each format against the carrier of the call.
//
// This header uses no ATen and is valid Metal, like the other common/
// headers, so the GEMM .cu files that include it pay for no ATen headers,
// and the MPS backend's block kernels (mps/block.metal, mps/gemm.metal) are
// compiled from it. MPTORCH_BLOCK_DEVICE names the address space of an
// operand pointer on Metal and is empty elsewhere.

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

// The float operations that the decode and the pack apply to values that
// can be, or can produce, a subnormal of the carrier: a code's value times
// its scale (an E8M0 scale goes down to 2^-127), a scale times a tensor
// scale, an input divided by its tile's scale, and a decoded value minus its
// input. On the host and CUDA, each is the operation it names, rounded to
// nearest even, with F the carrier's FloatTraits. An Apple GPU flushes
// binary32 subnormals to zero in its arithmetic, so on Metal each is
// bit_helper.h's exact version (metal_mul_rn, metal_div_rn, metal_sub_rn),
// which returns the float unit's result wherever no subnormal is involved.
#if defined(__METAL_VERSION__)
#define MPTORCH_BLOCK_MUL(F, a, b) metal_mul_rn(a, b)
#define MPTORCH_BLOCK_DIV(a, b) metal_div_rn(a, b)
#define MPTORCH_BLOCK_SUB(F, a, b) metal_sub_rn(a, b)
#else
#define MPTORCH_BLOCK_MUL(F, a, b) F::rn_mul(a, b)
#define MPTORCH_BLOCK_DIV(a, b) ((a) / (b))
#define MPTORCH_BLOCK_SUB(F, a, b) F::rn_sub(a, b)
#endif

namespace mptorch::block
{

  // The order of the fields of the `int[] fmt` descriptor that every block op
  // takes. It has one writer (mptorch/quant/ops/block.py's
  // _block_format_ints) and one reader (make_block_format_params);
  // tests/test_block_pack.py checks the writer's output field by field.
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

  // How a tile's scale is chosen from its largest magnitude `amax`, matching
  // mptorch.ScaleRounding by name and value. With r = amax / elem_max
  // (divided by the tensor scale too, for a cast scale) and T = elem_max plus
  // half the element format's step above it (BlockCastT::elem_fit):
  //
  //   OCP        power-of-two scales only: 2^(floor(log2(amax)) - emax).
  //   NEAREST    r rounded to nearest even in the scale format.
  //   UP         the smallest scale >= r, so that no element exceeds
  //              elem_max.
  //   SELECTIVE  NEAREST, or the next scale up if the tile's largest element
  //              would otherwise fall above T: the one case where clamping
  //              to elem_max moves it further than rounding to nearest would.
  //
  // The result of every rule is then clamped to the scale's codes. No rule
  // depends on the rounding mode of the elements.
  //
  // The three FOUR_OVER_SIX rules (Cook et al., arXiv:2512.02010) work
  // differently: they read the tile's elements, not only its amax. A tile has
  // two candidate scales: NEAREST's, which maps amax to elem_max (6 for E2M1),
  // and the scale nearest to r * elem_max / 2^emax, which maps amax to
  // elem_max's power of two (4). The tile is quantized under both, in the
  // call's rounding mode, and takes the second scale if its error is strictly
  // smaller. The error is the sum of squared errors (MSE), the sum of absolute
  // errors (MAE) or the largest absolute error (ABS_MAX) over the tile's
  // elements, of each decoded value against its input (four_over_six_low,
  // candidate_error_units). These rules need a MINIFLOAT scale, since a power
  // of two cannot express the factor between the candidates, and apply to 1D
  // blocks and to 2D tiles.
  enum class ScaleRounding : int
  {
    OCP = 0,
    NEAREST = 1,
    UP = 2,
    SELECTIVE = 3,
    FOUR_OVER_SIX_MSE = 4,
    FOUR_OVER_SIX_MAE = 5,
    FOUR_OVER_SIX_ABS_MAX = 6,
  };

  CUDA_HOST_DEVICE_INLINE bool is_four_over_six(int32_t rule)
  {
    return rule >= static_cast<int32_t>(ScaleRounding::FOUR_OVER_SIX_MSE);
  }

  // What `elem_nan_code` and `elem_inf_code` hold for a format without one.
  constexpr MPTORCH_CONSTANT uint32_t NO_CODE = 0xFFFFFFFFu;

  // Everything that decoding and encoding one operand needs, as a flat
  // struct passed by value. It is flat because a nested aggregate copied
  // across the GEMM's instantiations once took nvcc's cicc from 18 s to ten
  // minutes (see the note in cast_binaryK.h). The block GEMM carries one per
  // operand in its policy, and its kernel reads it, as a kernel parameter,
  // only in the tile load. The four floating-point fields are values of the
  // carrier T; the binary32 struct keeps the layout it had before the
  // binary64 one existed, field for field.
  template <class T>
  struct BlockFormatParamsT
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
    T elem_max;
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
    T scale_max;
    T scale_min; // MINIFLOAT: the smallest positive scale
    T tensor_scale;
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

  using BlockFormatParams = BlockFormatParamsT<float>;

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
  // [-149, 127] in binary32 and [-1074, 1023] in binary64, built on the word.
  // FloatTraits::pow2 stops at the smallest normal, and an E8M0 scale goes
  // down to 2^-127.
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
    return 31 - static_cast<int>(metal::clz(w));
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
  // subnormals included, computed exactly on the word.
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
  // or a value above every exponent for an infinity (which the POW2 scale
  // then clamps to its largest code).
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

  // The magnitude that a minifloat magnitude code encodes (NaN and infinity
  // aside), built on the word. A superfp supernormal (mag below super_codes)
  // is 2^(super_cutoff + mag - 1). A binaryK subnormal is
  // m * 2^(1 - bias - man_bits), an integer below 2^16 times a normal power
  // of two, so the product is exact.
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

  // The magnitude code of `q`, a positive value of the format (on its grid,
  // finite, and a normal carrier value, which every layout a BlockFormat
  // accepts guarantees), or 0 for a zero. This is the inverse of
  // decode_minifloat: a superfp value below its normal binades is a power of
  // two, whose code is its exponent's distance from the smallest
  // supernormal's exponent, plus one.
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
    // A subnormal: the significand, implicit bit included, shifted to the
    // subnormal grid, whose step is 2^(1 - bias - man_bits).
    const W sig = mant | (W(1) << F::MAN_BITS);
    return static_cast<uint32_t>(sig >> (F::MAN_BITS - man_bits + 1 - field));
  }

  // An element code's value, built on the word: NaN and infinity first, then
  // the magnitude, then the sign. The code with only the sign bit set decodes
  // to +0.0 (no cast in the library returns -0.0), and a NaN code to the
  // canonical quiet NaN.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_elem(uint32_t code, const MPTORCH_THREAD BlockFormatParamsT<T> &p)
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

  // The factor of a stored scale code: the power of two, or the minifloat
  // times the tensor scale, in that order. The scale's NaN code decodes to
  // NaN, and so does any POW2 code above its largest.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_scale(uint32_t sc, const MPTORCH_THREAD BlockFormatParamsT<T> &p)
  {
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
      return static_cast<int32_t>(sc) > p.scale_code_max ? block_quiet_nan<T>()
                                                          : block_pow2<T>(static_cast<int>(sc) - p.scale_bias);
    const uint32_t mag = sc & p.scale_mag_mask;
    if (mag == p.scale_nan_code)
      return block_quiet_nan<T>();
    return MPTORCH_BLOCK_MUL(FloatTraits<T>,
                             decode_minifloat<T>(mag, p.scale_man_bits, p.scale_bias, p.scale_extended,
                                                 p.scale_super_codes, p.scale_super_cutoff),
                             static_cast<T>(p.tensor_scale));
  }

  // The value that an element code and its tile's scale code decode to:
  // decode_elem(code) * decode_scale(scale), one product in the carrier,
  // which is exact for every OCP format. A product that underflows to zero
  // is +0.0, never -0.0.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T decode_value(uint32_t code, uint32_t sc,
                                         const MPTORCH_THREAD BlockFormatParamsT<T> &p)
  {
    using F = FloatTraits<T>;
    T v = decode_elem<T>(code, p);
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::NONE))
      return v;
    v = MPTORCH_BLOCK_MUL(F, v, decode_scale<T>(sc, p));
    return (block_word(v) & F::ABS_MASK) == 0 ? T(0) : v;
  }

  // --- the bit stream ------------------------------------------------------

  // The code of element i of the block at `blk`: bits
  // [i * bits, (i + 1) * bits) of the block's little-endian bit stream. Codes
  // have at most 16 bits, so a code spans at most three bytes, and only the
  // bytes it spans are read, never any past the block.
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

  // ORs element i's code into a zeroed block buffer: the inverse of
  // extract_code. The CPU pack uses it; the device kernels assemble their
  // bytes across lanes instead.
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

  // Byte j of a block, from its codes, for a kernel that holds the codes in
  // an array: the OR of every code whose bit range overlaps bits
  // [8j, 8j + 8).
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
  // the data, element c along the packed axis. `data` and `scales` point to
  // this batch element's arrays.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T load_block_elem(const MPTORCH_BLOCK_DEVICE uint8_t *data,
                                            const MPTORCH_BLOCK_DEVICE uint8_t *scales, int64_t r,
                                            int64_t c, const MPTORCH_THREAD BlockFormatParamsT<T> &p)
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

  // Element (i, k) of a GEMM operand, which is logically [rows of the
  // result, K]: the data's row i and its element k, or, for an operand stored
  // with K as its rows (`trans`), row k and element i. The same function
  // reads A (i = m) and B (i = n), since the block GEMM reads both with K
  // along their packed axis unless `trans` says otherwise.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T load_block_operand(const MPTORCH_BLOCK_DEVICE uint8_t *data,
                                               const MPTORCH_BLOCK_DEVICE uint8_t *scales, int64_t i,
                                               int64_t k, const MPTORCH_THREAD BlockFormatParamsT<T> &p)
  {
    return p.trans ? load_block_elem<T>(data, scales, k, i, p) : load_block_elem<T>(data, scales, i, k, p);
  }

  // --- writing one -----------------------------------------------------------

  // The casts that a pack runs, built once per launch: the element format's
  // (under SAT_FINITE, followed by the clamp to elem_max) and a MINIFLOAT
  // scale's, each in the family its descriptor names. The other family's
  // constants are carried but never read, so the kernels choose a cast with
  // one uniform branch rather than one instantiation per family. The scale
  // rule and the values it compares with are here too, rather than in
  // BlockFormatParamsT, which the block GEMM carries per operand and which a
  // field used only by the pack would change.
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
    // scaled element that rounding to nearest would still map to elem_max.
    T elem_fit;
    // POW2: elem_max's significand, and elem_fit divided by elem_max's power
    // of two, which the rules compare an amax's significand with.
    T max_sig;
    T fit_sig;
  };

  // A tile's scale: the code to store, and the factor that brings an element
  // to the element format's grid. An element is multiplied by `factor` for a
  // POW2 scale (2^-shared_exp, so the product is exact), or divided by it for
  // a MINIFLOAT scale (the decoded scale times the tensor scale).
  // `nan_block` marks a tile that holds a NaN for which its element format
  // has no code; that tile's scale is then the scale format's NaN.
  template <class T>
  struct TileScale
  {
    uint32_t code;
    T factor;
    bool nan_block;
  };

  // The step, -1, 0 or 1, that a POW2 rule other than OCP takes from OCP's
  // shared exponent e = floor(log2(amax)) - emax. With amax = sig * 2^(e +
  // emax) and elem_max = max_sig * 2^emax, amax / elem_max =
  // (sig / max_sig) * 2^e, so the rules compare sig with multiples of
  // max_sig, all exactly. NEAREST's midpoints are 0.75 and 1.5 times max_sig
  // (a tie between two powers of two goes to the even exponent, as every
  // cast in the library breaks such a tie), UP's bound is max_sig itself,
  // and SELECTIVE raises NEAREST's choice by one if amax at that scale,
  // sig * 2^(emax - d), is above elem_fit.
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

  // A MINIFLOAT scale's cast of `ratio`, to nearest even or up, then
  // clamped: to at most scale_max, and to the smallest positive scale in
  // place of a zero.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T cast_scale(T ratio, bool up, const MPTORCH_THREAD BlockFormatParamsT<T> &p,
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

  // The scale of a tile whose largest magnitude is `amax` (leaving out NaNs,
  // which `has_nan` reports), by the rule c.scale_rounding (ScaleRounding
  // above). `c` holds the MINIFLOAT scale's cast, built once per launch under
  // SAT_FINITE; the scale's rounding never depends on the elements'.
  //
  //   POW2       amax == 0 gives code 0. Otherwise
  //              shared = floor(log2(amax)) - emax, moved by pow2_scale_step
  //              for a rule other than OCP, and clamped to the codes; an
  //              element becomes v * 2^-shared.
  //   MINIFLOAT  s = amax / (elem_max * S_t), where S_t is the tensor scale,
  //              cast to nearest even (or up, for UP) in the scale format,
  //              then min(s, scale_max), with the smallest positive scale in
  //              place of a zero. SELECTIVE casts up instead where the tile's
  //              largest element, amax / (s * S_t), which is exactly the
  //              quotient scaled_elem computes, is above elem_fit. An element
  //              becomes v / (s * S_t).
  template <class T>
  CUDA_HOST_DEVICE_INLINE TileScale<T> tile_scale(T amax, bool has_nan,
                                                  const MPTORCH_THREAD BlockFormatParamsT<T> &p,
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
      const T ratio = MPTORCH_BLOCK_DIV(amax, MPTORCH_BLOCK_MUL(F, static_cast<T>(p.elem_max), ts));
      T sv = cast_scale(ratio, c.scale_rounding == static_cast<int32_t>(ScaleRounding::UP), p, c);
      T factor = MPTORCH_BLOCK_MUL(F, sv, ts);
      if (c.scale_rounding == static_cast<int32_t>(ScaleRounding::SELECTIVE) &&
          MPTORCH_BLOCK_DIV(amax, factor) > c.elem_fit)
      {
        sv = cast_scale(ratio, true, p, c);
        factor = MPTORCH_BLOCK_MUL(F, sv, ts);
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

  // An element scaled by its tile's scale, ready for the element format's
  // cast. A POW2 product is exact unless it falls below the carrier's
  // normals, which needs a block whose values span more than 2^125 in
  // binary32 (2^1021 in binary64). A product that underflows to zero is
  // replaced by the smallest subnormal of the same sign, so that the directed
  // modes still round it away from zero as they would the exact value. That
  // is safe because every element format's smallest value is at least 2^-125
  // in binary32 and 2^-1021 in binary64, so every such input is below half
  // of it.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T scaled_elem(T v, const MPTORCH_THREAD TileScale<T> &s,
                                        const MPTORCH_THREAD BlockFormatParamsT<T> &p)
  {
    using F = FloatTraits<T>;
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::POW2))
    {
      const T x = MPTORCH_BLOCK_MUL(F, v, s.factor);
      const auto vw = block_word(v);
      if ((block_word(x) & F::ABS_MASK) == 0 && (vw & F::ABS_MASK) != 0)
        return block_float<T>((vw & F::SIGN_MASK) | 1u);
      return x;
    }
    if (p.scale_kind == static_cast<int32_t>(ScaleKind::MINIFLOAT))
      return MPTORCH_BLOCK_DIV(v, s.factor);
    return v;
  }

  // The element format's cast in rounding mode RM, a template parameter as in
  // the GEMM's policies (gemm_policy.h), so that a kernel holds one cast body
  // per family. `rbits` is RoundMode::SR's random word, which no other mode
  // reads. `is_signed` is a parameter rather than c.elem_signed so that the
  // CPU kernel, which has an instantiation per sign, can pass a constant.
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
  // computes so that the rounding mode stays a template parameter of its
  // kernel. The clamp maps the codes above elem_max, which OCP spends on NaN
  // and infinity, to elem_max. A NaN input gets the element format's NaN
  // code; without one, code 0 if the scale is a NaN, and elem_max's code if
  // the format has no scale.
  template <class T>
  CUDA_HOST_DEVICE_INLINE uint32_t encode_elem(T v, T rounded, const MPTORCH_THREAD BlockFormatParamsT<T> &p)
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

  // --- Four Over Six --------------------------------------------------------

  // Four Over Six's second candidate scale for a tile whose first candidate,
  // `six`, is tile_scale's (NEAREST's): the scale nearest to
  // amax / (2^emax * S_t), under which the tile's largest element maps to
  // elem_max's power of two. The ratio is the first candidate's ratio times
  // elem_max's significand (1.5 for E2M1), as the reference implementation
  // computes it. The two candidates are the same scale when the scale format
  // clamps both, when it cannot tell them apart, or when elem_max is a power
  // of two; a caller then sees equal codes and skips the error pass.
  template <class T>
  CUDA_HOST_DEVICE_INLINE TileScale<T> four_over_six_low(T amax, const MPTORCH_THREAD TileScale<T> &six,
                                                         const MPTORCH_THREAD BlockFormatParamsT<T> &p,
                                                         const MPTORCH_THREAD BlockCastT<T> &c)
  {
    using F = FloatTraits<T>;
    TileScale<T> s = six;
    if (six.nan_block || p.scale_kind != static_cast<int32_t>(ScaleKind::MINIFLOAT))
      return s;
    const T ts = static_cast<T>(p.tensor_scale);
    const T ratio = MPTORCH_BLOCK_DIV(amax, MPTORCH_BLOCK_MUL(F, static_cast<T>(p.elem_max), ts));
    const T sv = cast_scale(MPTORCH_BLOCK_MUL(F, ratio, c.max_sig), false, p, c);
    s.code = encode_minifloat(sv, p.scale_man_bits, p.scale_bias, p.scale_extended, p.scale_super_codes,
                              p.scale_super_cutoff);
    s.factor = MPTORCH_BLOCK_MUL(F, sv, ts);
    return s;
  }

  // An element's error is counted in fixed point, in units of
  // 2^(amax_exp - ERROR_UNIT_BITS) with amax_exp = floor(log2(amax)),
  // truncated, and capped at ERROR_UNITS_MAX (an error of 2^(amax_exp + 2),
  // more than twice amax). So a tile's largest magnitude is 2^22 to 2^23
  // units. The error is an integer so that a tile's total is exact, and so
  // the same in whatever order its elements are added: by a CPU loop, by a
  // warp's shuffles, or along either axis of a square tile. A float sum
  // would let two backends choose differently at a near tie. A squared error
  // is at most 2^48, and a tile has at most 2^14 elements, so a sum fits in
  // 64 bits.
  constexpr MPTORCH_CONSTANT int ERROR_UNIT_BITS = 22;
  constexpr MPTORCH_CONSTANT uint32_t ERROR_UNITS_MAX = 1u << 24;

  // |err| in those units, computed on the word: err = sig * 2^(E - MAN_BITS),
  // with sig its significand as an integer, which is shifted right by the
  // remaining difference of exponents.
  template <class T>
  CUDA_HOST_DEVICE_INLINE uint32_t block_error_units(T err, int amax_exp)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    const W w = block_word(err) & F::ABS_MASK;
    if (w == 0)
      return 0u;
    if (w >= F::INF_BITS)
      return ERROR_UNITS_MAX;
    const int E = block_floor_log2(err);
    if (E - amax_exp >= 2)
      return ERROR_UNITS_MAX;
    const W sig = (w >> F::MAN_BITS) == 0 ? w << (F::MAN_BITS - block_leading_bit(w))
                                          : (w & F::MAN_MASK) | (W(1) << F::MAN_BITS);
    const int rs = F::MAN_BITS - ERROR_UNIT_BITS + amax_exp - E;
    return rs > F::MAN_BITS ? 0u : static_cast<uint32_t>(sig >> rs);
  }

  // One element's error under a candidate scale: what block_unpack would
  // return for the code that the pack would write, against the input. The
  // cast is the one the encode pass runs, in the call's rounding mode and
  // with the element's own random word, so the candidate that is judged is
  // the candidate that is written. `code` returns that code, for a caller
  // that keeps it instead of casting again. A NaN adds no error under either
  // candidate.
  template <class T, RoundMode RM>
  CUDA_HOST_DEVICE_INLINE uint32_t candidate_error_units(T v, const MPTORCH_THREAD TileScale<T> &s,
                                                         int amax_exp, bool is_signed,
                                                         typename FloatTraits<T>::word_t rbits,
                                                         const MPTORCH_THREAD BlockFormatParamsT<T> &p,
                                                         const MPTORCH_THREAD BlockCastT<T> &c,
                                                         MPTORCH_THREAD uint32_t &code)
  {
    using F = FloatTraits<T>;
    const bool nan = block_isnan(v);
    const T q = nan ? T(0) : block_cast<T, RM>(scaled_elem(v, s, p), is_signed, rbits, c);
    code = encode_elem(v, q, p);
    if (nan)
      return 0u;
    return block_error_units(MPTORCH_BLOCK_SUB(F, decode_value<T>(code, s.code, p), v), amax_exp);
  }

  // Adds one element's error to a tile's error so far, and merges two
  // partial errors, by the rule's measure.
  CUDA_HOST_DEVICE_INLINE uint64_t add_block_error(uint64_t acc, uint32_t units, int32_t rule)
  {
    if (rule == static_cast<int32_t>(ScaleRounding::FOUR_OVER_SIX_MSE))
      return acc + static_cast<uint64_t>(units) * units;
    if (rule == static_cast<int32_t>(ScaleRounding::FOUR_OVER_SIX_MAE))
      return acc + units;
    return units > acc ? units : acc;
  }

  CUDA_HOST_DEVICE_INLINE uint64_t merge_block_error(uint64_t a, uint64_t b, int32_t rule)
  {
    if (rule == static_cast<int32_t>(ScaleRounding::FOUR_OVER_SIX_ABS_MAX))
      return a > b ? a : b;
    return a + b;
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
  // zero's code included), and the exponent of the smallest, from the cast's
  // own region cutoffs.
  inline void set_supernormals(uint32_t &codes, int32_t &cutoff, int man_bits, int exp_bits,
                               int normal_binades, int bias)
  {
    codes = static_cast<uint32_t>(((1 << exp_bits) - normal_binades) << man_bits);
    cutoff = superfp_region_cutoffs(man_bits, exp_bits, normal_binades, bias).supernormal_cutoff;
  }

  // The parameters of one operand in the carrier T, from its descriptor
  // (`f`, in BlockFmtField's order, which the caller has validated for that
  // carrier), its three floating-point values and its extents: `cols` along
  // the packed axis and `rows` across it.
  template <class T>
  BlockFormatParamsT<T> make_block_format_params(const int64_t *f, double elem_max, double scale_max,
                                                 double tensor_scale, int64_t cols, int64_t rows,
                                                 bool trans = false)
  {
    auto at = [f](BlockFmtField k) { return f[static_cast<int>(k)]; };
    BlockFormatParamsT<T> p{};
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
    p.elem_max = static_cast<T>(elem_max);
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
    p.scale_max = static_cast<T>(scale_max);
    p.tensor_scale = T(1);
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
      // smallest normal value under NORMALS, or the first code above the
      // extended binade's zero.
      const uint32_t first = p.scale_super_codes == 0 && s_sub == static_cast<int64_t>(SubnormalsMode::NORMALS)
                                 ? (1u << p.scale_man_bits)
                                 : 1u;
      p.scale_min = decode_minifloat<T>(first, p.scale_man_bits, p.scale_bias, p.scale_extended,
                                        p.scale_super_codes, p.scale_super_cutoff);
      p.tensor_scale = static_cast<T>(tensor_scale);
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

  // The casts that a pack runs (BlockCastT), from the descriptor and
  // elem_max: the element format's under SAT_FINITE, and a MINIFLOAT
  // scale's. Constants are built only for the family the descriptor names;
  // the other family's stay zero and are never read. Then the scale rule's
  // constants. elem_fit is elem_max plus half the step to the value the next
  // code would have if the format continued: that next value is 2 * elem_max
  // where the format steps by powers of two (a superfp supernormal, or no
  // mantissa bits), elem_max plus the subnormal step below the normals, and
  // elem_max plus one unit in the last place of elem_max's binade otherwise.
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
                                          c.elem_signed, SaturationMode::SAT_FINITE);
    else
      c.elem_bk = make_binaryK_params<T>(at(BlockFmtField::ELEM_MAN_BITS), at(BlockFmtField::ELEM_EXP_BITS),
                                         at(BlockFmtField::ELEM_BIAS), c.elem_signed, SaturationMode::SAT_FINITE,
                                         c.elem_sub == SubnormalsMode::EXTENDED_NORMALS);
    c.scale_family = at(BlockFmtField::SCALE_FAMILY);
    if (at(BlockFmtField::SCALE_KIND) == static_cast<int>(ScaleKind::MINIFLOAT))
    {
      if (c.scale_family == static_cast<int>(BlockFamily::SUPERFP))
        c.scale_sfp = make_superfp_params<T>(
            at(BlockFmtField::SCALE_MAN_BITS), at(BlockFmtField::SCALE_EXP_BITS),
            at(BlockFmtField::SCALE_NORMAL_BINADES), at(BlockFmtField::SCALE_BIAS),
            at(BlockFmtField::SCALE_SIGNED) != 0, SaturationMode::SAT_FINITE);
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
