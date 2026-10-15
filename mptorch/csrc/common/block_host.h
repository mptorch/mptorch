#pragma once

// The host side of the elementwise block ops, shared by the CPU and CUDA
// entry points: the descriptor's validation, the dtype, the shapes and the
// output allocation. The descriptor is validated here, not only in Python,
// because torch.ops.mptorch.* is a public entry point and every check below
// guards an invariant the kernels rely on: that every code decodes to a
// normal value of the call's carrier (binary32, or binary64 for a float64
// tensor or result) assembled on the word, and that a block fills whole
// bytes. The carrier is the call's, so the same descriptor can pass in
// binary64 and fail in binary32.

#include "block_kernels.h"
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <array>
#include <cmath>
#include <vector>

namespace mptorch::block
{
  using at::Tensor;

  constexpr int BLOCK_FMT_COUNT = static_cast<int>(BlockFmtField::COUNT);
  using Fmt = std::array<int64_t, BLOCK_FMT_COUNT>;

  // The most elements one scale covers, block_rows * block_size. A Four Over
  // Six rule adds a tile's errors up in 64 bits, each at most ERROR_UNITS_MAX
  // and squared under the MSE measure (block_decode.h), so the largest tile's
  // sum has to fit: 2^48 a term, 2^62 in all.
  constexpr int64_t MAX_TILE_ELEMS = 16384;
  static_assert(static_cast<uint64_t>(ERROR_UNITS_MAX) * ERROR_UNITS_MAX <=
                    UINT64_MAX / static_cast<uint64_t>(MAX_TILE_ELEMS),
                "a Four Over Six error sum over the largest tile must fit in 64 bits");

  inline bool is_pow2_in(int64_t v, int64_t lo, int64_t hi)
  {
    return v >= lo && v <= hi && (v & (v - 1)) == 0;
  }

  // What the checks below read of a carrier: the exponents of its smallest
  // normal and its largest finite power of two, its stored significand bits,
  // the widest exponent field a layout may have in it, and its name.
  struct CarrierBounds
  {
    int min_exp, max_exp, man_bits, max_exp_bits;
    bool wide;
    const char *name;
  };

  inline CarrierBounds carrier_bounds(bool wide)
  {
    return wide ? CarrierBounds{-1022, 1023, 52, 11, true, "binary64"}
                : CarrierBounds{-126, 127, 23, 8, false, "binary32"};
  }

  // One code layout of the descriptor, as the checks below read it.
  struct Layout
  {
    int64_t family, sg, e, m, bias, sub, nb;
    bool extended() const { return family == 0 && sub == static_cast<int64_t>(SubnormalsMode::EXTENDED_NORMALS); }
    uint32_t super_codes() const
    {
      return family == static_cast<int64_t>(BlockFamily::SUPERFP) ? static_cast<uint32_t>(((int64_t(1) << e) - nb) << m)
                                                                   : 0u;
    }
    int super_cutoff() const
    {
      return family == static_cast<int64_t>(BlockFamily::SUPERFP)
                 ? superfp_region_cutoffs(static_cast<int>(m), static_cast<int>(e), static_cast<int>(nb),
                                          static_cast<int>(bias))
                       .supernormal_cutoff
                 : 0;
    }
  };

  // Whether the fields name a layout the kernels read in a carrier: a family,
  // a sign bit, widths with at most cb.max_exp_bits exponent bits, a binaryK
  // SubnormalsMode or a superfp count of normal binades that leaves
  // supernormal codes, and every value a normal value of the carrier on the
  // word (the decode assembles it there, and the encode reads its exponent
  // field), the subnormal step and the smallest supernormal included.
  inline bool layout_ok(const Layout &l, const CarrierBounds &cb)
  {
    if (!(l.family == 0 || l.family == 1) || !(l.sg == 0 || l.sg == 1) || l.e < 1 || l.e > cb.max_exp_bits ||
        l.m < 0)
      return false;
    const int64_t top_field = (int64_t(1) << l.e) - 1;
    if (l.family == static_cast<int64_t>(BlockFamily::SUPERFP))
    {
      if (l.sub != 0 || l.nb < 1 || l.nb > top_field)
        return false;
      const int64_t lowest_normal_field = (int64_t(1) << l.e) - l.nb;
      return lowest_normal_field - l.bias >= cb.min_exp && top_field - l.bias <= cb.max_exp &&
             l.super_cutoff() >= cb.min_exp;
    }
    if (l.nb != 0 || l.sub < 0 || l.sub > 2)
      return false;
    const int64_t lowest_field = l.extended() ? 0 : 1;
    return lowest_field - l.bias >= cb.min_exp && top_field - l.bias <= cb.max_exp &&
           1 - l.bias - l.m >= cb.min_exp;
  }

  // Whether a double is exactly a value of a layout below its specials: a
  // positive, finite, normal value of the carrier (for binary32, a float32
  // value) that survives the encode and decode round trip with a code under
  // `special_min`.
  template <class T>
  bool is_layout_value_in(double v, const Layout &l, uint32_t special_min, int min_exp)
  {
    if (!(v > 0) || !std::isfinite(v) || static_cast<double>(static_cast<T>(v)) != v)
      return false;
    const T f = static_cast<T>(v);
    if (!(f >= std::ldexp(T(1), min_exp)))
      return false;
    const int m = static_cast<int>(l.m), bias = static_cast<int>(l.bias);
    const uint32_t code = encode_minifloat(f, m, bias, l.extended(), l.super_codes(), l.super_cutoff());
    return code < special_min &&
           decode_minifloat<T>(code, m, bias, l.extended(), l.super_codes(), l.super_cutoff()) == f;
  }

  inline bool is_layout_value(double v, const Layout &l, uint32_t special_min, const CarrierBounds &cb)
  {
    return cb.wide ? is_layout_value_in<double>(v, l, special_min, cb.min_exp)
                   : is_layout_value_in<float>(v, l, special_min, cb.min_exp);
  }

  // The descriptor, checked against every invariant the kernels read it
  // with in the call's carrier (binary64 when `wide`). `elem_max`,
  // `scale_max` and `tensor_scale` are the schema's floats.
  inline Fmt checked_block_fmt(c10::IntArrayRef fmt, double elem_max, double scale_max,
                               double tensor_scale, const char *op, bool wide)
  {
    const CarrierBounds cb = carrier_bounds(wide);
    TORCH_CHECK(static_cast<int>(fmt.size()) == BLOCK_FMT_COUNT, op, ": fmt must have ",
                BLOCK_FMT_COUNT, " entries (mptorch/quant/block.py's _block_format_ints), got ",
                fmt.size());
    Fmt f;
    for (int i = 0; i < BLOCK_FMT_COUNT; ++i)
      f[i] = fmt[i];
    auto at = [&f](BlockFmtField k) { return f[static_cast<int>(k)]; };

    const int64_t bits = at(BlockFmtField::ELEM_BITS);
    const Layout el{at(BlockFmtField::ELEM_FAMILY),   at(BlockFmtField::ELEM_SIGNED),
                    at(BlockFmtField::ELEM_EXP_BITS), at(BlockFmtField::ELEM_MAN_BITS),
                    at(BlockFmtField::ELEM_BIAS),     at(BlockFmtField::ELEM_SUBNORMALS),
                    at(BlockFmtField::ELEM_NORMAL_BINADES)};
    TORCH_CHECK(bits == el.sg + el.e + el.m && bits >= 2 && bits <= 16 && layout_ok(el, cb), op,
                ": the element code must be 2 to 16 bits of sign, exponent and mantissa of a binaryK "
                "or superfp layout whose every value is a normal ", cb.name, " value, got family=",
                el.family, ", bits=", bits, ", signed=", el.sg, ", exp_bits=", el.e, ", man_bits=", el.m,
                ", bias=", el.bias, ", subnormals=", el.sub, ", normal_binades=", el.nb);
    const int64_t mag_mask = (int64_t(1) << (bits - el.sg)) - 1;
    const int64_t nan = at(BlockFmtField::ELEM_NAN_CODE), inf = at(BlockFmtField::ELEM_INF_CODE);
    TORCH_CHECK((nan == -1 || (nan > 0 && nan <= mag_mask)) && (inf == -1 || (inf > 0 && inf <= mag_mask)) &&
                    (nan == -1 || nan != inf),
                op, ": nan_code and inf_code must be distinct magnitude codes or -1, got ", nan,
                " and ", inf);
    uint32_t special_min = static_cast<uint32_t>(mag_mask) + 1u;
    if (nan >= 0 && static_cast<uint32_t>(nan) < special_min)
      special_min = static_cast<uint32_t>(nan);
    if (inf >= 0 && static_cast<uint32_t>(inf) < special_min)
      special_min = static_cast<uint32_t>(inf);
    TORCH_CHECK(is_layout_value(elem_max, el, special_min, cb), op,
                ": elem_max must be a positive value of the element format below its NaN and "
                "infinity codes, got ",
                elem_max);
    const int64_t prng = at(BlockFmtField::ELEM_PRNG_BITS);
    TORCH_CHECK(prng >= 0 && el.m + prng <= cb.man_bits, op, ": prng_bits must be in [0, ",
                cb.man_bits - el.m, "] for a ", el.m, "-bit mantissa rounded in ", cb.name, ", got ", prng);

    const int64_t kind = at(BlockFmtField::SCALE_KIND);
    const Layout sl{at(BlockFmtField::SCALE_FAMILY),   at(BlockFmtField::SCALE_SIGNED),
                    at(BlockFmtField::SCALE_EXP_BITS), at(BlockFmtField::SCALE_MAN_BITS),
                    at(BlockFmtField::SCALE_BIAS),     at(BlockFmtField::SCALE_SUBNORMALS),
                    at(BlockFmtField::SCALE_NORMAL_BINADES)};
    const int64_t bs = at(BlockFmtField::BLOCK_SIZE), br = at(BlockFmtField::BLOCK_ROWS);
    TORCH_CHECK(kind >= 0 && kind <= 2, op, ": scale kind must be 0 (none), 1 (power of two) or 2 "
                                            "(minifloat), got ", kind);
    const int64_t rule = at(BlockFmtField::SCALE_ROUNDING);
    TORCH_CHECK(rule >= 0 && rule <= 6 &&
                    (rule != static_cast<int64_t>(ScaleRounding::OCP) ||
                     kind != static_cast<int64_t>(ScaleKind::MINIFLOAT)),
                op, ": scale_rounding must be 0 (OCP, a power-of-two scale only), 1 (nearest), 2 (up), "
                    "3 (selective) or 4 to 6 (Four Over Six), got ", rule, " for scale kind ", kind);
    TORCH_CHECK(!is_four_over_six(static_cast<int32_t>(rule)) ||
                    kind == static_cast<int64_t>(ScaleKind::MINIFLOAT),
                op, ": a Four Over Six scale_rounding (4 to 6) takes a cast scale, which can hold the "
                    "factor between its two candidates, got scale kind ", kind);
    if (kind == static_cast<int64_t>(ScaleKind::POW2))
    {
      // the pack multiplies by the reciprocal, 2^-X, which is a value of the
      // carrier too only down to 2^-max_exp (2^-127 in binary32)
      TORCH_CHECK(sl.family == static_cast<int64_t>(BlockFamily::BINARYK) && sl.sg == 0 && sl.m == 0 &&
                      sl.e >= 1 && sl.e <= 8 && -sl.bias >= -cb.max_exp &&
                      (int64_t(1) << sl.e) - 2 - sl.bias <= cb.max_exp,
                  op, ": a power-of-two scale is a binaryK unsigned exponent of at most 8 bits whose "
                      "values lie in 2^-", cb.max_exp, " .. 2^", cb.max_exp, " in ", cb.name,
                  ", got exp_bits=", sl.e, ", bias=", sl.bias);
    }
    else if (kind == static_cast<int64_t>(ScaleKind::MINIFLOAT))
    {
      TORCH_CHECK(layout_ok(sl, cb) && sl.sg + sl.e + sl.m <= 8 &&
                      (sl.family == static_cast<int64_t>(BlockFamily::SUPERFP) || sl.m >= 1),
                  op, ": a cast scale is at most 8 bits, a binaryK with a mantissa or a superfp, whose "
                      "values are normal ", cb.name, " values, got family=", sl.family, ", signed=", sl.sg,
                  ", exp_bits=", sl.e, ", man_bits=", sl.m, ", bias=", sl.bias);
      const uint32_t s_nan = (1u << (sl.e + sl.m)) - 1u;
      TORCH_CHECK(is_layout_value(scale_max, sl, s_nan, cb), op,
                  ": scale_max must be a positive value of the scale format below its NaN, got ", scale_max);
      const bool ts_ok =
          wide ? std::isfinite(tensor_scale * scale_max * elem_max)
               : static_cast<double>(static_cast<float>(tensor_scale)) == tensor_scale &&
                     std::isfinite(static_cast<float>(tensor_scale) * static_cast<float>(scale_max) *
                                   static_cast<float>(elem_max));
      TORCH_CHECK(tensor_scale > 0 && std::isfinite(tensor_scale) && ts_ok, op,
                  ": tensor_scale must be a positive ", wide ? "float64" : "float32",
                  " value whose product with the largest scale and element is finite, got ", tensor_scale);
    }
    TORCH_CHECK(is_pow2_in(bs, 2, 128) && (bits * bs) % 8 == 0, op,
                ": block_size must be a power of two in [2, 128] whose block fills whole bytes, got ",
                bs, " for ", bits, "-bit codes");
    TORCH_CHECK(is_pow2_in(br, 1, 128) && br * bs <= MAX_TILE_ELEMS &&
                    (br == 1 || kind != static_cast<int64_t>(ScaleKind::NONE)),
                op, ": block_rows must be a power of two in [1, 128] with block_rows * block_size <= "
                    "16384, and 1 without a scale, got ", br);
    return f;
  }

  template <class T = float>
  BlockFormatParamsT<T> block_params(const Fmt &f, double elem_max, double scale_max, double tensor_scale,
                                     int64_t cols, int64_t rows, bool trans = false)
  {
    return make_block_format_params<T>(f.data(), elem_max, scale_max, tensor_scale, cols, rows, trans);
  }

  template <class T = float>
  BlockCastT<T> make_block_cast(const Fmt &f, double elem_max)
  {
    return make_block_cast<T>(f.data(), elem_max);
  }

  // The dtype of a tensor a block op reads or writes, which is also its
  // carrier's: binary64 for float64, binary32 for the other three (nothing
  // narrows a float64 tensor into binary32, and a narrower tensor reaches
  // binary64 through Python, which widens it). A build with MPTORCH_NO_FP64
  // has no binary64 block kernels and says so.
  inline mptorch::GemmDtype block_dtype(at::ScalarType st, const char *op)
  {
    switch (st)
    {
    case at::kFloat:
      return mptorch::GemmDtype::Float;
    case at::kHalf:
      return mptorch::GemmDtype::Half;
    case at::kBFloat16:
      return mptorch::GemmDtype::BFloat16;
    case at::kDouble:
#if defined(MPTORCH_NO_FP64)
      TORCH_CHECK(false, op, ": this build has no float64 block kernels (it was built with "
                             "MPTORCH_NO_FP64=1); use a float32, float16 or bfloat16 tensor, or "
                             "rebuild without it");
#else
      return mptorch::GemmDtype::Double;
#endif
    default:
      TORCH_CHECK(false, op, ": expected a float32, float64, float16 or bfloat16 tensor, got ", st);
    }
  }

  // The [batch, rows, cols] a rank-2 or rank-3 tensor is read as, with the
  // packed axis last, and whether it has the batch dimension.
  struct BlockExtents
  {
    int64_t batch = 1, rows = 0, cols = 0;
    bool batched = false;
  };

  inline BlockExtents block_extents(const Tensor &x, const char *op)
  {
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, op, " expects a 2D or 3D tensor with the packed axis "
                                                  "last, got ", x.dim(), "D (mptorch.quant.block_pack "
                                                  "moves an axis and folds leading dimensions)");
    BlockExtents e;
    e.batched = x.dim() == 3;
    e.batch = e.batched ? x.size(0) : 1;
    e.rows = x.size(-2);
    e.cols = x.size(-1);
    return e;
  }

  // Sizes of a [batch,] d1, d2 tensor.
  inline std::vector<int64_t> block_sizes(const BlockExtents &e, int64_t d1, int64_t d2)
  {
    if (e.batched)
      return {e.batch, d1, d2};
    return {d1, d2};
  }

  // A pack or quantize job over x, with x's strides, and o's when given.
  inline BlockPackJob pack_job(const Tensor &x, const BlockExtents &e, mptorch::GemmDtype dt,
                               int64_t round_mode)
  {
    BlockPackJob j;
    j.x = x.data_ptr();
    j.dt = dt;
    j.batch = e.batch;
    j.rows = e.rows;
    j.cols = e.cols;
    j.xs0 = e.batched ? x.stride(0) : 0;
    j.xs1 = x.stride(-2);
    j.xs2 = x.stride(-1);
    j.rm = static_cast<RoundMode>(round_mode);
    return j;
  }

  inline void set_output(BlockPackJob &j, Tensor &o, const BlockExtents &e)
  {
    j.o = o.data_ptr();
    j.os0 = e.batched ? o.stride(0) : 0;
    j.os1 = o.stride(-2);
    j.os2 = o.stride(-1);
    j.quant = true;
  }

  // block_unpack's operands: data and scales as block_pack wrote them, for a
  // packed axis of `cols` elements. Returns the extents and fills `p`.
  template <class T>
  BlockExtents unpack_extents(const Tensor &data, const Tensor &scales, int64_t cols, const Fmt &f,
                              double elem_max, double scale_max, double tensor_scale,
                              BlockFormatParamsT<T> &p, const char *op)
  {
    TORCH_CHECK(data.scalar_type() == at::kByte && scales.scalar_type() == at::kByte, op,
                ": data and scales are uint8, got ", data.scalar_type(), " and ",
                scales.scalar_type());
    TORCH_CHECK(data.device() == scales.device(), op, ": data and scales must be on one device");
    TORCH_CHECK((data.dim() == 2 || data.dim() == 3) && scales.dim() == data.dim(), op,
                ": data and scales are both 2D or both 3D, got ", data.dim(), "D and ",
                scales.dim(), "D");
    TORCH_CHECK(cols >= 0, op, ": cols must not be negative, got ", cols);
    BlockExtents e;
    e.batched = data.dim() == 3;
    e.batch = e.batched ? data.size(0) : 1;
    e.rows = data.size(-2);
    e.cols = cols;
    TORCH_CHECK(!e.batched || scales.size(0) == e.batch, op, ": data has ", e.batch,
                " batch elements and scales ", scales.size(0));
    p = block_params<T>(f, elem_max, scale_max, tensor_scale, cols, e.rows);
    TORCH_CHECK(data.size(-1) == p.row_bytes && scales.size(-2) == p.row_tiles &&
                    scales.size(-1) == p.scale_cols,
                op, ": ", cols, " elements per row of this format pack into data [..., ", e.rows,
                ", ", p.row_bytes, "] and scales [..., ", p.row_tiles, ", ", p.scale_cols,
                "], got ", data.sizes(), " and ", scales.sizes());
    return e;
  }

} // namespace mptorch::block
