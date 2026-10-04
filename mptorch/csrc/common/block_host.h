#pragma once

// The host side of the elementwise block ops, shared by the CPU and CUDA
// entry points: the descriptor's validation, the dtype, the shapes and the
// output allocation. The descriptor is validated here, not only in Python,
// because torch.ops.mptorch.* is a public entry point and every check below
// guards an invariant the kernels rely on: that every code decodes to a
// binary32 value assembled on the word, and that a block fills whole bytes.

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

  inline bool is_pow2_in(int64_t v, int64_t lo, int64_t hi)
  {
    return v >= lo && v <= hi && (v & (v - 1)) == 0;
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

  // Whether the fields name a layout the kernels read: a family, a sign bit,
  // widths with at most 8 exponent bits, a binaryK SubnormalsMode or a superfp
  // count of normal binades that leaves supernormal codes, and every value a
  // normal binary32 value on the word (the decode assembles it there, and the
  // encode reads its exponent field), the subnormal step and the smallest
  // supernormal included.
  inline bool layout_ok(const Layout &l)
  {
    if (!(l.family == 0 || l.family == 1) || !(l.sg == 0 || l.sg == 1) || l.e < 1 || l.e > 8 || l.m < 0)
      return false;
    const int64_t top_field = (int64_t(1) << l.e) - 1;
    if (l.family == static_cast<int64_t>(BlockFamily::SUPERFP))
    {
      if (l.sub != 0 || l.nb < 1 || l.nb > top_field)
        return false;
      const int64_t lowest_normal_field = (int64_t(1) << l.e) - l.nb;
      return lowest_normal_field - l.bias >= -126 && top_field - l.bias <= 127 && l.super_cutoff() >= -126;
    }
    if (l.nb != 0 || l.sub < 0 || l.sub > 2)
      return false;
    const int64_t lowest_field = l.extended() ? 0 : 1;
    return lowest_field - l.bias >= -126 && top_field - l.bias <= 127 && 1 - l.bias - l.m >= -126;
  }

  // Whether a float is exactly a value of a layout below its specials: a
  // positive, finite binary32 value that survives the encode and decode round
  // trip with a code under `special_min`.
  inline bool is_layout_value(double v, const Layout &l, uint32_t special_min)
  {
    if (!(v > 0) || !std::isfinite(v) || static_cast<double>(static_cast<float>(v)) != v)
      return false;
    const float f = static_cast<float>(v);
    if (!(f >= std::ldexp(1.0f, -126)))
      return false;
    const int m = static_cast<int>(l.m), bias = static_cast<int>(l.bias);
    const uint32_t code = encode_minifloat(f, m, bias, l.extended(), l.super_codes(), l.super_cutoff());
    return code < special_min &&
           decode_minifloat<float>(code, m, bias, l.extended(), l.super_codes(), l.super_cutoff()) == f;
  }

  // The descriptor, checked against every invariant the kernels read it
  // with. `elem_max`, `scale_max` and `tensor_scale` are the schema's floats.
  inline Fmt checked_block_fmt(c10::IntArrayRef fmt, double elem_max, double scale_max,
                               double tensor_scale, const char *op)
  {
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
    TORCH_CHECK(bits == el.sg + el.e + el.m && bits >= 2 && bits <= 16 && layout_ok(el), op,
                ": the element code must be 2 to 16 bits of sign, exponent and mantissa of a binaryK "
                "or superfp layout whose every value is a normal binary32 value, got family=",
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
    TORCH_CHECK(is_layout_value(elem_max, el, special_min), op,
                ": elem_max must be a positive value of the element format below its NaN and "
                "infinity codes, got ",
                elem_max);
    const int64_t prng = at(BlockFmtField::ELEM_PRNG_BITS);
    TORCH_CHECK(prng >= 0 && el.m + prng <= 23, op, ": prng_bits must be in [0, ", 23 - el.m,
                "] for a ", el.m, "-bit mantissa rounded in binary32, got ", prng);

    const int64_t kind = at(BlockFmtField::SCALE_KIND);
    const Layout sl{at(BlockFmtField::SCALE_FAMILY),   at(BlockFmtField::SCALE_SIGNED),
                    at(BlockFmtField::SCALE_EXP_BITS), at(BlockFmtField::SCALE_MAN_BITS),
                    at(BlockFmtField::SCALE_BIAS),     at(BlockFmtField::SCALE_SUBNORMALS),
                    at(BlockFmtField::SCALE_NORMAL_BINADES)};
    const int64_t bs = at(BlockFmtField::BLOCK_SIZE), br = at(BlockFmtField::BLOCK_ROWS);
    TORCH_CHECK(kind >= 0 && kind <= 2, op, ": scale kind must be 0 (none), 1 (power of two) or 2 "
                                            "(minifloat), got ", kind);
    const int64_t rule = at(BlockFmtField::SCALE_ROUNDING);
    TORCH_CHECK(rule >= 0 && rule <= 3 &&
                    (rule != static_cast<int64_t>(ScaleRounding::OCP) ||
                     kind != static_cast<int64_t>(ScaleKind::MINIFLOAT)),
                op, ": scale_rounding must be 0 (OCP, a power-of-two scale only), 1 (nearest), 2 (up) "
                    "or 3 (selective), got ", rule, " for scale kind ", kind);
    if (kind == static_cast<int64_t>(ScaleKind::POW2))
    {
      // the pack multiplies by the reciprocal, 2^-X, which is a binary32
      // value too only down to 2^-127
      TORCH_CHECK(sl.family == static_cast<int64_t>(BlockFamily::BINARYK) && sl.sg == 0 && sl.m == 0 &&
                      sl.e >= 1 && sl.e <= 8 && -sl.bias >= -127 && (int64_t(1) << sl.e) - 2 - sl.bias <= 127,
                  op, ": a power-of-two scale is a binaryK unsigned exponent of at most 8 bits whose "
                      "values lie in 2^-127 .. 2^127, got exp_bits=", sl.e, ", bias=", sl.bias);
    }
    else if (kind == static_cast<int64_t>(ScaleKind::MINIFLOAT))
    {
      TORCH_CHECK(layout_ok(sl) && sl.sg + sl.e + sl.m <= 8 &&
                      (sl.family == static_cast<int64_t>(BlockFamily::SUPERFP) || sl.m >= 1),
                  op, ": a cast scale is at most 8 bits, a binaryK with a mantissa or a superfp, whose "
                      "values are normal binary32 values, got family=", sl.family, ", signed=", sl.sg,
                  ", exp_bits=", sl.e, ", man_bits=", sl.m, ", bias=", sl.bias);
      const uint32_t s_nan = (1u << (sl.e + sl.m)) - 1u;
      TORCH_CHECK(is_layout_value(scale_max, sl, s_nan), op,
                  ": scale_max must be a positive value of the scale format below its NaN, got ", scale_max);
      TORCH_CHECK(tensor_scale > 0 && std::isfinite(tensor_scale) &&
                      static_cast<double>(static_cast<float>(tensor_scale)) == tensor_scale &&
                      std::isfinite(static_cast<float>(tensor_scale) * static_cast<float>(scale_max) *
                                    static_cast<float>(elem_max)),
                  op, ": tensor_scale must be a positive float32 value whose product with the "
                      "largest scale and element is finite, got ", tensor_scale);
    }
    TORCH_CHECK(is_pow2_in(bs, 2, 128) && (bits * bs) % 8 == 0, op,
                ": block_size must be a power of two in [2, 128] whose block fills whole bytes, got ",
                bs, " for ", bits, "-bit codes");
    TORCH_CHECK(is_pow2_in(br, 1, 128) && br * bs <= 16384 &&
                    (br == 1 || kind != static_cast<int64_t>(ScaleKind::NONE)),
                op, ": block_rows must be a power of two in [1, 128] with block_rows * block_size <= "
                    "16384, and 1 without a scale, got ", br);
    return f;
  }

  inline BlockFormatParams block_params(const Fmt &f, double elem_max, double scale_max,
                                        double tensor_scale, int64_t cols, int64_t rows,
                                        bool trans = false)
  {
    return make_block_format_params(f.data(), elem_max, scale_max, tensor_scale, cols, rows, trans);
  }

  inline BlockCast make_block_cast(const Fmt &f, double elem_max)
  {
    return make_block_cast<float>(f.data(), elem_max);
  }

  // The dtype of a tensor a block op reads or writes. binary32 is the only
  // carrier the block kernels have so far, and nothing narrows a float64
  // tensor into it.
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
      TORCH_CHECK(false, op, ": block formats have binary32 kernels only so far, and a float64 "
                             "tensor rounds in binary64 (dev/continuation_plan.md, phase G); use a "
                             "float32, float16 or bfloat16 tensor");
    default:
      TORCH_CHECK(false, op, ": expected a float32, float16 or bfloat16 tensor, got ", st);
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
  inline BlockExtents unpack_extents(const Tensor &data, const Tensor &scales, int64_t cols,
                                     const Fmt &f, double elem_max, double scale_max,
                                     double tensor_scale, BlockFormatParams &p, const char *op)
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
    p = block_params(f, elem_max, scale_max, tensor_scale, cols, e.rows);
    TORCH_CHECK(data.size(-1) == p.row_bytes && scales.size(-2) == p.row_tiles &&
                    scales.size(-1) == p.scale_cols,
                op, ": ", cols, " elements per row of this format pack into data [..., ", e.rows,
                ", ", p.row_bytes, "] and scales [..., ", p.row_tiles, ", ", p.scale_cols,
                "], got ", data.sizes(), " and ", scales.sizes());
    return e;
  }

} // namespace mptorch::block
