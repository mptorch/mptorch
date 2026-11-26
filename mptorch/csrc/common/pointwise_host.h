#pragma once

// The host side of the fused pointwise ops (common/pointwise.h states them):
// the operands' checks, written on symbolic sizes so that the Meta kernels
// (meta/pointwise_meta.cpp) run the same functions with the same messages;
// the result, contiguous in the broadcast shape; and the geometry the
// kernels index the operands with. Each entry point makes its elementwise
// twin's format checks first (stochastic_host.h's check_quantizer_format),
// then these.

#include "modes.h"
#include "pointwise.h"
#include <ATen/ExpandUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

namespace mptorch::pointwise
{
  // The op code, the carrier rounding, the operands' dtypes and devices, and
  // which second operand the op takes: a tensor b or a number for ADD, SUB,
  // MUL and DIV, a number for RSUB and RDIV (which put it first), nothing for
  // SQRT.
  inline void check_pointwise(const at::Tensor &a, const std::optional<at::Tensor> &b,
                              std::optional<double> scalar, int64_t op, int64_t carrier_rounding,
                              const char *name)
  {
    TORCH_CHECK(op >= 0 && op < POINTWISE_OPS, name,
                ": op is 0 (a + b), 1 (a - b), 2 (a * b), 3 (a / b), 4 (sqrt(a)), 5 (number - a) "
                "or 6 (number / a), got ", op);
    TORCH_CHECK(carrier_rounding == static_cast<int64_t>(RoundMode::RNE) ||
                    carrier_rounding == static_cast<int64_t>(RoundMode::RO),
                name, ": carrier_rounding is RoundMode.RNE (", static_cast<int>(RoundMode::RNE),
                ") or RoundMode.RO (", static_cast<int>(RoundMode::RO), "), got ", carrier_rounding);
    const at::ScalarType st = a.scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
                name, ": expected a float32, float64, float16 or bfloat16 tensor, got ", st);
    if (op == SQRT)
    {
      TORCH_CHECK(!b.has_value() && !scalar.has_value(), name,
                  ": sqrt (op 4) takes one operand, got a second");
    }
    else if (op == RSUB || op == RDIV)
    {
      TORCH_CHECK(!b.has_value() && scalar.has_value(), name, ": op ", op,
                  " takes the tensor a and a number, and no tensor b");
    }
    else
    {
      TORCH_CHECK(b.has_value() != scalar.has_value(), name, ": op ", op,
                  " takes one second operand, a tensor b or a number, got ",
                  b.has_value() ? "both" : "neither");
    }
    if (b.has_value())
    {
      TORCH_CHECK(b->scalar_type() == st, name, ": a and b share one dtype, got ", st, " and ",
                  b->scalar_type());
      TORCH_CHECK(b->device() == a.device(), name, ": b is on ", b->device(), " and a on ",
                  a.device());
    }
  }

  // The table ops' checks: x's dtype as the arithmetic ops check it, the
  // tails, the range, and the entries, a 1D tensor of at least two values in
  // x's carrier on x's device.
  inline void check_pwl(const at::Tensor &x, const at::Tensor &entries, double lo, double hi,
                        int64_t tails, const char *name)
  {
    const at::ScalarType st = x.scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16,
                name, ": expected a float32, float64, float16 or bfloat16 tensor, got ", st);
    TORCH_CHECK(tails >= 0 && tails < PWL_TAILS, name,
                ": tails is 0 (extend), 1 (clamp) or 2 (relu), got ", tails);
    TORCH_CHECK(std::isfinite(lo) && std::isfinite(hi) && lo < hi, name,
                ": the table's range is finite with lo < hi, got [", lo, ", ", hi, "]");
    const at::ScalarType carrier = st == at::kDouble ? at::kDouble : at::kFloat;
    TORCH_CHECK(entries.scalar_type() == carrier, name, ": the entries are values of the carrier, ",
                carrier, " for a ", st, " tensor, got ", entries.scalar_type());
    TORCH_CHECK(entries.dim() == 1, name, ": the entries are a 1D tensor, got ", entries.dim(),
                " dimensions");
    TORCH_CHECK(entries.device() == x.device(), name, ": the entries are on ", entries.device(),
                " and x on ", x.device());
    TORCH_SYM_CHECK(entries.sym_size(0).sym_ge(2) & entries.sym_size(0).sym_lt(int64_t(1) << 31),
                    name, ": a table has 2 to 2^31 - 1 entries, got ", entries.sym_size(0));
  }

  // The result: contiguous, in a's dtype, of the shape a and b broadcast to
  // (torch's rule and message, at::infer_size).
  inline at::Tensor pointwise_result(const at::Tensor &a, const std::optional<at::Tensor> &b)
  {
    if (!b.has_value())
      return at::empty_symint(a.sym_sizes(), a.options());
    return at::empty_symint(at::infer_size_symdimvector(a.sym_sizes(), b->sym_sizes()),
                            a.options());
  }

  // The geometry of a checked call whose result `o` is nonempty, with
  // `b` null for a number operand. It returns false when the kernels cannot
  // index the operands as they are: more than POINTWISE_MAX_DIMS dimensions
  // after merging, or, on a device (`device`), a geometry that is not flat
  // with 2^31 elements or more (FastDivmod's range). The caller then copies
  // the broadcast operands into contiguous tensors, whose geometry is flat.
  inline bool make_geom(const at::Tensor &a, const at::Tensor *b, const at::Tensor &o, bool device,
                        PointwiseGeom &g)
  {
    g = PointwiseGeom{};
    g.has_b = b != nullptr;
    g.numel = o.numel();
    const int64_t n = o.dim();
    // Each operand's stride for each of the result's dimensions: its own,
    // right-aligned, or 0 where it is missing or broadcast.
    auto stride_of = [&](const at::Tensor &t, int64_t d) -> int64_t
    {
      const int64_t td = d - (n - t.dim());
      if (td < 0 || (t.size(td) == 1 && o.size(d) != 1))
        return 0;
      return t.stride(td);
    };
    // Merged dimensions, innermost first, without those of size 1.
    c10::SmallVector<int64_t, POINTWISE_MAX_DIMS> size, sa, sb;
    for (int64_t d = n - 1; d >= 0; --d)
    {
      const int64_t s = o.size(d);
      if (s == 1)
        continue;
      const int64_t ta = stride_of(a, d), tb = b ? stride_of(*b, d) : 0;
      if (!size.empty() && ta == sa.back() * size.back() && tb == sb.back() * size.back())
      {
        size.back() *= s;
        continue;
      }
      size.push_back(s);
      sa.push_back(ta);
      sb.push_back(tb);
    }
    if (size.empty()) // one element
    {
      size.push_back(1);
      sa.push_back(0);
      sb.push_back(0);
    }
    const int m = static_cast<int>(size.size());
    if (m > POINTWISE_MAX_DIMS)
      return false;
    g.ndim = m;
    for (int d = 0; d < m; ++d)
    {
      g.size[d] = size[m - 1 - d];
      g.sa[d] = sa[m - 1 - d];
      g.sb[d] = sb[m - 1 - d];
    }
    g.flat = m == 1 && (g.size[0] == 1 || (g.sa[0] == 1 && (!b || g.sb[0] == 1)));
    if (g.flat)
      return true;
    if (device && g.numel >= (int64_t(1) << 31))
      return false;
    if (g.numel < (int64_t(1) << 31))
      for (int d = 1; d < m; ++d)
        g.by[d] = mptorch::gemm::make_fast_divmod(static_cast<int32_t>(g.size[d]));
    return true;
  }

  // The number operand in the carrier T: converted to nearest, as torch
  // converts a number, and under RO first rounded to the operands' dtype
  // scalar_t, as mptorch.quant.pointwise rounds one (torch.full in that
  // dtype), so that both operands have that dtype's precision.
  template <class T, class scalar_t>
  T number_in_carrier(std::optional<double> scalar, bool odd)
  {
    if (!scalar.has_value())
      return T(0);
    return odd ? static_cast<T>(static_cast<scalar_t>(*scalar)) : static_cast<T>(*scalar);
  }
} // namespace mptorch::pointwise
