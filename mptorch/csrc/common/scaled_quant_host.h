#pragma once

// The scaled quantizers (binaryK_quant_scaled, superfp_quant_scaled,
// fixedpoint_quant_scaled): each element of `a` is multiplied by a scale,
// rounded to the format and divided by the same scale, in one pass,
//
//     o[i] = storage(div(cast(mul(carrier(a[i]), s)), s)),
//
// where every operation rounds to nearest in the carrier (binary32 for
// float32, float16 and bfloat16 tensors, binary64 for float64 ones) and
// `storage` is the one conversion back to a's dtype. That is
// mptorch.quant.ScaledQuant's statement, which it otherwise computes as four
// separate passes over the tensor (multiply, round in place, divide,
// narrow), and these ops give the same bits: the same operations in the same
// order, and under stochastic rounding the same random word for each element,
// which the drivers key on the element's index as the elementwise quantizers'
// drivers do.
//
// The layout. `a` is a 2D tensor of rows, and `scale`, in the carrier's
// dtype, holds `per_row` scales for each row: row r's element c takes
// scale[r][c / block]. A block_size of 0 means one scale for the whole row
// (block = the row's length, per_row = 1), which with one row is a scale for
// the whole tensor; otherwise per_row = ceil(cols / block_size), and a row's
// last block may be shorter. The Python caller views its tensor this way
// (the scaled axis last), so these ops take no sizes: a size argument would
// be a symbolic integer under torch.compile, and the shapes already say it.
//
// The checks are written on symbolic sizes, so the Meta kernel
// (meta/meta_ops.cpp) runs the same function with the same messages.

#include <ATen/core/Tensor.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <cstdint>

namespace mptorch
{

  // What one row of the scale tensor covers: `per_row` scales of `block`
  // elements each, for a row of `cols` elements (all three > 0 on a
  // nonempty tensor).
  struct ScaleLayout
  {
    int64_t cols;
    int64_t block;
    int64_t per_row;
  };

  // Check a scaled quantizer's two operands and its block size, before
  // anything reads them: `a` a 2D float tensor, `scale` a 2D tensor in a's
  // carrier, on a's device, with a's rows and `per_row` columns.
  inline void check_scaled_operands(const at::Tensor &a, const at::Tensor &scale,
                                    int64_t block_size, const char *op)
  {
    const at::ScalarType st = a.scalar_type();
    TORCH_CHECK(st == at::kFloat || st == at::kDouble || st == at::kHalf || st == at::kBFloat16, op,
                ": expected a float32, float64, float16 or bfloat16 tensor, got ", st);
    TORCH_CHECK(a.dim() == 2, op, ": expected the tensor as a 2D tensor of rows, got ", a.dim(),
                " dimensions");
    const at::ScalarType carrier = st == at::kDouble ? at::kDouble : at::kFloat;
    TORCH_CHECK(scale.scalar_type() == carrier, op, ": the scales are values of the carrier, ",
                carrier, " for a ", st, " tensor, got ", scale.scalar_type());
    TORCH_CHECK(scale.dim() == 2, op, ": expected the scales as a 2D tensor, one row per row of "
                "the tensor, got ", scale.dim(), " dimensions");
    TORCH_CHECK(scale.device() == a.device(), op, ": the scales are on ", scale.device(),
                " and the tensor on ", a.device());
    TORCH_CHECK(block_size >= 0, op, ": block_size is 0 (one scale per row) or a block length, "
                "got ", block_size);
    const c10::SymInt cols = a.sym_size(1);
    // ceil(cols / block_size), written so that no block size overflows it
    const c10::SymInt per_row =
        block_size == 0 ? c10::SymInt(1) : cols / block_size + (cols % block_size).sym_ne(0).guard_bool(__FILE__, __LINE__);
    TORCH_SYM_CHECK(scale.sym_size(0).sym_eq(a.sym_size(0)) & scale.sym_size(1).sym_eq(per_row), op,
                    ": a tensor of ", a.sym_size(0), " rows of ", cols, " elements in blocks of ",
                    block_size, " takes ", a.sym_size(0), " x ", per_row, " scales, got ",
                    scale.sym_sizes());
  }

  // The layout of a checked, nonempty pair of operands. A block longer than
  // the row is the row.
  inline ScaleLayout scale_layout(const at::Tensor &a, int64_t block_size)
  {
    const int64_t cols = a.size(1);
    const int64_t block = block_size == 0 || block_size > cols ? cols : block_size;
    return ScaleLayout{cols, block, (cols + block - 1) / block};
  }

} // namespace mptorch
