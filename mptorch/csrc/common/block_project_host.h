#pragma once

// The host side of block_project (common/block_project.h states the op),
// shared by its CPU, CUDA, MPS and Meta entry points: the checks, in the
// order every backend makes them, and the result's sizes. The format is
// checked as block_pack checks it (block_host.h's checked_block_fmt, in the
// call's carrier); the scale codes are checked here, on symbolic sizes, so
// that the Meta kernel runs the same function with the same messages. A file
// of its own, included by block_project's entry files only, so that no
// existing object changes.

#include "block_host.h"
#include "block_project.h"
#include "stochastic_host.h"
#include <ATen/core/Tensor.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <vector>

namespace mptorch::block
{

  // The checks of a block_project call, before anything reads a tensor:
  // the rounding mode, x (2D or 3D, a float dtype that decides the carrier),
  // the format in that carrier, with SR_B and SR_C made SR (in `round_mode`
  // and the format's prng_bits, as for block_pack), and the scale
  // codes, which are uint8 on x's device, shaped as block_pack would write
  // them for x: [batch,] row_tiles, scale_cols. Returns the canonical
  // descriptor, and sets `wide` for the binary64 carrier.
  inline Fmt check_block_project(const Tensor &x, const Tensor &scales, c10::IntArrayRef fmt, double elem_max,
                                 double scale_max, double tensor_scale, int64_t &round_mode, const char *op,
                                 bool &wide)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
    TORCH_CHECK(x.dim() == 2 || x.dim() == 3, op, " expects a 2D or 3D tensor with the packed axis "
                                                  "last, got ", x.dim(), "D (mptorch.quant.block_project "
                                                  "moves an axis and folds leading dimensions)");
    wide = block_dtype(x.scalar_type(), op) == mptorch::GemmDtype::Double;
    Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, wide);
    mptorch::sr_canonical_fmt(f, static_cast<int>(BlockFmtField::ELEM_PRNG_BITS), round_mode, op);

    TORCH_CHECK(scales.scalar_type() == at::kByte, op, ": the scale codes are uint8, got ",
                scales.scalar_type());
    TORCH_CHECK(scales.device() == x.device(), op, ": the scale codes are on ", scales.device(),
                " and the tensor on ", x.device());
    TORCH_CHECK(scales.dim() == x.dim(), op, ": the scale codes of a ", x.dim(), "D tensor are ", x.dim(),
                "D, got ", scales.dim(), "D");
    auto at = [&f](BlockFmtField k) { return f[static_cast<int>(k)]; };
    const int64_t bs = at(BlockFmtField::BLOCK_SIZE), br = at(BlockFmtField::BLOCK_ROWS);
    const c10::SymInt rows = x.sym_size(-2), cols = x.sym_size(-1);
    const c10::SymInt n_blocks = (cols + (bs - 1)) / bs;
    const c10::SymInt row_tiles = (rows + (br - 1)) / br;
    const c10::SymInt scale_cols =
        at(BlockFmtField::SCALE_KIND) == static_cast<int64_t>(ScaleKind::NONE) ? c10::SymInt(0) : n_blocks;
    c10::SymBool ok = scales.sym_size(-2).sym_eq(row_tiles) & scales.sym_size(-1).sym_eq(scale_cols);
    if (x.dim() == 3)
      ok = ok & scales.sym_size(0).sym_eq(x.sym_size(0));
    TORCH_SYM_CHECK(ok, op, ": a tensor of sizes ", x.sym_sizes(), " in blocks of ", bs, " over ", br,
                    " rows takes scale codes of sizes [", x.dim() == 3 ? "batch, " : "", row_tiles, ", ",
                    scale_cols, "], as block_pack writes them, got ", scales.sym_sizes());
    return f;
  }

  // The sizes of block_project's result, block_pack's data: [batch,] rows,
  // row_bytes, with row_bytes = ceil(cols / block_size) * block_bytes.
  inline std::vector<c10::SymInt> block_project_sizes(const Tensor &x, const Fmt &f)
  {
    auto at = [&f](BlockFmtField k) { return f[static_cast<int>(k)]; };
    const int64_t bs = at(BlockFmtField::BLOCK_SIZE);
    const int64_t block_bytes = bs * at(BlockFmtField::ELEM_BITS) / 8;
    const c10::SymInt row_bytes = (x.sym_size(-1) + (bs - 1)) / bs * block_bytes;
    if (x.dim() == 3)
      return {x.sym_size(0), x.sym_size(1), row_bytes};
    return {x.sym_size(0), row_bytes};
  }

  // The message for a NaN that a tile's scale does not make NaN, in a format
  // whose elements have no NaN code.
  inline void refuse_unstored_nan(bool unstored, const char *op)
  {
    TORCH_CHECK(!unstored, op, ": an element is NaN and its block's scale is not, and this format's "
                               "elements have no NaN code to store it in: give that block the scale "
                               "format's NaN code, as block_pack does, or remove the NaN");
  }

  // A job over x with its strides and the scale codes, for the kernels.
  inline BlockProjectJob project_job(const Tensor &x, const Tensor &scales, mptorch::GemmDtype dt,
                                     int64_t round_mode)
  {
    BlockProjectJob j;
    j.x = x.data_ptr();
    j.scales = scales.numel() != 0 ? scales.data_ptr<uint8_t>() : nullptr;
    j.dt = dt;
    j.batch = x.dim() == 3 ? x.size(0) : 1;
    j.rows = x.size(-2);
    j.cols = x.size(-1);
    j.xs0 = x.dim() == 3 ? x.stride(0) : 0;
    j.xs1 = x.stride(-2);
    j.xs2 = x.stride(-1);
    j.rm = static_cast<RoundMode>(round_mode);
    return j;
  }

} // namespace mptorch::block
