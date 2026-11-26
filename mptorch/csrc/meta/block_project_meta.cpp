// The Meta kernel of block_project (common/block_project.h), which
// meta_ops.cpp registers: the CPU entry point's checks, in its order and
// with its messages (common/block_project_host.h, on symbolic sizes), and the
// codes it allocates, with nothing computed and no RNG state drawn. The NaN
// check reads the elements, so it is the real kernels' only.
// A file of its own, so that meta_ops.cpp's other kernels are what they were.

#include "../common/block_project_host.h"
#include "meta_ops.h"
#include <ATen/ops/empty.h>

Tensor block_project_meta(Tensor x, Tensor scales, c10::IntArrayRef fmt, double elem_max, double scale_max,
                          double tensor_scale, int64_t round_mode)
{
  bool wide = false;
  const mptorch::block::Fmt f = mptorch::block::check_block_project(
      x, scales, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_project", wide);
  return at::empty_symint(mptorch::block::block_project_sizes(x, f), x.options().dtype(at::kByte));
}
