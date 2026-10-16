// The MPS entry points of the elementwise block ops: block_pack,
// block_unpack, block_quant and block_quant_ (common/block_decode.h has the
// format; mptorch/quant/block.py the Python side). The block GEMM's is
// custom_matmul_block_entry.cpp.
//
// Each observes what its CPU twin (cpu/block_kernel.cpp) does, in the same
// order and with the same checks (common/block_host.h), so an MPS call
// raises what a CPU call raises: the descriptor is validated for binary32,
// the outputs are allocated, an empty tensor returns before the SR seed is
// drawn, and the seed is drawn from the CPU generator as the CPU kernel
// draws it. MPS has no float64 tensors, so block_unpack refuses a float64
// result by name.
//
// The kernels are block.metal's, compiled per instantiation: the storage
// dtype, the rounding mode, whether the call quantizes or packs, whether the
// scale rule is Four Over Six, and the casts the pack runs, written out as
// literals by block_cast_source below so that their constants fold. The
// layout (BlockFormatParams, which holds the extents as well as the format)
// is bound as a buffer, byte for byte as the host builds it; gemm.metal
// asserts its size and offsets on the GPU side.

#include "../common/block_host.h"
#include "../cpu/utils.h" // draw_cpu_seed
#include "../quant_ops.h"
#include "launch_params.h"
#include "metal_runtime.h"
#include <ATen/ops/empty_like.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

using at::Tensor;
using namespace mptorch::block;

namespace
{

  // A binary32 value as a Metal literal that is exactly it: its word.
  std::string word_literal(float v)
  {
    uint32_t w;
    std::memcpy(&w, &v, sizeof w);
    char buf[48];
    std::snprintf(buf, sizeof buf, "as_type<float>(0x%08Xu)", w);
    return buf;
  }

  std::string b(bool v) { return v ? "true" : "false"; }
  std::string i(int64_t v) { return std::to_string(v); }

  // The Metal definition of `mpt_block_cast()`: block_decode.h's
  // make_block_cast, with each cast built on the device by the function
  // the host builds it with, from the descriptor's widths, and the scale
  // rule's three constants copied from the host's, word for word.
  std::string block_cast_source(const Fmt &f, double elem_max)
  {
    static_assert(sizeof(BlockCastT<float>) == 324, "write every BlockCastT field in block_cast_source");
    auto at = [&f](BlockFmtField k) { return f[static_cast<int>(k)]; };
    const BlockCastT<float> host = make_block_cast<float>(f, elem_max);
    std::string s = "inline mptorch::block::BlockCastT<float> mpt_block_cast()\n{\n    mptorch::block::BlockCastT<float> c{};\n";
    s += "    c.elem_family = " + i(host.elem_family) + ";\n";
    s += "    c.elem_signed = " + b(host.elem_signed) + ";\n";
    s += "    c.elem_sub = SubnormalsMode(" + i(static_cast<int>(host.elem_sub)) + ");\n";
    s += "    c.prng_bits = " + i(host.prng_bits) + ";\n";
    if (host.elem_family == static_cast<int>(BlockFamily::SUPERFP))
      s += "    c.elem_sfp = make_superfp_params<float>(" + i(at(BlockFmtField::ELEM_MAN_BITS)) + ", " +
           i(at(BlockFmtField::ELEM_EXP_BITS)) + ", " + i(at(BlockFmtField::ELEM_NORMAL_BINADES)) + ", " +
           i(at(BlockFmtField::ELEM_BIAS)) + ", SaturationMode::SAT_FINITE);\n";
    else
      s += "    c.elem_bk = make_binaryK_params<float>(" + i(at(BlockFmtField::ELEM_MAN_BITS)) + ", " +
           i(at(BlockFmtField::ELEM_EXP_BITS)) + ", " + i(at(BlockFmtField::ELEM_BIAS)) + ", " +
           b(host.elem_signed) + ", SaturationMode::SAT_FINITE, " +
           b(host.elem_sub == SubnormalsMode::EXTENDED_NORMALS) + ");\n";
    s += "    c.scale_family = " + i(host.scale_family) + ";\n";
    if (at(BlockFmtField::SCALE_KIND) == static_cast<int64_t>(ScaleKind::MINIFLOAT))
    {
      if (host.scale_family == static_cast<int>(BlockFamily::SUPERFP))
        s += "    c.scale_sfp = make_superfp_params<float>(" + i(at(BlockFmtField::SCALE_MAN_BITS)) + ", " +
             i(at(BlockFmtField::SCALE_EXP_BITS)) + ", " + i(at(BlockFmtField::SCALE_NORMAL_BINADES)) + ", " +
             i(at(BlockFmtField::SCALE_BIAS)) + ", SaturationMode::SAT_FINITE);\n";
      else
        s += "    c.scale_bk = make_binaryK_params<float>(" + i(at(BlockFmtField::SCALE_MAN_BITS)) + ", " +
             i(at(BlockFmtField::SCALE_EXP_BITS)) + ", " + i(at(BlockFmtField::SCALE_BIAS)) + ", " +
             b(at(BlockFmtField::SCALE_SIGNED) != 0) + ", SaturationMode::SAT_FINITE, " +
             b(at(BlockFmtField::SCALE_SUBNORMALS) == static_cast<int64_t>(SubnormalsMode::EXTENDED_NORMALS)) +
             ");\n";
    }
    s += "    c.scale_rounding = " + i(host.scale_rounding) + ";\n";
    s += "    c.elem_fit = " + word_literal(host.elem_fit) + ";\n";
    s += "    c.max_sig = " + word_literal(host.max_sig) + ";\n";
    s += "    c.fit_sig = " + word_literal(host.fit_sig) + ";\n";
    return s + "    return c;\n}\n";
  }

  // The BlockFormatParams binding: the host's struct, byte for byte.
  static_assert(sizeof(BlockFormatParams) == 184, "BlockFormatParams must have one layout on the host and the GPU");

  // The CPU's draw, for the CPU's draw order: an SR call draws one seed,
  // and only once it has elements to round.
  uint64_t seed_for(int64_t round_mode)
  {
    return static_cast<RoundMode>(round_mode) == RoundMode::SR ? draw_cpu_seed() : 0;
  }

  void refuse_float64(at::ScalarType st, const char *op)
  {
    TORCH_CHECK(st != at::kDouble, op, ": MPS has no float64 tensors, so there is no binary64 block kernel "
                                       "on MPS; unpack to float32, float16 or bfloat16, or on a CPU or CUDA "
                                       "tensor");
  }

  // Launches the pack over x's tiles: codes into data and scales (`o`
  // null), or decoded values into o, which is x itself in place. Each tile
  // gets one thread (mpt_block_pack) when the call has at least 256 tiles
  // per element of a tile, and one threadgroup otherwise
  // (mpt_block_pack_tile, 64 elements a thread, from one simdgroup to
  // BLOCK_TILE_THREADS threads). One thread runs its tile in series, so the
  // GPU needs many tiles to hide the latency; given them, it is the cheaper
  // kernel, with no barrier and no idle thread. On an M3 Pro, over 256^2 to
  // 4096^2 elements, 1D blocks of 16 to 128 and 2D tiles of 64 to 16,384
  // elements, the rule picked the faster kernel or one within 3% of it. A
  // 128 x 128 tile always gets a threadgroup: under Four Over Six that takes
  // 6.4 ms at 4096^2, where one thread took 72. A 1D block of 16 gets one
  // thread from 256^2, and NVFP4's 16 x 16 tile from 4096^2: there one
  // thread takes 2.8 ms and a threadgroup 3.1, and at 1024^2 one thread
  // takes 0.62 ms and a threadgroup 0.44.
  void run_pack(const Tensor &x, const Tensor *o, const Tensor *data, const Tensor *scales, const Fmt &f,
                const BlockFormatParams &p, const BlockExtents &e, double elem_max, int64_t round_mode,
                const char *op)
  {
    const bool quant = o != nullptr;
    mptorch_mps::BlockPackLaunch L{};
    L.batch = e.batch;
    L.rows = e.rows;
    L.cols = e.cols;
    L.xs0 = e.batched ? x.stride(0) : 0;
    L.xs1 = x.stride(-2);
    L.xs2 = x.stride(-1);
    if (quant)
    {
      L.os0 = e.batched ? o->stride(0) : 0;
      L.os1 = o->stride(-2);
      L.os2 = o->stride(-1);
    }
    L.tiles = static_cast<uint64_t>(e.batch * p.row_tiles * p.n_blocks);
    TORCH_CHECK(L.tiles <= std::numeric_limits<uint32_t>::max(), op,
                ": an MPS block pack runs one thread or threadgroup per tile, at most 2^32 - 1 of them, got ",
                L.tiles);
    const uint32_t tile_elems = static_cast<uint32_t>(p.block_rows * p.block_size);
    const bool grouped = L.tiles < uint64_t{256} * tile_elems;
    const char *storage = mptorch_mps::metal_storage_type(x.scalar_type(), op);
    L.seed = seed_for(round_mode);
    const std::string tail =
        std::string("using mpt_storage_t = ") + storage + ";\n#define MPT_UNPACK 0\n#define MPT_ROUND_MODE " +
        mptorch_mps::metal_round_mode(round_mode) + "\n#define MPT_QUANT " + (quant ? "1" : "0") +
        "\n#define MPT_FOS " +
        (is_four_over_six(static_cast<int32_t>(f[static_cast<int>(BlockFmtField::SCALE_ROUNDING)])) ? "1" : "0") +
        "\n#define MPT_TILE " + (grouped ? "1" : "0") + "\n" + block_cast_source(f, elem_max);
    // The buffers block.metal does not use on this path are bound to x, as
    // are a scale-free format's empty scales.
    const Tensor &scales_or_x = !quant && scales->numel() != 0 ? *scales : x;
    const std::vector<mptorch_mps::KernelArg> args{x, quant ? *o : x, quant ? x : *data, scales_or_x,
                                                   mptorch_mps::KernelArg::params(p),
                                                   mptorch_mps::KernelArg::params(L)};
    if (!grouped)
    {
      mptorch_mps::launch(mptorch_mps::KernelSource::BlockPack, tail, args, L.tiles);
      return;
    }
    const uint32_t threads = std::clamp<uint32_t>(tile_elems / 64, 32, mptorch_mps::BLOCK_TILE_THREADS);
    mptorch_mps::launch_groups(mptorch_mps::KernelSource::BlockPackTile, tail, args, L.tiles, 1, 1, threads, 1);
  }

  // The body of block_quant and block_quant_: every element of x rounded
  // into o, which is x itself for the in-place op.
  void quantize_into(const Tensor &x, const Tensor &o, c10::IntArrayRef fmt, double elem_max,
                     double scale_max, double tensor_scale, int64_t round_mode, const char *op)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
    const BlockExtents e = block_extents(x, op);
    block_dtype(x.scalar_type(), op);
    const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, false);
    if (x.numel() == 0)
      return;
    const BlockFormatParams p = block_params<float>(f, elem_max, scale_max, tensor_scale, e.cols, e.rows);
    run_pack(x, &o, nullptr, nullptr, f, p, e, elem_max, round_mode, op);
  }

} // namespace

std::tuple<Tensor, Tensor> block_pack_mps(Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                                          double tensor_scale, int64_t round_mode)
{
  constexpr const char *op = "block_pack";
  TORCH_CHECK(mptorch::is_round_mode(round_mode), op, ": ", round_mode, " is not a RoundMode");
  const BlockExtents e = block_extents(x, op);
  block_dtype(x.scalar_type(), op);
  const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, false);
  const BlockFormatParams p = block_params<float>(f, elem_max, scale_max, tensor_scale, e.cols, e.rows);
  Tensor data = at::empty(block_sizes(e, e.rows, p.row_bytes), x.options().dtype(at::kByte));
  Tensor scales = at::empty(block_sizes(e, p.row_tiles, p.scale_cols), x.options().dtype(at::kByte));
  if (x.numel() == 0)
    return {data, scales};
  run_pack(x, nullptr, &data, &scales, f, p, e, elem_max, round_mode, op);
  return {data, scales};
}

Tensor block_unpack_mps(Tensor data, Tensor scales, int64_t cols, c10::IntArrayRef fmt, double elem_max,
                        double scale_max, double tensor_scale, at::ScalarType dtype)
{
  constexpr const char *op = "block_unpack";
  refuse_float64(dtype, op);
  block_dtype(dtype, op);
  const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op, false);
  BlockFormatParams p;
  const BlockExtents e = unpack_extents(data, scales, cols, f, elem_max, scale_max, tensor_scale, p, op);
  Tensor o = at::empty(block_sizes(e, e.rows, e.cols), data.options().dtype(dtype));
  if (o.numel() == 0)
    return o;
  const Tensor d = data.contiguous(), s = scales.contiguous();
  const uint64_t rows = static_cast<uint64_t>(e.batch * e.rows);
  TORCH_CHECK(rows <= std::numeric_limits<uint32_t>::max() && static_cast<uint64_t>(e.cols) <= (uint64_t(1) << 31),
              op, ": an MPS unpack covers at most 2^32 - 1 rows of at most 2^31 elements, got ", rows, " and ",
              e.cols);
  const mptorch_mps::BlockUnpackLaunch L{e.batch, e.rows, e.cols};
  const std::string tail = std::string("using mpt_storage_t = ") + mptorch_mps::metal_storage_type(dtype, op) +
                           ";\n#define MPT_UNPACK 1\n";
  constexpr uint32_t threads = 256;
  // An unquantized scale-free format still binds a scales buffer; its
  // tensor may be empty, and then the data stands in for it (never read).
  mptorch_mps::launch_groups(mptorch_mps::KernelSource::BlockUnpack, tail,
                             {d, s.numel() != 0 ? s : d, o, mptorch_mps::KernelArg::params(p),
                              mptorch_mps::KernelArg::params(L)},
                             (static_cast<uint64_t>(e.cols) + threads - 1) / threads, rows, 1, threads, 1);
  return o;
}

Tensor block_quantize_mps(Tensor x, c10::IntArrayRef fmt, double elem_max, double scale_max, double tensor_scale,
                          int64_t round_mode)
{
  // A result laid out as x is, as on the CPU.
  Tensor o = at::empty_like(x);
  quantize_into(x, o, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant");
  return o;
}

Tensor &block_quantize_mps_(Tensor &x, c10::IntArrayRef fmt, double elem_max, double scale_max,
                            double tensor_scale, int64_t round_mode)
{
  TORCH_CHECK(x.is_non_overlapping_and_dense(), "block_quant_ writes its argument in place and "
              "needs a tensor whose elements do not overlap, got strides ", x.strides(),
              " for sizes ", x.sizes(), ": use block_quant, which writes a new tensor");
  quantize_into(x, x, fmt, elem_max, scale_max, tensor_scale, round_mode, "block_quant_");
  return x;
}
