#pragma once

// The host side of the block GEMM, custom_matmul_block (common/gemm_block.h):
// the operands' checks and derivation, the output allocation, and the driver
// that hands a BlockGemmArgs to the backend. The GEMM's own pieces are reused
// as they are: the batch rule, the accumulation checks, the dtype-free RNG
// draw and the launch context all come from common/gemm_host.h, and the block
// descriptor's checks from common/block_host.h.
//
// A header of its own so that the entry points of the other GEMM ops, which
// include gemm_host.h, compile to what they compiled before this op existed.
// Only {cpu,cuda}/custom_matmul_block_entry.cpp include it.
//
//   custom_matmul_block(a_data, a_scales, a_cols, trans_a,
//                       b_data, b_scales, b_cols, trans_b,
//                       a_fmt, a_elem_max, a_scale_max, a_tensor_scale,
//                       b_fmt, b_elem_max, b_scale_max, b_tensor_scale,
//                       fused, accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
//                       accumulate_algorithm, round_mode, acc_saturation_mode,
//                       acc_subnormals_mode, acc_prng_bits, <the *_accumulated tail>)
//
// computes C [batch, M, N] = A @ B^T, A logically [M, K] and B [N, K], each
// stored as block_pack wrote it: data [.., rows, row_bytes] and scales
// [.., row_tiles, scale_cols], rank 2 or 3. `*_cols` is the unpadded length of
// the operand's packed axis, which row_bytes rounds up to whole blocks, and
// `trans_*` says the operand is stored with K as its rows: A as [K, M], B as
// [K, N]. Each operand has its own format, so any pairing is allowed. The
// result is float32, the carrier's.

#include "block_host.h"
#include "gemm_block.h"
#include "gemm_host.h"

namespace mptorch::gemm
{
  using at::Tensor;

  // One packed operand, checked, made contiguous and described.
  struct BlockGemmOperand
  {
    Tensor data;
    Tensor scales;
    mptorch::block::BlockFormatParams p{};
    int64_t batch = 1;
    bool batched = false;
  };

  inline BlockGemmOperand block_gemm_operand(const Tensor &data, const Tensor &scales, int64_t cols,
                                             bool trans, c10::IntArrayRef fmt, double elem_max,
                                             double scale_max, double tensor_scale,
                                             const char *op_name, const char *which)
  {
    using namespace mptorch::block;
    TORCH_CHECK(data.scalar_type() == at::kByte && scales.scalar_type() == at::kByte, op_name, ": ",
                which, "'s data and scales are uint8, got ", data.scalar_type(), " and ",
                scales.scalar_type());
    TORCH_CHECK((data.dim() == 2 || data.dim() == 3) && scales.dim() == data.dim(), op_name, ": ",
                which, "'s data and scales are both 2D or both 3D, got ", data.dim(), "D and ",
                scales.dim(), "D");
    TORCH_CHECK(cols >= 0, op_name, ": ", which, "_cols must not be negative, got ", cols);
    const Fmt f = checked_block_fmt(fmt, elem_max, scale_max, tensor_scale, op_name);
    BlockGemmOperand o;
    o.batched = data.dim() == 3;
    o.batch = o.batched ? data.size(0) : 1;
    TORCH_CHECK(!o.batched || scales.size(0) == o.batch, op_name, ": ", which, "'s data has ",
                o.batch, " batch elements and its scales ", scales.size(0));
    o.p = block_params(f, elem_max, scale_max, tensor_scale, cols, data.size(-2), trans);
    constexpr int64_t LIM = (int64_t(1) << 31) - 1;
    TORCH_CHECK(o.p.rows <= LIM && o.p.cols <= LIM && o.p.row_bytes <= LIM && o.p.scale_cols <= LIM,
                op_name, ": ", which, " has ", o.p.rows, " rows of ", o.p.cols,
                " elements; the tile loads address a batch element in 32-bit coordinates, so both "
                "must be below 2^31");
    TORCH_CHECK(o.p.elem_bits <= BLOCK_TABLE_BITS, op_name, ": ", which, "'s element codes are ",
                o.p.elem_bits, " bits, and the block GEMM decodes codes of at most ",
                BLOCK_TABLE_BITS, " (every OCP and NVFP4 format's) through tables; unpack a wider "
                "format and use a float GEMM");
    TORCH_CHECK(data.size(-1) == o.p.row_bytes && scales.size(-2) == o.p.row_tiles &&
                    scales.size(-1) == o.p.scale_cols,
                op_name, ": ", which, ": ", cols, " elements per row of its format pack into data "
                "[..., ", o.p.rows, ", ", o.p.row_bytes, "] and scales [..., ", o.p.row_tiles, ", ",
                o.p.scale_cols, "], got ", data.sizes(), " and ", scales.sizes());
    o.data = data.contiguous();
    o.scales = scales.contiguous();
    return o;
  }

  // The operands and formats of one call, as the schema passes them.
  struct BlockGemmCall
  {
    Tensor a_data, a_scales;
    int64_t a_cols;
    bool trans_a;
    Tensor b_data, b_scales;
    int64_t b_cols;
    bool trans_b;
    c10::IntArrayRef a_fmt;
    double a_elem_max, a_scale_max, a_tensor_scale;
    c10::IntArrayRef b_fmt;
    double b_elem_max, b_scale_max, b_tensor_scale;
  };

  // Runs the block GEMM on the Args of its mac (BlockSplitArgs,
  // BinaryKFusedArgs, or an AccumulateArgs of either). The order is the GEMM
  // driver's: the checks, the output, the empty return, the RNG draw.
  template <class Backend, class Inner>
  Tensor run_custom_matmul_block(const char *op_name, const Inner &inner, const BlockGemmCall &call,
                                 int64_t round_mode)
  {
    TORCH_CHECK(mptorch::is_round_mode(round_mode), op_name, ": ", round_mode, " is not a RoundMode");
    const BlockGemmOperand A = block_gemm_operand(call.a_data, call.a_scales, call.a_cols, call.trans_a,
                                                  call.a_fmt, call.a_elem_max, call.a_scale_max,
                                                  call.a_tensor_scale, op_name, "a");
    const BlockGemmOperand B = block_gemm_operand(call.b_data, call.b_scales, call.b_cols, call.trans_b,
                                                  call.b_fmt, call.b_elem_max, call.b_scale_max,
                                                  call.b_tensor_scale, op_name, "b");
    TORCH_CHECK(A.data.device() == A.scales.device() && B.data.device() == A.data.device() &&
                    B.scales.device() == A.data.device(),
                op_name, ": every operand tensor must be on one device");

    const int64_t M = call.trans_a ? A.p.cols : A.p.rows;
    const int64_t K = call.trans_a ? A.p.rows : A.p.cols;
    const int64_t N = call.trans_b ? B.p.cols : B.p.rows;
    const int64_t K_b = call.trans_b ? B.p.rows : B.p.cols;
    TORCH_CHECK(K == K_b, op_name, ": inner dimensions must match (got ", K, " vs ", K_b, ")");

    int64_t batch;
    if (A.batch == B.batch)
      batch = A.batch;
    else if (A.batch == 1)
      batch = B.batch;
    else if (B.batch == 1)
      batch = A.batch;
    else
      TORCH_CHECK(false, op_name, ": batch dimensions must match or be 1 (got ", A.batch, " and ",
                  B.batch, ")");
    const bool batched = A.batched || B.batched;

    Tensor c = at::empty(matmul_output_sizes(batch, M, N, batched), A.data.options().dtype(at::kFloat));
    if (batch == 0 || M == 0 || N == 0)
      return c;

    GemmShape s;
    s.M = M;
    s.K = K;
    s.N = N;
    s.batch = batch;
    // The data's batch strides, in bytes: the blocked tile loads add them to
    // a byte pointer. 0 broadcasts an operand, as for the GEMM.
    s.stride_a = A.batch == 1 ? 0 : A.p.rows * A.p.row_bytes;
    s.stride_b = B.batch == 1 ? 0 : B.p.rows * B.p.row_bytes;
    s.trans_a = call.trans_a;
    s.trans_b = call.trans_b;
    s.rm = static_cast<RoundMode>(round_mode);
    s.use_rng = (s.rm == RoundMode::SR);
    s.dt = mptorch::GemmDtype::Float;

    BlockGemmArgs<Inner> args;
    args.inner = inner;
    args.a = BlockOperand{A.p, A.scales.data_ptr<uint8_t>(), A.batch == 1 ? 0 : A.p.row_tiles * A.p.scale_cols};
    args.b = BlockOperand{B.p, B.scales.data_ptr<uint8_t>(), B.batch == 1 ? 0 : B.p.row_tiles * B.p.scale_cols};

    typename Backend::LaunchContext ctx = Backend::make_context(
        s.use_rng, draws_per_k_step_of(inner) * words_per_draw(s.dt) * static_cast<uint64_t>(K));
    s.a = A.data.data_ptr();
    s.b = B.data.data_ptr();
    s.c = c.data_ptr();
    bind_tensors(ctx, A.data, B.data, c, nullptr);
    Backend::launch(s, args, ctx);
    return c;
  }

  // The whole schema: the mac packed from its arguments (split or fused),
  // NAIVE run on it and KAHAN, BLOCK and TREE on its AccumulateArgs, with the
  // accumulation checked as the *_accumulated ops check it.
  template <class Backend>
  Tensor custom_matmul_block_entry(const BlockGemmCall &call, bool fused, bool accumulate_quant,
                                   int64_t acc_K, int64_t acc_P, int64_t acc_bias, bool acc_is_signed,
                                   int64_t accumulate_algorithm, int64_t round_mode,
                                   int64_t acc_saturation_mode, int64_t acc_subnormals_mode,
                                   int64_t acc_prng_bits,
                                   const AccumulateTail<BinaryKWidths, BinaryKCommon> &tail)
  {
    constexpr const char *op = "custom_matmul_block";
    TORCH_CHECK(mptorch::is_accumulate_algorithm(accumulate_algorithm), op, ": ",
                accumulate_algorithm, " is not an AccumulateAlgorithm");
    TORCH_CHECK(tail.block_size >= 0 && tail.block_size <= (int64_t(1) << 30), op, ": block_size ",
                tail.block_size, " is out of range");
    const auto alg = static_cast<AccumulateAlgorithm>(accumulate_algorithm);
    auto run = [&](const auto &base) -> Tensor
    {
      using Base = std::remove_cvref_t<decltype(base)>;
      if (alg == AccumulateAlgorithm::NAIVE)
      {
        TORCH_CHECK(tail.block_size == 0 && !tail.outer_quant, op,
                    ": AccumulateAlgorithm.NAIVE takes no block_size and no outer format");
        return run_custom_matmul_block<Backend>(op, base, call, round_mode);
      }
      AccumulateArgs<Base> args;
      args.base = base;
      args.alg = alg;
      args.block_size = static_cast<int>(tail.block_size);
      args.outer_quant = tail.outer_quant;
      args.outer = tail.outer;
      args.outer_c = tail.outer_c;
      check_accumulate_algorithm(args, op, accumulate_algorithm);
      return run_custom_matmul_block<Backend>(op, args, call, round_mode);
    };
    if (fused)
      return run(pack_binaryK_fused(accumulate_quant, acc_K, acc_P, acc_bias, acc_is_signed,
                                    acc_saturation_mode, acc_subnormals_mode, acc_prng_bits));
    BlockSplitArgs split;
    split.accumulate_quant = accumulate_quant;
    split.acc = binaryK_widths(acc_K, acc_P, acc_bias, acc_is_signed);
    split.acc_c = BinaryKCommon{acc_is_signed, static_cast<SaturationMode>(acc_saturation_mode),
                                static_cast<SubnormalsMode>(acc_subnormals_mode),
                                static_cast<int>(acc_prng_bits)};
    return run(split);
  }

} // namespace mptorch::gemm
