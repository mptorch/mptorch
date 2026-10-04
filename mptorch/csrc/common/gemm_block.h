#pragma once

// The block GEMM (custom_matmul_block): C = op(A) @ op(B)^T over two packed
// block-format operands (common/block_decode.h), each decoded in the tile
// load, in binary32, and multiplied and summed by the GEMM kernel's own K-loop
// and Mac/Accumulator policies. So every rounding mode and every accumulate
// algorithm applies, and the SR keying, the batch dimension and the grid-z
// chunking are the GEMM's.
//
// The decode rides on a Blocked<Acc> policy that derives from the accumulator
// it wraps, the way the conv ops' geometry rides on Gathered<Acc>
// (common/gemm_gather.h): the kernels take the block load behind
// `if constexpr (is_blocked_v<Accumulator>)`, which no other policy
// satisfies, so the kernels every other op runs compile to the bytes they
// compiled before (dev/gemm_roadmap.md, R-3). A template parameter of the
// kernels would have renamed all of them.
//
// Operands. A is logically [M, K] and B [N, K], both with K along the packed
// axis unless their `trans` flag says the operand is stored with K as its
// rows ([K, M] or [K, N]); decoding is random access, so either orientation
// is a select in the load. The multiply is the carrier's own, rounded once
// (IdentityMultiplier): what a hardware MX unit does exactly or in a wider
// format, the block formats' values having few bits; the sum is rounded to
// an accumulate format, or left in the carrier. No palette.
//
// This header is ATen-free and includes nothing the policies do not; the
// Metal prelude does not include it (the block GEMM raises on MPS until
// dev/continuation_plan.md's phase H).

#include "block_decode.h"
#include "gemm_accumulate.h"
#include "gemm_args.h"

// The multiply of a block GEMM: the product in the carrier, rounded once and
// by nothing else. rn_mul (__fmul_rn on the device) rather than `a * b`, so
// that nvcc cannot contract the split mac's `acc + a * b` into an FMA and turn
// it into the fused one; the host builds with -ffp-contract=off already.
template <class T>
struct IdentityMultiplier
{
    using value_t = T;
    CUDA_HOST_DEVICE_INLINE T operator()(T a, T b, MPTORCH_THREAD PhiloxEngine & /*rng*/) const
    {
        return FloatTraits<T>::rn_mul(a, b);
    }
};

namespace mptorch::gemm
{

  // The split block mac: the carrier's product, then the sum rounded to the
  // accumulate format (or not rounded). The fused one is BinaryKFusedArgs, as
  // it is for the flat binaryK op: a fused step has no multiply format to
  // differ in.
  struct BlockSplitArgs
  {
    static constexpr MPTORCH_CONSTANT bool mixed = false;
    // The multiply draws nothing, so a step draws at most once, for the sum.
    static constexpr MPTORCH_CONSTANT uint64_t draws_per_k_step = 1;

    bool accumulate_quant = false;
    BinaryKWidths acc{};
    BinaryKCommon acc_c{};

    template <class T, RoundMode RM, class F>
    void with_accumulator(MPTORCH_THREAD F &&f) const
    {
      if (accumulate_quant)
      {
        using Mac = SplitMac<IdentityMultiplier<T>, BinaryKAdderT<T, RM>>;
        f(NaiveAccumulator<Mac>{Mac{IdentityMultiplier<T>{}, make_add<T, RM>(acc, acc_c)}, T(0)});
      }
      else
      {
        using Mac = SplitMac<IdentityMultiplier<T>, IdentityAdder<T>>;
        f(NaiveAccumulator<Mac>{Mac{IdentityMultiplier<T>{}, IdentityAdder<T>{}}, T(0)});
      }
    }
  };

  template <>
  struct AccumulateFamily<BlockSplitArgs>
  {
    using Widths = BinaryKWidths;
    using Common = BinaryKCommon;
    static constexpr MPTORCH_CONSTANT bool has_product = true;
  };

  // One packed operand as the tile load reads it: its format and layout, and
  // its scales with their batch stride in scale bytes (the data's batch
  // stride is the kernel's own stride_a / stride_b, in bytes here).
  struct BlockOperand
  {
    mptorch::block::BlockFormatParams p;
    const MPTORCH_BLOCK_DEVICE uint8_t *scales;
    int64_t scale_stride;
  };

  // An Accumulator policy whose operands are packed block formats. It is the
  // wrapped policy in every other respect, like Gathered.
  template <class Acc>
  struct Blocked : Acc
  {
    static constexpr MPTORCH_CONSTANT bool blocked = true;
    BlockOperand block_a;
    BlockOperand block_b;
  };

#if !defined(__METAL_VERSION__)
  template <class Accumulator>
  inline constexpr bool is_blocked_v = requires { Accumulator::blocked; };

  // The Args of the block GEMM: the Args of the mac it runs (BlockSplitArgs,
  // BinaryKFusedArgs, or their AccumulateArgs for KAHAN, BLOCK and TREE) and
  // the two operands. The backends' launch_block_as builds `inner`'s policy
  // and wraps it in a Blocked.
  template <class Inner>
  struct BlockGemmArgs
  {
    static constexpr bool mixed = false;
    Inner inner{};
    BlockOperand a{};
    BlockOperand b{};
  };
#endif

  // The decode tables a CTA (or a CPU task) builds once and its tile loads
  // read: per operand, each element code's value and each scale code's
  // factor. Filled by decode_elem and decode_scale themselves, so a load
  // through them is the arithmetic decode's value, bit for bit, for a table
  // read and a multiply in place of the decode's thirty-odd instructions. The
  // block GEMM therefore takes element codes of at most BLOCK_TABLE_BITS bits,
  // which every OCP and NVFP4 format's are, and its driver refuses a wider
  // one: with the arithmetic decode compiled in beside the tables as a
  // fallback, the kernels held 48-80 registers against 30-38 without it
  // (sm_89), which cost a third of their resident blocks. The elementwise
  // ops take codes of up to 16 bits.
  constexpr MPTORCH_CONSTANT int BLOCK_TABLE_BITS = 8;
  constexpr MPTORCH_CONSTANT int BLOCK_TABLE = 1 << BLOCK_TABLE_BITS;

  // Entry i of an operand's two tables: [0, BLOCK_TABLE) its element values,
  // [BLOCK_TABLE, 2 * BLOCK_TABLE) its scale factors (1 without a scale).
  template <class T>
  CUDA_HOST_DEVICE_INLINE void fill_block_table(MPTORCH_THREAD T *table, int i,
                                                const MPTORCH_THREAD mptorch::block::BlockFormatParams &p)
  {
    using mptorch::block::ScaleKind;
    table[i] = p.elem_bits <= BLOCK_TABLE_BITS && i < (1 << p.elem_bits)
                   ? mptorch::block::decode_elem<T>(static_cast<uint32_t>(i), p)
                   : T(0);
    table[BLOCK_TABLE + i] = p.scale_kind == static_cast<int32_t>(ScaleKind::NONE)
                                 ? T(1)
                                 : mptorch::block::decode_scale<T>(static_cast<uint32_t>(i), p);
  }

  // Logical element (i, k) of a packed operand for the tile loads, i being
  // A's row or B's column: `data` is the operand's base, `data_off` its batch
  // element's offset in bytes, and `bId` that batch element.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_load(const MPTORCH_BLOCK_DEVICE void *data,
                                         const MPTORCH_THREAD BlockOperand &o, int64_t bId,
                                         int64_t data_off, int64_t i, int64_t k)
  {
    return mptorch::block::load_block_operand<T>(
        static_cast<const MPTORCH_BLOCK_DEVICE uint8_t *>(data) + data_off,
        o.scales + bId * o.scale_stride, i, k, o.p);
  }

  // block_load through an operand's tables (fill_block_table): the element's
  // code and its tile's scale code, read from the packed arrays, and their
  // value and factor from the tables. What decode_value computes from them,
  // bit for bit: the one product, and a zero that is +0.0. The element code
  // is at most BLOCK_TABLE_BITS bits (the driver's check).
  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_load_table(const MPTORCH_BLOCK_DEVICE void *data,
                                             const MPTORCH_THREAD BlockOperand &o,
                                             const MPTORCH_THREAD T *table, int64_t bId,
                                             int64_t data_off, int64_t i, int64_t k)
  {
    const MPTORCH_THREAD mptorch::block::BlockFormatParams &p = o.p;
    // 32-bit coordinates, widened only where they meet a stride: the driver
    // holds a batch element's rows, columns and row bytes below 2^31.
    const int32_t r = static_cast<int32_t>(p.trans ? k : i);
    const int32_t c = static_cast<int32_t>(p.trans ? i : k);
    const int32_t blk = c >> p.log2_block_size;
    const int32_t bit = (c & (p.block_size - 1)) * p.elem_bits;
    const MPTORCH_BLOCK_DEVICE uint8_t *b = static_cast<const MPTORCH_BLOCK_DEVICE uint8_t *>(data) + data_off +
                                            static_cast<int64_t>(r) * static_cast<int32_t>(p.row_bytes) +
                                            blk * p.block_bytes + (bit >> 3);
    const int sh = bit & 7;
    uint32_t w = b[0];
    if (sh + p.elem_bits > 8)
      w |= static_cast<uint32_t>(b[1]) << 8;
    const uint32_t code = (w >> sh) & ((1u << p.elem_bits) - 1u);
    const uint32_t sc =
        p.scale_cols != 0
            ? static_cast<uint32_t>(o.scales[bId * o.scale_stride +
                                             static_cast<int64_t>(r >> p.log2_block_rows) *
                                                 static_cast<int32_t>(p.scale_cols) +
                                             blk])
            : 0u;
    const T v = FloatTraits<T>::rn_mul(table[code], table[BLOCK_TABLE + sc]);
    return (mptorch::block::block_word(v) & FloatTraits<T>::ABS_MASK) == 0 ? T(0) : v;
  }

} // namespace mptorch::gemm
