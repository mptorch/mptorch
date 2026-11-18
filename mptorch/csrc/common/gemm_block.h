#pragma once

// The block GEMM (custom_matmul_block): C = op(A) @ op(B)^T over two
// operands packed in block formats (block_decode.h). Each operand is decoded
// in the tile load, into the carrier (binary32, or binary64 when the call
// asks for it), and multiplied and summed by the GEMM kernel's own K-loop and
// its Mac and accumulator policies. So every rounding mode and every
// accumulate algorithm applies, and the keying of SR's random streams, the
// batch dimension and the splitting of a large batch over several CUDA
// launches (for the grid's z limit) are the GEMM's.
//
// The decode is attached to a Blocked<Acc> policy, which derives from the
// accumulator it wraps, as the conv ops' geometry is attached to
// Gathered<Acc> (gemm_gather.h). The kernels run the block load behind
// `if constexpr (is_blocked_v<Accumulator>)`, which no other policy
// satisfies, so the kernels of every other op compile exactly as they would
// without it. A new template parameter of the kernels would have changed
// all of their names.
//
// The operands. A is logically [M, K] and B [N, K], both with K along the
// packed axis, unless an operand's `trans` flag says it is stored with K as
// its rows ([K, M] or [K, N]). Decoding reads any element directly, so either
// orientation costs only a select in the load. The product is the carrier's
// own multiplication, rounded once (IdentityMultiplier). Since block-format
// values have few significand bits, that is what a hardware MX unit
// computes, exactly or in a wider format. The sum is rounded to an
// accumulate format, or left in the carrier. There is no palette version.
//
// This header uses no ATen, and includes nothing that the policies do not.
// The MPS backend's GEMM (mps/gemm.metal) takes its decode, tables and mac
// from here, and takes the operands as buffers of its own (Blocked is used
// only by the CPU and CUDA).

#include "block_decode.h"
#include "gemm_accumulate.h"
#include "gemm_args.h"

// The multiply of a block GEMM: the product in the carrier, rounded once by
// the carrier and to no format. It is rn_mul (__fmul_rn on the device) rather
// than `a * b`, so that nvcc cannot contract the split mac's `acc + a * b`
// into an FMA, which would turn it into the fused mac; the host already
// compiles with -ffp-contract=off.
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
  // accumulate format (or not rounded). The fused block mac uses
  // BinaryKFusedArgs, as the binaryK GEMM op does: a fused step has no
  // multiply format, so there is nothing for a block version to change.
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

  // One packed operand as the tile load reads it in carrier T: its format and
  // layout, and its scales, with their batch stride in bytes of scales. (The
  // batch stride of the element data is the kernel's own stride_a or
  // stride_b, counted in bytes here.)
  template <class T>
  struct BlockOperandT
  {
    mptorch::block::BlockFormatParamsT<T> p;
    const MPTORCH_BLOCK_DEVICE uint8_t *scales;
    int64_t scale_stride;
  };

  using BlockOperand = BlockOperandT<float>;

#if !defined(__METAL_VERSION__)
  // An accumulator policy whose operands are packed in block formats. In
  // every other respect it is the policy it wraps, like Gathered. The
  // operands' parameters are in the accumulator's carrier, through a member
  // type rather than a template parameter of Blocked, so a kernel's name
  // depends only on its accumulator. Only the CPU and CUDA use it, as with
  // Gathered (mps/gemm.metal takes the operands' scales and parameters as
  // buffers of their own).
  template <class Acc>
  struct Blocked : Acc
  {
    static constexpr MPTORCH_CONSTANT bool blocked = true;
    BlockOperandT<typename Acc::value_t> block_a;
    BlockOperandT<typename Acc::value_t> block_b;
  };

  template <class Accumulator>
  inline constexpr bool is_blocked_v = requires { Accumulator::blocked; };

  // The Args of the block GEMM: the Args of the mac it runs (BlockSplitArgs,
  // BinaryKFusedArgs, or their AccumulateArgs for KAHAN, BLOCK and TREE), and
  // the two operands in the carrier the call runs in: `a` and `b` for
  // binary32, `a64` and `b64` for binary64 (the driver fills in one pair).
  // The backends' launch_block_as builds `inner`'s policy and wraps it in a
  // Blocked over the pair for its carrier, operand_a<T>() and
  // operand_b<T>().
  template <class Inner>
  struct BlockGemmArgs
  {
    static constexpr bool mixed = false;
    Inner inner{};
    BlockOperandT<float> a{};
    BlockOperandT<float> b{};
    BlockOperandT<double> a64{};
    BlockOperandT<double> b64{};

    template <class T>
    const BlockOperandT<T> &operand_a() const
    {
      if constexpr (std::is_same_v<T, double>)
        return a64;
      else
        return a;
    }

    template <class T>
    const BlockOperandT<T> &operand_b() const
    {
      if constexpr (std::is_same_v<T, double>)
        return b64;
      else
        return b;
    }
  };
#endif

  // The decode tables that a CTA (a CUDA thread block) or a CPU task builds
  // once and its tile loads read: per operand, the value of each element code
  // and the factor of each scale code. decode_elem and decode_scale fill
  // them, so a load through the tables gives the arithmetic decode's value
  // bit for bit, for a table read and a multiply instead of the decode's
  // thirty-odd instructions. So the block GEMM takes element codes of at most
  // BLOCK_TABLE_BITS bits, as every OCP and NVFP4 format has, and its driver
  // refuses wider ones. With the arithmetic decode compiled in next to the
  // tables as a fallback, the kernels used 48-80 registers against 30-38
  // without it (sm_89), which cost a third of their resident blocks. The
  // elementwise block ops take codes of up to 16 bits.
  constexpr MPTORCH_CONSTANT int BLOCK_TABLE_BITS = 8;
  constexpr MPTORCH_CONSTANT int BLOCK_TABLE = 1 << BLOCK_TABLE_BITS;

  // The tables' address space: a CTA's shared memory on CUDA, a
  // threadgroup's memory on Metal, which, unlike CUDA's, has to be named on
  // the pointer (mps/gemm.metal), and a task's stack on the CPU.
#if defined(__METAL_VERSION__)
#define MPTORCH_BLOCK_TABLE threadgroup
#else
#define MPTORCH_BLOCK_TABLE
#endif

  // Fills entry i of an operand's two tables: [0, BLOCK_TABLE) holds its
  // element values, and [BLOCK_TABLE, 2 * BLOCK_TABLE) its scale factors (1
  // for a format without a scale).
  template <class T>
  CUDA_HOST_DEVICE_INLINE void fill_block_table(MPTORCH_BLOCK_TABLE T *table, int i,
                                                const MPTORCH_THREAD mptorch::block::BlockFormatParamsT<T> &p)
  {
    using mptorch::block::ScaleKind;
    table[i] = p.elem_bits <= BLOCK_TABLE_BITS && i < (1 << p.elem_bits)
                   ? mptorch::block::decode_elem<T>(static_cast<uint32_t>(i), p)
                   : T(0);
    table[BLOCK_TABLE + i] = p.scale_kind == static_cast<int32_t>(ScaleKind::NONE)
                                 ? T(1)
                                 : mptorch::block::decode_scale<T>(static_cast<uint32_t>(i), p);
  }

  // Logical element (i, k) of a packed operand, for the tile loads, where i
  // is A's row or B's column. `data` is the operand's base, `data_off` the
  // offset of its batch element in bytes, and `bId` that batch element.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_load(const MPTORCH_BLOCK_DEVICE void *data,
                                         const MPTORCH_THREAD BlockOperandT<T> &o, int64_t bId,
                                         int64_t data_off, int64_t i, int64_t k)
  {
    return mptorch::block::load_block_operand<T>(
        static_cast<const MPTORCH_BLOCK_DEVICE uint8_t *>(data) + data_off,
        o.scales + bId * o.scale_stride, i, k, o.p);
  }

  // block_load through an operand's tables (fill_block_table). The element's
  // code and its tile's scale code are read from the packed arrays, and their
  // value and factor from the tables. The result is bit for bit what
  // decode_value computes from them: the one product, with a zero returned as
  // +0.0. The element code has at most BLOCK_TABLE_BITS bits, which the
  // driver checks.
  template <class T>
  CUDA_HOST_DEVICE_INLINE T block_load_table(const MPTORCH_BLOCK_DEVICE void *data,
                                             const MPTORCH_THREAD BlockOperandT<T> &o,
                                             const MPTORCH_BLOCK_TABLE T *table, int64_t bId,
                                             int64_t data_off, int64_t i, int64_t k)
  {
    const MPTORCH_THREAD mptorch::block::BlockFormatParamsT<T> &p = o.p;
    // The coordinates are 32-bit, widened only where they are multiplied by
    // a stride: the driver keeps a batch element's rows, columns and bytes
    // per row below 2^31.
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
    const T v = MPTORCH_BLOCK_MUL(FloatTraits<T>, table[code], table[BLOCK_TABLE + sc]);
    return (mptorch::block::block_word(v) & FloatTraits<T>::ABS_MASK) == 0 ? T(0) : v;
  }

} // namespace mptorch::gemm
