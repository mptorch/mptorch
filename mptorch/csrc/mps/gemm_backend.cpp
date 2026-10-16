#include "gemm_backend.h"
#include "../cpu/utils.h" // draw_cpu_seed
#include "launch_params.h"
#include "metal_runtime.h"
#include <cstddef>
#include <limits>
#include <vector>

namespace mptorch::gemm_mps
{
  using namespace mptorch::gemm;

  namespace
  {

    // Writes `mpt_args()` for gemm.metal: the Args struct value-initialized,
    // then every field the schema set, by name, so that the Metal compiler
    // checks each name against the struct it is compiled from. What cannot
    // be checked there is a field this file does not know about, which would
    // stay zero on the device; the size assertions in the args_source
    // functions below catch the struct growing one.
    class ArgsWriter
    {
    public:
      explicit ArgsWriter(const char *type)
          : type_(type)
      {
        out_ = "inline mptorch::gemm::" + type_ + " mpt_args()\n{\n    mptorch::gemm::" + type_ +
               " a{};\n";
      }

      void set(const std::string &field, int64_t v) { line(field, std::to_string(v)); }
      void set(const std::string &field, bool v) { line(field, v ? "true" : "false"); }
      void set(const std::string &field, SaturationMode v)
      {
        line(field, "SaturationMode(" + std::to_string(static_cast<int>(v)) + ")");
      }
      void set(const std::string &field, SubnormalsMode v)
      {
        line(field, "SubnormalsMode(" + std::to_string(static_cast<int>(v)) + ")");
      }
      void set(const std::string &field, const BinaryKWidths &w)
      {
        set(field + ".man_bits", int64_t{w.man_bits});
        set(field + ".exp_bits", int64_t{w.exp_bits});
        set(field + ".bias", int64_t{w.bias});
      }
      void set(const std::string &field, const SuperfpWidths &w)
      {
        set(field + ".man_bits", int64_t{w.man_bits});
        set(field + ".exp_bits", int64_t{w.exp_bits});
        set(field + ".normal_binades", int64_t{w.normal_binades});
        set(field + ".bias", int64_t{w.bias});
      }
      void set(const std::string &field, const BinaryKCommon &c)
      {
        set(field + ".is_signed", c.is_signed);
        set(field + ".sat", c.sat);
        set(field + ".sub", c.sub);
        set(field + ".prng_bits", int64_t{c.prng_bits});
      }
      void set(const std::string &field, const SuperfpCommon &c)
      {
        set(field + ".is_signed", c.is_signed);
        set(field + ".sat", c.sat);
        set(field + ".prng_bits", int64_t{c.prng_bits});
      }
      // A palette's per-slot list: only the n_fmt slots the op fills.
      template <class W>
      void set_slots(const std::string &field, const W (&slots)[MAX_GEMM_FORMATS], int n_fmt)
      {
        for (int i = 0; i < n_fmt; ++i)
          set(field + "[" + std::to_string(i) + "]", slots[i]);
      }

      void set(const std::string &field, AccumulateAlgorithm v)
      {
        line(field, "AccumulateAlgorithm(" + std::to_string(static_cast<int>(v)) + ")");
      }

      std::string str() const { return out_ + "    return a;\n}\n"; }

    private:
      void line(const std::string &field, const std::string &value)
      {
        out_ += "    a." + field + " = " + value + ";\n";
      }

      std::string type_;
      std::string out_;
    };

    // The component structs, whose every field the writer above sets.
    static_assert(sizeof(BinaryKWidths) == 3 * sizeof(int), "set() every new BinaryKWidths field");
    static_assert(sizeof(SuperfpWidths) == 4 * sizeof(int), "set() every new SuperfpWidths field");
    static_assert(sizeof(BinaryKCommon) == 16, "set() every new BinaryKCommon field");
    static_assert(sizeof(SuperfpCommon) == 12, "set() every new SuperfpCommon field");

    // The fields of the single-format Args, under `at` ("" for the Args
    // itself, "base." inside an AccumulateArgs). The size assertions are each
    // struct's size with the fields written here; a new field changes it and
    // stops the build here.
    void write(ArgsWriter &w, const std::string &at, const BinaryKSplitArgs &a)
    {
      static_assert(sizeof(BinaryKSplitArgs) == 60, "write every BinaryKSplitArgs field");
      w.set(at + "mul", a.mul);
      w.set(at + "mul_c", a.mul_c);
      w.set(at + "accumulate_quant", a.accumulate_quant);
      w.set(at + "acc", a.acc);
      w.set(at + "acc_c", a.acc_c);
    }

    void write(ArgsWriter &w, const std::string &at, const BinaryKFusedArgs &a)
    {
      static_assert(sizeof(BinaryKFusedArgs) == 32, "write every BinaryKFusedArgs field");
      w.set(at + "fma_quant", a.fma_quant);
      w.set(at + "fma", a.fma);
      w.set(at + "fma_c", a.fma_c);
    }

    void write(ArgsWriter &w, const std::string &at, const SuperfpSplitArgs &a)
    {
      static_assert(sizeof(SuperfpSplitArgs) == 60, "write every SuperfpSplitArgs field");
      w.set(at + "mul", a.mul);
      w.set(at + "mul_c", a.mul_c);
      w.set(at + "accumulate_quant", a.accumulate_quant);
      w.set(at + "acc", a.acc);
      w.set(at + "acc_c", a.acc_c);
    }

    void write(ArgsWriter &w, const std::string &at, const SuperfpFusedArgs &a)
    {
      static_assert(sizeof(SuperfpFusedArgs) == 32, "write every SuperfpFusedArgs field");
      w.set(at + "fma_quant", a.fma_quant);
      w.set(at + "fma", a.fma);
      w.set(at + "fma_c", a.fma_c);
    }

    // The cross-family split macs (common/gemm_cross.h): their accumulate
    // format is always there, so there is no flag.
    template <class MulW, class MulC, class AccW, class AccC>
    void write(ArgsWriter &w, const std::string &at, const CrossSplitArgs<MulW, MulC, AccW, AccC> &a)
    {
      static_assert(sizeof(BinaryKSuperfpArgs) == 56 && sizeof(SuperfpBinaryKArgs) == 56,
                    "write every CrossSplitArgs field");
      w.set(at + "mul", a.mul);
      w.set(at + "mul_c", a.mul_c);
      w.set(at + "acc", a.acc);
      w.set(at + "acc_c", a.acc_c);
    }

    // The block GEMM's split mac (common/gemm_block.h): the carrier's
    // product, and the sum's format.
    void write(ArgsWriter &w, const std::string &at, const BlockSplitArgs &a)
    {
      static_assert(sizeof(BlockSplitArgs) == 32, "write every BlockSplitArgs field");
      w.set(at + "accumulate_quant", a.accumulate_quant);
      w.set(at + "acc", a.acc);
      w.set(at + "acc_c", a.acc_c);
    }

    // An AccumulateArgs: its base's fields, then the accumulation's.
    template <class Base>
    std::string accumulated_source(const char *base, const AccumulateArgs<Base> &a)
    {
      static_assert(sizeof(AccumulateArgs<Base>) ==
                        sizeof(Base) + 3 * sizeof(int) + sizeof(typename AccumulateArgs<Base>::Widths) +
                            sizeof(typename AccumulateArgs<Base>::Common),
                    "write every AccumulateArgs field");
      ArgsWriter w((std::string("AccumulateArgs<mptorch::gemm::") + base + ">").c_str());
      write(w, "base.", a.base);
      w.set("alg", a.alg);
      w.set("block_size", int64_t{a.block_size});
      w.set("outer_quant", a.outer_quant);
      // An outer format that is off arrives as zero widths, which the
      // device's Builder never derives a cast from (gemm_accumulate.h).
      w.set("outer", a.outer);
      w.set("outer_c", a.outer_c);
      return w.str();
    }

  } // namespace

  std::string MpsBackend::args_source(const BinaryKSplitArgs &a)
  {
    ArgsWriter w("BinaryKSplitArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const BinaryKSplitMixedArgs &a)
  {
    static_assert(sizeof(BinaryKSplitMixedArgs) == 232, "write every BinaryKSplitMixedArgs field");
    ArgsWriter w("BinaryKSplitMixedArgs");
    w.set("n_fmt", int64_t{a.n_fmt});
    w.set_slots("mul", a.mul, a.n_fmt);
    w.set("mul_c", a.mul_c);
    w.set("accumulate_quant", a.accumulate_quant);
    w.set_slots("acc", a.acc, a.n_fmt);
    w.set("acc_c", a.acc_c);
    return w.str();
  }

  std::string MpsBackend::args_source(const BinaryKFusedArgs &a)
  {
    ArgsWriter w("BinaryKFusedArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const BinaryKFusedMixedArgs &a)
  {
    static_assert(sizeof(BinaryKFusedMixedArgs) == 116, "write every BinaryKFusedMixedArgs field");
    ArgsWriter w("BinaryKFusedMixedArgs");
    w.set("n_fmt", int64_t{a.n_fmt});
    w.set_slots("fma", a.fma, a.n_fmt);
    w.set("fma_c", a.fma_c);
    return w.str();
  }

  std::string MpsBackend::args_source(const SuperfpSplitArgs &a)
  {
    ArgsWriter w("SuperfpSplitArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const SuperfpSplitMixedArgs &a)
  {
    static_assert(sizeof(SuperfpSplitMixedArgs) == 288, "write every SuperfpSplitMixedArgs field");
    ArgsWriter w("SuperfpSplitMixedArgs");
    w.set("n_fmt", int64_t{a.n_fmt});
    w.set_slots("mul", a.mul, a.n_fmt);
    w.set("mul_c", a.mul_c);
    w.set("accumulate_quant", a.accumulate_quant);
    w.set_slots("acc", a.acc, a.n_fmt);
    w.set("acc_c", a.acc_c);
    return w.str();
  }

  std::string MpsBackend::args_source(const SuperfpFusedArgs &a)
  {
    ArgsWriter w("SuperfpFusedArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const SuperfpFusedMixedArgs &a)
  {
    static_assert(sizeof(SuperfpFusedMixedArgs) == 144, "write every SuperfpFusedMixedArgs field");
    ArgsWriter w("SuperfpFusedMixedArgs");
    w.set("n_fmt", int64_t{a.n_fmt});
    w.set_slots("fma", a.fma, a.n_fmt);
    w.set("fma_c", a.fma_c);
    return w.str();
  }

  std::string MpsBackend::args_source(const BinaryKSuperfpArgs &a)
  {
    ArgsWriter w("BinaryKSuperfpArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const SuperfpBinaryKArgs &a)
  {
    ArgsWriter w("SuperfpBinaryKArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const BlockSplitArgs &a)
  {
    ArgsWriter w("BlockSplitArgs");
    write(w, "", a);
    return w.str();
  }

  std::string MpsBackend::args_source(const AccumulateArgs<BinaryKSplitArgs> &a)
  {
    return accumulated_source("BinaryKSplitArgs", a);
  }

  std::string MpsBackend::args_source(const AccumulateArgs<BinaryKFusedArgs> &a)
  {
    return accumulated_source("BinaryKFusedArgs", a);
  }

  std::string MpsBackend::args_source(const AccumulateArgs<SuperfpSplitArgs> &a)
  {
    return accumulated_source("SuperfpSplitArgs", a);
  }

  std::string MpsBackend::args_source(const AccumulateArgs<SuperfpFusedArgs> &a)
  {
    return accumulated_source("SuperfpFusedArgs", a);
  }

  std::string MpsBackend::args_source(const AccumulateArgs<BinaryKSuperfpArgs> &a)
  {
    return accumulated_source("BinaryKSuperfpArgs", a);
  }

  std::string MpsBackend::args_source(const AccumulateArgs<SuperfpBinaryKArgs> &a)
  {
    return accumulated_source("SuperfpBinaryKArgs", a);
  }

  std::string MpsBackend::args_source(const AccumulateArgs<BlockSplitArgs> &a)
  {
    return accumulated_source("BlockSplitArgs", a);
  }

  std::string MpsBackend::accumulate_line(AccumulateAlgorithm alg)
  {
    static const char *const names[] = {"NAIVE", "KAHAN", "BLOCK", "TREE"};
    const int i = static_cast<int>(alg);
    TORCH_INTERNAL_ASSERT(i >= 1 && i <= 3, "mptorch: an AccumulateArgs is never a NAIVE reduction");
    return std::string("#define MPT_ACCUMULATE AccumulateAlgorithm::") + names[i] + "\n";
  }

  MpsBackend::LaunchContext MpsBackend::make_context(bool use_rng, uint64_t /*draws_per_thread*/)
  {
    LaunchContext ctx;
    ctx.seed = use_rng ? draw_cpu_seed() : 0;
    return ctx;
  }

  namespace
  {
    // The kernel counts in 32 bits, and a tile's indices run up to
    // GEMM_TILE past the edge of C and of K.
    void check_extents(const GemmShape &s)
    {
      constexpr int64_t most = std::numeric_limits<uint32_t>::max();
      constexpr int64_t most_mkn = most - mptorch_mps::GEMM_TILE;
      TORCH_CHECK(s.M <= most_mkn && s.K <= most_mkn && s.N <= most_mkn && s.batch <= most,
                  "mptorch: an MPS GEMM's M, K and N must each be at most 2^32 - 17, and its batch "
                  "at most 2^32 - 1, got ",
                  s.M, ", ", s.K, ", ", s.N, " and ", s.batch);
    }

    // The tail's first lines: C's dtype (the operands' too, except for the
    // block GEMM, whose operands are codes) and the rounding mode.
    std::string tail_head(const GemmShape &s, const at::Tensor &c)
    {
      return std::string("using mpt_storage_t = ") +
             mptorch_mps::metal_storage_type(c.scalar_type(), "mptorch GEMM") +
             ";\n#define MPT_ROUND_MODE " + mptorch_mps::metal_round_mode(static_cast<int64_t>(s.rm)) +
             "\n";
    }

    mptorch_mps::GemmLaunch launch_of(const GemmShape &s, uint64_t seed)
    {
      mptorch_mps::GemmLaunch g{};
      g.M = static_cast<uint32_t>(s.M);
      g.K = static_cast<uint32_t>(s.K);
      g.N = static_cast<uint32_t>(s.N);
      g.batch = static_cast<uint32_t>(s.batch);
      g.trans_a = s.trans_a;
      g.trans_b = s.trans_b;
      g.use_rng = s.use_rng;
      g.stride_a = static_cast<uint64_t>(s.stride_a);
      g.stride_b = static_cast<uint64_t>(s.stride_b);
      g.seed = seed;
      g.idx_row_stride = static_cast<uint64_t>(s.idx_row_stride);
      g.idx_col_stride = static_cast<uint64_t>(s.idx_col_stride);
      g.idx_batch_stride = static_cast<uint64_t>(s.idx_batch_stride);
      return g;
    }

    // One launch over (ceil(N / tile), ceil(M / tile), batch) threadgroups:
    // the operands, C, the launch parameters and the precision map at
    // buffers 0 to 4 (gemm.metal binds a map on every op and reads it on the
    // mixed ones only; the others bind C there, which is never read), and
    // the variant's buffers after them.
    void launch_gemm(const GemmShape &s, const std::string &tail, const MpsBackend::LaunchContext &ctx,
                     std::vector<mptorch_mps::KernelArg> extra)
    {
      TORCH_INTERNAL_ASSERT(ctx.a && ctx.b && ctx.c, "mptorch: the MPS GEMM was not bound its tensors");
      check_extents(s);
      const mptorch_mps::GemmLaunch g = launch_of(s, ctx.seed);
      const at::Tensor &idx = ctx.prec_idx ? *ctx.prec_idx : *ctx.c;
      std::vector<mptorch_mps::KernelArg> args{*ctx.a, *ctx.b, *ctx.c, mptorch_mps::KernelArg::params(g), idx};
      args.insert(args.end(), extra.begin(), extra.end());
      constexpr uint64_t tile = mptorch_mps::GEMM_TILE;
      mptorch_mps::launch_groups(mptorch_mps::KernelSource::Gemm, tail_head(s, *ctx.c) + tail, args,
                                 (static_cast<uint64_t>(s.N) + tile - 1) / tile,
                                 (static_cast<uint64_t>(s.M) + tile - 1) / tile,
                                 static_cast<uint64_t>(s.batch), tile, tile);
    }

    // The host's ConvGeom and BlockFormatParams are bound byte for byte, and
    // gemm.metal asserts the same layout.
    static_assert(sizeof(ConvGeom) == 336 && offsetof(ConvGeom, IN) == 96 && offsetof(ConvGeom, res_N) == 328,
                  "ConvGeom must have one layout on the host and the GPU (gemm.metal)");
    static_assert(sizeof(mptorch::block::BlockFormatParams) == 184 &&
                      offsetof(mptorch::block::BlockFormatParams, tensor_scale) == 104 &&
                      offsetof(mptorch::block::BlockFormatParams, trans) == 176,
                  "BlockFormatParams must have one layout on the host and the GPU (gemm.metal)");

    const char *conv_pass_name(ConvPass p)
    {
      switch (p)
      {
      case ConvPass::IGRAD:
        return "IGRAD";
      case ConvPass::WGRAD:
        return "WGRAD";
      default:
        return "FWD";
      }
    }

  } // namespace

  void MpsBackend::launch_source(const GemmShape &s, const std::string &args, const LaunchContext &ctx)
  {
    launch_gemm(s, args, ctx, {});
  }

  void MpsBackend::launch_conv(const GemmShape &s, const std::string &args, const ConvGeom &geom,
                               const LaunchContext &ctx)
  {
    const std::string tail = args + "#define MPT_GATHERED 1\n#define MPT_CONV_PASS mptorch::gemm::ConvPass::" +
                             conv_pass_name(geom.pass) + "\n";
    launch_gemm(s, tail, ctx, {mptorch_mps::KernelArg::params(geom)});
  }

  void MpsBackend::launch_block(const GemmShape &s, const std::string &args, const BlockOperand &a,
                                const BlockOperand &b, const LaunchContext &ctx)
  {
    TORCH_INTERNAL_ASSERT(ctx.scales_a && ctx.scales_b, "mptorch: the MPS block GEMM was not bound its scales");
    const mptorch_mps::BlockGemmLaunch bl{a.scale_stride, b.scale_stride};
    launch_gemm(s, args + "#define MPT_BLOCKED 1\n", ctx,
                {mptorch_mps::KernelArg::params(a.p), mptorch_mps::KernelArg::params(b.p), *ctx.scales_a,
                 *ctx.scales_b, mptorch_mps::KernelArg::params(bl)});
  }

} // namespace mptorch::gemm_mps
