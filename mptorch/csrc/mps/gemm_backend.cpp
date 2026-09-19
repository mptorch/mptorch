#include "gemm_backend.h"
#include "../cpu/utils.h" // draw_cpu_seed
#include "launch_params.h"
#include "metal_runtime.h"
#include <limits>

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

  } // namespace

  // One per Args type. The size assertions are each struct's size with the
  // fields written here; a new field changes it and stops the build here.

  std::string MpsBackend::args_source(const BinaryKSplitArgs &a)
  {
    static_assert(sizeof(BinaryKSplitArgs) == 60, "write every BinaryKSplitArgs field");
    ArgsWriter w("BinaryKSplitArgs");
    w.set("mul", a.mul);
    w.set("mul_c", a.mul_c);
    w.set("accumulate_quant", a.accumulate_quant);
    w.set("acc", a.acc);
    w.set("acc_c", a.acc_c);
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
    static_assert(sizeof(BinaryKFusedArgs) == 32, "write every BinaryKFusedArgs field");
    ArgsWriter w("BinaryKFusedArgs");
    w.set("fma_quant", a.fma_quant);
    w.set("fma", a.fma);
    w.set("fma_c", a.fma_c);
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
    static_assert(sizeof(SuperfpSplitArgs) == 60, "write every SuperfpSplitArgs field");
    ArgsWriter w("SuperfpSplitArgs");
    w.set("mul", a.mul);
    w.set("mul_c", a.mul_c);
    w.set("accumulate_quant", a.accumulate_quant);
    w.set("acc", a.acc);
    w.set("acc_c", a.acc_c);
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
    static_assert(sizeof(SuperfpFusedArgs) == 32, "write every SuperfpFusedArgs field");
    ArgsWriter w("SuperfpFusedArgs");
    w.set("fma_quant", a.fma_quant);
    w.set("fma", a.fma);
    w.set("fma_c", a.fma_c);
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

  MpsBackend::LaunchContext MpsBackend::make_context(bool use_rng, uint64_t /*draws_per_thread*/)
  {
    LaunchContext ctx;
    ctx.seed = use_rng ? draw_cpu_seed() : 0;
    return ctx;
  }

  void MpsBackend::launch_source(const GemmShape &s, const std::string &args,
                                 const LaunchContext &ctx)
  {
    TORCH_INTERNAL_ASSERT(ctx.a && ctx.b && ctx.c, "mptorch: the MPS GEMM was not bound its tensors");
    constexpr int64_t most = std::numeric_limits<uint32_t>::max();
    TORCH_CHECK(s.M <= most && s.K <= most && s.N <= most && s.batch <= most,
                "mptorch: an MPS GEMM's M, K, N and batch must each fit in 32 bits, got ", s.M,
                ", ", s.K, ", ", s.N, " and ", s.batch);

    const std::string tail = std::string("using mpt_storage_t = ") +
                             mptorch_mps::metal_storage_type(ctx.a->scalar_type(), "mptorch GEMM") +
                             ";\n#define MPT_ROUND_MODE " +
                             mptorch_mps::metal_round_mode(static_cast<int64_t>(s.rm)) + "\n" + args;

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
    g.seed = ctx.seed;
    g.idx_row_stride = static_cast<uint64_t>(s.idx_row_stride);
    g.idx_col_stride = static_cast<uint64_t>(s.idx_col_stride);
    g.idx_batch_stride = static_cast<uint64_t>(s.idx_batch_stride);

    // gemm.metal binds a precision map at index 4 on every op and reads it
    // on the mixed ones only; the others bind C there, which is never read.
    const at::Tensor &idx = ctx.prec_idx ? *ctx.prec_idx : *ctx.c;
    mptorch_mps::launch(mptorch_mps::KernelSource::Gemm, tail,
                        {*ctx.a, *ctx.b, *ctx.c, mptorch_mps::KernelArg::params(g), idx},
                        static_cast<uint64_t>(s.N), static_cast<uint64_t>(s.M),
                        static_cast<uint64_t>(s.batch));
  }

} // namespace mptorch::gemm_mps
