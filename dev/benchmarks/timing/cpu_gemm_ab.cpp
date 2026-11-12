// The shipped CPU GEMM kernel, timed outside torch, so a change to the
// kernel or to the casts it inlines can be weighed against the build before
// it without rebuilding the extension (written with the vectorized RNE tiles
// of cpu/gemm_rne_tile.cpp, eafa9aa; the CPU twin of gemm_sr_ab.cu).
//
// It calls CpuBackend::launch_as itself, which instantiates the kernel from
// whatever headers -I names. So one source, compiled against two trees, is an
// A/B of those headers and nothing else: a tree from
// `git archive <commit> mptorch/csrc | tar -x -C DIR`, and the working tree.
// Compile with the extension's own host flags (build/temp.*/build.ninja's
// cflags and post_cflags), or the comparison is of something else:
//
//   T=.venv/lib/python3.13/site-packages/torch
//   for tree in DIR/mptorch/csrc mptorch/csrc; do
//     g++ -O3 -std=c++20 -fopenmp -fPIC -DNDEBUG -fno-omit-frame-pointer \
//         -mno-omit-leaf-frame-pointer --param inline-unit-growth=400 \
//         -ffp-contract=off -DMPTORCH_FAST_CAST=1 -I $tree -I $T/include \
//         -I $T/include/torch/csrc/api/include dev/benchmarks/timing/cpu_gemm_ab.cpp \
//         $(ls $tree/cpu/gemm_rne_tile.cpp 2>/dev/null) \
//         -L $T/lib -ltorch_cpu -lc10 -Wl,-rpath,$T/lib -o ...; done
//
// A tree with the vectorized RNE tiles (cpu/gemm_rne_tile.cpp, since
// eafa9aa, linked in as above) also takes MPTORCH_CPU_SIMD=0, which keeps the
// scalar loop: one binary run both ways is the tiles' A/B.
//
// Run each with `--threads N --reps R [--n SIZE] [--only SUBSTRING]`,
// alternating the binaries; --only runs the cases whose name contains it (for
// `perf stat` on one of them).
// Each op is timed at SIZE^3 (min over the reps) and its output's hash is
// printed, so two variants meant to compute alike can be compared too. The
// formats are sr_variants.py's: binaryK split Binary8p4 x Binary16p11 and
// Binary16p11 x Binary16p11, fused Binary16p11, SAT_NONE and SUBNORMALS, on
// N(0, 1) inputs, in RNE, RU and SR (8 random bits).

#include "cpu/custom_matmul_kernel.h"

#include <ATen/Parallel.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace mptorch::gemm;
using mptorch::gemm_cpu::CpuBackend;

namespace
{
  const char *only = nullptr; // --only SUBSTRING: run the cases whose name has it

  template <class Args>
  void run(const char *name, GemmShape s, const Args &args, int reps)
  {
    if (only && !std::strstr(name, only))
      return;
    CpuBackend::LaunchContext ctx;
    ctx.seed = s.use_rng ? 1234 : 0;
    CpuBackend::launch_as<float, Args>(s, args, ctx); // warm
    double best = 1e30;
    for (int i = 0; i < reps; ++i)
    {
      auto t0 = std::chrono::steady_clock::now();
      CpuBackend::launch_as<float, Args>(s, args, ctx);
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
    }
    const uint32_t *out = static_cast<const uint32_t *>(s.c);
    uint64_t h = 1469598103934665603ull;
    for (int64_t i = 0; i < s.M * s.N; ++i)
      h = (h ^ out[i]) * 1099511628211ull;
    const double steps = double(s.M) * double(s.K) * double(s.N);
    std::printf("%-30s %9.3f ms  %7.3f ns/mac  hash %016llx\n", name, best * 1e3,
                best * 1e9 / steps, (unsigned long long)h);
  }

  BinaryKSplitArgs split(BinaryKWidths mul, BinaryKWidths acc, int prng_bits)
  {
    BinaryKSplitArgs a;
    a.mul = mul;
    a.mul_c = {true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, prng_bits};
    a.accumulate_quant = true;
    a.acc = acc;
    a.acc_c = {true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, prng_bits};
    return a;
  }
} // namespace

int main(int argc, char **argv)
{
  int64_t n = 256;
  int reps = 5, threads = 1;
  for (int i = 1; i + 1 < argc; ++i)
  {
    if (!std::strcmp(argv[i], "--reps"))
      reps = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--threads"))
      threads = std::atoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--n"))
      n = std::atoll(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--only"))
      only = argv[i + 1];
  }
  at::set_num_threads(threads);

  // N(0,1) by Box-Muller on a fixed LCG (gemm_sr_ab.cu's)
  std::vector<float> a(static_cast<size_t>(n * n)), b(a.size()), c(a.size());
  uint64_t st = 0x9E3779B97F4A7C15ull;
  auto u01 = [&]
  {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    return ((st >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  };
  for (auto *v : {&a, &b})
    for (auto &x : *v)
      x = static_cast<float>(std::sqrt(-2.0 * std::log(u01())) * std::cos(6.283185307179586 * u01()));

  GemmShape s;
  s.a = a.data();
  s.b = b.data();
  s.c = c.data();
  s.dt = mptorch::GemmDtype::Float;
  s.M = s.K = s.N = n;
  std::printf("%lld^3, %d thread(s), min of %d\n", (long long)n, threads, reps);

  const BinaryKWidths b8p4{3, 4, 8}, b16p11{10, 5, 16};
  for (RoundMode rm : {RoundMode::RNE, RoundMode::RU, RoundMode::SR})
  {
    const bool sr = rm == RoundMode::SR;
    const int pb = sr ? 8 : 0;
    s.rm = rm;
    s.use_rng = sr;
    const char *mode = rm == RoundMode::RNE ? "RNE" : rm == RoundMode::RU ? "RU" : "SR";
    char name[64];
    std::snprintf(name, sizeof name, "split 8p4/16p11 %s", mode);
    run(name, s, split(b8p4, b16p11, pb), reps);
    std::snprintf(name, sizeof name, "split 16p11/16p11 %s", mode);
    run(name, s, split(b16p11, b16p11, pb), reps);
    BinaryKSplitArgs mul_only = split(b16p11, b16p11, pb);
    mul_only.accumulate_quant = false;
    std::snprintf(name, sizeof name, "split 16p11/carrier %s", mode);
    run(name, s, mul_only, reps);
    BinaryKFusedArgs f;
    f.fma_quant = true;
    f.fma = b16p11;
    f.fma_c = {true, SaturationMode::SAT_NONE, SubnormalsMode::SUBNORMALS, pb};
    std::snprintf(name, sizeof name, "fused 16p11 %s", mode);
    run(name, s, f, reps);
  }

  return 0;
}
