// Every binary32 word through the vectorized RNE tiles of cpu/gemm_rne_tile.cpp,
// against the scalar step the CPU kernel runs in their place.
//
// The tiles are the shipped file, compiled in here, so this checks the code
// the extension runs: the compiler's vectorization of cast_binaryK_rne_fast,
// lane by lane, on every input. Each word goes through two tiles:
//
//   fused   a = 0, b = -1, sum = x:  cast(fma(0, -1, x)) = cast(x) exactly
//   split   a = 1, b = x, sum = 0:   cast(0 + cast(1 * x))
//
// and each result must be the scalar expression's word for word, on the
// scalar kernel's own cast (cast_binaryK_nearest_even, which takes the fast
// path for every format here; the program says so if one does not). A block
// whose tile ends with a NaN is handed back (gemm_rne_tile.h): it must then be
// as it was, and each of its lanes is run alone, to match the scalar step or
// be handed back with a NaN the scalar step makes too. Built with the
// extension's host flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_rne_tile_sweep.cpp \
//       mptorch/csrc/cpu/gemm_rne_tile.cpp -o /tmp/cpu_rne_tile_sweep
//   /tmp/cpu_rne_tile_sweep                         # the widest ISA the CPU has
//   MPTORCH_CPU_SIMD=avx2 /tmp/cpu_rne_tile_sweep   # the AVX2 tiles on an AVX-512 CPU
//
// About 5 s per format on 16 threads. MPTORCH_CPU_SIMD=0 leaves the tiles
// null, and the program then says it has nothing to check.

#include "cpu/gemm_rne_tile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using mptorch::gemm_cpu::rne_tiles;

namespace
{
  struct Format
  {
    int man_bits, exp_bits, bias;
    SaturationMode sat;
    const char *name;
  };

  float bits_float(uint32_t w)
  {
    float f;
    std::memcpy(&f, &w, 4);
    return f;
  }
  uint32_t float_bits(float f)
  {
    uint32_t w;
    std::memcpy(&w, &f, 4);
    return w;
  }

  float want_fused(float x, const BinaryKParams &p)
  {
    return cast_binaryK_nearest_even(std::fma(0.0f, -1.0f, x), true, SubnormalsMode::SUBNORMALS, p);
  }
  float want_split(float x, const BinaryKParams &p)
  {
    return cast_binaryK_nearest_even(
        0.0f + cast_binaryK_nearest_even(1.0f * x, true, SubnormalsMode::SUBNORMALS, p), true,
        SubnormalsMode::SUBNORMALS, p);
  }

  // One tile call's results against the scalar step, `in` its sums before
  // and `out` after. A call that hands its block back must leave the sums as
  // they were; its lanes are then run one at a time, and a lane handed back
  // alone must be one the scalar step makes a NaN of (it is the scalar
  // loop's to run).
  template <class Call, class Want>
  uint64_t check(bool done, const float *in, const float *out, int64_t n, Call one_lane, Want want)
  {
    uint64_t bad = 0;
    if (done)
    {
      for (int64_t l = 0; l < n; ++l)
        bad += float_bits(out[l]) != float_bits(want(in[l]));
      return bad;
    }
    for (int64_t l = 0; l < n; ++l)
      bad += float_bits(out[l]) != float_bits(in[l]);
    for (int64_t l = 0; l < n; ++l)
    {
      float r;
      if (one_lane(in[l], r))
        bad += float_bits(r) != float_bits(want(in[l]));
      else
        bad += float_bits(r) != float_bits(in[l]) || !std::isnan(want(in[l]));
    }
    return bad;
  }

  // Mismatches over every word for one format, through both tiles.
  uint64_t sweep(const BinaryKParams &p)
  {
    constexpr int64_t TI = 32, TJ = 32, LANES = TI * TJ;
    uint64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
    for (int64_t chunk = 0; chunk < (int64_t(1) << 32) / LANES; ++chunk)
    {
      alignas(64) float x[LANES], sums[LANES];
      alignas(64) float zero_a[TI], one_a[TI], minus_one_b[TJ];
      for (int i = 0; i < TI; ++i)
        zero_a[i] = 0.0f, one_a[i] = 1.0f;
      for (int j = 0; j < TJ; ++j)
        minus_one_b[j] = -1.0f;
      for (int64_t l = 0; l < LANES; ++l)
        x[l] = sums[l] = bits_float(static_cast<uint32_t>(chunk * LANES + l));
      const float zero = 0.0f, one = 1.0f, minus_one = -1.0f;

      // fused: one K-step of a = 0, b = -1 over sum = x
      const bool fused = rne_tiles.fused(zero_a, minus_one_b, sums, TI, 1, TJ, p, p);
      bad += check(fused, x, sums, LANES,
                   [&](float v, float &r)
                   {
                     r = v;
                     return rne_tiles.fused(&zero, &minus_one, &r, 1, 1, 1, p, p);
                   },
                   [&](float v) { return want_fused(v, p); });

      // split: one K-step of a = 1 against b = a row of words, sum = 0
      for (int64_t i = 0; i < TI; ++i)
      {
        alignas(64) float out[TJ] = {};
        const bool split = rne_tiles.split(one_a + i, x + i * TJ, out, 1, 1, TJ, p, p);
        for (int64_t j = 0; j < TJ; ++j)
        {
          const float v = x[i * TJ + j];
          if (split)
            bad += float_bits(out[j]) != float_bits(want_split(v, p));
          else
          {
            bad += float_bits(out[j]) != 0u; // handed back: the zeros as they were
            float r = 0.0f;
            if (rne_tiles.split(&one, &v, &r, 1, 1, 1, p, p))
              bad += float_bits(r) != float_bits(want_split(v, p));
            else
              bad += float_bits(r) != 0u || !std::isnan(want_split(v, p));
          }
        }
      }
    }
    return bad;
  }
} // namespace

int main()
{
  if (!rne_tiles.split)
  {
    std::printf("no vectorized tile on this CPU or build (isa %s): nothing to check\n",
                rne_tiles.isa);
    return 0;
  }
  const Format formats[] = {
      {3, 4, 7, SaturationMode::SAT_NONE, "e4m3-like, SAT_NONE"},
      {3, 4, 8, SaturationMode::SAT_FINITE, "binary8p4, SAT_FINITE"},
      {10, 5, 15, SaturationMode::SAT_NONE, "binary16"},
      {10, 5, 16, SaturationMode::SAT_FINITE_EXTENDED, "binary16p11, SAT_FINITE_EXTENDED"},
      {7, 7, 63, SaturationMode::SAT_NONE, "P = 8, 7 exponent bits"},
      {1, 5, 15, SaturationMode::SAT_NONE, "P = 2"},
      {2, 3, 4, SaturationMode::SAT_NONE, "binary6p3 (overflows on N(0, 1) products)"},
      {22, 7, 63, SaturationMode::SAT_FINITE, "P = 23, 7 exponent bits"},
      {5, 6, 31, SaturationMode::SAT_NONE, "P = 6, 6 exponent bits"},
  };
  std::printf("isa %s, every binary32 word through the fused and split tiles\n", rne_tiles.isa);
  uint64_t total = 0;
  int checked = 0;
  for (const Format &f : formats)
  {
    const BinaryKParams p =
        make_binaryK_params<float>(f.man_bits, f.exp_bits, f.bias, true, f.sat, false);
    if (!p.fast_rne)
    {
      std::printf("  %-36s the fast gate is off: not the tile's case, skipped\n", f.name);
      continue;
    }
    const uint64_t bad = sweep(p);
    std::printf("  %-36s %llu mismatches\n", f.name, (unsigned long long)bad);
    total += bad;
    ++checked;
  }
  std::printf("%s: %d formats checked, %llu mismatches\n", total ? "FAIL" : "PASS", checked,
              (unsigned long long)total);
  return total ? 1 : 0;
}
