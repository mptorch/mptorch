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
// be handed back with a NaN the scalar step makes too.
//
// Those calls are one K-step each, which the tiles run with their sizes at
// run time. A whole tile (32 x 32, a K-block of 32) runs a copy with them
// fixed, so a second mode runs whole tiles: a sample of the words (one in 64,
// by an odd multiplier, the NaNs made finite) as the sums entering a K-block
// of 32 steps, its operands moderate values in most blocks, small values,
// zeros and subnormals in some, and spreads over the whole range with
// infinities in others (operand(), below), through all three tiles. Each block runs whole, and again as 32
// one-step calls, and both must be the scalar steps' result word for word, a
// handed-back block or step taking the scalar steps as the kernel's loop
// does; the program prints how many blocks ran whole. Built with the
// extension's host flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_rne_tile_sweep.cpp \
//       mptorch/csrc/cpu/gemm_rne_tile.cpp -o /tmp/cpu_rne_tile_sweep
//   /tmp/cpu_rne_tile_sweep                         # the widest ISA the CPU has
//   MPTORCH_CPU_SIMD=avx2 /tmp/cpu_rne_tile_sweep   # the AVX2 tiles on an AVX-512 CPU
//
// About 5 s per format on 16 threads, and as long again for the whole tiles. MPTORCH_CPU_SIMD=0 leaves the tiles
// null, and the program then says it has nothing to check.

#include "cpu/gemm_rne_tile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
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

  // The operands of a whole tile's K-block, by the block's kind:
  //   0  moderate values, |x| in [2^-6, 2): every product is below 4 and so
  //      finite in every format here, and such a block makes no NaN from a
  //      finite sum and runs whole;
  //   1  small ones, zeros and subnormals: |x| in [2^-80, 1), products that
  //      underflow, and still no NaN;
  //   2  a spread over the whole range, zeros, subnormals, values near the
  //      top and infinities, which is mostly handed back.
  float operand(uint64_t &st, int kind)
  {
    st = st * 6364136223846793005ull + 1442695040888963407ull;
    const uint32_t r = static_cast<uint32_t>(st >> 32);
    const float m = 1.0f + float(r & 0x7FFFFF) * 0x1p-23f;
    const float sign = (r >> 31) ? -1.0f : 1.0f;
    if (kind == 0)
      return sign * std::ldexp(m, int((r >> 23) % 6) - 6);
    switch ((r >> 23) % 16)
    {
    case 0:
      return 0.0f;
    case 1:
      return sign * std::ldexp(1.0f, -149) * float(1 + (r & 7)); // subnormals
    case 2:
      return kind == 1 ? sign * 0x1p-126f : sign * std::numeric_limits<float>::max() * 0.5f;
    case 3:
      return kind == 1 ? -0.0f : sign * std::numeric_limits<float>::infinity();
    default:
      return kind == 1 ? sign * std::ldexp(m, -int((r >> 8) % 80) - 1)
                       : sign * std::ldexp(m, int((r >> 8) % 250) - 125);
    }
  }

  // The scalar kernel's steps, the tile's kind by kind.
  float want_step(int kind, float a, float b, float s, const BinaryKParams &p)
  {
    auto cast = [&](float v) { return cast_binaryK_nearest_even(v, true, SubnormalsMode::SUBNORMALS, p); };
    if (kind == 1)
      return cast(s + cast(a * b));
    if (kind == 2)
      return s + cast(a * b);
    return cast(std::fma(a, b, s));
  }

  // Mismatches over whole tiles for one format, the header's second mode.
  uint64_t whole_tiles(const BinaryKParams &p, uint64_t &ran, uint64_t &blocks)
  {
    constexpr int64_t T = 32, LANES = T * T, SAMPLE = 64;
    const mptorch::gemm_cpu::RneTileFn tiles[] = {rne_tiles.split, rne_tiles.mul_only,
                                                  rne_tiles.fused};
    uint64_t bad = 0, kept = 0;
    const int64_t chunks = (int64_t(1) << 32) / (LANES * SAMPLE);
#pragma omp parallel for reduction(+ : bad, kept) schedule(dynamic, 16)
    for (int64_t chunk = 0; chunk < chunks; ++chunk)
    {
      alignas(64) float a[LANES], b[LANES], x[LANES], whole[LANES], steps[LANES], want[LANES];
      alignas(64) float a_k[T];
      uint64_t st = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(chunk);
      const int kind_of_block = chunk % 8 < 6 ? 0 : chunk % 8 == 6 ? 1 : 2;
      for (int64_t l = 0; l < LANES; ++l)
      {
        a[l] = operand(st, kind_of_block); // a[i * T + k]
        b[l] = operand(st, kind_of_block); // b[k * T + j]
        // a NaN word would hand every block back: the first mode has those
        const uint32_t w = static_cast<uint32_t>((chunk * LANES + l) * 0x9E3779B1u);
        x[l] = bits_float((w & 0x7FFFFFFFu) > 0x7F800000u ? w & ~0x00800000u : w);
      }
      for (int kind = 1; kind <= 3; ++kind)
      {
        const auto tile = tiles[kind - 1];
        // the scalar steps
        std::memcpy(want, x, sizeof want);
        for (int64_t k = 0; k < T; ++k)
          for (int64_t l = 0; l < LANES; ++l)
            want[l] = want_step(kind, a[(l / T) * T + k], b[k * T + l % T], want[l], p);
        // whole, or handed back to the scalar steps
        std::memcpy(whole, x, sizeof whole);
        const bool whole_ran = tile(a, b, whole, T, T, T, p, p);
        kept += whole_ran;
        if (!whole_ran)
        {
          for (int64_t l = 0; l < LANES; ++l)
            bad += float_bits(whole[l]) != float_bits(x[l]);
          std::memcpy(whole, want, sizeof whole);
          bool nan = false;
          for (int64_t l = 0; l < LANES; ++l)
            nan |= std::isnan(want[l]);
          bad += !nan; // handed back with no NaN at the block's end
        }
        // one K-step a call, each handed back to the scalar step
        std::memcpy(steps, x, sizeof steps);
        for (int64_t k = 0; k < T; ++k)
        {
          for (int64_t i = 0; i < T; ++i)
            a_k[i] = a[i * T + k];
          if (!tile(a_k, b + k * T, steps, T, 1, T, p, p))
            for (int64_t l = 0; l < LANES; ++l)
              steps[l] = want_step(kind, a[(l / T) * T + k], b[k * T + l % T], steps[l], p);
        }
        // A NaN's payload is the scalar loop's (a sum of two NaNs returns one
        // by operand order, which the compiler may commute in it too): a
        // block whose sums end with a NaN is handed back, so a block the tile
        // kept has none, and NaN = NaN is all a handed-back one can check.
        auto same = [](float u, float v)
        { return float_bits(u) == float_bits(v) || (std::isnan(u) && std::isnan(v)); };
        for (int64_t l = 0; l < LANES; ++l)
          bad += !same(whole[l], want[l]) + !same(steps[l], want[l]);
      }
    }
    ran = kept;
    blocks = 3 * chunks;
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
  std::printf("isa %s, every binary32 word through the fused and split tiles, and whole tiles\n",
              rne_tiles.isa);
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
    uint64_t ran = 0, blocks = 0;
    const uint64_t bad = sweep(p) + whole_tiles(p, ran, blocks);
    std::printf("  %-36s %llu mismatches (whole tiles: %llu of %llu blocks ran whole)\n", f.name,
                (unsigned long long)bad, (unsigned long long)ran, (unsigned long long)blocks);
    total += bad;
    ++checked;
  }
  std::printf("%s: %d formats checked, %llu mismatches\n", total ? "FAIL" : "PASS", checked,
              (unsigned long long)total);
  return total ? 1 : 0;
}
