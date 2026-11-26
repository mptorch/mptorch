// Every binary32 word through the palette GEMMs' lane tiles of
// cpu/gemm_palette_lanes.cpp, each lane on a format of its own, against the
// scalar step the CPU kernel runs in their place.
//
// The lane tiles spell cast_binaryK_rne_fast with each lane's constants read
// from LaneParams (cast_lane), so this checks that spelling and the
// compiler's vectorization of it on every input, as cpu_rne_tile_sweep.cpp
// checks the single-format tiles. The formats are that sweep's admitted ones,
// and lane l of a chunk takes format (l + r) % F for each rotation r < F, so
// every word meets every format, and neighbouring lanes always differ. Each
// word goes through two one-step tiles, per element and per column:
//
//   fused   a = 0, b = -1, sum = x:  cast_l(fma(0, -1, x)) = cast_l(x)
//   split   a = 1, b = x, sum = 0:   cast_l(0 + cast_l(1 * x))
//
// and each lane must be the scalar expression's word for word, on lane l's
// own cast (cast_binaryK_nearest_even). A block whose tile ends with a NaN is
// handed back: it must be as it was, and its lanes are run alone, as in
// cpu_rne_tile_sweep.cpp. A second mode runs whole tiles (32 x 32, a K-block
// of 32) of the three macs with a format per element from a palette of three,
// as cpu_rne_tile_sweep.cpp's whole_tiles runs them with one, against the
// scalar steps. `control` hands every lane its neighbour's format, so the
// comparison must fail. Built with the extension's host flags:
//
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc dev/benchmarks/sweeps/cpu_palette_tile_sweep.cpp \
//       mptorch/csrc/cpu/gemm_palette_lanes.cpp mptorch/csrc/cpu/gemm_rne_tile.cpp \
//       -o /tmp/cpu_palette_tile_sweep
//   /tmp/cpu_palette_tile_sweep                         # the widest ISA the CPU has
//   MPTORCH_CPU_SIMD=avx2 /tmp/cpu_palette_tile_sweep   # the AVX2 tiles on an AVX-512 CPU
//   /tmp/cpu_palette_tile_sweep control                 # must report mismatches
//
// About 2 minutes on 16 threads. MPTORCH_CPU_SIMD=0 leaves the tiles null, and
// the program then says it has nothing to check.

#include "cpu/gemm_palette_tile.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using mptorch::gemm_cpu::lane_tiles;
using mptorch::gemm_cpu::LaneParams;

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

  float cast(float v, const BinaryKParams &p)
  {
    return cast_binaryK_nearest_even(v, true, SubnormalsMode::SUBNORMALS, p);
  }

  // The tiles here are 32 x 32.
  constexpr int64_t T = 32, LANES = T * T;

  // A lane of a handed-back block, alone on its format (`p`): its one-lane
  // tile must give `want`, or hand the lane back with a NaN the scalar step
  // makes too (it is the scalar loop's to run).
  uint64_t one_lane(bool split, float v, const BinaryKParams &p, float want)
  {
    static thread_local LaneParams one;
    mptorch::gemm_cpu::set_lane(one, 0, p);
    const float zero = 0.0f, one_f = 1.0f, minus_one = -1.0f;
    float r = split ? 0.0f : v;
    const bool ran = split ? lane_tiles.split(&one_f, &v, &r, 1, 1, 1, false, one, one)
                           : lane_tiles.fused(&zero, &minus_one, &r, 1, 1, 1, false, one, one);
    if (ran)
      return float_bits(r) != float_bits(want);
    return float_bits(r) != float_bits(split ? 0.0f : v) || !std::isnan(want);
  }

  // Mismatches over every word for every format, through both one-step
  // tiles, per element and per column. `shift` 1 is the control: the
  // expectation takes lane l's neighbour's format.
  uint64_t sweep(const std::vector<BinaryKParams> &ps, int shift)
  {
    const int64_t F = static_cast<int64_t>(ps.size());
    uint64_t bad = 0;
#pragma omp parallel for reduction(+ : bad) schedule(dynamic, 64)
    for (int64_t chunk = 0; chunk < (int64_t(1) << 32) / LANES; ++chunk)
    {
      alignas(64) float x[LANES], sums[LANES];
      alignas(64) float zero_a[T], one_a[T], minus_one_b[T];
      for (int i = 0; i < T; ++i)
        zero_a[i] = 0.0f, one_a[i] = 1.0f, minus_one_b[i] = -1.0f;
      for (int64_t l = 0; l < LANES; ++l)
        x[l] = bits_float(static_cast<uint32_t>(chunk * LANES + l));
      static thread_local LaneParams lp;
      for (int64_t r = 0; r < F; ++r)
      {
        auto fmt = [&](int64_t lane) { return (lane + r) % F; };
        auto want_of = [&](int64_t lane) { return ps[(fmt(lane) + shift) % F]; };
        for (int cols = 0; cols < 2; ++cols)
        {
          // per element: lane l on format fmt(l); per column: column j on fmt(j)
          for (int64_t l = 0; l < (cols ? T : LANES); ++l)
            mptorch::gemm_cpu::set_lane(lp, l, ps[fmt(l)]);
          auto lane_of = [&](int64_t l) { return cols ? l % T : l; };

          // fused: one K-step of a = 0, b = -1 over sum = x
          std::memcpy(sums, x, sizeof sums);
          const bool fused = lane_tiles.fused(zero_a, minus_one_b, sums, T, 1, T, cols, lp, lp);
          for (int64_t l = 0; l < LANES; ++l)
          {
            const BinaryKParams &p = want_of(lane_of(l));
            if (fused)
              bad += float_bits(sums[l]) != float_bits(cast(x[l], p));
            else
              bad += (float_bits(sums[l]) != float_bits(x[l])) +
                     one_lane(false, x[l], ps[fmt(lane_of(l))], cast(x[l], p));
          }

          // split: one K-step of a = 1 against b = a row of words, sum = 0
          for (int64_t i = 0; i < T; ++i)
          {
            alignas(64) float out[T] = {};
            // a row's lanes: by element, row i's; by column, the columns'
            static thread_local LaneParams row;
            for (int64_t j = 0; j < T; ++j)
              mptorch::gemm_cpu::set_lane(row, j, ps[fmt(cols ? j : i * T + j)]);
            const bool split = lane_tiles.split(one_a + i, x + i * T, out, 1, 1, T, false, row, row);
            for (int64_t j = 0; j < T; ++j)
            {
              const float v = x[i * T + j];
              const BinaryKParams &p = want_of(cols ? j : i * T + j);
              const float want = cast(0.0f + cast(1.0f * v, p), p);
              if (split)
                bad += float_bits(out[j]) != float_bits(want);
              else
                bad += (float_bits(out[j]) != 0u) +
                       one_lane(true, v, ps[fmt(cols ? j : i * T + j)], want);
            }
          }
        }
      }
    }
    return bad;
  }

  // cpu_rne_tile_sweep.cpp's operands of a whole tile, by the block's kind.
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
      return sign * std::ldexp(1.0f, -149) * float(1 + (r & 7));
    case 2:
      return kind == 1 ? sign * 0x1p-126f : sign * std::numeric_limits<float>::max() * 0.5f;
    case 3:
      return kind == 1 ? -0.0f : sign * std::numeric_limits<float>::infinity();
    default:
      return kind == 1 ? sign * std::ldexp(m, -int((r >> 8) % 80) - 1)
                       : sign * std::ldexp(m, int((r >> 8) % 250) - 125);
    }
  }

  float want_step(int kind, float a, float b, float s, const BinaryKParams &pm,
                  const BinaryKParams &pa)
  {
    if (kind == 1)
      return cast(s + cast(a * b, pm), pa);
    if (kind == 2)
      return s + cast(a * b, pm);
    return cast(std::fma(a, b, s), pa);
  }

  // Whole tiles of the three macs, a palette of three (multiply, accumulate)
  // pairs, an entry per element (or per column) drawn per block.
  uint64_t whole_tiles(const std::vector<BinaryKParams> &ps, uint64_t &ran, uint64_t &blocks)
  {
    constexpr int64_t SAMPLE = 64;
    const mptorch::gemm_cpu::LaneTileFn tiles[] = {lane_tiles.split, lane_tiles.mul_only,
                                                   lane_tiles.fused};
    const int64_t F = static_cast<int64_t>(ps.size());
    uint64_t bad = 0, kept = 0;
    const int64_t chunks = (int64_t(1) << 32) / (LANES * SAMPLE);
#pragma omp parallel for reduction(+ : bad, kept) schedule(dynamic, 16)
    for (int64_t chunk = 0; chunk < chunks; ++chunk)
    {
      alignas(64) float a[LANES], b[LANES], x[LANES], whole[LANES], want[LANES];
      static thread_local LaneParams lm, la;
      int entry_m[LANES], entry_a[LANES];
      uint64_t st = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(chunk);
      const int kind_of_block = chunk % 8 < 6 ? 0 : chunk % 8 == 6 ? 1 : 2;
      const bool cols = chunk % 2 == 1;
      for (int64_t l = 0; l < LANES; ++l)
      {
        a[l] = operand(st, kind_of_block);
        b[l] = operand(st, kind_of_block);
        const uint32_t w = static_cast<uint32_t>((chunk * LANES + l) * 0x9E3779B1u);
        x[l] = bits_float((w & 0x7FFFFFFFu) > 0x7F800000u ? w & ~0x00800000u : w);
      }
      for (int64_t l = 0; l < (cols ? T : LANES); ++l)
      {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        const int e = static_cast<int>((st >> 33) % 3);
        entry_m[l] = static_cast<int>((chunk + e) % F);
        entry_a[l] = static_cast<int>((chunk + 2 * e + 1) % F);
        mptorch::gemm_cpu::set_lane(lm, l, ps[entry_m[l]]);
        mptorch::gemm_cpu::set_lane(la, l, ps[entry_a[l]]);
      }
      for (int kind = 1; kind <= 3; ++kind)
      {
        std::memcpy(want, x, sizeof want);
        for (int64_t k = 0; k < T; ++k)
          for (int64_t l = 0; l < LANES; ++l)
          {
            const int64_t q = cols ? l % T : l;
            want[l] = want_step(kind, a[(l / T) * T + k], b[k * T + l % T], want[l],
                                ps[entry_m[q]], ps[entry_a[q]]);
          }
        std::memcpy(whole, x, sizeof whole);
        const bool whole_ran = tiles[kind - 1](a, b, whole, T, T, T, cols, lm, la);
        kept += whole_ran;
        if (!whole_ran)
        {
          for (int64_t l = 0; l < LANES; ++l)
            bad += float_bits(whole[l]) != float_bits(x[l]);
          bool nan = false;
          for (int64_t l = 0; l < LANES; ++l)
            nan |= std::isnan(want[l]);
          bad += !nan; // handed back with no NaN at the block's end
          continue;
        }
        for (int64_t l = 0; l < LANES; ++l)
          bad += float_bits(whole[l]) != float_bits(want[l]);
      }
    }
    ran = kept;
    blocks = 3 * chunks;
    return bad;
  }
} // namespace

int main(int argc, char **argv)
{
  const bool control = argc > 1 && std::strcmp(argv[1], "control") == 0;
  if (!lane_tiles.split)
  {
    std::printf("no lane tile on this CPU or build (isa %s): nothing to check\n", lane_tiles.isa);
    return 0;
  }
  const Format formats[] = {
      {3, 4, 7, SaturationMode::SAT_NONE, "e4m3-like, SAT_NONE"},
      {3, 4, 8, SaturationMode::SAT_FINITE, "binary8p4, SAT_FINITE"},
      {10, 5, 15, SaturationMode::SAT_NONE, "binary16"},
      {10, 5, 16, SaturationMode::SAT_FINITE_EXTENDED, "binary16p11, SAT_FINITE_EXTENDED"},
      {7, 7, 63, SaturationMode::SAT_NONE, "P = 8, 7 exponent bits"},
      {1, 5, 15, SaturationMode::SAT_NONE, "P = 2"},
      {2, 3, 4, SaturationMode::SAT_NONE, "binary6p3"},
      {22, 7, 63, SaturationMode::SAT_FINITE, "P = 23, 7 exponent bits"},
      {5, 6, 31, SaturationMode::SAT_NONE, "P = 6, 6 exponent bits"},
  };
  std::vector<BinaryKParams> ps;
  for (const Format &f : formats)
  {
    const BinaryKParams p =
        make_binaryK_params<float>(f.man_bits, f.exp_bits, f.bias, true, f.sat, false);
    if (p.fast_rne)
      ps.push_back(p);
    else
      std::printf("  %-36s the fast gate is off: not the tile's case, skipped\n", f.name);
  }
  std::printf("isa %s, %zu formats, a format per lane%s\n", lane_tiles.isa, ps.size(),
              control ? ": the control, each lane checked on its neighbour's format" : "");
  const uint64_t words = sweep(ps, control ? 1 : 0);
  std::printf("  every word, one-step tiles, per element and per column: %llu mismatches\n",
              (unsigned long long)words);
  if (control)
  {
    std::printf("%s: the control %s\n", words ? "PASS" : "FAIL",
                words ? "mismatches, so the comparison can fail" : "found no mismatch");
    return words ? 0 : 1;
  }
  uint64_t ran = 0, blocks = 0;
  const uint64_t tiles = whole_tiles(ps, ran, blocks);
  std::printf("  whole tiles, a palette of three per block: %llu mismatches (%llu of %llu blocks "
              "ran whole)\n",
              (unsigned long long)tiles, (unsigned long long)ran, (unsigned long long)blocks);
  const uint64_t total = words + tiles;
  std::printf("%s: %llu mismatches\n", total ? "FAIL" : "PASS", (unsigned long long)total);
  return total ? 1 : 0;
}
