// The shipped casts under the undefined behaviour sanitizer: every binaryK,
// superfp and fixed-point cast of common/cast_*.h, in every rounding mode
// (StochasticA, B and C included), in binary32 and binary64, over a sample
// of inputs that reaches every arm. It answers what the value sweeps cannot:
// whether a cast executes an operation the language leaves undefined, such as
// a shift by the word's width or more, or a signed overflow. A compiler may
// assume such an operation never happens, even where its result is then
// discarded, so a cast can return the right word in every sweep and still be
// wrong code.
//
// Build & run (from the repository root):
//   g++ -O2 -std=c++20 -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all \
//       -I mptorch/csrc dev/benchmarks/sweeps/cast_ubsan.cpp -o /tmp/cast_ubsan && /tmp/cast_ubsan
// It has been run with GCC only; clang documents the same flags. The first
// two flags are setup.py's host pairing. -fno-sanitize-recover makes the first
// report end the run with a non-zero exit, naming the file and line;
// UBSAN_OPTIONS=print_stacktrace=1 adds the cast that called it.
//
// No argument: the gate, ~3.5 s. It runs in two parts:
//   1. The negative control, in a child process: it calls
//      round_bitwise_nearest_even with a man_bits below its documented
//      bound, which shifts by more than the word's width. It must be reported.
//      If it runs clean, the binary was built without the sanitizer, and the
//      run fails (exit 2) rather than pass vacuously.
//   2. The check. Each family and carrier is named before it runs, so a
//      report follows the name of the family it came from (exit 1). A family
//      that completes prints how many casts ran and a checksum of every
//      result word, which two header trees meant to round alike must agree on.
// `--recover`, for a build without -fno-sanitize-recover: the control passes
// on its report alone, and the check runs to the end, leaving every report on
// stderr (UBSan prints each source location once). It lists them all, and
// can neither pass nor fail.
//
// The inputs, in each carrier:
//   * every exponent field below binary32's 140, or binary64's 1100, each
//     with zero, one, all-ones and tie mantissas and random ones, both signs;
//     that is every value below every format's normal range here, and so
//     every shift a subnormal or underflow arm computes;
//   * random words over the whole word;
//   * every power of two, the infinities and NaNs among them.
// The formats are edge and middle ones:
//   * binaryK from 0 to 23 (52) mantissa bits;
//   * exponent fields up to 30 bits;
//   * negative biases, and biases that put the bottom of the range below the
//     carrier's;
//   * signed and unsigned, two saturation modes, all three SubnormalsModes;
//   * superfp from 0 to 22 (51) mantissa bits;
//   * fixed point with negative and over-wide fraction lengths, both
//     signednesses, symmetric or not;
//   * SR with 0, 1 and the most random bits the format allows.
// Each binary32 binaryK and superfp format runs twice: as built, which takes
// the float-arithmetic RNE fast path where the format admits it, and with
// fast_rne cleared, which is the integer path the device and Metal take for a
// format the gate refuses.
//
// Only formats mptorch.number accepts are used. In particular, man_bits plus
// prng_bits never exceed the carrier's mantissa. The C++ ops leave that check
// to Python: stochastic_host.h holds prng_bits to [0, 255] only. So a raw
// torch.ops call past it still reaches a negative shift in stochastic.h's
// make_stochastic.
//
// Results (2026-11-18, AMD Ryzen 9 7940HS, GCC 15.2):
//   * Before that day's fix to cast_binaryK.h's subnormal arms, the check
//     stopped in its first family, at round_bitwise_nearest_away. Under
//     --recover it listed every site: the nearest-even, nearest-away and
//     down helpers shifted by up to 149 bits in binary32 and 1059 in
//     binary64, and overflowed their masks' `- 1`. All of these calls came
//     from binaryK's three subnormal arms, with a man_bits far below -1.
//   * After the fix, binaryK in both carriers and superfp in binary32 run
//     clean, their checksums unchanged. The check stops in superfp binary64,
//     at cast_superfp_odd's supernormal arm. SuperFP(51, 4, 2, 7)'s
//     supernormal cutoff lies below int's range, and superfp_region_cutoffs
//     clamps it to INT_MIN + 1. `target_exp - supernormal_cutoff + 1` then
//     overflows.
//   * The clamp also flips the parity of that code index. So round to odd
//     returns the wrong neighbour for every inexact supernormal input of
//     such a format (1.5 rounds to 2, not 1), on the CPU and on CUDA alike.
//     reference_casts.h calls the same superfp_region_cutoffs, so the sweeps
//     agree with it.
//   * Under --recover, the fixed-point casts run clean too.
#include "common/cast_binaryK.h"
#include "common/cast_fixedpoint.h"
#include "common/cast_superfp.h"
#include "common/modes.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
  uint64_t rng_state = 0x9E3779B97F4A7C15ull;
  uint64_t next_u64()
  {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
  }

  // FNV-1a over each result word's bytes, and the number of casts that fed it.
  struct Digest
  {
    uint64_t h = 1469598103934665603ull;
    uint64_t casts = 0;
    template <class T>
    void add(T x)
    {
      typename FloatTraits<T>::word_t w;
      std::memcpy(&w, &x, sizeof w);
      for (unsigned i = 0; i < sizeof w; ++i)
      {
        h ^= (uint64_t)((w >> (8 * i)) & 0xFF);
        h *= 1099511628211ull;
      }
      ++casts;
    }
  };

  template <class T>
  std::vector<T> inputs(int low_fields, int per_field, int random)
  {
    using F = FloatTraits<T>;
    using W = typename F::word_t;
    std::vector<W> words;
    for (int field = 0; field < low_fields; ++field)
      for (int i = 0; i < per_field; ++i)
        for (W sign : {W(0), (W)F::SIGN_MASK})
        {
          const W edges[] = {W(0), W(1), (W)F::MAN_MASK, (W)F::TIE};
          const W man = i < 4 ? edges[i] : (W)(next_u64() & F::MAN_MASK);
          words.push_back(sign | ((W)field << F::MAN_BITS) | man);
        }
    for (int i = 0; i < random; ++i)
      words.push_back((W)next_u64());
    for (int field = 0; field <= (int)F::FIELD_MASK; ++field)
      for (W man : {W(0), W(1)})
        words.push_back(((W)field << F::MAN_BITS) | man);
    std::vector<T> values(words.size());
    std::memcpy(values.data(), words.data(), words.size() * sizeof(T));
    return values;
  }

  // The random-bit counts a format takes under SR: 0, 1 and the most that fit
  // beside its mantissa, without repeats.
  std::vector<int> prng_counts(int most)
  {
    std::vector<int> n = {0};
    if (most >= 1)
      n.push_back(1);
    if (most > 1)
      n.push_back(most);
    return n;
  }

  template <class T>
  typename FloatTraits<T>::word_t draw()
  {
    return (typename FloatTraits<T>::word_t)next_u64();
  }

  struct BinaryKFmt
  {
    int man, exp, bias;
  };

  template <class T>
  void binaryK_casts(Digest &d, const std::vector<T> &xs, bool is_signed, SubnormalsMode sub,
                     const BinaryKParamsT<T> &p)
  {
    const std::vector<int> ns = prng_counts(FloatTraits<T>::MAN_BITS - p.man_bits);
    for (T x : xs)
    {
      d.add(cast_binaryK_nearest_even(x, is_signed, sub, p));
      d.add(cast_binaryK_nearest_away(x, is_signed, sub, p));
      d.add(cast_binaryK_odd(x, is_signed, sub, p));
      d.add(cast_binaryK_up(x, is_signed, sub, p));
      d.add(cast_binaryK_down(x, is_signed, sub, p));
      d.add(cast_binaryK_zero(x, is_signed, sub, p));
      for (int n : ns)
        for (int variant = 0; variant < 3; ++variant)
          d.add(cast_binaryK_stochastic(x, draw<T>(), sr_spec(n, variant), is_signed, sub, p));
    }
  }

  template <class T>
  Digest run_binaryK(const std::vector<BinaryKFmt> &fmts, const std::vector<T> &xs)
  {
    Digest d;
    for (const BinaryKFmt &f : fmts)
      for (bool is_signed : {true, false})
        for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE})
          for (SubnormalsMode sub :
               {SubnormalsMode::SUBNORMALS, SubnormalsMode::NORMALS, SubnormalsMode::EXTENDED_NORMALS})
          {
            BinaryKParamsT<T> p = make_binaryK_params<T>(f.man, f.exp, f.bias, is_signed, sat,
                                                         sub == SubnormalsMode::EXTENDED_NORMALS);
            binaryK_casts(d, xs, is_signed, sub, p);
            if constexpr (std::is_same_v<T, float>)
              if (p.fast_rne)
              {
                p.fast_rne = false;
                binaryK_casts(d, xs, is_signed, sub, p);
              }
          }
    return d;
  }

  struct SuperfpFmt
  {
    int man, exp, binades, bias;
  };

  template <class T>
  void superfp_casts(Digest &d, const std::vector<T> &xs, bool is_signed, const SuperfpParamsT<T> &p, int man)
  {
    const std::vector<int> ns = prng_counts(FloatTraits<T>::MAN_BITS - man);
    for (T x : xs)
    {
      d.add(cast_superfp_nearest_even(x, is_signed, p));
      d.add(cast_superfp_nearest_away(x, is_signed, p));
      d.add(cast_superfp_odd(x, is_signed, p));
      d.add(cast_superfp_up(x, is_signed, p));
      d.add(cast_superfp_down(x, is_signed, p));
      d.add(cast_superfp_zero(x, is_signed, p));
      for (int n : ns)
        for (int variant = 0; variant < 3; ++variant)
          d.add(cast_superfp_stochastic(x, draw<T>(), sr_spec(n, variant), is_signed, p));
    }
  }

  template <class T>
  Digest run_superfp(const std::vector<SuperfpFmt> &fmts, const std::vector<T> &xs)
  {
    Digest d;
    for (const SuperfpFmt &f : fmts)
      for (bool is_signed : {true, false})
        for (SaturationMode sat : {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE})
        {
          SuperfpParamsT<T> p = make_superfp_params<T>(f.man, f.exp, f.binades, f.bias, sat);
          superfp_casts(d, xs, is_signed, p, f.man);
          if constexpr (std::is_same_v<T, float>)
            if (p.fast_rne)
            {
              p.fast_rne = false;
              superfp_casts(d, xs, is_signed, p, f.man);
            }
        }
    return d;
  }

  struct FixedPointFmt
  {
    int wl, fl;
  };

  template <class T>
  Digest run_fixedpoint(const std::vector<FixedPointFmt> &fmts, const std::vector<T> &xs)
  {
    Digest d;
    for (const FixedPointFmt &f : fmts)
      for (bool is_signed : {true, false})
        for (bool symmetric : {false, true})
        {
          const FixedPointParamsT<T> p = make_fixedpoint_params<T>(f.wl, f.fl, is_signed, symmetric);
          // a fixed-point format's magnitude bits and random bits share the
          // carrier's significand, the implicit bit included
          const std::vector<int> ns = prng_counts(FloatTraits<T>::MAN_BITS + 1 - (f.wl - (is_signed ? 1 : 0)));
          for (T x : xs)
          {
            d.add(cast_fixedpoint_nearest_even(x, is_signed, p));
            d.add(cast_fixedpoint_nearest_away(x, is_signed, p));
            d.add(cast_fixedpoint_odd(x, is_signed, p));
            d.add(cast_fixedpoint_up(x, is_signed, p));
            d.add(cast_fixedpoint_down(x, is_signed, p));
            d.add(cast_fixedpoint_zero(x, is_signed, p));
            for (int n : ns)
              for (int variant = 0; variant < 3; ++variant)
                d.add(cast_fixedpoint_stochastic(x, draw<T>(), sr_spec(n, variant), is_signed, p));
          }
        }
    return d;
  }

  // Names the family before running it, so that a report, which ends the
  // run, follows the name of the family it came from.
  template <class Run>
  void report(const char *family, const char *carrier, Run run)
  {
    std::printf("  %-11s %-8s ", family, carrier);
    std::fflush(stdout);
    const Digest d = run();
    std::printf("%11llu casts  checksum %016llx\n", (unsigned long long)d.casts, (unsigned long long)d.h);
  }

  // Undefined on purpose: man_bits = -20 shifts a 32-bit mask by 43.
  int control()
  {
    volatile int man_bits = -20;
    const uint32_t r = round_bitwise_nearest_even<uint32_t>(0x3F800001u, man_bits);
    std::printf("%08x\n", r);
    return 0;
  }

  // Runs the control in a child and reports whether the sanitizer caught it:
  // a report that ended the child or, in a recovering build, a report alone.
  bool control_is_reported(bool recover)
  {
    std::fflush(stdout);
    int fds[2];
    if (pipe(fds) != 0)
      return false;
    const pid_t pid = fork();
    if (pid == 0)
    {
      close(fds[0]);
      dup2(fds[1], STDERR_FILENO);
      dup2(fds[1], STDOUT_FILENO);
      _exit(control());
    }
    close(fds[1]);
    std::string out;
    char buf[512];
    ssize_t got;
    while ((got = read(fds[0], buf, sizeof buf)) > 0)
      out.append(buf, (size_t)got);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    const size_t at = out.find("runtime error");
    if (at != std::string::npos && (recover || (WIFEXITED(status) && WEXITSTATUS(status) != 0)))
    {
      const size_t newline = out.rfind('\n', at);
      const size_t from = newline == std::string::npos ? 0 : newline + 1;
      const size_t to = out.find('\n', at);
      std::printf("control: reported, as it must be:\n  %s\n", out.substr(from, to - from).c_str());
      return true;
    }
    std::printf("FAIL: the control was not reported (exit status %d). Build with "
                "-fsanitize=undefined,float-cast-overflow -fno-sanitize-recover=all, or pass --recover "
                "for a build without -fno-sanitize-recover.\n",
                WIFEXITED(status) ? WEXITSTATUS(status) : -1);
    return false;
  }
} // namespace

int main(int argc, char **argv)
{
  const bool recover = argc > 1 && !std::strcmp(argv[1], "--recover");
  // A report ends the process without flushing stdio, which would drop
  // whatever this had printed.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (!control_is_reported(recover))
    return 2;

  const std::vector<BinaryKFmt> bk32 = {{0, 5, 15},  {1, 2, 1},   {2, 5, 15},  {3, 4, 7},          {3, 4, 8},
                                        {3, 4, 21},  {7, 7, 63},  {10, 5, 15}, {13, 7, 64},        {22, 7, 63},
                                        {2, 7, 120}, {7, 8, 127}, {0, 8, 127}, {23, 8, 127},       {5, 10, 1},
                                        {3, 4, -20}, {3, 4, 140}, {4, 30, 1 << 29}};
  const std::vector<BinaryKFmt> bk64 = {{2, 5, 15},   {3, 4, 7},     {10, 5, 15},   {23, 8, 127},
                                        {51, 10, 511}, {52, 11, 1023}, {2, 10, 1000}, {7, 11, 1023},
                                        {3, 4, -200}, {5, 30, 1 << 29}};
  const std::vector<SuperfpFmt> sfp32 = {{3, 4, 8, 7}, {3, 4, 2, 7},  {1, 2, 1, 1},  {9, 3, 1, 3},
                                         {3, 5, 1, 7}, {0, 4, 2, 7},  {22, 3, 1, 3}, {3, 8, 4, 100}};
  const std::vector<SuperfpFmt> sfp64 = {{3, 4, 8, 7}, {3, 5, 1, 7}, {20, 6, 4, 31}, {51, 4, 2, 7}, {3, 10, 8, 500}};
  const std::vector<FixedPointFmt> fx32 = {{2, 1},   {4, 0},   {8, 4},   {8, -4}, {8, 20},
                                           {12, 12}, {16, 8},  {20, 10}, {24, 12}};
  const std::vector<FixedPointFmt> fx64 = {{8, 4}, {8, -4}, {32, 16}, {40, 20}, {53, 26}};

  const std::vector<float> x32 = inputs<float>(140, 32, 1 << 15);
  const std::vector<double> x64 = inputs<double>(1100, 6, 1 << 14);

  std::printf("check: %zu binary32 and %zu binary64 inputs\n", x32.size(), x64.size());
  report("binaryK", "binary32", [&] { return run_binaryK<float>(bk32, x32); });
  report("binaryK", "binary64", [&] { return run_binaryK<double>(bk64, x64); });
  report("superfp", "binary32", [&] { return run_superfp<float>(sfp32, x32); });
  report("superfp", "binary64", [&] { return run_superfp<double>(sfp64, x64); });
  report("fixedpoint", "binary32", [&] { return run_fixedpoint<float>(fx32, x32); });
  report("fixedpoint", "binary64", [&] { return run_fixedpoint<double>(fx64, x64); });
  if (recover)
    std::printf("done: every report is on stderr, each source location once\n");
  else
    std::printf("PASS: no undefined behaviour reported\n");
  return 0;
}
