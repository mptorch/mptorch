// The binary64 casts against reference_casts.h's binary64 instantiation, on
// inputs chosen for the arms.
//
// cast_all_modes_sweep.cu checks the binary32 casts over every input, and its
// `image64` mode carries that across to the binary64 instantiation wherever
// binary32 can say anything. What neither reaches is the part of binary64
// binary32 does not have: mantissas past 23 bits, exponent fields past 255,
// formats up to P = 53 and ten exponent bits, a floor at 2^-1021, and the SR
// arms whose intermediates binary32 rounds (see image64's note). There are
// 2^64 inputs, so this checks a structured subset of them against the
// reference -- which is written apart from the shipped casts, down to its own
// derivation of the carrier's constants -- and counts, as the binary32 sweep
// does, mismatches per mode with the first input that shows one, and -0.0
// results on either side.
//
// The inputs, about 1.44e9 per configuration, generated in the kernel from
// their index so nothing is stored:
//
//   structured  every one of the 2048 exponent fields, both signs, and in
//               each of those binades these mantissas:
//                 * ten fixed ones: 0, 1, 2, 3, the tie and its two
//                   neighbours, the top two, and an alternating pattern;
//                 * the format's own grid in that binade -- every code when it
//                   has at most 4096 there, otherwise its bottom and top codes
//                   and a hashed sample of 4096 -- each exactly, one ulp above
//                   and one below, and the midpoint to the next code likewise,
//                   which is where every rounding mode decides something;
//                 * 2^16 random ones.
//   random      2^30 words with the sign, the exponent field and the mantissa
//               drawn independently, so the fields are uniform rather than
//               concentrated where finite values are.
//
// "The format's grid in that binade" is read from the format rather than from
// the cast: man_bits significand bits in the normal region, fewer in a
// SUBNORMALS format's subnormal binades (the step is fixed there), and none --
// powers of two -- in superfp's supernormal region.
//
// Stochastic rounding gets the same draw on both sides, a hash of the input
// word, and rand_bits wide (30, or whatever man_bits leaves of 52), since the
// binary64 draw is 52 bits where binary32's was 23.
//
// Build & run (from the repo root):
//   nvcc -O3 -std=c++17 -arch=sm_89 --expt-relaxed-constexpr -I mptorch/csrc \
//        -I dev/benchmarks/sweeps dev/benchmarks/sweeps/cast_double_sweep.cu \
//        -o /tmp/cast_double_sweep && /tmp/cast_double_sweep spot
//   g++ -O3 -std=c++20 -fopenmp -ffp-contract=off -DMPTORCH_FAST_CAST=1 \
//       -I mptorch/csrc -I dev/benchmarks/sweeps -x c++ dev/benchmarks/sweeps/cast_double_sweep.cu \
//       -o /tmp/cast_double_sweep_host && /tmp/cast_double_sweep_host host
// `short` is ~40 edge and middle formats and eighteen fixed-point ones (a
// section of their own below), the gate for a change (`short-host`
// five of them for the host build); `spot` is the grid below, `sweep` a denser
// one, `host` a ten-format list for the host build -- what a host run adds is
// g++'s codegen for the same casts, as in cast_all_modes_sweep.cu -- `quick`
// the spot grid on every 1021st input, and `edges` the three binary32-edge
// formats alone.
#include "common/cast_binaryK.h"
#include "common/cast_fixedpoint.h"
#include "common/cast_superfp.h"
#include "common/modes.h"
#include "reference_casts.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#if defined(__CUDACC__)
#define CHECK(x) do{cudaError_t e_=(x); if(e_!=cudaSuccess){printf("ERR %s %d\n",cudaGetErrorString(e_),__LINE__);exit(1);} }while(0)
#endif

constexpr int NMODE = 9;
static const char *MODE_NAME[NMODE] = {"RNE", "RNA", "RU", "RD", "RZ", "RO", "SR", "SRB", "SRC"};

struct Counts
{
    unsigned long long bad[NMODE];
    unsigned long long nzero_shipped[NMODE];
    unsigned long long nzero_reference[NMODE];
    unsigned first_bad[NMODE]; // an input index, not a word: every index fits in 32 bits
};

struct Cfg
{
    bool is_binaryK;
    int man_bits, exp_bits, bias;
    int normal_binades; // superfp only
    bool is_signed;
    SaturationMode sat;
    SubnormalsMode subm; // binaryK only
    int rand_bits;       // SR
    int normal_cutoff;   // superfp only: superfp_region_cutoffs', filled in by run_case
};

constexpr uint64_t SIGN = 0x8000000000000000u;
constexpr uint64_t MAN_MASK = 0x000FFFFFFFFFFFFFu;
constexpr uint64_t M_FIXED = 10, GRID_CODES = 4096, M_GRID = 6 * GRID_CODES, M_RANDOM = 1u << 16;
constexpr uint64_t M_PER_BINADE = M_FIXED + M_GRID + M_RANDOM;
constexpr uint64_t N_STRUCTURED = 2048 * M_PER_BINADE * 2;
constexpr uint64_t N_RANDOM = 1u << 30;
constexpr uint64_t N_TOTAL = N_STRUCTURED + N_RANDOM;

// `quick` visits every STRIDE-th index instead of every one: the same grid in
// about a thousandth of the time, for checking the sweep before running it.
static uint64_t stride = 1;

CUDA_HOST_DEVICE_INLINE uint64_t splitmix64(uint64_t z)
{
    z += 0x9E3779B97F4A7C15u;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9u;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBu;
    return z ^ (z >> 31);
}

CUDA_HOST_DEVICE_INLINE double value_of(uint64_t w)
{
    double d;
    memcpy(&d, &w, sizeof d);
    return d;
}

CUDA_HOST_DEVICE_INLINE uint64_t bits_of(double d)
{
    uint64_t w;
    memcpy(&w, &d, sizeof w);
    return w;
}

// Significand bits of the format's grid in the carrier binade `field` -- the
// format's definition, not the cast's (field 0 is the carrier's subnormals,
// whose spacing is binade -1022's).
CUDA_HOST_DEVICE_INLINE int grid_bits(const Cfg &c, int field)
{
    const int e = field == 0 ? -1022 : field - 1023;
    int s = c.man_bits;
    if (c.is_binaryK)
    {
        const int min_exp = 1 - c.bias;
        if (c.subm == SubnormalsMode::SUBNORMALS && e < min_exp)
            s = c.man_bits - (min_exp - e);
    }
    else if (e < c.normal_cutoff)
        s = 0;
    return s < 0 ? 0 : (s > 52 ? 52 : s);
}

// A fixed-point format (cast_fixedpoint.h): one step, 2^-fl, everywhere, so
// binade e holds e + fl significand bits of it.
struct FxCfg
{
    int wl, fl;
    bool is_signed, symmetric;
    int rand_bits; // SR's N; mag_bits + N <= 52 is the carrier check
};

CUDA_HOST_DEVICE_INLINE int grid_bits(const FxCfg &c, int field)
{
    const int s = (field == 0 ? -1022 : field - 1023) + c.fl;
    return s < 0 ? 0 : (s > 52 ? 52 : s);
}

// The format's grid comes from grid_bits, which is the only thing input_word
// reads of its configuration.
template <class C>
CUDA_HOST_DEVICE_INLINE uint64_t input_word(const C &c, uint64_t i)
{
    if (i >= N_STRUCTURED)
    {
        const uint64_t j = i - N_STRUCTURED;
        const uint64_t a = splitmix64(2 * j + 1), b = splitmix64(2 * j + 2);
        return (b & SIGN) | (((b >> 16) % 2048) << 52) | (a & MAN_MASK);
    }
    const uint64_t sign = (i & 1) << 63;
    const uint64_t j = (i >> 1) % M_PER_BINADE;
    const uint64_t field = (i >> 1) / M_PER_BINADE;
    uint64_t man;
    if (j < M_FIXED)
    {
        switch (j)
        {
        case 0: man = 0; break;
        case 1: man = 1; break;
        case 2: man = 2; break;
        case 3: man = 3; break;
        case 4: man = (MAN_MASK >> 1); break;     // the tie, less one
        case 5: man = (MAN_MASK >> 1) + 1; break; // the tie
        case 6: man = (MAN_MASK >> 1) + 2; break; // the tie, plus one
        case 7: man = MAN_MASK - 1; break;
        case 8: man = MAN_MASK; break;
        default: man = 0x0005555555555555u; break;
        }
    }
    else if (j < M_FIXED + M_GRID)
    {
        const uint64_t g = (j - M_FIXED) / 6, variant = (j - M_FIXED) % 6;
        const int s = grid_bits(c, (int)field);
        const uint64_t codes = (uint64_t)1 << s;
        uint64_t k;
        if (codes <= GRID_CODES)
            k = g % codes;
        else if (g < 8)
            k = g < 4 ? g : codes - 8 + g; // the bottom four codes and the top four
        else
            k = splitmix64(g ^ (field << 20)) & (codes - 1);
        const uint64_t step = (uint64_t)1 << (52 - s);
        const uint64_t code = k << (52 - s);
        const uint64_t mid = code + (step >> 1);
        switch (variant)
        {
        case 0: man = code; break;
        case 1: man = code + 1; break;
        case 2: man = code - 1; break; // wraps to the binade's top word at k == 0, which is an input too
        case 3: man = mid; break;
        case 4: man = mid + 1; break;
        default: man = mid - 1; break;
        }
    }
    else
        man = splitmix64(i);
    return sign | (field << 52) | (man & MAN_MASK);
}

CUDA_HOST_DEVICE_INLINE double shipped_cast(int mode, double x, uint64_t rnd, const Cfg &c,
                                            const BinaryKParamsT<double> &bp, const SuperfpParamsT<double> &sp)
{
    if (c.is_binaryK)
        switch (mode)
        {
        case 0: return cast_binaryK_nearest_even(x, c.is_signed, c.subm, bp);
        case 1: return cast_binaryK_nearest_away(x, c.is_signed, c.subm, bp);
        case 2: return cast_binaryK_up(x, c.is_signed, c.subm, bp);
        case 3: return cast_binaryK_down(x, c.is_signed, c.subm, bp);
        case 4: return cast_binaryK_zero(x, c.is_signed, c.subm, bp);
        case 5: return cast_binaryK_odd(x, c.is_signed, c.subm, bp);
        default: return cast_binaryK_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), c.is_signed, c.subm, bp);
        }
    switch (mode)
    {
    case 0: return cast_superfp_nearest_even(x, c.is_signed, sp);
    case 1: return cast_superfp_nearest_away(x, c.is_signed, sp);
    case 2: return cast_superfp_up(x, c.is_signed, sp);
    case 3: return cast_superfp_down(x, c.is_signed, sp);
    case 4: return cast_superfp_zero(x, c.is_signed, sp);
    case 5: return cast_superfp_odd(x, c.is_signed, sp);
    default: return cast_superfp_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), c.is_signed, sp);
    }
}

CUDA_HOST_DEVICE_INLINE double reference_cast(int mode, double x, uint64_t rnd, const Cfg &c)
{
    const int m = c.man_bits, e = c.exp_bits, b = c.bias, nb = c.normal_binades;
    if (c.is_binaryK)
        switch (mode)
        {
        case 0: return cast_binaryK_nearest_even(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 1: return cast_binaryK_nearest_away(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 2: return cast_binaryK_up(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 3: return cast_binaryK_down(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 4: return cast_binaryK_zero(x, m, e, b, c.is_signed, c.sat, c.subm);
        case 5: return cast_binaryK_odd(x, m, e, b, c.is_signed, c.sat, c.subm);
        default: return cast_binaryK_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), m, e, b, c.is_signed, c.sat, c.subm);
        }
    switch (mode)
    {
    case 0: return cast_superfp_nearest_even(x, m, e, nb, b, c.is_signed, c.sat);
    case 1: return cast_superfp_nearest_away(x, m, e, nb, b, c.is_signed, c.sat);
    case 2: return cast_superfp_up(x, m, e, nb, b, c.is_signed, c.sat);
    case 3: return cast_superfp_down(x, m, e, nb, b, c.is_signed, c.sat);
    case 4: return cast_superfp_zero(x, m, e, nb, b, c.is_signed, c.sat);
    case 5: return cast_superfp_odd(x, m, e, nb, b, c.is_signed, c.sat);
    default: return cast_superfp_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), m, e, nb, b, c.is_signed, c.sat);
    }
}

CUDA_HOST_DEVICE_INLINE void one_input(uint64_t i, const Cfg &c, const BinaryKParamsT<double> &bp,
                                       const SuperfpParamsT<double> &sp, unsigned long long *bad,
                                       unsigned long long *nzs, unsigned long long *nzr, uint32_t *first)
{
    const uint64_t w = input_word(c, i);
    const double x = value_of(w);
    const uint64_t rnd = splitmix64(w ^ 0xD1B54A32D192ED03u);
    for (int mode = 0; mode < NMODE; ++mode)
    {
        const uint64_t s = bits_of(shipped_cast(mode, x, rnd, c, bp, sp));
        const uint64_t r = bits_of(reference_cast(mode, x, rnd, c));
        if (s != r)
        {
            ++bad[mode];
            if ((uint32_t)i < first[mode])
                first[mode] = (uint32_t)i;
        }
        if (s == SIGN) ++nzs[mode];
        if (r == SIGN) ++nzr[mode];
    }
}

#if defined(__CUDACC__)
constexpr int PER_THREAD = 64;

__global__ void sweep_kernel(Cfg c, uint64_t stride, BinaryKParamsT<double> bp, SuperfpParamsT<double> sp, Counts *out)
{
    const uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long bad[NMODE] = {0}, nzs[NMODE] = {0}, nzr[NMODE] = {0};
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
    for (int k = 0; k < PER_THREAD && (base + k) * stride < N_TOTAL; ++k)
        one_input((base + k) * stride, c, bp, sp, bad, nzs, nzr, first);
    for (int m = 0; m < NMODE; ++m)
    {
        if (bad[m]) { atomicAdd(&out->bad[m], bad[m]); atomicMin(&out->first_bad[m], first[m]); }
        if (nzs[m]) atomicAdd(&out->nzero_shipped[m], nzs[m]);
        if (nzr[m]) atomicAdd(&out->nzero_reference[m], nzr[m]);
    }
}

static void run_sweep(const Cfg &c, const BinaryKParamsT<double> &bp, const SuperfpParamsT<double> &sp, Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    Counts zero;
    memset(&zero, 0, sizeof zero);
    for (int m = 0; m < NMODE; ++m) zero.first_bad[m] = 0xFFFFFFFFu;
    CHECK(cudaMemcpy(dc, &zero, sizeof zero, cudaMemcpyHostToDevice));
    const uint64_t per_block = 256 * PER_THREAD, n = (N_TOTAL + stride - 1) / stride;
    sweep_kernel<<<(unsigned)((n + per_block - 1) / per_block), 256>>>(c, stride, bp, sp, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
static void run_sweep(const Cfg &c, const BinaryKParamsT<double> &bp, const SuperfpParamsT<double> &sp, Counts *h)
{
    memset(h, 0, sizeof *h);
    for (int m = 0; m < NMODE; ++m) h->first_bad[m] = 0xFFFFFFFFu;
#pragma omp parallel
    {
        unsigned long long b[NMODE] = {0}, s[NMODE] = {0}, r[NMODE] = {0};
        uint32_t f[NMODE];
        for (int m = 0; m < NMODE; ++m) f[m] = 0xFFFFFFFFu;
#pragma omp for schedule(static, 1 << 16)
        for (int64_t i = 0; i < (int64_t)((N_TOTAL + stride - 1) / stride); ++i)
            one_input((uint64_t)i * stride, c, bp, sp, b, s, r, f);
#pragma omp critical
        for (int m = 0; m < NMODE; ++m)
        {
            h->bad[m] += b[m]; h->nzero_shipped[m] += s[m]; h->nzero_reference[m] += r[m];
            if (f[m] < h->first_bad[m]) h->first_bad[m] = f[m];
        }
    }
}
#endif

static const char *sat_name(SaturationMode s)
{
    return s == SaturationMode::SAT_NONE ? "SAT_NONE" : s == SaturationMode::SAT_FINITE ? "SAT_FINITE" : "SAT_PROPAGATE";
}
static const char *sub_name(SubnormalsMode s)
{
    return s == SubnormalsMode::SUBNORMALS ? "SUBNORMALS" : s == SubnormalsMode::NORMALS ? "NORMALS" : "EXTENDED_NORMALS";
}

static int failures = 0, checked = 0;

static void run_case(Cfg c)
{
    c.normal_cutoff = superfp_region_cutoffs(c.man_bits, c.exp_bits, c.normal_binades, c.bias).normal_cutoff;
    const BinaryKParamsT<double> bp = make_binaryK_params<double>(c.man_bits, c.exp_bits, c.bias, c.is_signed, c.sat,
                                                                  c.subm == SubnormalsMode::EXTENDED_NORMALS);
    const SuperfpParamsT<double> sp = make_superfp_params<double>(c.man_bits, c.exp_bits, c.normal_binades, c.bias, c.sat);
    Counts h;
    run_sweep(c, bp, sp, &h);
    ++checked;

    char head[160];
    if (c.is_binaryK)
        snprintf(head, sizeof head, "binaryK m%-2d e%-2d b%-5d %s %-13s %-16s r%d", c.man_bits, c.exp_bits, c.bias,
                 c.is_signed ? "signed  " : "unsigned", sat_name(c.sat), sub_name(c.subm), c.rand_bits);
    else
        snprintf(head, sizeof head, "superfp m%-2d e%-2d n%-4d b%-9d cutoff %-11d %s %-13s r%d", c.man_bits, c.exp_bits,
                 c.normal_binades, c.bias, sp.supernormal_cutoff, c.is_signed ? "signed  " : "unsigned",
                 sat_name(c.sat), c.rand_bits);

    bool ok = true;
    for (int m = 0; m < NMODE; ++m)
        if (h.bad[m] || h.nzero_shipped[m] || h.nzero_reference[m]) ok = false;
    printf("  %s  %s\n", head, ok ? "ok" : "FAIL");
    fflush(stdout);
    if (ok)
        return;
    ++failures;
    for (int m = 0; m < NMODE; ++m)
    {
        if (!h.bad[m] && !h.nzero_shipped[m] && !h.nzero_reference[m]) continue;
        printf("      %-3s  mismatches %-12llu  -0.0 shipped %-10llu  reference %-10llu", MODE_NAME[m], h.bad[m],
               h.nzero_shipped[m], h.nzero_reference[m]);
        if (h.bad[m])
        {
            const uint64_t w = input_word(c, h.first_bad[m]);
            const double x = value_of(w);
            const uint64_t rnd = splitmix64(w ^ 0xD1B54A32D192ED03u);
            printf("  e.g. 0x%016llx = %.17g: shipped 0x%016llx, reference 0x%016llx", (unsigned long long)w, x,
                   (unsigned long long)bits_of(shipped_cast(m, x, rnd, c, bp, sp)),
                   (unsigned long long)bits_of(reference_cast(m, x, rnd, c)));
        }
        printf("\n");
    }
}


// ===========================================================================
// Fixed point: cast_fixedpoint.h against reference_casts.h's ref_fixedpoint,
// in binary64, over the same structured inputs (grid_bits above puts them on
// the format's grid, ties and their neighbours included) and random words.
// ===========================================================================
CUDA_HOST_DEVICE_INLINE double fx_shipped(int mode, double x, uint64_t rnd, const FxCfg &c,
                                          const FixedPointParamsT<double> &p)
{
    switch (mode)
    {
    case 0: return cast_fixedpoint_nearest_even(x, c.is_signed, p);
    case 1: return cast_fixedpoint_nearest_away(x, c.is_signed, p);
    case 2: return cast_fixedpoint_up(x, c.is_signed, p);
    case 3: return cast_fixedpoint_down(x, c.is_signed, p);
    case 4: return cast_fixedpoint_zero(x, c.is_signed, p);
    case 5: return cast_fixedpoint_odd(x, c.is_signed, p);
    default: return cast_fixedpoint_stochastic(x, rnd, sr_spec(c.rand_bits, mode - 6), c.is_signed, p);
    }
}

CUDA_HOST_DEVICE_INLINE double fx_reference(int mode, double x, uint64_t rnd, const FxCfg &c)
{
    switch (mode)
    {
    case 0: return cast_fixedpoint_nearest_even(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 1: return cast_fixedpoint_nearest_away(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 2: return cast_fixedpoint_up(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 3: return cast_fixedpoint_down(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 4: return cast_fixedpoint_zero(x, c.wl, c.fl, c.is_signed, c.symmetric);
    case 5: return cast_fixedpoint_odd(x, c.wl, c.fl, c.is_signed, c.symmetric);
    default: return ref_fixedpoint(x, c.wl, c.fl, c.is_signed, c.symmetric, (RoundMode)mode, rnd, c.rand_bits);
    }
}

CUDA_HOST_DEVICE_INLINE void fx_one_input(uint64_t i, const FxCfg &c, const FixedPointParamsT<double> &p,
                                          unsigned long long *bad, unsigned long long *nzs,
                                          unsigned long long *nzr, uint32_t *first)
{
    const uint64_t w = input_word(c, i);
    const double x = value_of(w);
    const uint64_t rnd = splitmix64(w ^ 0xD1B54A32D192ED03u);
    for (int mode = 0; mode < NMODE; ++mode)
    {
        const uint64_t s = bits_of(fx_shipped(mode, x, rnd, c, p));
        const uint64_t r = bits_of(fx_reference(mode, x, rnd, c));
        if (s != r)
        {
            ++bad[mode];
            if ((uint32_t)i < first[mode])
                first[mode] = (uint32_t)i;
        }
        if (s == SIGN) ++nzs[mode];
        if (r == SIGN) ++nzr[mode];
    }
}

#if defined(__CUDACC__)
__global__ void fx_sweep_kernel(FxCfg c, uint64_t stride, FixedPointParamsT<double> p, Counts *out)
{
    const uint64_t base = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) * PER_THREAD;
    unsigned long long bad[NMODE] = {0}, nzs[NMODE] = {0}, nzr[NMODE] = {0};
    uint32_t first[NMODE];
    for (int m = 0; m < NMODE; ++m) first[m] = 0xFFFFFFFFu;
    for (int k = 0; k < PER_THREAD && (base + k) * stride < N_TOTAL; ++k)
        fx_one_input((base + k) * stride, c, p, bad, nzs, nzr, first);
    for (int m = 0; m < NMODE; ++m)
    {
        if (bad[m]) { atomicAdd(&out->bad[m], bad[m]); atomicMin(&out->first_bad[m], first[m]); }
        if (nzs[m]) atomicAdd(&out->nzero_shipped[m], nzs[m]);
        if (nzr[m]) atomicAdd(&out->nzero_reference[m], nzr[m]);
    }
}

static void fx_run_sweep(const FxCfg &c, const FixedPointParamsT<double> &p, Counts *h)
{
    static Counts *dc = nullptr;
    if (!dc) CHECK(cudaMalloc(&dc, sizeof(Counts)));
    Counts zero;
    memset(&zero, 0, sizeof zero);
    for (int m = 0; m < NMODE; ++m) zero.first_bad[m] = 0xFFFFFFFFu;
    CHECK(cudaMemcpy(dc, &zero, sizeof zero, cudaMemcpyHostToDevice));
    const uint64_t per_block = 256 * PER_THREAD, n = (N_TOTAL + stride - 1) / stride;
    fx_sweep_kernel<<<(unsigned)((n + per_block - 1) / per_block), 256>>>(c, stride, p, dc);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, dc, sizeof(Counts), cudaMemcpyDeviceToHost));
}
#else
static void fx_run_sweep(const FxCfg &c, const FixedPointParamsT<double> &p, Counts *h)
{
    memset(h, 0, sizeof *h);
    for (int m = 0; m < NMODE; ++m) h->first_bad[m] = 0xFFFFFFFFu;
#pragma omp parallel
    {
        unsigned long long b[NMODE] = {0}, s[NMODE] = {0}, r[NMODE] = {0};
        uint32_t f[NMODE];
        for (int m = 0; m < NMODE; ++m) f[m] = 0xFFFFFFFFu;
#pragma omp for schedule(static, 1 << 16)
        for (int64_t i = 0; i < (int64_t)((N_TOTAL + stride - 1) / stride); ++i)
            fx_one_input((uint64_t)i * stride, c, p, b, s, r, f);
#pragma omp critical
        for (int m = 0; m < NMODE; ++m)
        {
            h->bad[m] += b[m]; h->nzero_shipped[m] += s[m]; h->nzero_reference[m] += r[m];
            if (f[m] < h->first_bad[m]) h->first_bad[m] = f[m];
        }
    }
}
#endif

static void fx_run_case(const FxCfg &c)
{
    const FixedPointParamsT<double> p = make_fixedpoint_params<double>(c.wl, c.fl, c.is_signed, c.symmetric);
    Counts h;
    fx_run_sweep(c, p, &h);
    ++checked;
    char head[160];
    snprintf(head, sizeof head, "fixed   wl%-2d fl%-6d %s r%d", c.wl, c.fl,
             !c.is_signed ? "unsigned " : c.symmetric ? "symmetric" : "signed   ", c.rand_bits);
    bool ok = true;
    for (int m = 0; m < NMODE; ++m)
        if (h.bad[m] || h.nzero_shipped[m] || h.nzero_reference[m]) ok = false;
    printf("  %s  %s\n", head, ok ? "ok" : "FAIL");
    fflush(stdout);
    if (ok)
        return;
    ++failures;
    for (int m = 0; m < NMODE; ++m)
    {
        if (!h.bad[m] && !h.nzero_shipped[m] && !h.nzero_reference[m]) continue;
        printf("      %-3s  mismatches %-12llu  -0.0 shipped %-10llu  reference %-10llu", MODE_NAME[m], h.bad[m],
               h.nzero_shipped[m], h.nzero_reference[m]);
        if (h.bad[m])
        {
            const uint64_t w = input_word(c, h.first_bad[m]);
            const double x = value_of(w);
            const uint64_t rnd = splitmix64(w ^ 0xD1B54A32D192ED03u);
            printf("  e.g. 0x%016llx = %.17g: shipped %.17g, reference %.17g", (unsigned long long)w, x,
                   fx_shipped(m, x, rnd, c, p), fx_reference(m, x, rnd, c));
        }
        printf("\n");
    }
}

// The fixed-point edges in binary64: each end of the word (wl 1 unsigned, 2
// signed, mag_bits 52 at the carrier's mantissa), the three ranges, fl
// negative, zero, inside the word and past it, binary32's mantissa edge as an
// ordinary format, the step at 2^-1021 (the last one inside the carrier
// check's warning) and the top at 2^1023, and SR with no random bits and with
// all of them.
static void fx_short_list()
{
    const int before = checked, before_bad = failures;
    printf("short, fixed point\n");
    const FxCfg rows[] = {
        {4, 2, true, false, 49},     {4, 2, true, true, 49},     {4, 2, false, false, 48},
        {2, 0, true, false, 51},     {1, 0, false, false, 0},    {8, 6, true, true, 45},
        {8, -3, true, false, 30},    {16, 8, true, false, 37},   {24, 20, true, false, 29},
        {32, 30, false, false, 20},  {53, 49, true, false, 0},   {52, 50, false, false, 0},
        {53, 0, true, true, 0},      {40, 60, true, false, 10},  {8, 1021, true, false, 30},
        {8, -1016, true, false, 30}, {8, -1016, true, true, 30}, {16, -1007, false, false, 20},
    };
    for (const FxCfg &c : rows) fx_run_case(c);
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

// binary64's analogs of cast_all_modes_sweep.cu's fx_beyond_list: formats past
// the fixed-point checks as they stood when SR was rewritten (2026-11-09).
static void fx_beyond_list()
{
    const int before = checked, before_bad = failures;
    printf("beyond the checks, fixed point\n");
    const FxCfg rows[] = {
        {54, 50, true, false, 0}, {53, 50, false, false, 0}, {54, 0, true, true, 0},
        {53, 51, true, false, 1}, {32, 16, true, false, 22}, {8, -1017, true, true, 30},
        {7, -1017, false, false, 30},
    };
    for (const FxCfg &c : rows) fx_run_case(c);
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
}

static void fx_short_host_list()
{
    const FxCfg rows[] = {
        {4, 2, true, false, 49}, {53, 49, true, false, 0}, {16, -1007, false, false, 20}, {8, 1021, true, false, 30},
    };
    for (const FxCfg &c : rows) fx_run_case(c);
}

static Cfg bk(int man, int e, int bias, bool sgn, SaturationMode sat, SubnormalsMode sub)
{
    return Cfg{true, man, e, bias, 1, sgn, sat, sub, man <= 22 ? 30 : 52 - man, 0};
}

static Cfg sf(int man, int e, int nb, int bias, bool sgn, SaturationMode sat)
{
    return Cfg{false, man, e, bias, nb, sgn, sat, SubnormalsMode::SUBNORMALS, man <= 22 ? 30 : 52 - man, 0};
}

// The bias that puts a superfp format's smallest supernormal at 2^cutoff.
static int superfp_bias_for_cutoff(int man, int e, int nb, int cutoff)
{
    // normal_cutoff = 2^e - bias - nb, supernormal_cutoff = normal_cutoff - (2^e - nb) * 2^man + 1
    return (1 << e) - nb - (cutoff - 1 + ((1 << e) - nb) * (1 << man));
}

// Three formats on binary32's NORMALS/EXTENDED_NORMALS edge whose binary32
// casts cast_all_modes_sweep.cu's image64 mode found unfaithful -- a binary32
// subnormal read through its exponent field as a tie, a floor the binary32
// params leave dead, and a half-floor binary32 cannot hold -- so there the
// binary64 cast is the only one that is right, and this is what says so.
static void binary32_edges()
{
    const SaturationMode F = SaturationMode::SAT_FINITE, P = SaturationMode::SAT_PROPAGATE;
    printf("binary32's edge formats, in binary64\n");
    run_case(bk(0, 7, 127, false, F, SubnormalsMode::NORMALS));
    run_case(bk(0, 7, 127, true, P, SubnormalsMode::EXTENDED_NORMALS));
    run_case(bk(23, 7, 126, false, P, SubnormalsMode::EXTENDED_NORMALS));
}

// binaryK: each man_bits in four roles, rotating the saturation mode and the
// signedness through them -- the floor at binary64's faithful edge under
// SUBNORMALS (smallest value 2^-1021, then a binade and two below it as the
// rotation moves), the NORMALS and EXTENDED_NORMALS floors around their edges
// (2^-1022 and a binade either side), and the top at 2^1023 and a binade above.
// Exponent bits are as many as the reference's code count allows, eleven where
// man_bits leaves room. superfp: the supernormal floor at -1021, -1022, -1023
// and far below, with one normal binade and with all but one, and three wide
// mantissas whose supernormal region is out of binary64's reach altogether.
static void grid(bool dense)
{
    const SaturationMode sats[] = {SaturationMode::SAT_NONE, SaturationMode::SAT_FINITE, SaturationMode::SAT_PROPAGATE};
    const SubnormalsMode subs[] = {SubnormalsMode::SUBNORMALS, SubnormalsMode::NORMALS, SubnormalsMode::EXTENDED_NORMALS};
    const int mans[] = {0, 1, 2, 3, 7, 10, 23, 24, 31, 42, 51, 52};

    printf("binaryK -- %.3g inputs x 9 rounding modes per configuration\n", (double)N_TOTAL);
    for (int mi = 0; mi < 12; ++mi)
    {
        const int man = mans[mi];
        const int emax = 63 - man < 11 ? 63 - man : 11;
        const int e10 = emax < 10 ? emax : 10;
        const int reps = dense ? 3 : 1;
        for (int r = 0; r < reps; ++r)
        {
            const int k = dense ? r : mi % 3;
            run_case(bk(man, emax, 1022 - man + k, mi % 2 == 0, sats[(mi + r) % 3], SubnormalsMode::SUBNORMALS));
            run_case(bk(man, e10, 1022 + k, mi % 2 == 1, sats[(mi + r + 1) % 3], SubnormalsMode::NORMALS));
            run_case(bk(man, emax, 1021 + k, true, sats[(mi + r + 2) % 3], SubnormalsMode::EXTENDED_NORMALS));
            run_case(bk(man, emax, (1 << emax) - 1024 - (k % 2), false, sats[(mi + r) % 3], subs[(mi + r) % 3]));
        }
    }
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);

    const int before = checked, before_bad = failures;
    printf("superfp\n");
    const int superfp_mans[] = {0, 1, 3, 8, 20};
    const int cutoffs[] = {-1021, -1022, -1023, -1100};
    for (int mi = 0; mi < 5; ++mi)
        for (int ci = 0; ci < 4; ++ci)
            for (int wide = 0; wide < (dense ? 2 : 1); ++wide)
            {
                const int man = superfp_mans[mi];
                const int e = 2 + (mi + ci) % 3;
                const bool all_normal = dense ? wide == 1 : (mi + ci) % 2 == 1;
                const int nb = all_normal ? (1 << e) - 1 : 1;
                run_case(sf(man, e, nb, superfp_bias_for_cutoff(man, e, nb, cutoffs[ci]), ci % 2 == 0, sats[(mi + ci) % 3]));
            }
    run_case(sf(23, 11, 1, 1023, true, SaturationMode::SAT_NONE));
    run_case(sf(42, 11, 2, 1023, false, SaturationMode::SAT_FINITE));
    run_case(sf(52, 11, 1, 1024, true, SaturationMode::SAT_PROPAGATE));
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
    binary32_edges();
}

// Ten configurations for the host build, one per arm.
static void host_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE, P = SaturationMode::SAT_PROPAGATE;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    printf("host list -- %.3g inputs x 9 rounding modes per configuration\n", (double)N_TOTAL);
    run_case(bk(3, 11, 1022 - 3, true, O, S));
    run_case(bk(42, 11, 1023, true, F, N));
    run_case(bk(51, 11, 1022, false, P, E));
    run_case(bk(0, 11, 1022, true, O, S));
    run_case(bk(52, 11, 1024, true, F, S));
    run_case(bk(23, 10, 511, false, O, S));
    run_case(sf(3, 4, 1, superfp_bias_for_cutoff(3, 4, 1, -1021), true, O));
    run_case(sf(0, 3, 7, superfp_bias_for_cutoff(0, 3, 7, -1022), false, F));
    run_case(sf(20, 2, 3, superfp_bias_for_cutoff(20, 2, 3, -1100), true, P));
    run_case(sf(52, 11, 1, 1023, true, O));
}

// The per-change gate, as in cast_all_modes_sweep.cu: binary64's edges, each
// named for what it is there for, and a few ordinary formats. The grids above
// run once, at the end of a piece of work.
static void short_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE, P = SaturationMode::SAT_PROPAGATE;
    const SubnormalsMode S = SubnormalsMode::SUBNORMALS, N = SubnormalsMode::NORMALS,
                         E = SubnormalsMode::EXTENDED_NORMALS;
    printf("short, binaryK -- %.3g inputs x 9 rounding modes per configuration\n", (double)N_TOTAL);
    run_case(bk(3, 4, 8, true, O, S));        // the middle: Binary8p4, a 16-bit format,
    run_case(bk(10, 5, 15, true, F, S));
    run_case(bk(23, 8, 127, false, P, S));    // binary32's shape, and binary64's own
    run_case(bk(52, 11, 1023, true, F, S));
    run_case(bk(0, 11, 1022, true, O, S));    // man_bits 0: SUBNORMALS' floor at 2^-1021,
    run_case(bk(0, 11, 1022, false, F, N));   // P = 1's under NORMALS (binary32's 2^-125 bound, transposed),
    run_case(bk(0, 11, 1023, true, P, N));    // the tie at 2^-1023 a binade below it,
    run_case(bk(0, 11, 1023, true, O, E));    // min_exponent_store 0 with the floor NORMALS',
    run_case(bk(0, 2, -1023, true, O, E));    // and 2046, the floor past the carrier
    run_case(bk(1, 11, 1024, false, O, S));   // the reserved codes outnumbering the top binade's
    run_case(bk(31, 11, 991, true, P, S));    // SUBNORMALS' floor at 2^-1021 for wide mantissas
    run_case(bk(51, 11, 971, false, O, S));
    run_case(bk(52, 11, 970, true, F, S));
    run_case(bk(52, 11, 1023, false, P, N));  // NORMALS' at 2^-1022
    run_case(bk(3, 11, 1024, true, F, N));    // its underflow test dead
    run_case(bk(3, 11, 1022, true, F, E));    // EXTENDED_NORMALS' at 2^-1022,
    run_case(bk(51, 11, 1022, true, O, E));
    run_case(bk(52, 11, 1021, false, P, E));  // at P = 53 a binade higher (binary32's at P = 24, transposed),
    run_case(bk(52, 11, 1022, true, O, E));   // and the half-floor binary64 does not hold
    run_case(bk(3, 2, -1022, true, O, N));    // the floor on the top field
    run_case(bk(0, 11, 1024, true, F, S));    // the top at 2^1023,
    run_case(bk(52, 11, 1024, true, P, N));
    run_case(bk(3, 11, 1023, false, O, S));   // past it,
    run_case(bk(3, 4, 2000, true, O, S));     // and no finite normal at all
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked, failures);

    const int before = checked, before_bad = failures;
    printf("short, superfp\n");
    run_case(sf(3, 4, 1, 7, true, O));                                            // the middle
    run_case(sf(3, 4, 1, superfp_bias_for_cutoff(3, 4, 1, -1021), true, F));      // the supernormal floor at 2^-1021,
    run_case(sf(0, 3, 7, superfp_bias_for_cutoff(0, 3, 7, -1022), false, F));     // below it, man_bits 0 and one
    run_case(sf(8, 4, 15, superfp_bias_for_cutoff(8, 4, 15, -1023), true, P));    // supernormal binade left,
    run_case(sf(20, 2, 3, superfp_bias_for_cutoff(20, 2, 3, -1100), true, P));    // and far below
    // the supernormal floor between binary64's smallest subnormal and its
    // normals: a binary64 subnormal takes the arm below 2^c with a negative
    // shift (sh = 1073 + c - N, N = 30 here), and at or above 2^c goes up
    run_case(sf(3, 4, 1, superfp_bias_for_cutoff(3, 4, 1, -1060), true, O));     // sh = -17
    run_case(sf(0, 6, 2, superfp_bias_for_cutoff(0, 6, 2, -1072), false, F));    // sh = -29, the window's bottom
    run_case(sf(2, 4, 16, 7, true, O));                                           // no supernormal codes
    run_case(sf(23, 11, 1, 1023, true, O));                                       // wide mantissas
    run_case(sf(42, 11, 2, 1023, false, F));
    run_case(sf(52, 11, 1, 1024, true, P));
    run_case(sf(2, 3, 1, -1016, true, F));                                        // the top at 2^1023,
    run_case(sf(2, 3, 1, -1017, true, P));                                        // and past it
    printf("  %d configurations, %d with a mismatch or a -0.0\n", checked - before, failures - before_bad);
    binary32_edges();
    fx_short_list();
}

// One per kind of arm, for g++'s codegen.
static void short_host_list()
{
    const SaturationMode O = SaturationMode::SAT_NONE, F = SaturationMode::SAT_FINITE, P = SaturationMode::SAT_PROPAGATE;
    printf("short host list -- %.3g inputs x 9 rounding modes per configuration\n", (double)N_TOTAL);
    run_case(bk(3, 11, 1019, true, O, SubnormalsMode::SUBNORMALS));
    run_case(bk(0, 11, 1023, false, F, SubnormalsMode::EXTENDED_NORMALS));
    run_case(bk(52, 11, 1021, true, P, SubnormalsMode::EXTENDED_NORMALS));
    run_case(sf(3, 4, 1, superfp_bias_for_cutoff(3, 4, 1, -1021), true, O));
    run_case(sf(52, 11, 1, 1023, false, F));
    fx_short_host_list();
}

int main(int argc, char **argv)
{
    const std::string what = argc > 1 ? argv[1] : "spot";
    if (what == "quick")
        stride = 1021;
    if (what == "host") host_list();
    else if (what == "short") short_list();
    else if (what == "short-host") short_host_list();
    else if (what == "fixed") fx_short_list();       // the fixed-point rows alone,
    else if (what == "fixed-host") fx_short_host_list(); // for cast_fixedpoint.h only
    else if (what == "fixed-beyond") fx_beyond_list();
    else if (what == "edges") binary32_edges();
    else grid(what == "sweep");
    printf("\n%s: %d configurations checked, %d bad\n", failures ? "FAIL" : "PASS", checked, failures);
    return failures ? 1 : 0;
}
